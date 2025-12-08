import os
import hashlib
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from textwrap import dedent
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from ..utils.config_loader import load_config
from ..utils.ollama_api import get_ollama_api
from ..utils.doc_id import generate_doc_id, get_normalized_path
from ..utils.docstore import get_docstore
from ..utils.text_preprocessing import preprocess_for_embedding


SUMMARY_COLLECTION_SUFFIX = "_summaries"


class SummaryIndexer:
    def __init__(self, base_dir: str = None):
        self.config = load_config()
        self.client = QdrantClient(
            host=self.config['qdrant']['host'],
            port=self.config['qdrant']['port'],
            timeout=self.config['qdrant']['timeout']
        )
        self.embedding_model = self.config['embedding']['model']
        self.ollama_api = get_ollama_api()
        self.docstore = get_docstore()
        self.base_dir = base_dir
        
        summary_config = self.config.get('summary', {})
        self.summary_word_count = summary_config.get('word_count', 175)
        self.embed_summaries = summary_config.get('embed_summaries', True)
        
        llm_config = self.config.get('llm', {})
        self.llm = Ollama(
            model=llm_config.get('primary_model', 'llama3.2'),
            base_url=f"http://{self.config.get('embedding', {}).get('ollama_host', 'localhost')}:11434",
            request_timeout=llm_config.get('timeout', 300)
        )
        self._ensure_summary_collections()
    
    def _get_summary_collection_name(self, collection_name: str) -> str:
        if collection_name.endswith(SUMMARY_COLLECTION_SUFFIX):
            return collection_name
        return f"{collection_name}{SUMMARY_COLLECTION_SUFFIX}"
    
    def _ensure_summary_collections(self):
        for collection_name in self.config['qdrant']['collections'].values():
            summary_collection = self._get_summary_collection_name(collection_name)
            try:
                self.client.get_collection(summary_collection)
            except Exception:
                test_embedding = self._get_embedding("test")
                if not test_embedding:
                    vector_size = 1024 if 'bge-m3' in self.embedding_model else 768
                else:
                    vector_size = len(test_embedding)
                self.client.create_collection(
                    collection_name=summary_collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print(f"Created summary collection '{summary_collection}'")
    
    def _get_embedding(self, text: str) -> List[float]:
        try:
            processed = preprocess_for_embedding([text], 'document', self.config.get('embedding', {}))[0]
            embedding = self.ollama_api.get_embeddings(model=self.embedding_model, prompt=processed)
            return embedding if embedding else []
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def generate_summary(self, text: str, title: str = None) -> str:
        prompt = dedent(f"""
            Summarize the following document in approximately {self.summary_word_count} words.
            Focus on the key topics, main points, and important details.
            
            {"Title: " + title if title else ""}
            
            Document:
            {text[:8000]}
            
            Summary:
        """).strip()
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""
    
    def index_document(self, file_path: str, collection_name: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if not text or len(text.strip()) < 50:
                return False
            
            doc_id = generate_doc_id(file_path, self.base_dir)
            source_path = get_normalized_path(file_path, self.base_dir)
            title = Path(file_path).stem
            
            if file_path.endswith('.md'):
                for line in text.split('\n')[:5]:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
            
            summary = self.generate_summary(text, title)
            if not summary:
                return False
            
            file_stat = os.stat(file_path)
            last_modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            summary_collection = self._get_summary_collection_name(collection_name)
            
            payload = {
                'doc_id': doc_id, 'source_path': source_path, 'collection_name': collection_name,
                'title': title, 'last_modified': last_modified, 'summary': summary
            }
            
            if self.embed_summaries:
                embedding = self._get_embedding(summary)
                if not embedding:
                    return False
                # Use deterministic ID based on doc_id to prevent duplicates
                summary_id = hashlib.sha256(f"{doc_id}:summary".encode()).hexdigest()[:32]
                point = PointStruct(id=summary_id, vector=embedding, payload=payload)
                self.client.upsert(collection_name=summary_collection, points=[point])
                print(f"Ingested summary for '{title}' into collection '{summary_collection}'")
            return True
        except Exception as e:
            print(f"Error indexing document summary {file_path}: {e}")
            return False
    
    def index_directory(self, directory: str, collection_name: str, file_extensions: List[str] = None) -> Dict:
        if file_extensions is None:
            file_extensions = ['.md', '.txt']
        if self.base_dir is None:
            self.base_dir = str(directory)
        
        stats = {'total_files': 0, 'success_files': 0, 'failed_files': 0}
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return stats
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                if 'readme' in file_path.name.lower() or file_path.name.startswith('._'):
                    continue
                stats['total_files'] += 1
                if self.index_document(str(file_path), collection_name):
                    stats['success_files'] += 1
                else:
                    stats['failed_files'] += 1
        return stats


def ingest_summaries(data_directories: List[str] = None, collection_name: str = None) -> Dict:
    config = load_config()
    if data_directories is None:
        data_directories = [p for k, p in config.get('data', {}).items() if isinstance(p, str) and os.path.exists(p)]
    if collection_name is None:
        collections = config['qdrant']['collections']
        collection_name = list(collections.values())[0] if collections else 'default'
    
    indexer = SummaryIndexer()
    total = {'total_files': 0, 'success_files': 0, 'failed_files': 0}
    
    for directory in data_directories:
        print(f"\n=== Indexing summaries from {directory} ===")
        stats = indexer.index_directory(directory, collection_name)
        total['total_files'] += stats['total_files']
        total['success_files'] += stats['success_files']
        total['failed_files'] += stats['failed_files']
    return total
