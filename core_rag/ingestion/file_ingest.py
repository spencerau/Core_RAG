import json
import hashlib
from pathlib import Path
from typing import Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from ..utils.docstore import DocStore
from .content_extract import extract_content
from .chunking import AdvancedChunker
from .embedding import EmbeddingGenerator
from .json_extract import JSONContentExtractor
from .ingest_helpers import prepare_doc_metadata, get_collection_name, extract_markdown_title


class FileIngestor:
    def __init__(self, client: QdrantClient, config: dict, embedding_gen: EmbeddingGenerator,
                 chunker: AdvancedChunker, json_extractor: JSONContentExtractor,
                 docstore: DocStore, metadata_extractor, base_dir: str = None):
        self.client = client
        self.config = config
        self.embedding_gen = embedding_gen
        self.chunker = chunker
        self.json_extractor = json_extractor
        self.docstore = docstore
        self.metadata_extractor = metadata_extractor
        self.base_dir = base_dir
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID to prevent duplicates"""
        combined = f"{doc_id}:chunk:{chunk_index}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def ingest_file(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        if ext == '.json':
            return self.ingest_json_file(file_path)
        elif ext == '.pdf':
            return self.ingest_pdf_file(file_path)
        elif ext == '.md':
            return self.ingest_markdown_file(file_path)
        print(f"Unsupported file type: {ext}")
        return False
    
    def ingest_json_file(self, file_path: str) -> bool:
        try:
            print(f"Ingesting JSON: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
                json_data = json.loads(json_content)
            
            file_metadata = self.metadata_extractor.extract_metadata_from_path(file_path)
            content_items = self.json_extractor.extract_content_for_embedding(json_data)
            if not content_items:
                print(f"  Warning: No content extracted from {file_path}")
                return False
            
            collection = get_collection_name(file_metadata['DocumentType'], self.config)
            doc_id, source_path, last_modified = prepare_doc_metadata(file_path, self.base_dir)
            
            self.docstore.put(doc_id, json_content, {
                'source_path': source_path, 'title': file_metadata.get('title', Path(file_path).stem),
                'collection_name': collection, 'last_modified': last_modified, 'content_type': 'json'
            })
            
            points = []
            for idx, item in enumerate(content_items):
                embedding = self.embedding_gen.get_embedding(item['text'])
                if not embedding:
                    continue
                meta = {**file_metadata, 'doc_id': doc_id, 'source_path': source_path, 'chunk_index': idx,
                        'section_type': item.get('section_type', 'unknown'),
                        'section_name': item.get('section_name', 'Unknown Section'),
                        'section_classification': item.get('section_classification', 'Program Requirements'),
                        'content_type': 'structured_json', 'chunk_text': item['text'], 'total_chunks': len(content_items)}
                chunk_id = self._generate_chunk_id(doc_id, idx)
                points.append(PointStruct(id=chunk_id, vector=embedding, payload=meta))
            
            return self._upsert_points(points, collection, file_path)
        except Exception as e:
            print(f"Error ingesting JSON {file_path}: {e}")
            return False
    
    def ingest_markdown_file(self, file_path: str) -> bool:
        try:
            print(f"Ingesting Markdown: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if not text or len(text.strip()) < 10:
                print(f"  Warning: No content in {file_path}")
                return False
            
            file_metadata = self.metadata_extractor.extract_metadata_from_path(file_path)
            collection = get_collection_name(file_metadata.get('DocumentType', 'default'), self.config)
            doc_id, source_path, last_modified = prepare_doc_metadata(file_path, self.base_dir)
            title = extract_markdown_title(text, file_path)
            
            self.docstore.put(doc_id, text, {
                'source_path': source_path, 'title': title, 'collection_name': collection,
                'last_modified': last_modified, 'content_type': 'markdown'
            })
            
            chunk_data = self.chunker.chunk_text(text, {**file_metadata, 'doc_id': doc_id, 'source_path': source_path})
            points = []
            for idx, (chunk_text, chunk_meta) in enumerate(chunk_data):
                embedding = self.embedding_gen.get_embedding(chunk_text)
                if not embedding:
                    continue
                chunk_meta.update({'doc_id': doc_id, 'source_path': source_path, 'chunk_index': idx,
                                   'chunk_text': chunk_text, 'total_chunks': len(chunk_data)})
                chunk_id = self._generate_chunk_id(doc_id, idx)
                points.append(PointStruct(id=chunk_id, vector=embedding, payload=chunk_meta))
            
            return self._upsert_points(points, collection, file_path)
        except Exception as e:
            print(f"Error ingesting markdown {file_path}: {e}")
            return False
    
    def ingest_pdf_file(self, file_path: str) -> bool:
        try:
            print(f"Ingesting PDF: {file_path}")
            text, tika_metadata = extract_content(file_path)
            if not text or len(text.strip()) < 10:
                print(f"  Warning: No content extracted from {file_path}")
                return False
            
            file_metadata = self.metadata_extractor.extract_metadata_from_path(file_path)
            combined_metadata = {**tika_metadata, **file_metadata}
            collection = get_collection_name(file_metadata['DocumentType'], self.config)
            doc_id, source_path, last_modified = prepare_doc_metadata(file_path, self.base_dir)
            
            self.docstore.put(doc_id, text, {
                'source_path': source_path, 'title': file_metadata.get('title', Path(file_path).stem),
                'collection_name': collection, 'last_modified': last_modified, 'content_type': 'pdf'
            })
            
            combined_metadata['doc_id'] = doc_id
            combined_metadata['source_path'] = source_path
            chunk_data = self.chunker.chunk_text(text, combined_metadata)
            
            points = []
            for idx, (chunk_text, chunk_meta) in enumerate(chunk_data):
                embedding = self.embedding_gen.get_embedding(chunk_text)
                if not embedding:
                    continue
                chunk_meta.update({'doc_id': doc_id, 'source_path': source_path, 'chunk_index': idx,
                                   'chunk_text': chunk_text, 'total_chunks': len(chunk_data)})
                chunk_id = self._generate_chunk_id(doc_id, idx)
                points.append(PointStruct(id=chunk_id, vector=embedding, payload=chunk_meta))
            
            return self._upsert_points(points, collection, file_path)
        except Exception as e:
            print(f"Error ingesting PDF {file_path}: {e}")
            return False
    
    def _upsert_points(self, points: List[PointStruct], collection: str, file_path: str) -> bool:
        if points:
            self.client.upsert(collection_name=collection, points=points)
            print(f"Ingested {len(points)} chunks into collection '{collection}'")
            return True
        print(f"No valid chunks created for {file_path}")
        return False
