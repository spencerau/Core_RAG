import os
from pathlib import Path
from typing import Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from ..utils.config_loader import load_config
from ..utils.docstore import get_docstore
from .chunking import AdvancedChunker
from .edit_metadata import MetadataExtractor
from .embedding import EmbeddingGenerator
from .json_extract import JSONContentExtractor
from .file_ingest import FileIngestor

try:
    from ..summary import SummaryIndexer, LLAMAINDEX_AVAILABLE
except ImportError:
    SummaryIndexer = None
    LLAMAINDEX_AVAILABLE = False


class UnifiedIngestion:
    def __init__(self, base_dir: str = None, collection_name: str = None, config_path: str = None):
        self.config = load_config(config_path) if config_path else load_config()
        self.client = QdrantClient(
            host=self.config['qdrant']['host'],
            port=self.config['qdrant']['port'],
            timeout=self.config['qdrant']['timeout']
        )
        self.base_dir = base_dir
        self.collection_name = collection_name
        self.embedding_gen = EmbeddingGenerator(self.config)
        self.chunker = AdvancedChunker(self.config.get('chunker', {}))
        self.json_extractor = JSONContentExtractor(self.config)
        self.metadata_extractor = MetadataExtractor()
        self.docstore = get_docstore()
        
        summary_config = self.config.get('summary', {})
        self.enable_summaries = summary_config.get('enable_summary_gating', False)
        if self.enable_summaries and LLAMAINDEX_AVAILABLE and SummaryIndexer:
            try:
                self.summary_indexer = SummaryIndexer(base_dir=base_dir)
                print("Summary indexer initialized")
            except Exception as e:
                print(f"Warning: Could not initialize summary indexer: {e}")
                self.summary_indexer = None
        else:
            self.summary_indexer = None
        
        self._ensure_collections_exist()
        self.file_ingestor = FileIngestor(
            client=self.client, config=self.config, embedding_gen=self.embedding_gen,
            chunker=self.chunker, json_extractor=self.json_extractor, docstore=self.docstore,
            metadata_extractor=self.metadata_extractor, base_dir=self.base_dir,
            collection_name=self.collection_name
        )
    
    def _ensure_collections_exist(self):
        for collection_name in self.config['qdrant']['collections'].values():
            try:
                self.client.get_collection(collection_name)
                print(f"Collection '{collection_name}' already exists")
            except Exception:
                vector_size = self.embedding_gen.get_vector_size()
                print(f"Creating collection '{collection_name}' with vector size {vector_size}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
    
    def ingest_file(self, file_path: str) -> bool:
        success = self.file_ingestor.ingest_file(file_path)
        
        if success and self.summary_indexer and file_path.endswith(('.md', '.txt')):
            try:
                collection_name = self.collection_name or self.file_ingestor.get_last_used_collection() or list(self.config['qdrant']['collections'].values())[0]
                self.summary_indexer.index_document(file_path, collection_name)
            except Exception as e:
                print(f"Warning: Could not generate summary for {file_path}: {e}")
        
        return success
    
    def ingest_json_file(self, file_path: str) -> bool:
        return self.file_ingestor.ingest_json_file(file_path)
    
    def ingest_markdown_file(self, file_path: str) -> bool:
        return self.file_ingestor.ingest_markdown_file(file_path)
    
    def ingest_pdf_file(self, file_path: str) -> bool:
        return self.file_ingestor.ingest_pdf_file(file_path)
    
    def ingest_directory(self, directory: str, file_extensions: List[str] = None) -> Dict:
        if file_extensions is None:
            file_extensions = ['.pdf', '.json', '.md']
        if self.base_dir is None and self.collection_name is None:
            self.base_dir = str(directory)
            self.file_ingestor.base_dir = self.base_dir
            if self.summary_indexer:
                self.summary_indexer.base_dir = self.base_dir
        
        stats = {'total_files': 0, 'success_files': 0, 'failed_files': 0, 'total_chunks': 0, 'ingested_chunks': 0, 'collections_used': set()}
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return stats
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                if 'readme' in file_path.name.lower() or file_path.name.startswith('._'):
                    continue
                stats['total_files'] += 1
                result = self.ingest_file(str(file_path))
                if result:
                    stats['success_files'] += 1
                    if hasattr(self.file_ingestor, '_last_chunk_stats'):
                        chunk_stats = self.file_ingestor._last_chunk_stats
                        stats['total_chunks'] += chunk_stats.get('total', 0)
                        stats['ingested_chunks'] += chunk_stats.get('ingested', 0)
                else:
                    stats['failed_files'] += 1
        return stats
    
    def bulk_ingest(self, data_directories: List[str]) -> Dict:
        total = {'total_files': 0, 'success_files': 0, 'failed_files': 0, 'collections_used': set()}
        for directory in data_directories:
            print(f"\n=== Ingesting from {directory} ===")
            stats = self.ingest_directory(directory)
            total['total_files'] += stats['total_files']
            total['success_files'] += stats['success_files']
            total['failed_files'] += stats['failed_files']
            total['collections_used'].update(stats['collections_used'])
            print(f"Directory stats: {stats['success_files']}/{stats['total_files']} files successful")
        return total
    
    def print_collection_summary(self):
        print("\n=== Collection Summary ===")
        for collection_name in self.config['qdrant']['collections'].values():
            try:
                info = self.client.get_collection(collection_name)
                print(f"{collection_name}: {info.points_count} documents")
            except Exception as e:
                print(f"{collection_name}: Error - {e}")
    
    def clear_collections(self):
        for collection_name in self.config['qdrant']['collections'].values():
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted collection '{collection_name}'")
            except Exception as e:
                print(f"Error deleting collection '{collection_name}': {e}")
        self._ensure_collections_exist()


def main():
    print("Starting unified RAG ingestion...")
    ingestion = UnifiedIngestion()
    config = load_config()
    data_dirs = [p for k, p in config.get('data', {}).items() if isinstance(p, str) and os.path.exists(p)]
    for k, p in config.get('data', {}).items():
        if isinstance(p, str) and os.path.exists(p):
            print(f"Added {k} directory: {p}")
    if not data_dirs:
        print("No data directories found to process!")
        return
    stats = ingestion.bulk_ingest(data_dirs)
    print(f"\n=== Final Results ===")
    print(f"Total: {stats['total_files']}, Success: {stats['success_files']}, Failed: {stats['failed_files']}")
    ingestion.print_collection_summary()


if __name__ == "__main__":
    main()
