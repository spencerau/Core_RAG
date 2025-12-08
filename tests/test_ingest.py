import pytest
from core_rag.ingestion.ingest import UnifiedIngestion
from qdrant_client import QdrantClient


@pytest.fixture(scope="module")
def ingestion():
    """Create an ingestion instance once per test module"""
    ing = UnifiedIngestion()
    
    try:
        for collection in ['main_collection', 'main_collection_summaries', 'docstore']:
            try:
                ing.client.delete_collection(collection)
                print(f"Cleared collection: {collection}")
            except Exception:
                pass
        
        ing._ensure_collections_exist()
        ing.docstore._ensure_collection()
        
        if ing.summary_indexer:
            ing.summary_indexer._ensure_summary_collections()
    except Exception as e:
        print(f"Warning: Could not clear collections: {e}")
    
    return ing


def test_ingestion_initialization(ingestion):
    """Test that ingestion initializes properly"""
    assert ingestion is not None
    assert ingestion.config is not None
    assert ingestion.client is not None
    assert ingestion.embedding_gen is not None


def test_ingest_directory(ingestion):
    """Test ingesting documents from a directory"""
    result = ingestion.ingest_directory("data/main_collection")
    
    assert result is not None
    assert 'total_files' in result
    assert 'success_files' in result
    assert 'failed_files' in result
    assert result['total_files'] > 0
    assert result['success_files'] > 0


def test_collection_has_documents(ingestion):
    """Test that documents were successfully added to the collection"""
    info = ingestion.client.get_collection("main_collection")
    assert info.points_count > 0
    print(f"\n✓ Collection 'main_collection' has {info.points_count} documents")

