import pytest
from core_rag.ingestion.ingest import UnifiedIngestion
from core_rag.utils.doc_id import generate_doc_id, get_normalized_path
from qdrant_client import QdrantClient


@pytest.fixture(scope="module")
def ingestion():
    """Create an ingestion instance once per test module"""
    ing = UnifiedIngestion(base_dir="data")
    
    try:
        all_collections = ing.client.get_collections().collections
        for collection in all_collections:
            try:
                ing.client.delete_collection(collection.name)
                print(f"Deleted collection: {collection.name}")
            except Exception as e:
                print(f"Could not delete {collection.name}: {e}")
    except Exception as e:
        print(f"Warning: Could not list collections: {e}")
    
    try:
        ing._ensure_collections_exist()
        ing.docstore._ensure_collection()
        
        if ing.summary_indexer:
            ing.summary_indexer._ensure_summary_collections()
    except Exception as e:
        print(f"Warning: Could not create collections: {e}")
    
    return ing


def test_ingestion_initialization(ingestion):
    """Test that ingestion initializes properly"""
    assert ingestion is not None
    assert ingestion.config is not None
    assert ingestion.client is not None
    assert ingestion.embedding_gen is not None


def test_doc_id_generation_consistency():
    """Test that doc_id generation is consistent with the same base_dir"""
    base_dir = "data"
    file_path1 = "data/main_collection/krabby_patty_instructions.md"
    file_path2 = "data/main_collection_2/PDF Powerpoint.PDF The Art of Job Coaching.md"
    
    doc_id1 = generate_doc_id(file_path1, base_dir)
    doc_id2 = generate_doc_id(file_path1, base_dir)
    assert doc_id1 == doc_id2, "Same file with same base_dir should generate same doc_id"
    
    normalized_path1 = get_normalized_path(file_path1, base_dir)
    assert normalized_path1 == "main_collection/krabby_patty_instructions.md"
    
    normalized_path2 = get_normalized_path(file_path2, base_dir)
    assert normalized_path2 == "main_collection_2/PDF Powerpoint.PDF The Art of Job Coaching.md"
    
    doc_id3 = generate_doc_id(file_path1, None)
    assert doc_id1 != doc_id3, "Different base_dir should generate different doc_id"
    
    print(f"doc_id generation is consistent")



def test_ingest_directory(ingestion):
    """Test ingesting documents from multiple directories into different collections"""
    result1 = ingestion.ingest_directory("data/main_collection")
    
    assert result1 is not None
    assert 'total_files' in result1
    assert 'success_files' in result1
    assert result1['total_files'] > 0
    assert result1['success_files'] > 0
    if 'total_chunks' in result1 and result1['total_chunks'] > 0:
        assert result1['ingested_chunks'] == result1['total_chunks'], \
            f"Only {result1['ingested_chunks']} out of {result1['total_chunks']} chunks were ingested"
    
    result2 = ingestion.ingest_directory("data/main_collection_2")
    
    assert result2 is not None
    assert 'total_files' in result2
    assert 'success_files' in result2
    assert result2['total_files'] > 0
    assert result2['success_files'] > 0
    if 'total_chunks' in result2 and result2['total_chunks'] > 0:
        assert result2['ingested_chunks'] == result2['total_chunks'], \
            f"Only {result2['ingested_chunks']} out of {result2['total_chunks']} chunks were ingested"
    
    if ingestion.summary_indexer:
        print("\nRegenerating summaries after ingestion...")
        ingestion.summary_indexer.index_directory("data/main_collection", "main_collection", [".md", ".pdf"])
        ingestion.summary_indexer.index_directory("data/main_collection_2", "main_collection_2", [".md", ".pdf"])


def test_collection_has_documents(ingestion):
    info1 = ingestion.client.get_collection("main_collection")
    assert info1.points_count > 0
    print(f"\nCollection 'main_collection' has {info1.points_count} chunks")
    
    info2 = ingestion.client.get_collection("main_collection_2")
    assert info2.points_count > 0
    print(f"Collection 'main_collection_2' has {info2.points_count} chunks")


def test_doc_id_consistency_between_chunks_and_docstore(ingestion):
    """Test that doc_ids in chunks match doc_ids in docstore"""
    client = ingestion.client
    docstore = ingestion.docstore
    
    for collection_name in ["main_collection", "main_collection_2"]:
        points = client.scroll(collection_name=collection_name, limit=10, with_payload=True)
        
        doc_ids_from_chunks = set()
        for point in points[0]:
            doc_id = point.payload.get('doc_id')
            assert doc_id is not None, f"Chunk missing doc_id in {collection_name}"
            doc_ids_from_chunks.add(doc_id)
        
        for doc_id in doc_ids_from_chunks:
            doc = docstore.get(doc_id)
            assert doc is not None, f"doc_id {doc_id} from {collection_name} not found in docstore"
            assert 'text' in doc, f"Document {doc_id} in docstore missing text field"
            assert len(doc['text']) > 0, f"Document {doc_id} in docstore has empty text"
        
        print(f"All {len(doc_ids_from_chunks)} doc_ids from {collection_name} exist in docstore")


def test_doc_id_consistency_between_summaries_and_docstore(ingestion):
    """Test that doc_ids in summary collections match doc_ids in docstore"""
    if not ingestion.summary_indexer:
        pytest.skip("Summary indexer not available")
    
    client = ingestion.client
    docstore = ingestion.docstore
    
    for collection_name in ["main_collection", "main_collection_2"]:
        summary_collection = f"{collection_name}_summaries"
        
        try:
            summary_points = client.scroll(collection_name=summary_collection, limit=10, with_payload=True)
        except Exception as e:
            pytest.skip(f"Summary collection {summary_collection} not available: {e}")
            continue
        
        doc_ids_from_summaries = set()
        for point in summary_points[0]:
            doc_id = point.payload.get('doc_id')
            assert doc_id is not None, f"Summary missing doc_id in {summary_collection}"
            doc_ids_from_summaries.add(doc_id)
        
        for doc_id in doc_ids_from_summaries:
            doc = docstore.get(doc_id)
            assert doc is not None, f"doc_id {doc_id} from {summary_collection} not found in docstore"
            assert 'text' in doc, f"Document {doc_id} in docstore missing text field"
            assert len(doc['text']) > 0, f"Document {doc_id} in docstore has empty text"
        
        print(f"All {len(doc_ids_from_summaries)} doc_ids from {summary_collection} exist in docstore")


def test_doc_id_consistency_across_all_three(ingestion):
    """Test that the same document has the same doc_id in chunks, summaries, and docstore"""
    if not ingestion.summary_indexer:
        pytest.skip("Summary indexer not available")
    
    client = ingestion.client
    docstore = ingestion.docstore
    
    for collection_name in ["main_collection", "main_collection_2"]:
        summary_collection = f"{collection_name}_summaries"
        
        try:
            chunk_points = client.scroll(collection_name=collection_name, limit=10, with_payload=True)
            summary_points = client.scroll(collection_name=summary_collection, limit=10, with_payload=True)
        except Exception as e:
            pytest.skip(f"Collections not available: {e}")
            continue
        
        doc_ids_from_chunks = {}
        for point in chunk_points[0]:
            doc_id = point.payload.get('doc_id')
            source_path = point.payload.get('source_path')
            if source_path:
                doc_ids_from_chunks[source_path] = doc_id
        
        doc_ids_from_summaries = {}
        for point in summary_points[0]:
            doc_id = point.payload.get('doc_id')
            source_path = point.payload.get('source_path')
            if source_path:
                doc_ids_from_summaries[source_path] = doc_id
        
        for source_path, chunk_doc_id in doc_ids_from_chunks.items():
            if source_path in doc_ids_from_summaries:
                summary_doc_id = doc_ids_from_summaries[source_path]
                assert chunk_doc_id == summary_doc_id, \
                    f"doc_id mismatch for {source_path}: chunk={chunk_doc_id}, summary={summary_doc_id}"
                
                doc = docstore.get(chunk_doc_id)
                assert doc is not None, f"doc_id {chunk_doc_id} not found in docstore"
        
        print(f"doc_ids are consistent across chunks, summaries, and docstore for {collection_name}")
