import pytest
from core_rag.ingestion.ingest import UnifiedIngestion
from core_rag.utils.doc_id import generate_doc_id, get_normalized_path

COLLECTIONS = ["job_coaching", "major_catalogs", "recipes"]


@pytest.fixture(scope="module")
def ingestion():
    """Create an ingestion instance, wipe all existing collections, then recreate them."""
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


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_ingestion_initialization(ingestion):
    assert ingestion is not None
    assert ingestion.config is not None
    assert ingestion.client is not None
    assert ingestion.embedding_gen is not None


# ---------------------------------------------------------------------------
# doc_id helpers
# ---------------------------------------------------------------------------

def test_doc_id_generation_consistency():
    """doc_id must be stable for the same file+base_dir and differ when base_dir changes."""
    base_dir = "data"
    file_path_a = "data/recipes/krabby_patty_instructions.md"
    file_path_b = "data/job_coaching/art_of_job_coaching.md"
    file_path_c = "data/major_catalogs/2025_cs.md"

    # Same path → same id
    assert generate_doc_id(file_path_a, base_dir) == generate_doc_id(file_path_a, base_dir)

    # Normalized paths strip the base prefix
    assert get_normalized_path(file_path_a, base_dir) == "recipes/krabby_patty_instructions.md"
    assert get_normalized_path(file_path_b, base_dir) == "job_coaching/art_of_job_coaching.md"
    assert get_normalized_path(file_path_c, base_dir) == "major_catalogs/2025_cs.md"

    # Different base_dir → different id
    assert generate_doc_id(file_path_a, base_dir) != generate_doc_id(file_path_a, None)

    print("doc_id generation is consistent")


def test_doc_ids_differ_across_collections():
    """Files in different collections should always get different doc_ids."""
    base_dir = "data"
    ids = [
        generate_doc_id(f"data/{col}/sample.md", base_dir)
        for col in COLLECTIONS
    ]
    assert len(ids) == len(set(ids)), "doc_ids should be unique across collections"
    print("doc_ids are unique across collections")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def test_ingest_directory(ingestion):
    """Ingest all three collections and verify every chunk was stored."""
    for collection in COLLECTIONS:
        result = ingestion.ingest_directory(f"data/{collection}")

        assert result is not None, f"No result returned for {collection}"
        assert result.get('total_files', 0) > 0, f"No files found in data/{collection}"
        assert result.get('success_files', 0) > 0, f"No files successfully ingested from data/{collection}"

        total = result.get('total_chunks', 0)
        ingested = result.get('ingested_chunks', 0)
        if total > 0:
            assert ingested == total, \
                f"{collection}: only {ingested}/{total} chunks ingested"

        print(f"\n{collection}: {result['success_files']} file(s), {ingested} chunk(s)")

    if ingestion.summary_indexer:
        print("\nRegenerating summaries after ingestion...")
        coll_cfg = ingestion.config.get('collection_config', {})
        summary_collections = [c for c in COLLECTIONS if coll_cfg.get(c, {}).get('summary_enabled', False)]
        for collection in summary_collections:
            ingestion.summary_indexer.index_directory(
                f"data/{collection}", collection, [".md", ".pdf"]
            )


def test_collection_has_documents(ingestion):
    """Every collection must be non-empty after ingestion."""
    for collection in COLLECTIONS:
        info = ingestion.client.get_collection(collection)
        assert info.points_count > 0, f"Collection '{collection}' is empty after ingestion"
        print(f"\n✓ '{collection}': {info.points_count} chunks")


def test_collections_have_distinct_content(ingestion):
    """Verify each collection holds different documents (no cross-contamination)."""
    all_doc_ids = {}
    for collection in COLLECTIONS:
        points, _ = ingestion.client.scroll(
            collection_name=collection, limit=100, with_payload=True
        )
        doc_ids = {p.payload.get('doc_id') for p in points if p.payload.get('doc_id')}
        assert len(doc_ids) > 0, f"No doc_ids found in {collection}"
        all_doc_ids[collection] = doc_ids

    # No doc_id should appear in more than one collection
    for i, col_a in enumerate(COLLECTIONS):
        for col_b in COLLECTIONS[i + 1:]:
            overlap = all_doc_ids[col_a] & all_doc_ids[col_b]
            assert not overlap, \
                f"doc_ids appear in both {col_a} and {col_b}: {overlap}"

    print("\nCollections contain distinct documents")


# ---------------------------------------------------------------------------
# Docstore consistency
# ---------------------------------------------------------------------------

def test_doc_id_consistency_between_chunks_and_docstore(ingestion):
    """Every doc_id referenced by a chunk must exist in the docstore."""
    for collection in COLLECTIONS:
        points, _ = ingestion.client.scroll(
            collection_name=collection, limit=10, with_payload=True
        )
        doc_ids = set()
        for point in points:
            doc_id = point.payload.get('doc_id')
            assert doc_id is not None, f"Chunk missing doc_id in {collection}"
            doc_ids.add(doc_id)

        for doc_id in doc_ids:
            doc = ingestion.docstore.get(doc_id)
            assert doc is not None, f"doc_id {doc_id} from {collection} not found in docstore"
            assert doc.get('text'), f"Document {doc_id} in docstore has empty text"

        print(f"{len(doc_ids)} doc_ids from {collection} verified in docstore")


def test_doc_id_consistency_between_summaries_and_docstore(ingestion):
    """Every doc_id referenced by a summary must exist in the docstore."""
    if not ingestion.summary_indexer:
        pytest.skip("Summary indexer not available")

    coll_cfg = ingestion.config.get('collection_config', {})
    summary_collections = [c for c in COLLECTIONS if coll_cfg.get(c, {}).get('summary_enabled', False)]
    if not summary_collections:
        pytest.skip("No collections have summary_enabled: true")

    for collection in summary_collections:
        summary_collection = f"{collection}_summaries"
        try:
            points, _ = ingestion.client.scroll(
                collection_name=summary_collection, limit=10, with_payload=True
            )
        except Exception as e:
            pytest.skip(f"Summary collection {summary_collection} not available: {e}")

        doc_ids = set()
        for point in points:
            doc_id = point.payload.get('doc_id')
            assert doc_id is not None, f"Summary missing doc_id in {summary_collection}"
            doc_ids.add(doc_id)

        for doc_id in doc_ids:
            doc = ingestion.docstore.get(doc_id)
            assert doc is not None, f"doc_id {doc_id} from {summary_collection} not found in docstore"
            assert doc.get('text'), f"Document {doc_id} in docstore has empty text"

        print(f"{len(doc_ids)} doc_ids from {summary_collection} verified in docstore")


def test_doc_id_consistency_across_chunks_summaries_docstore(ingestion):
    """The same source file must have the same doc_id in chunks, summaries, and docstore."""
    if not ingestion.summary_indexer:
        pytest.skip("Summary indexer not available")

    coll_cfg = ingestion.config.get('collection_config', {})
    summary_collections = [c for c in COLLECTIONS if coll_cfg.get(c, {}).get('summary_enabled', False)]
    if not summary_collections:
        pytest.skip("No collections have summary_enabled: true")

    for collection in summary_collections:
        summary_collection = f"{collection}_summaries"
        try:
            chunk_points, _ = ingestion.client.scroll(
                collection_name=collection, limit=10, with_payload=True
            )
            summary_points, _ = ingestion.client.scroll(
                collection_name=summary_collection, limit=10, with_payload=True
            )
        except Exception as e:
            pytest.skip(f"Collections not available: {e}")

        chunks_by_path = {
            p.payload['source_path']: p.payload['doc_id']
            for p in chunk_points
            if p.payload.get('source_path') and p.payload.get('doc_id')
        }
        summaries_by_path = {
            p.payload['source_path']: p.payload['doc_id']
            for p in summary_points
            if p.payload.get('source_path') and p.payload.get('doc_id')
        }

        for source_path, chunk_doc_id in chunks_by_path.items():
            if source_path in summaries_by_path:
                assert chunk_doc_id == summaries_by_path[source_path], \
                    f"doc_id mismatch for {source_path}: chunk={chunk_doc_id}, summary={summaries_by_path[source_path]}"
                assert ingestion.docstore.get(chunk_doc_id) is not None, \
                    f"doc_id {chunk_doc_id} not found in docstore"

        print(f"{collection}: doc_ids consistent across chunks, summaries, and docstore")