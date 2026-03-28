import pytest
from core_rag.retrieval.unified_rag import UnifiedRAG

COLLECTIONS = ["job_coaching", "major_catalogs", "recipes"]

# One representative query per collection
COLLECTION_QUERIES = {
    "job_coaching": "What are effective strategies for helping someone find employment?",
    "major_catalogs": "What courses are required for a computer science degree?",
    "recipes": "What are the steps and ingredients for making a krabby patty burger?",
}

# Cross-collection query that should pull from multiple collections
MULTI_COLLECTION_QUERY = "I'm a student interested in food science — what CS courses and culinary skills should I focus on?"


@pytest.fixture(scope="module")
def rag():
    return UnifiedRAG()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_rag_initialization(rag):
    assert rag is not None
    assert rag.config is not None
    assert rag.client is not None
    assert rag.ollama_api is not None


# ---------------------------------------------------------------------------
# Per-collection search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("collection", COLLECTIONS)
def test_search_collection(rag, collection):
    """Each collection should return results for a relevant query."""
    query = COLLECTION_QUERIES[collection]
    results = rag.search_collection(query=query, collection_name=collection, top_k=5)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0, f"No results from '{collection}' for query: {query}"
    print(f"\n'{collection}': {len(results)} result(s)")


@pytest.mark.parametrize("collection", COLLECTIONS)
def test_collection_stats(rag, collection):
    """Each collection must report a non-zero point count."""
    stats = rag.get_collection_stats(collection)
    assert stats is not None
    if 'error' in stats:
        pytest.skip(f"Collection stats API error: {stats['error']}")
    assert 'points_count' in stats
    assert stats['points_count'] > 0, f"'{collection}' appears empty"
    print(f"\n'{collection}': {stats['points_count']} points")


# ---------------------------------------------------------------------------
# Answer generation — per collection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("collection,query", COLLECTION_QUERIES.items())
def test_answer_question_per_collection(rag, collection, query):
    """RAG should produce a non-empty answer for each collection's domain query."""
    answer = rag.answer_question(query, selected_collections=[collection], stream=False)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert answer != "I couldn't generate a response."
    print(f"\n'{collection}' answer ({len(answer)} chars)")


def test_answer_question_streaming(rag):
    """Streaming should yield tokens that assemble into a valid answer."""
    query = COLLECTION_QUERIES["recipes"]
    stream = rag.answer_question(query, selected_collections=["recipes"], stream=True)

    assert stream is not None
    full_answer = "".join(stream)
    assert isinstance(full_answer, str)
    assert len(full_answer) > 0
    assert full_answer != "I couldn't generate a response."
    print(f"\nStreamed answer ({len(full_answer)} chars)")


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

def test_query_router_returns_valid_collections(rag):
    """Router should always return a subset of known collections."""
    if not hasattr(rag, 'query_router') or rag.query_router is None:
        pytest.skip("Query router not configured")

    for collection, query in COLLECTION_QUERIES.items():
        result = rag.query_router.route_query(query)
        assert 'collections' in result
        assert 'token_allocation' in result
        assert 'confidence' in result
        assert isinstance(result['collections'], list)
        assert len(result['collections']) > 0
        for col in result['collections']:
            assert col in COLLECTIONS, f"Router returned unknown collection: {col}"
        assert 150 <= result['token_allocation'] <= 2000
        assert 0.0 <= result['confidence'] <= 1.0
        print(f"\n'{collection}' routed to: {result['collections']} (confidence={result['confidence']:.2f})")


def test_query_router_routes_to_correct_collection(rag):
    """Router should preferentially select the most relevant collection."""
    if not hasattr(rag, 'query_router') or rag.query_router is None:
        pytest.skip("Query router not configured")

    expected = {
        "job_coaching": "What should I put on my resume for an entry-level position?",
        "major_catalogs": "What are the prerequisite courses for advanced algorithms?",
        "recipes": "How do I make a sourdough starter from scratch?",
    }

    for expected_collection, query in expected.items():
        result = rag.query_router.route_query(query)
        assert expected_collection in result['collections'], \
            f"Expected '{expected_collection}' in routing result for: {query!r}\nGot: {result['collections']}"
        print(f"\n'{expected_collection}' correctly identified for: {query!r}")


def test_query_router_simple_fallback(rag):
    """Simple (keyword) routing should always return valid collections."""
    if not hasattr(rag, 'query_router') or rag.query_router is None:
        pytest.skip("Query router not configured")

    for query in COLLECTION_QUERIES.values():
        result = rag.query_router.route_simple(query)
        assert 'collections' in result
        assert len(result['collections']) > 0
        for col in result['collections']:
            assert col in COLLECTIONS


def test_query_router_structured_output_fields(rag):
    """Router output must always contain all expected fields with correct types."""
    if not hasattr(rag, 'query_router') or rag.query_router is None:
        pytest.skip("Query router not configured")

    result = rag.query_router.route_query(COLLECTION_QUERIES["job_coaching"])
    assert isinstance(result['collections'], list)
    assert isinstance(result['token_allocation'], int)
    assert isinstance(result['reasoning'], str)
    assert isinstance(result['confidence'], float)
    print(f"\nStructured output fields verified: {result}")


# ---------------------------------------------------------------------------
# Multi-collection / debug
# ---------------------------------------------------------------------------

def test_answer_with_debug_info(rag):
    """Debug info should report routing details."""
    answer, _, debug = rag.answer_question(
        COLLECTION_QUERIES["major_catalogs"],
        return_debug_info=True,
        stream=False
    )
    assert isinstance(answer, str) and len(answer) > 0
    assert debug is not None
    assert 'collections_searched' in debug
    assert isinstance(debug['collections_searched'], list)
    print(f"\nDebug info: collections={debug['collections_searched']}, "
          f"routing_used={debug.get('routing_used')}, "
          f"token_allocation={debug.get('token_allocation')}")


def test_answer_all_collections(rag):
    """Searching all collections should still produce a coherent answer."""
    answer = rag.answer_question(
        MULTI_COLLECTION_QUERY,
        selected_collections=COLLECTIONS,
        stream=False
    )
    assert isinstance(answer, str) and len(answer) > 0
    print(f"\nAll-collections answer ({len(answer)} chars)")