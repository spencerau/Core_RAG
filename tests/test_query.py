import pytest
from core_rag.retrieval.unified_rag import UnifiedRAG


@pytest.fixture(scope="module")
def rag():
    """Create a RAG instance once per test module"""
    return UnifiedRAG()


def test_rag_initialization(rag):
    """Test that RAG initializes properly"""
    assert rag is not None
    assert rag.config is not None
    assert rag.client is not None
    assert rag.ollama_api is not None


def test_answer_question(rag):
    """Test answering a question with streaming"""
    answer_stream = rag.answer_question(
        "What are the instructions for making a Krabby Patty?",
        stream=True
    )
    
    assert answer_stream is not None
    full_answer = ""
    for chunk in answer_stream:
        full_answer += chunk
        print(chunk, end="", flush=True)
    
    assert isinstance(full_answer, str)
    assert len(full_answer) > 0
    assert full_answer != "I couldn't generate a response."
    print(f"\n✓ Streamed answer received ({len(full_answer)} chars)")


def test_search_collection(rag):
    """Test searching a specific collection"""
    results = rag.search_collection(
        query="Krabby Patty ingredients",
        collection_name="main_collection",
        top_k=5
    )
    
    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0
    print(f"\n✓ Found {len(results)} search results")


def test_collection_stats(rag):
    """Test getting collection statistics"""
    stats = rag.get_collection_stats("main_collection")
    
    assert stats is not None
    if 'error' in stats:
        pytest.skip(f"Collection stats API error: {stats['error']}")
    assert 'points_count' in stats
    assert stats['points_count'] > 0
    print(f"\n✓ Collection stats: {stats['points_count']} points")

