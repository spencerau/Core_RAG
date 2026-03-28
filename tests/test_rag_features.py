"""
Integration tests for per-collection RAG feature flags.

Verifies that summary gating, reranking, and hybrid search behave according
to each collection's collection_config settings:
  - job_coaching:    summary_enabled=True,  reranking_enabled=True,  hybrid_enabled=False
  - recipes:         summary_enabled=True,  reranking_enabled=True,  hybrid_enabled=False
  - major_catalogs:  summary_enabled=False, reranking_enabled=False, hybrid_enabled=False

Prerequisites
-------------
- Qdrant running locally at localhost:6333 with ingested collections
- Ollama running locally with embedding + LLM models available
- Run test_ingest.py first to populate collections and summary collections

Run with:
    pytest tests/test_rag_features.py -v -s
"""

import pytest
from core_rag.retrieval.unified_rag import UnifiedRAG
from core_rag.summary import SummaryRetriever, SUMMARY_AVAILABLE

SUMMARY_COLLECTIONS = ["job_coaching", "recipes"]
NO_SUMMARY_COLLECTIONS = ["major_catalogs"]
ALL_COLLECTIONS = SUMMARY_COLLECTIONS + NO_SUMMARY_COLLECTIONS

COLLECTION_QUERIES = {
    "job_coaching": "What are effective strategies for helping someone find employment?",
    "recipes": "What are the steps and ingredients for making a krabby patty burger?",
    "major_catalogs": "What courses are required for a computer science degree?",
}


@pytest.fixture(scope="module")
def rag():
    return UnifiedRAG()


@pytest.fixture(scope="module")
def summary_retriever():
    if not SUMMARY_AVAILABLE:
        pytest.skip("Summary module not available")
    return SummaryRetriever()


# ---------------------------------------------------------------------------
# Summary retriever — direct tests
# ---------------------------------------------------------------------------

def test_summary_available():
    """Summary module must be importable (no llama_index required)."""
    assert SUMMARY_AVAILABLE, "SUMMARY_AVAILABLE is False — check summary_index.py imports"


def test_summary_retriever_finds_docs_for_summary_collections(summary_retriever):
    """search_summaries() must return results for collections with summary_enabled=True."""
    for collection in SUMMARY_COLLECTIONS:
        results = summary_retriever.search_summaries(
            query=COLLECTION_QUERIES[collection],
            collection_names=[collection],
            top_n=3
        )
        assert len(results) > 0, \
            f"No summary results for '{collection}' — was test_ingest.py run first?"
        for r in results:
            assert r.get('doc_id'), f"Summary result missing doc_id in '{collection}'"
            assert 0.0 <= r.get('score', 0) <= 1.0, "Score out of range"
        print(f"\n'{collection}': {len(results)} summary result(s), "
              f"top score={results[0]['score']:.3f}")


def test_summary_retriever_returns_full_doc_text(summary_retriever):
    """get_documents_by_summaries() must return full document text, not empty strings."""
    for collection in SUMMARY_COLLECTIONS:
        docs = summary_retriever.get_documents_by_summaries(
            query=COLLECTION_QUERIES[collection],
            collection_names=[collection],
            top_n=2
        )
        assert len(docs) > 0, f"No documents returned via summary gating for '{collection}'"
        for doc in docs:
            assert doc.get('text'), f"Document has empty text in '{collection}'"
            assert len(doc['text']) > 100, \
                f"Document text suspiciously short ({len(doc['text'])} chars) in '{collection}'"
            assert doc.get('doc_id'), "Document missing doc_id"
        print(f"\n'{collection}': {len(docs)} doc(s), "
              f"first doc text length={len(docs[0]['text'])} chars")


def test_no_summary_collection_for_major_catalogs(summary_retriever):
    """major_catalogs has summary_enabled=False — its summary collection must not exist in Qdrant."""
    existing = {c.name for c in summary_retriever.client.get_collections().collections}
    assert "major_catalogs_summaries" not in existing, \
        "major_catalogs_summaries should not exist (summary_enabled=False)"
    print("\nmajor_catalogs_summaries correctly absent from Qdrant")


# ---------------------------------------------------------------------------
# Per-collection debug flags via answer_question
# ---------------------------------------------------------------------------

def test_summary_gating_enabled_for_summary_collections(rag):
    """Collections with summary_enabled=True must report summary_gating_used=True in debug."""
    if not rag.summary_retriever:
        pytest.skip("Summary retriever not initialized on RAG instance")

    for collection in SUMMARY_COLLECTIONS:
        answer, _, debug = rag.answer_question(
            COLLECTION_QUERIES[collection],
            selected_collections=[collection],
            return_debug_info=True,
            stream=False
        )
        assert isinstance(answer, str) and len(answer) > 0
        assert debug.get('summary_gating_used') is True, \
            f"Expected summary_gating_used=True for '{collection}', got: {debug.get('summary_gating_used')}"
        print(f"\n'{collection}': summary_gating_used={debug['summary_gating_used']}")


def test_summary_gating_disabled_for_major_catalogs(rag):
    """major_catalogs has summary_enabled=False — summary_gating_used must be False."""
    answer, _, debug = rag.answer_question(
        COLLECTION_QUERIES["major_catalogs"],
        selected_collections=["major_catalogs"],
        return_debug_info=True,
        stream=False
    )
    assert isinstance(answer, str) and len(answer) > 0
    assert debug.get('summary_gating_used') is False, \
        f"Expected summary_gating_used=False for major_catalogs, got: {debug.get('summary_gating_used')}"
    print(f"\nmajor_catalogs: summary_gating_used={debug['summary_gating_used']}")


def test_reranking_enabled_for_summary_collections(rag):
    """Collections with reranking_enabled=True must report reranking_enabled=True in debug."""
    for collection in SUMMARY_COLLECTIONS:
        answer, _, debug = rag.answer_question(
            COLLECTION_QUERIES[collection],
            selected_collections=[collection],
            return_debug_info=True,
            stream=False
        )
        assert isinstance(answer, str) and len(answer) > 0
        assert debug.get('reranking_enabled') is True, \
            f"Expected reranking_enabled=True for '{collection}', got: {debug.get('reranking_enabled')}"
        print(f"\n'{collection}': reranking_enabled={debug['reranking_enabled']}")


def test_reranking_disabled_for_major_catalogs(rag):
    """major_catalogs has reranking_enabled=False — reranking must not fire."""
    answer, _, debug = rag.answer_question(
        COLLECTION_QUERIES["major_catalogs"],
        selected_collections=["major_catalogs"],
        return_debug_info=True,
        stream=False
    )
    assert isinstance(answer, str) and len(answer) > 0
    assert debug.get('reranking_enabled') is False, \
        f"Expected reranking_enabled=False for major_catalogs, got: {debug.get('reranking_enabled')}"
    print(f"\nmajor_catalogs: reranking_enabled={debug['reranking_enabled']}")


# ---------------------------------------------------------------------------
# Summary fallback — no results → falls back to chunk search
# ---------------------------------------------------------------------------

def test_summary_fallback_to_chunks_on_no_results(rag):
    """
    If the summary retriever returns nothing, the system must fall back to
    standard chunk search and still produce an answer.
    """
    if not rag.summary_retriever:
        pytest.skip("Summary retriever not initialized")

    # Use an off-topic query that is unlikely to match any summary
    answer, chunks, debug = rag.answer_question(
        "xyzzy frobozz nonsense query that matches nothing",
        selected_collections=SUMMARY_COLLECTIONS,
        return_debug_info=True,
        stream=False
    )
    # May return no-results message, but must not raise
    assert isinstance(answer, str) and len(answer) > 0
    print(f"\nFallback test: answer='{answer[:80]}...', chunks={len(chunks)}, "
          f"summary_gating_used={debug.get('summary_gating_used')}")


# ---------------------------------------------------------------------------
# Mixed-collection query
# ---------------------------------------------------------------------------

def test_mixed_collection_query_uses_summary_for_enabled_collections(rag):
    """
    When querying both summary-enabled and non-summary collections together,
    the system should use summary gating for the enabled ones.
    """
    if not rag.summary_retriever:
        pytest.skip("Summary retriever not initialized")

    answer, _, debug = rag.answer_question(
        "What courses and job skills should I focus on as a student?",
        selected_collections=ALL_COLLECTIONS,
        return_debug_info=True,
        stream=False
    )
    assert isinstance(answer, str) and len(answer) > 0
    # At least one summary-enabled collection should trigger gating
    assert debug.get('summary_gating_used') is True, \
        f"Expected summary_gating_used=True for mixed query, got: {debug}"
    print(f"\nMixed query: summary_gating_used={debug['summary_gating_used']}, "
          f"collections={debug.get('collections_searched')}")


# ---------------------------------------------------------------------------
# Collection config integrity
# ---------------------------------------------------------------------------

def test_collection_config_flags_present(rag):
    """collection_config must exist and have the expected keys for all collections."""
    coll_cfg = rag.config.get('collection_config', {})
    assert coll_cfg, "collection_config missing from config"

    expected_flags = ['summary_enabled', 'reranking_enabled', 'hybrid_enabled']
    for collection in ALL_COLLECTIONS:
        cfg = coll_cfg.get(collection)
        assert cfg is not None, f"No collection_config entry for '{collection}'"
        for flag in expected_flags:
            assert flag in cfg, f"Flag '{flag}' missing from collection_config['{collection}']"

    # Verify the expected values
    for collection in SUMMARY_COLLECTIONS:
        assert coll_cfg[collection]['summary_enabled'] is True
        assert coll_cfg[collection]['reranking_enabled'] is True

    assert coll_cfg['major_catalogs']['summary_enabled'] is False
    assert coll_cfg['major_catalogs']['reranking_enabled'] is False
    print(f"\nCollection config verified for {ALL_COLLECTIONS}")
