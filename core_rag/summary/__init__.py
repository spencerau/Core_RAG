try:
    from .summary_index import SummaryIndexer, ingest_summaries
    from .summary_retriever import SummaryRetriever
    SUMMARY_AVAILABLE = True
    LLAMAINDEX_AVAILABLE = True  # backwards-compat alias
except ImportError:
    SUMMARY_AVAILABLE = False
    LLAMAINDEX_AVAILABLE = False
    SummaryIndexer = None
    SummaryRetriever = None
    ingest_summaries = None

__all__ = ['SummaryIndexer', 'SummaryRetriever', 'ingest_summaries', 'SUMMARY_AVAILABLE', 'LLAMAINDEX_AVAILABLE']
