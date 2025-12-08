import os
from typing import Any, Dict, List
from qdrant_client import QdrantClient
from ..utils.config_loader import load_config
from ..utils.ollama_api import get_ollama_api
from ..utils.docstore import get_docstore
from .bm25 import BM25Retriever
from .reranker import BGEReranker
from .search import SearchEngine
from .llm_handler import LLMHandler, format_system_prompt
from .answer import AnswerGenerator

try:
    from ..summary import SummaryRetriever, LLAMAINDEX_AVAILABLE
except ImportError:
    SummaryRetriever = None
    LLAMAINDEX_AVAILABLE = False

try:
    from .query_router import QueryRouter
    QUERY_ROUTER_IMPORT_ERROR = None
except ImportError as e:
    QueryRouter = None
    QUERY_ROUTER_IMPORT_ERROR = e


class UnifiedRAG:
    def __init__(self):
        self.config = load_config()
        self.client = QdrantClient(host=self.config['qdrant']['host'], port=self.config['qdrant']['port'],
                                   timeout=self.config['qdrant']['timeout'])
        self.embedding_model = self.config['embedding']['model']
        self.collections = self.config['qdrant']['collections']
        self.ollama_api = get_ollama_api(timeout=self.config.get('llm', {}).get('timeout', 300))
        self.docstore = get_docstore()
        self.hybrid_disabled = os.getenv('HYBRID_DISABLED', 'false').lower() == 'true'
        self.rerank_disabled = os.getenv('RERANK_DISABLED', 'false').lower() == 'true'
        
        summary_cfg = self.config.get('summary', {})
        self.enable_summary_gating = summary_cfg.get('enable_summary_gating', False)
        self.summary_top_n = summary_cfg.get('summary_top_n', 5)
        self.return_parent_docs = summary_cfg.get('return_parent_docs', False)
        
        self._init_bm25()
        self._init_query_router()
        self._init_summary_retriever()
        self.reranker = None
        
        self.search_engine = SearchEngine(self.client, self.config, self.collections, self.ollama_api,
                                          self.embedding_model, self.bm25_retriever, self.hybrid_disabled)
        self.system_prompt = format_system_prompt(self.config)
        self.llm_handler = LLMHandler(self.config, self.ollama_api, self.system_prompt)
        self.answer_gen = AnswerGenerator(self.config, self.search_engine, self.llm_handler,
                                          self._get_reranker, self.query_router, self.summary_retriever,
                                          self.docstore, self.enable_summary_gating, self.summary_top_n,
                                          self.return_parent_docs, self.rerank_disabled)
    
    def _init_bm25(self):
        self.bm25_retriever = None
        if not self.hybrid_disabled and os.getenv('OPENSEARCH_URL'):
            try:
                self.bm25_retriever = BM25Retriever()
            except Exception as e:
                print(f"BM25 initialization failed: {e}")
    
    def _init_query_router(self):
        self.query_router = None
        if QueryRouter is None:
            if QUERY_ROUTER_IMPORT_ERROR:
                print(f"Warning: Query router disabled: {QUERY_ROUTER_IMPORT_ERROR}")
            return
        try:
            from utils.ollama_api import get_intermediate_ollama_api
            self.query_router = QueryRouter(get_intermediate_ollama_api(timeout=30))
            print("Query router initialized")
        except Exception as e:
            print(f"Warning: Query router initialization failed: {e}")
    
    def _init_summary_retriever(self):
        self.summary_retriever = None
        if LLAMAINDEX_AVAILABLE and SummaryRetriever and self.enable_summary_gating:
            try:
                self.summary_retriever = SummaryRetriever()
                print("Summary retriever initialized")
            except Exception as e:
                print(f"Summary retriever initialization failed: {e}")
    
    def _get_reranker(self):
        if self.reranker is None and not self.rerank_disabled:
            try:
                print("Initializing reranker...")
                self.reranker = BGEReranker()
            except Exception as e:
                print(f"Reranker initialization failed: {e}")
                self.reranker = False
        return self.reranker if self.reranker is not False else None
    
    def search_collection(self, query: str, collection_name: str, user_context: Dict = None,
                          top_k: int = 10, document_type: str = None) -> List[Dict]:
        return self.search_engine.search_collection(query, collection_name, user_context, top_k, document_type)
    
    def search_multiple_collections(self, query: str, collection_names: List[str],
                                    user_context: Dict = None, top_k_per_collection: int = 8) -> List[Dict]:
        return self.search_engine.search_multiple_collections(query, collection_names, user_context, top_k_per_collection)
    
    def get_parent_documents(self, chunks: List[Dict], max_docs: int = 5) -> List[Dict]:
        doc_ids = []
        for c in chunks:
            d = c.get('metadata', {}).get('doc_id') or c.get('doc_id')
            if d and d not in doc_ids:
                doc_ids.append(d)
                if len(doc_ids) >= max_docs:
                    break
        if not doc_ids:
            return []
        docs = self.docstore.batch_get(doc_ids)
        return [{'doc_id': d, 'text': docs[d].get('text', ''), 'source_path': docs[d].get('source_path', ''),
                 'title': docs[d].get('title', ''), 'collection_name': docs[d].get('collection_name', ''),
                 'metadata': docs[d].get('metadata', {})} for d in doc_ids if d in docs]
    
    def search_with_summary_gating(self, query: str, collection_names: List[str] = None, top_n: int = None) -> List[Dict]:
        if not self.summary_retriever:
            return []
        return self.summary_retriever.get_documents_by_summaries(query, collection_names, top_n or self.summary_top_n)
    
    def answer_question(self, query: str, conversation_history: List[Dict] = None, user_context: Dict = None,
                        stream: bool = False, selected_collections: List[str] = None, top_k: int = None,
                        enable_thinking: bool = True, show_thinking: bool = False, enable_reranking: bool = None,
                        return_debug_info: bool = False, use_summary_gating: bool = None,
                        use_parent_docs: bool = None) -> Any:
        return self.answer_gen.answer_question(query, conversation_history, user_context, stream, selected_collections,
                                               top_k, enable_thinking, show_thinking, enable_reranking,
                                               return_debug_info, use_summary_gating, use_parent_docs)
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        try:
            info = self.client.get_collection(self.collections.get(collection_name, collection_name))
            return {'points_count': info.points_count, 'status': info.status}
        except Exception as e:
            return {'error': str(e)}
    
    def list_collections(self) -> List[str]:
        return list(self.collections.keys())
