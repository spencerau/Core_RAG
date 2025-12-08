from typing import Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ..utils.text_preprocessing import preprocess_for_embedding


class SearchEngine:
    def __init__(self, client: QdrantClient, config: dict, collections: dict, ollama_api,
                 embedding_model: str, bm25_retriever=None, hybrid_disabled: bool = False):
        self.client = client
        self.config = config
        self.collections = collections
        self.ollama_api = ollama_api
        self.embedding_model = embedding_model
        self.bm25_retriever = bm25_retriever
        self.hybrid_disabled = hybrid_disabled
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            processed = preprocess_for_embedding([text], 'query', self.config.get('embedding', {}))[0]
            return self.ollama_api.get_embeddings(model=self.embedding_model, prompt=processed)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def build_filter(self, user_context: Dict = None, document_type: str = None) -> Optional[Filter]:
        if not user_context and not document_type:
            return None
        conditions = []
        if user_context:
            mappings = self.config.get('domain', {}).get('filter_mappings', {})
            for k, v in user_context.items():
                if v and k in mappings:
                    conditions.append(FieldCondition(key=mappings[k], match=MatchValue(value=str(v))))
        if document_type:
            conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=document_type)))
        return Filter(must=conditions) if conditions else None
    
    def dense_search(self, query: str, collection_name: str, user_context: Dict = None,
                     top_k: int = 10, document_type: str = None) -> List[Dict]:
        query_vector = self.get_embedding(query)
        if not query_vector:
            return []
        filter_obj = self.build_filter(user_context, document_type)
        collection = self.collections.get(collection_name, collection_name)
        results = self.client.query_points(collection_name=collection, query=query_vector,
                                           limit=top_k, query_filter=filter_obj)
        return [{'text': h.payload.get('text', ''), 'score': h.score,
                 'metadata': h.payload.get('metadata', {}), 'collection': collection_name}
                for h in results.points]
    
    def hybrid_search(self, query: str, collection_name: str, user_context: Dict = None,
                      top_k: int = 10, document_type: str = None) -> List[Dict]:
        from .fusion import HybridRetriever
        dense_results = self.dense_search(query, collection_name, user_context, top_k, document_type)
        try:
            sparse = self.bm25_retriever.search(query=query,
                collection_name=self.collections.get(collection_name, collection_name), top_k=top_k)
            fused = HybridRetriever().reciprocal_rank_fusion(dense_results, sparse, k=60)
            return fused[:top_k]
        except Exception as e:
            print(f"Hybrid search fallback to dense: {e}")
            return dense_results[:top_k]
    
    def search_collection(self, query: str, collection_name: str, user_context: Dict = None,
                          top_k: int = 10, document_type: str = None) -> List[Dict]:
        if self.hybrid_disabled or not self.bm25_retriever:
            return self.dense_search(query, collection_name, user_context, top_k, document_type)
        return self.hybrid_search(query, collection_name, user_context, top_k, document_type)
    
    def search_multiple_collections(self, query: str, collection_names: List[str],
                                    user_context: Dict = None, top_k_per_collection: int = 8,
                                    chunk_allocation: Dict[str, int] = None) -> List[Dict]:
        all_results = []
        for name in collection_names:
            top_k = (chunk_allocation or {}).get(name, top_k_per_collection)
            all_results.extend(self.search_collection(query, name, user_context, top_k))
        seen = {}
        for r in all_results:
            key = r.get('text', '')[:200]
            if key not in seen or r.get('score', 0) > seen[key].get('score', 0):
                seen[key] = r
        return sorted(seen.values(), key=lambda x: x.get('score', 0), reverse=True)
