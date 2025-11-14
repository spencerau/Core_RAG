from typing import List, Dict, Any


def reciprocal_rank_fusion(dense_results: List[Dict[str, Any]], 
                          sparse_results: List[Dict[str, Any]],
                          k: int = 60,
                          weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
    if weights is None:
        weights = {'dense': 0.6, 'sparse': 0.4}
    
    doc_scores = {}
    
    for rank, result in enumerate(dense_results):
        doc_id = result.get('doc_id') or result.get('text', '')[:50]
        rrf_score = weights['dense'] / (k + rank + 1)
        
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {
                'result': result,
                'score_rrf': 0,
                'score_dense': result.get('score', 0),
                'score_sparse': 0
            }
        doc_scores[doc_id]['score_rrf'] += rrf_score
    
    for rank, result in enumerate(sparse_results):
        doc_id = result.get('doc_id') or result.get('text', '')[:50]
        rrf_score = weights['sparse'] / (k + rank + 1)
        
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {
                'result': result,
                'score_rrf': 0,
                'score_dense': 0,
                'score_sparse': result.get('score', 0)
            }
        else:
            doc_scores[doc_id]['score_sparse'] = result.get('score', 0)
        
        doc_scores[doc_id]['score_rrf'] += rrf_score
    
    fused_results = []
    for doc_data in doc_scores.values():
        result = doc_data['result'].copy()
        result.update({
            'score_rrf': doc_data['score_rrf'],
            'score_dense': doc_data['score_dense'],
            'score_sparse': doc_data['score_sparse']
        })
        fused_results.append(result)
    
    fused_results.sort(key=lambda x: x['score_rrf'], reverse=True)
    return fused_results


class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, config: dict):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.config = config
        self.k_dense = config.get('k_dense', 40)
        self.k_sparse = config.get('k_sparse', 40)
        self.rrf_k = config.get('rrf_k', 60)
        self.fuse_weights = config.get('fuse_weights', {'dense': 0.6, 'sparse': 0.4})
    
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        dense_results = self.dense_retriever.search(query, top_k=self.k_dense, **kwargs)
        sparse_results = self.sparse_retriever.search(query, top_k=self.k_sparse)
        
        fused_results = reciprocal_rank_fusion(
            dense_results, sparse_results, 
            k=self.rrf_k, weights=self.fuse_weights
        )
        
        return fused_results[:top_k]
