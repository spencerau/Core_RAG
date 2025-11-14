import os
import math
import numpy as np
from typing import List, Dict, Any
from utils.config_loader import load_config


class BGEReranker:
    def __init__(self):
        self.config = load_config()
        self.model_name = self.config.get('reranker', {}).get('model', 'BAAI/bge-reranker-v2-m3')
        self.top_k = self.config.get('reranker', {}).get('top_k_rerank', 12)
        self.batch_size = self.config.get('reranker', {}).get('batch_size', 32)
        self.max_candidates = self.config.get('reranker', {}).get('max_candidates_for_rerank', 200)
        self.activation = self.config.get('reranker', {}).get('activation', 'sigmoid')
        self.model = None
        self.use_ollama = not self.model_name.startswith('BAAI/')
        self._initialize_model()
    
    def _initialize_model(self):
        if self.use_ollama:
            try:
                from utils.ollama_api import get_ollama_api
                self.ollama_api = get_ollama_api()
                print(f"Using Ollama reranker: {self.model_name}")
            except Exception as e:
                print(f"Failed to initialize Ollama reranker: {e}")
                self.ollama_api = None
        else:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name, trust_remote_code=True)
                print(f"Using HuggingFace reranker: {self.model_name}")
            except ImportError:
                print("sentence-transformers not available, using fallback reranker")
                self.model = None
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = None) -> List[Dict]:
        if not documents:
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        candidates = documents[:self.max_candidates]
        
        if self.use_ollama and self.ollama_api:
            scores = self._rerank_with_ollama(query, candidates)
        elif self.model is not None:
            scores = self._rerank_with_model(query, candidates)
        else:
            scores = self._fallback_rerank(query, candidates)
        
        reranked_docs = []
        for i, doc in enumerate(candidates):
            new_doc = doc.copy()
            new_doc['rerank_score'] = scores[i] if i < len(scores) else 0.0
            reranked_docs.append(new_doc)
        
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked_docs[:top_k]
    
    def _rerank_with_ollama(self, query: str, documents: List[Dict]) -> List[float]:
        scores = []
        
        for doc in documents:
            text = doc.get('text', '')
            
            try:
                combined_text = f"Query: {query}\nDocument: {text}"
                embedding = self.ollama_api.get_embeddings(self.model_name, combined_text)
                
                if embedding:
                    import numpy as np
                    embedding_array = np.array(embedding)
                    
                    if len(embedding_array) > 0:
                        score = float(embedding_array[0])
                        score = 1 / (1 + math.exp(-score))
                    else:
                        score = 0.0
                else:
                    score = self._calculate_simple_relevance(query, text)
                    
            except Exception as e:
                print(f"Ollama reranking failed for document: {e}")
                score = self._calculate_simple_relevance(query, text)
            
            scores.append(score)
        
        return scores
    
    def _rerank_with_model(self, query: str, documents: List[Dict]) -> List[float]:
        pairs = [[query, doc.get('text', '')] for doc in documents]
        
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch_pairs)
            
            if self.activation == 'sigmoid':
                batch_scores = [1 / (1 + math.exp(-score)) for score in batch_scores]
            
            all_scores.extend(batch_scores)
        
        return all_scores
    
    def _fallback_rerank(self, query: str, documents: List[Dict]) -> List[float]:
        scores = []
        for doc in documents:
            score = self._calculate_simple_relevance(query, doc.get('text', ''))
            scores.append(score)
        return scores
    
    def _calculate_simple_relevance(self, query: str, text: str) -> float:
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        phrase_boost = 0.0
        if query_lower in text_lower:
            phrase_boost = 0.3
        
        boost_terms = self.config.get('domain', {}).get('boost_terms', [])
        
        term_boost = 0.0
        if boost_terms:
            for term in boost_terms:
                if term.lower() in query_lower and term.lower() in text_lower:
                    term_boost += 0.1
        
        term_boost = min(term_boost, 0.2)
        
        return min(jaccard_score + phrase_boost + term_boost, 1.0)
    
    def rerank_with_weights(self, query: str, documents: List[Dict], 
                           pdf_weight: float = 1.0, course_weight: float = 1.0, 
                           year_plan_weight: float = 1.0) -> List[Dict]:
        reranked = self.rerank(query, documents)
        
        collection_weights = self.config.get('reranker', {}).get('collection_weights', {})
        
        for doc in reranked:
            collection = doc.get('collection', '')
            
            for collection_pattern, weight in collection_weights.items():
                if collection_pattern.lower() in collection.lower():
                    doc['rerank_score'] *= weight
                    break
        
        reranked.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked