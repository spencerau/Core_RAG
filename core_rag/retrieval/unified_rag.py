import os
import sys
from typing import List, Dict, Any, Optional
from textwrap import dedent
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..utils.config_loader import load_config
from ..utils.ollama_api import get_ollama_api
from ..utils.text_preprocessing import preprocess_for_embedding
from .bm25 import BM25Retriever
from .fusion import HybridRetriever
from .reranker import BGEReranker

try:
    from .query_router import QueryRouter
    QUERY_ROUTER_IMPORT_ERROR = None
except ImportError as import_error:
    QueryRouter = None
    QUERY_ROUTER_IMPORT_ERROR = import_error


class UnifiedRAG:
    def __init__(self):
        self.config = load_config()
        self.client = QdrantClient(
            host=self.config['qdrant']['host'],
            port=self.config['qdrant']['port'],
            timeout=self.config['qdrant']['timeout']
        )
        self.embedding_model = self.config['embedding']['model']
        self.collections = self.config['qdrant']['collections']
        llm_timeout = self.config.get('llm', {}).get('timeout', 300)
        self.ollama_api = get_ollama_api(timeout=llm_timeout)
        
        self.system_prompt = self._format_system_prompt()
        self.hybrid_disabled = os.getenv('HYBRID_DISABLED', 'false').lower() == 'true'
        self.rerank_disabled = os.getenv('RERANK_DISABLED', 'false').lower() == 'true'
        
        self._init_query_router()
        self._init_bm25()
        self.reranker = None
    
    def _init_query_router(self):
        if QueryRouter is None:
            if QUERY_ROUTER_IMPORT_ERROR:
                print(f"Warning: Query router disabled: {QUERY_ROUTER_IMPORT_ERROR}")
            self.query_router = None
            return
        
        try:
            from utils.ollama_api import get_intermediate_ollama_api
            intermediate_api = get_intermediate_ollama_api(timeout=30)
            self.query_router = QueryRouter(intermediate_api)
            print("Query router initialized")
        except Exception as e:
            print(f"Warning: Query router initialization failed: {e}")
            self.query_router = None
    
    def _init_bm25(self):
        self.bm25_retriever = None
        if not self.hybrid_disabled and os.getenv('OPENSEARCH_URL'):
            try:
                self.bm25_retriever = BM25Retriever()
            except Exception as e:
                print(f"BM25 initialization failed: {e}")
    
    def _format_system_prompt(self) -> str:
        prompt_template = self.config['llm']['system_prompt']
        domain = self.config.get('domain', {})
        
        return prompt_template.format(
            role=domain.get('role', 'assistant'),
            department=domain.get('department', 'organization'),
            contact_email=domain.get('contact_email', 'support')
        )
    
    def _get_reranker(self):
        if self.reranker is None and not self.rerank_disabled:
            try:
                print("Initializing reranker...")
                self.reranker = BGEReranker()
            except Exception as e:
                print(f"Reranker initialization failed: {e}")
                self.reranker = False
        return self.reranker if self.reranker is not False else None
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            processed_text = preprocess_for_embedding(
                [text], 'query', self.config.get('embedding', {})
            )[0]
            return self.ollama_api.get_embeddings(
                model=self.embedding_model,
                prompt=processed_text
            )
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _build_filter(self, user_context: Dict = None, 
                     document_type: str = None) -> Optional[Filter]:
        if not user_context and not document_type:
            return None
        
        conditions = []
        
        if user_context:
            filter_mappings = self.config.get('domain', {}).get('filter_mappings', {})
            for key, value in user_context.items():
                if value and key in filter_mappings:
                    metadata_key = filter_mappings[key]
                    conditions.append(
                        FieldCondition(key=metadata_key, match=MatchValue(value=str(value)))
                    )
        
        if document_type:
            conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=document_type))
            )
        
        return Filter(must=conditions) if conditions else None
    
    def _dense_search(self, query: str, collection_name: str, 
                     user_context: Dict = None, top_k: int = 10,
                     document_type: str = None) -> List[Dict]:
        query_vector = self._get_embedding(query)
        if not query_vector:
            return []
        
        filter_obj = self._build_filter(user_context, document_type)
        collection = self.collections.get(collection_name, collection_name)
        
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            query_filter=filter_obj
        )
        
        return [{
            'text': hit.payload.get('text', ''),
            'score': hit.score,
            'metadata': hit.payload.get('metadata', {}),
            'collection': collection_name
        } for hit in results]
    
    def _hybrid_search(self, query: str, collection_name: str,
                      user_context: Dict = None, top_k: int = 10,
                      document_type: str = None) -> List[Dict]:
        dense_results = self._dense_search(query, collection_name, user_context, top_k, document_type)
        
        try:
            sparse_results = self.bm25_retriever.search(
                query=query,
                collection_name=self.collections.get(collection_name, collection_name),
                top_k=top_k
            )
            
            hybrid_retriever = HybridRetriever()
            fused_results = hybrid_retriever.reciprocal_rank_fusion(
                dense_results, sparse_results, k=60
            )
            return fused_results[:top_k]
        except Exception as e:
            print(f"Hybrid search fallback to dense: {e}")
            return dense_results[:top_k]
    
    def search_collection(self, query: str, collection_name: str, 
                         user_context: Dict = None, top_k: int = 10,
                         document_type: str = None) -> List[Dict]:
        if self.hybrid_disabled or not self.bm25_retriever:
            return self._dense_search(query, collection_name, user_context, top_k, document_type)
        return self._hybrid_search(query, collection_name, user_context, top_k, document_type)
    
    def search_multiple_collections(self, query: str, collection_names: List[str],
                                   user_context: Dict = None, top_k_per_collection: int = 8) -> List[Dict]:
        all_results = []
        chunk_allocation = self._calculate_chunk_allocation(collection_names, query)
        
        for collection_name in collection_names:
            top_k = chunk_allocation.get(collection_name, top_k_per_collection)
            results = self.search_collection(query, collection_name, user_context, top_k)
            all_results.extend(results)
        
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results
    
    def _calculate_chunk_allocation(self, collection_names: List[str], query: str) -> Dict[str, int]:
        base_chunks = self.config.get('rag', {}).get('base_chunks_per_collection', 8)
        priority_boost = self.config.get('rag', {}).get('priority_boost', 4)
        priority_collections = self._get_priority_collections(collection_names, query)
        
        return {
            name: base_chunks + priority_boost if name in priority_collections else base_chunks
            for name in collection_names
        }
    
    def _get_priority_collections(self, collection_names: List[str], query: str) -> List[str]:
        priority_order = self.config.get('rag', {}).get('collection_priority', collection_names)
        return [name for name in priority_order if name in collection_names]
    
    def _format_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        for chunk in chunks:
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            collection = chunk.get('collection', '')
            
            meta_parts = []
            if collection:
                meta_parts.append(f"Collection: {collection}")
            
            source = metadata.get('file_name') or metadata.get('resourceName') or metadata.get('source')
            if source:
                meta_parts.append(f"Source: {source}")
            
            display_keys = self.config.get('rag', {}).get('metadata_display_keys', [])
            for key in display_keys:
                if metadata.get(key):
                    label = key.replace('_', ' ').title()
                    meta_parts.append(f"{label}: {metadata[key]}")
            
            prefix = f"[{', '.join(meta_parts)}] " if meta_parts else ""
            context_parts.append(f"{prefix}{text}")
        
        return "\n\n".join(context_parts)
    
    def _get_llm_response(self, prompt: str, conversation_history: List[Dict] = None, 
                         token_allocation: int = 600) -> str:
        messages = [{'role': 'system', 'content': self.system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            response = self.ollama_api.chat(
                model=self.config['llm']['primary_model'],
                messages=messages,
                stream=False,
                options={
                    'temperature': self.config['llm']['temperature'],
                    'num_predict': token_allocation
                }
            )
            return response.strip() if response else "I couldn't generate a response."
        except Exception as e:
            print(f"LLM error: {e}")
            return f"Error: {str(e)}"
    
    def _get_llm_response_stream(self, prompt: str, conversation_history: List[Dict] = None, 
                                 token_allocation: int = 600):
        messages = [{'role': 'system', 'content': self.system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            for chunk in self.ollama_api.chat(
                model=self.config['llm']['primary_model'],
                messages=messages,
                stream=True,
                options={
                    'temperature': self.config['llm']['temperature'],
                    'num_predict': token_allocation
                }
            ):
                yield chunk
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"Error: {str(e)}"
    
    def answer_question(self, query: str, conversation_history: List[Dict] = None,
                       user_context: Dict = None, stream: bool = False,
                       selected_collections: List[str] = None,
                       top_k: int = None,
                       enable_thinking: bool = True,
                       show_thinking: bool = False,
                       enable_reranking: bool = None,
                       return_debug_info: bool = False) -> Any:
        
        debug_info = {
            'query': query,
            'collections_searched': [],
            'total_results': 0,
            'reranking_enabled': False,
            'thinking_enabled': enable_thinking,
            'top_k_used': top_k
        }
        
        if self.query_router and not selected_collections:
            route_result = self.query_router.route_query(
                query, conversation_history=conversation_history, user_context=user_context
            )
            collection_names = route_result['collections']
            token_allocation = route_result['token_allocation']
            debug_info['routing_used'] = True
            debug_info['token_allocation'] = token_allocation
        else:
            collection_names = selected_collections or list(self.collections.keys())
            token_allocation = self.config.get('llm', {}).get('default_tokens', 600)
            debug_info['routing_used'] = False
            debug_info['token_allocation'] = token_allocation
        
        debug_info['collections_searched'] = collection_names
        
        if top_k is None:
            top_k = self.config.get('rag', {}).get('top_k', 20)
        debug_info['top_k_used'] = top_k
        
        all_results = self.search_multiple_collections(query, collection_names, user_context)
        debug_info['total_results'] = len(all_results)
        
        if not all_results:
            no_results_msg = "I couldn't find relevant information to answer your question."
            debug_info['error'] = 'no_results'
            if stream:
                def error_stream():
                    yield no_results_msg
                if return_debug_info:
                    return error_stream(), [], debug_info
                return error_stream()
            if return_debug_info:
                return no_results_msg, [], debug_info
            return no_results_msg
        
        if enable_reranking is None:
            enable_reranking = not self.rerank_disabled
        
        if enable_reranking:
            reranker = self._get_reranker()
            if reranker:
                reranked_chunks = reranker.rerank(query, all_results, top_k=top_k)
                debug_info['reranking_enabled'] = True
            else:
                reranked_chunks = all_results[:top_k]
                debug_info['reranking_enabled'] = False
        else:
            reranked_chunks = all_results[:top_k]
            debug_info['reranking_enabled'] = False
        
        debug_info['chunks_used'] = len(reranked_chunks)
        
        context = self._format_context(reranked_chunks)
        
        if enable_thinking and show_thinking:
            prompt = dedent(f"""
                Context:
                {context}
                
                Question: {query}
                
                Please think through your answer step by step, then provide your final response.
                
                Thinking: [Show your reasoning process here]
                
                Answer: [Provide your final answer here]
            """).strip()
        elif enable_thinking:
            prompt = dedent(f"""
                Context:
                {context}
                
                Question: {query}
                
                Think through your answer carefully using the provided context, then provide a clear and concise response.
                
                Answer:
            """).strip()
        else:
            prompt = dedent(f"""
                Context:
                {context}
                
                Question: {query}
                
                Answer:
            """).strip()
        
        debug_info['prompt_length'] = len(prompt)
        
        if stream:
            stream_gen = self._get_llm_response_stream(prompt, conversation_history, token_allocation)
            if return_debug_info:
                return stream_gen, reranked_chunks, debug_info
            return stream_gen
        
        answer = self._get_llm_response(prompt, conversation_history, token_allocation)
        debug_info['answer_length'] = len(answer)
        
        if return_debug_info:
            return answer, reranked_chunks, debug_info
        return answer
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        try:
            collection = self.collections.get(collection_name, collection_name)
            info = self.client.get_collection(collection)
            return {
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            return {'error': str(e)}
    
    def list_collections(self) -> List[str]:
        return list(self.collections.keys())
