from typing import Any, Dict, List
from .context_formatter import format_context, build_prompt, chunks_to_context_docs, parent_docs_to_context


class AnswerGenerator:
    def __init__(self, config: dict, search_engine, llm_handler, reranker_getter,
                 query_router=None, summary_retriever=None, docstore=None,
                 enable_summary_gating: bool = False, summary_top_n: int = 5,
                 return_parent_docs: bool = False, rerank_disabled: bool = False):
        self.config = config
        self.search_engine = search_engine
        self.llm_handler = llm_handler
        self.get_reranker = reranker_getter
        self.query_router = query_router
        self.summary_retriever = summary_retriever
        self.docstore = docstore
        self.enable_summary_gating = enable_summary_gating
        self.summary_top_n = summary_top_n
        self.return_parent_docs = return_parent_docs
        self.rerank_disabled = rerank_disabled
        self.collections = search_engine.collections
    
    def answer_question(self, query: str, conversation_history: List[Dict] = None,
                        user_context: Dict = None, stream: bool = False,
                        selected_collections: List[str] = None, top_k: int = None,
                        enable_thinking: bool = True, show_thinking: bool = False,
                        enable_reranking: bool = None, return_debug_info: bool = False,
                        use_summary_gating: bool = None, use_parent_docs: bool = None) -> Any:
        debug = {'query': query, 'collections_searched': [], 'total_results': 0, 'reranking_enabled': False,
                 'thinking_enabled': enable_thinking, 'top_k_used': top_k, 'summary_gating_used': False, 'parent_docs_used': False}
        
        collections, tokens = self._route_query(query, conversation_history, user_context, selected_collections, debug)
        if top_k is None:
            top_k = self.config.get('rag', {}).get('top_k', 20)
        debug['top_k_used'] = top_k
        if use_summary_gating is None:
            use_summary_gating = self.enable_summary_gating
        if use_parent_docs is None:
            use_parent_docs = self.return_parent_docs
        
        if use_summary_gating and self.summary_retriever:
            result = self._summary_gated_answer(query, collections, tokens, conversation_history,
                                                enable_thinking, show_thinking, stream, return_debug_info, debug)
            if result is not None:
                return result
        
        return self._standard_answer(query, collections, user_context, top_k, tokens, conversation_history,
                                     enable_thinking, show_thinking, stream, enable_reranking,
                                     use_parent_docs, return_debug_info, debug)
    
    def _route_query(self, query, history, user_ctx, selected, debug):
        if self.query_router and not selected:
            r = self.query_router.route_query(query, conversation_history=history, user_context=user_ctx)
            debug['routing_used'], debug['token_allocation'] = True, r['token_allocation']
            debug['collections_searched'] = r['collections']
            return r['collections'], r['token_allocation']
        cols = selected or list(self.collections.keys())
        tokens = self.config.get('llm', {}).get('max_tokens', 15000)
        debug['routing_used'], debug['token_allocation'], debug['collections_searched'] = False, tokens, cols
        return cols, tokens
    
    def _summary_gated_answer(self, query, collections, tokens, history, thinking, show, stream, ret_debug, debug):
        debug['summary_gating_used'] = True
        docs = self.summary_retriever.get_documents_by_summaries(query, collections, self.summary_top_n)
        if not docs:
            return None
        ctx_docs = chunks_to_context_docs(docs)
        debug['total_results'], debug['parent_docs_used'] = len(ctx_docs), True
        ctx = format_context(ctx_docs, self.config)
        prompt = build_prompt(ctx, query, thinking, show)
        debug['prompt_length'] = len(prompt)
        return self._generate(prompt, history, tokens, stream, ret_debug, ctx_docs, debug)
    
    def _standard_answer(self, query, collections, user_ctx, top_k, tokens, history,
                         thinking, show, stream, reranking, parent_docs, ret_debug, debug):
        alloc = self._chunk_allocation(collections)
        results = self.search_engine.search_multiple_collections(query, collections, user_ctx, chunk_allocation=alloc)
        debug['total_results'] = len(results)
        if not results:
            return self._no_results(stream, ret_debug, debug)
        
        reranked = self._apply_reranking(results, query, top_k, reranking, debug)
        debug['chunks_used'] = len(reranked)
        ctx = self._get_context(reranked, parent_docs, debug)
        prompt = build_prompt(ctx, query, thinking, show)
        debug['prompt_length'] = len(prompt)
        return self._generate(prompt, history, tokens, stream, ret_debug, reranked, debug)
    
    def _chunk_allocation(self, collections):
        base = self.config.get('rag', {}).get('base_chunks_per_collection', 8)
        boost = self.config.get('rag', {}).get('priority_boost', 4)
        priority = set(self.config.get('rag', {}).get('collection_priority', collections)) & set(collections)
        return {n: base + boost if n in priority else base for n in collections}
    
    def _apply_reranking(self, results, query, top_k, enable, debug):
        if enable is None:
            enable = not self.rerank_disabled
        if enable:
            reranker = self.get_reranker()
            if reranker:
                debug['reranking_enabled'] = True
                return reranker.rerank(query, results, top_k=top_k)
        debug['reranking_enabled'] = False
        return results[:top_k]
    
    def _get_context(self, chunks, use_parent, debug):
        if not use_parent or not self.docstore:
            return format_context(chunks, self.config)
        debug['parent_docs_used'] = True
        doc_ids = []
        for c in chunks:
            d = c.get('metadata', {}).get('doc_id') or c.get('doc_id')
            if d and d not in doc_ids:
                doc_ids.append(d)
                if len(doc_ids) >= self.summary_top_n:
                    break
        if not doc_ids:
            return format_context(chunks, self.config)
        docs = self.docstore.batch_get(doc_ids)
        if not docs:
            return format_context(chunks, self.config)
        return format_context(parent_docs_to_context(docs, doc_ids), self.config)
    
    def _generate(self, prompt, history, tokens, stream, ret_debug, chunks, debug):
        if stream:
            gen = self.llm_handler.get_response_stream(prompt, history, tokens)
            return (gen, chunks, debug) if ret_debug else gen
        ans = self.llm_handler.get_response(prompt, history, tokens)
        debug['answer_length'] = len(ans)
        return (ans, chunks, debug) if ret_debug else ans
    
    def _no_results(self, stream, ret_debug, debug):
        msg = "I couldn't find relevant information to answer your question."
        debug['error'] = 'no_results'
        if stream:
            def err():
                yield msg
            return (err(), [], debug) if ret_debug else err()
        return (msg, [], debug) if ret_debug else msg
