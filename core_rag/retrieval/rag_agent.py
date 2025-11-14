from ..utils.config_loader import load_config
from .reranker import BGEReranker
from .unified_rag import UnifiedRAG
from typing import Dict


class RAGAgent:
    
    def __init__(self, rag_system=None):
        if rag_system is None:
            self.rag = UnifiedRAG()
        else:
            self.rag = rag_system
        self.reranker = BGEReranker()
        self.config = load_config()
    
    def answer(self, query: str, user_context: Dict = None, 
               top_k: int = None, rerank_top_k: int = None, enable_thinking: bool = True,
               show_thinking: bool = False, use_streaming: bool = True) -> tuple:
        
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
        if rerank_top_k is None:
            rerank_top_k = self.config['retrieval']['rerank_top_k']
        
        answer, retrieved_chunks = self.rag.answer_question(
            query, user_context=user_context, top_k=top_k, 
            enable_thinking=enable_thinking, show_thinking=show_thinking,
            use_streaming=use_streaming
        )
        
        if self.reranker and retrieved_chunks:
            try:
                reranked_chunks = self.reranker.rerank(query, retrieved_chunks, rerank_top_k)
            except Exception as e:
                print(f"Reranking failed: {e}")
                reranked_chunks = retrieved_chunks[:rerank_top_k]
        else:
            reranked_chunks = retrieved_chunks[:rerank_top_k]
        
        context_parts = []
        for chunk in reranked_chunks:
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            collection = chunk.get('collection', '')
            
            metadata_str = ""
            if metadata or collection:
                meta_parts = []
                
                if collection:
                    meta_parts.append(f"Collection: {collection}")
                
                source_name = (
                    metadata.get('file_name') or 
                    metadata.get('resourceName') or
                    metadata.get('source')
                )
                
                if source_name:
                    if isinstance(source_name, str) and source_name.startswith("b'") and source_name.endswith("'"):
                        source_name = source_name[2:-1]
                    meta_parts.append(f"Source: {source_name}")
                
                display_keys = self.config.get('rag_agent', {}).get('metadata_display_keys', 
                                                                     ['year', 'Year', 'Subject', 'document_type', 'ProgramType', 'program_full'])
                for key in display_keys:
                    if metadata.get(key):
                        label = key.replace('_', ' ').title()
                        meta_parts.append(f"{label}: {metadata[key]}")
                
                if meta_parts:
                    metadata_str = f"[{', '.join(meta_parts)}] "
            
            context_parts.append(f"{metadata_str}{text}")
        
        context = "\n\n".join(context_parts)
        
        return answer, context, reranked_chunks
