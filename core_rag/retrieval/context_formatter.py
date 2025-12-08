from typing import Dict, List
from textwrap import dedent


def format_context(chunks: List[Dict], config: dict) -> str:
    parts = []
    display_keys = config.get('rag', {}).get('metadata_display_keys', [])
    for chunk in chunks:
        text = chunk.get('text', '')
        meta = chunk.get('metadata', {})
        coll = chunk.get('collection', '')
        meta_parts = []
        if coll:
            meta_parts.append(f"Collection: {coll}")
        source = meta.get('file_name') or meta.get('resourceName') or meta.get('source')
        if source:
            meta_parts.append(f"Source: {source}")
        for k in display_keys:
            if meta.get(k):
                meta_parts.append(f"{k.replace('_', ' ').title()}: {meta[k]}")
        prefix = f"[{', '.join(meta_parts)}] " if meta_parts else ""
        parts.append(f"{prefix}{text}")
    return "\n\n".join(parts)


def build_prompt(context: str, query: str, thinking: bool, show_thinking: bool) -> str:
    if thinking and show_thinking:
        return dedent(f"""
            Context:
            {context}
            
            Question: {query}
            
            Please think through your answer step by step, then provide your final response.
            
            Thinking: [Show your reasoning process here]
            
            Answer: [Provide your final answer here]
        """).strip()
    elif thinking:
        return dedent(f"""
            Context:
            {context}
            
            Question: {query}
            
            Think through your answer carefully using the provided context, then provide a clear and concise response.
            
            Answer:
        """).strip()
    return dedent(f"""
        Context:
        {context}
        
        Question: {query}
        
        Answer:
    """).strip()


def chunks_to_context_docs(chunks: List[Dict]) -> List[Dict]:
    return [{
        'text': c.get('text', ''), 'score': c.get('score', 0),
        'metadata': {'doc_id': c.get('doc_id', c.get('metadata', {}).get('doc_id', '')),
                     'source_path': c.get('source_path', c.get('metadata', {}).get('source_path', '')),
                     'title': c.get('title', c.get('metadata', {}).get('title', ''))},
        'collection': c.get('collection_name', c.get('collection', ''))
    } for c in chunks]


def parent_docs_to_context(documents: dict, doc_ids: List[str]) -> List[Dict]:
    return [{
        'text': documents[d].get('text', ''), 'score': 1.0,
        'metadata': {'doc_id': d, 'source_path': documents[d].get('source_path', ''),
                     'title': documents[d].get('title', '')},
        'collection': documents[d].get('collection_name', '')
    } for d in doc_ids if d in documents]
