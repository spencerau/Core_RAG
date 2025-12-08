from typing import List, Dict, Optional, Any

from qdrant_client import QdrantClient

from ..utils.config_loader import load_config
from ..utils.ollama_api import get_ollama_api
from ..utils.docstore import get_docstore
from ..utils.text_preprocessing import preprocess_for_embedding


SUMMARY_COLLECTION_SUFFIX = "_summaries"


class SummaryRetriever:
    def __init__(self):
        self.config = load_config()
        self.client = QdrantClient(
            host=self.config['qdrant']['host'],
            port=self.config['qdrant']['port'],
            timeout=self.config['qdrant']['timeout']
        )
        self.embedding_model = self.config['embedding']['model']
        self.ollama_api = get_ollama_api()
        self.docstore = get_docstore()
        self.collections = self.config['qdrant']['collections']
        
        summary_config = self.config.get('summary', {})
        self.summary_top_n = summary_config.get('summary_top_n', 5)
    
    def _get_summary_collection_name(self, collection_name: str) -> str:
        if collection_name.endswith(SUMMARY_COLLECTION_SUFFIX):
            return collection_name
        return f"{collection_name}{SUMMARY_COLLECTION_SUFFIX}"
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            processed_text = preprocess_for_embedding([text], 'query', self.config.get('embedding', {}))[0]
            return self.ollama_api.get_embeddings(
                model=self.embedding_model,
                prompt=processed_text
            )
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def search_summaries(self, query: str, collection_names: List[str] = None, top_n: int = None) -> List[Dict]:
        if top_n is None:
            top_n = self.summary_top_n
        
        if collection_names is None:
            collection_names = list(self.collections.keys())
        
        query_vector = self._get_embedding(query)
        if not query_vector:
            return []
        
        all_results = []
        
        for coll_key in collection_names:
            collection_name = self.collections.get(coll_key, coll_key)
            summary_collection = self._get_summary_collection_name(collection_name)
            
            try:
                results = self.client.query_points(
                    collection_name=summary_collection,
                    query=query_vector,
                    limit=top_n
                )
                
                for hit in results.points:
                    all_results.append({
                        'doc_id': hit.payload.get('doc_id', ''),
                        'source_path': hit.payload.get('source_path', ''),
                        'collection_name': hit.payload.get('collection_name', collection_name),
                        'title': hit.payload.get('title', ''),
                        'score': hit.score
                    })
            except Exception as e:
                print(f"Error searching summary collection {summary_collection}: {e}")
                continue
        
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:top_n]
    
    def get_documents_by_summaries(self, query: str, collection_names: List[str] = None, top_n: int = None) -> List[Dict]:
        summary_results = self.search_summaries(query, collection_names, top_n)
        
        if not summary_results:
            return []
        
        doc_ids = [r['doc_id'] for r in summary_results if r.get('doc_id')]
        doc_ids = list(dict.fromkeys(doc_ids))
        
        documents = self.docstore.batch_get(doc_ids)
        
        results = []
        for summary in summary_results:
            doc_id = summary.get('doc_id')
            if doc_id and doc_id in documents:
                doc = documents[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'text': doc.get('text', ''),
                    'source_path': doc.get('source_path', summary.get('source_path', '')),
                    'title': doc.get('title', summary.get('title', '')),
                    'collection_name': summary.get('collection_name', ''),
                    'score': summary.get('score', 0),
                    'metadata': doc.get('metadata', {})
                })
        
        return results
    
    def get_doc_ids_from_summaries(self, query: str, collection_names: List[str] = None, top_n: int = None) -> List[str]:
        summary_results = self.search_summaries(query, collection_names, top_n)
        doc_ids = [r['doc_id'] for r in summary_results if r.get('doc_id')]
        return list(dict.fromkeys(doc_ids))
