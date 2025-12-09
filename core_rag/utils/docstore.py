import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


class DocStore(ABC):
    @abstractmethod
    def put(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def batch_get(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        pass
    
    @abstractmethod
    def exists(self, doc_id: str) -> bool:
        pass


DOCSTORE_COLLECTION = "docstore"


class QdrantDocStore(DocStore):
    def __init__(self, host: str = None, port: int = None, collection_name: str = None):
        from .config_loader import load_config
        config = load_config()
        self.host = host or config['qdrant']['host']
        self.port = port or config['qdrant']['port']
        self.timeout = config['qdrant']['timeout']
        self.collection_name = collection_name or DOCSTORE_COLLECTION
        self.client = QdrantClient(host=self.host, port=self.port, timeout=self.timeout)
        self._ensure_collection()
    
    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE)
            )
            print(f"Created docstore collection '{self.collection_name}'")
    
    def put(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
        now = datetime.now(datetime.UTC).isoformat()
        payload = {
            'doc_id': doc_id,
            'text': text,
            'source_path': metadata.get('source_path', ''),
            'title': metadata.get('title', ''),
            'collection_name': metadata.get('collection_name', ''),
            'last_modified': metadata.get('last_modified', ''),
            'content_type': metadata.get('content_type', ''),
            'metadata': json.dumps(metadata),
            'updated_at': now
        }
        point = PointStruct(id=doc_id, vector=[0.0], payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.client.retrieve(collection_name=self.collection_name, ids=[doc_id])
            if not results:
                return None
            p = results[0].payload
            return {
                'doc_id': p.get('doc_id', ''),
                'text': p.get('text', ''),
                'source_path': p.get('source_path', ''),
                'title': p.get('title', ''),
                'collection_name': p.get('collection_name', ''),
                'last_modified': p.get('last_modified', ''),
                'metadata': json.loads(p.get('metadata', '{}'))
            }
        except Exception:
            return None
    
    def batch_get(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not doc_ids:
            return {}
        try:
            results = self.client.retrieve(collection_name=self.collection_name, ids=doc_ids)
            out = {}
            for r in results:
                p = r.payload
                out[p.get('doc_id', '')] = {
                    'doc_id': p.get('doc_id', ''),
                    'text': p.get('text', ''),
                    'source_path': p.get('source_path', ''),
                    'title': p.get('title', ''),
                    'collection_name': p.get('collection_name', ''),
                    'last_modified': p.get('last_modified', ''),
                    'metadata': json.loads(p.get('metadata', '{}'))
                }
            return out
        except Exception:
            return {}
    
    def delete(self, doc_id: str) -> bool:
        try:
            self.client.delete(collection_name=self.collection_name, points_selector=[doc_id])
            return True
        except Exception:
            return False
    
    def exists(self, doc_id: str) -> bool:
        try:
            results = self.client.retrieve(collection_name=self.collection_name, ids=[doc_id])
            return len(results) > 0
        except Exception:
            return False
    
    def list_all(self) -> List[Dict[str, Any]]:
        try:
            results = self.client.scroll(collection_name=self.collection_name, limit=10000)[0]
            return [{
                'doc_id': r.payload.get('doc_id', ''),
                'source_path': r.payload.get('source_path', ''),
                'title': r.payload.get('title', ''),
                'collection_name': r.payload.get('collection_name', ''),
                'last_modified': r.payload.get('last_modified', '')
            } for r in results]
        except Exception:
            return []
    
    def clear(self) -> int:
        try:
            count = self.client.count(collection_name=self.collection_name).count
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            return count
        except Exception:
            return 0


_default_docstore = None


def get_docstore(host: str = None, port: int = None) -> DocStore:
    global _default_docstore
    if host or port:
        return QdrantDocStore(host=host, port=port)
    if _default_docstore is None:
        _default_docstore = QdrantDocStore()
    return _default_docstore
