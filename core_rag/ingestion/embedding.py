from typing import List
from ..utils.ollama_api import get_ollama_api
from ..utils.text_preprocessing import preprocess_for_embedding


class EmbeddingGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = config['embedding']['model']
        self.ollama_api = get_ollama_api()
    
    def get_embedding(self, text: str, task_type: str = 'document') -> List[float]:
        try:
            processed_text = preprocess_for_embedding(
                [text], task_type, self.config.get('embedding', {})
            )[0]
            embedding = self.ollama_api.get_embeddings(model=self.embedding_model, prompt=processed_text)
            return embedding if embedding is not None else []
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str], task_type: str = 'document') -> List[List[float]]:
        try:
            processed_texts = preprocess_for_embedding(texts, task_type, self.config.get('embedding', {}))
            embeddings = []
            batch_size = self.config.get('embedding', {}).get('batch_size', 32)
            for i in range(0, len(processed_texts), batch_size):
                for text in processed_texts[i:i + batch_size]:
                    embeddings.append(self.ollama_api.get_embeddings(model=self.embedding_model, prompt=text))
            return embeddings
        except Exception as e:
            print(f"Error getting batch embeddings: {e}")
            return []
    
    def get_vector_size(self) -> int:
        test_embedding = self.get_embedding("test")
        if test_embedding and len(test_embedding) > 0:
            return len(test_embedding)
        if 'bge-m3' in self.embedding_model:
            return 1024
        elif 'nomic-embed' in self.embedding_model:
            return 768
        return 768
