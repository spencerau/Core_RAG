import os
import math
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import pickle
import json


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.doc_metadata = []
    
    def _tokenize(self, text: str) -> List[str]:
        course_code_pattern = r'\b([A-Z]{2,6}\s*\d{2,4}[A-Z]?)\b'
        
        course_codes = re.findall(course_code_pattern, text.upper())
        normalized_codes = []
        
        for code in course_codes:
            normalized = re.sub(r'\s+', '', code).lower()
            normalized_codes.append(normalized)
        
        modified_text = text
        for i, code in enumerate(course_codes):
            normalized = normalized_codes[i]
            modified_text = re.sub(
                re.escape(code), 
                normalized, 
                modified_text, 
                flags=re.IGNORECASE
            )
        
        modified_text = re.sub(r'[^\w\s]', ' ', modified_text.lower())
        
        tokens = modified_text.split()
        
        return tokens
    
    def fit(self, corpus: List[str], metadata: List[Dict[str, Any]] = None):
        self.corpus = corpus
        self.doc_metadata = metadata or [{} for _ in corpus]
        
        nd = len(corpus)
        doc_freqs = []
        df = defaultdict(int)
        
        for document in corpus:
            frequencies = Counter(self._tokenize(document))
            doc_freqs.append(frequencies)
            
            for word in frequencies.keys():
                df[word] += 1
        
        self.doc_freqs = doc_freqs
        
        idf = {}
        for word, freq in df.items():
            idf[word] = math.log(nd / freq)
        
        self.idf = idf
        self.doc_len = [len(self._tokenize(doc)) for doc in corpus]
        self.avgdl = sum(self.doc_len) / nd if nd > 0 else 0
    
    def search(self, query: str, top_k: int = 10, course_code_boost: float = 2.0) -> List[Dict[str, Any]]:
        if not self.corpus:
            return []
        
        query_tokens = self._tokenize(query)
        scores = []
        
        course_code_pattern = r'^[a-z]{2,6}\d{2,4}[a-z]?$'
        course_code_tokens = [token for token in query_tokens 
                            if re.match(course_code_pattern, token)]
        
        for i, doc_freqs in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_len[i]
            
            for word in query_tokens:
                if word in doc_freqs:
                    freq = doc_freqs[word]
                    idf = self.idf.get(word, 0)
                    
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    base_score = idf * numerator / denominator
                    
                    if word in course_code_tokens:
                        base_score *= course_code_boost
                    
                    score += base_score
            
            scores.append({
                'text': self.corpus[i],
                'score': score,
                'doc_id': i,
                'metadata': self.doc_metadata[i]
            })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def save(self, filepath: str):
        data = {
            'k1': self.k1,
            'b': self.b,
            'corpus': self.corpus,
            'doc_freqs': self.doc_freqs,
            'idf': self.idf,
            'doc_len': self.doc_len,
            'avgdl': self.avgdl,
            'doc_metadata': self.doc_metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.k1 = data['k1']
        self.b = data['b']
        self.corpus = data['corpus']
        self.doc_freqs = data['doc_freqs']
        self.idf = data['idf']
        self.doc_len = data['doc_len']
        self.avgdl = data['avgdl']
        self.doc_metadata = data['doc_metadata']


class BM25Retriever:
    def __init__(self, index_path: str = "bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = BM25()
        self.is_loaded = False
    
    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        self.bm25.fit(documents, metadata)
        self.bm25.save(self.index_path)
        self.is_loaded = True
    
    def load_index(self):
        if os.path.exists(self.index_path):
            self.bm25.load(self.index_path)
            self.is_loaded = True
            return True
        return False
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            if not self.load_index():
                return []
        
        results = self.bm25.search(query, top_k)
        for result in results:
            result['score_sparse'] = result['score']
        return results
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        if not self.is_loaded:
            self.load_index()
        
        existing_docs = self.bm25.corpus
        existing_metadata = self.bm25.doc_metadata
        
        all_docs = existing_docs + documents
        all_metadata = existing_metadata + (metadata or [{} for _ in documents])
        
        self.build_index(all_docs, all_metadata)
