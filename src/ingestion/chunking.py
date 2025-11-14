import tiktoken
from typing import List, Dict, Any, Tuple
import re


class AdvancedChunker:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.strategy = config.get('strategy', 'recursive')
        self.target_tokens = config.get('target_tokens', 768)
        self.min_tokens = config.get('min_tokens', 300)
        self.overlap_ratio = config.get('overlap_ratio', 0.15)
        self.respect_headings = config.get('respect_headings', True)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Tuple[str, Dict[str, Any]]]:
        if self.strategy == 'recursive':
            chunks = self._recursive_chunk(text, metadata or {})
        else:
            chunks = self._simple_chunk(text, metadata or {})

        collection_type = (metadata or {}).get('collection_type', '')
        target_chunks_config = self.config.get('chunking', {}).get('collection_targets', {})
        target = target_chunks_config.get(collection_type)

        if target is not None:
            text_to_split = text if isinstance(text, str) else '\n'.join([c[0] for c in chunks])
            min_length = self.min_tokens * target * 4
            if len(text_to_split) < min_length:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'chunk_index': 0,
                    'chunking_strategy': self.strategy
                })
                return [(text_to_split, chunk_metadata)]
            length = len(text_to_split)
            chunk_size = max(1, length // target)
            split_chunks = [text_to_split[i*chunk_size:(i+1)*chunk_size] for i in range(target)]
            if length > chunk_size * target:
                split_chunks[-1] += text_to_split[chunk_size*target:]
            result = []
            for i, chunk_text in enumerate(split_chunks):
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'chunking_strategy': self.strategy
                })
                result.append((chunk_text, chunk_metadata))
            return result
        return chunks

    def _force_chunk_count(self, texts: List[str], target: int) -> List[str]:
        total = len(texts)
        if total == 0:
            return []
        if total == target:
            return texts
        if total < target:
            merged = texts[:]
            merged[-1] = ' '.join(texts[-(target-total+1):])
            return merged[:target]
        avg = max(1, total // target)
        merged = []
        idx = 0
        for i in range(target):
            next_idx = min(total, idx + avg)
            merged.append(' '.join(texts[idx:next_idx]))
            idx = next_idx
        if idx < total:
            merged[-1] += ' ' + ' '.join(texts[idx:])
        return merged
    
    def _recursive_chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        if self.respect_headings:
            separators = ["\n\n\n", "\n\n", "\n", ". ", " "]
        else:
            separators = ["\n\n", "\n", ". ", " "]
        
        chunks = self._split_recursive(text, separators)
        
        result = []
        for i, chunk in enumerate(chunks):
            token_count = len(self.tokenizer.encode(chunk))
            if token_count >= self.min_tokens:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'token_count': token_count,
                    'chunking_strategy': self.strategy
                })
                result.append((chunk, chunk_metadata))
        
        return result
    
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        good_splits = []
        separator = separators[-1]
        new_separators = []
        
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if re.search(_s, text):
                separator = _s
                new_separators = separators[i + 1:]
                break
        
        splits = re.split(separator, text) if separator else [text]
        
        for s in splits:
            if self._get_char_size(s) < self._tokens_to_chars(self.target_tokens):
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = separator.join(good_splits)
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                if new_separators:
                    other_info = self._split_recursive(s, new_separators)
                    final_chunks.extend(other_info)
                else:
                    final_chunks.append(s)
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total_chars = 0
        
        for s in splits:
            char_len = self._get_char_size(s)
            if total_chars + char_len + len(separator) > self._tokens_to_chars(self.target_tokens) and current_doc:
                docs.append(separator.join(current_doc))
                overlap_size = int(len(current_doc) * self.overlap_ratio)
                current_doc = current_doc[-overlap_size:] if overlap_size > 0 else []
                total_chars = sum(self._get_char_size(x) for x in current_doc)
            
            current_doc.append(s)
            total_chars += char_len + len(separator)
        
        if current_doc:
            docs.append(separator.join(current_doc))
        
        return docs
    
    def _get_char_size(self, text: str) -> int:
        return len(text)
    
    def _simple_chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        tokens = self.tokenizer.encode(text)
        overlap_tokens = int(self.target_tokens * self.overlap_ratio)
        stride = self.target_tokens - overlap_tokens
        
        result = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + self.target_tokens]
            if len(chunk_tokens) >= self.min_tokens:
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i // stride,
                    'token_count': len(chunk_tokens),
                    'chunking_strategy': self.strategy
                })
                result.append((chunk_text, chunk_metadata))
        
        return result
    
    def _tokens_to_chars(self, tokens: int) -> int:
        return int(tokens * 4)
