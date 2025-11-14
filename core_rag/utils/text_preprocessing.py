import re
import unicodedata
from typing import List


def normalize_text(text: str, config: dict) -> str:
    if config.get('normalize_unicode', True):
        text = unicodedata.normalize('NFC', text)
    
    if config.get('preserve_course_codes', True):
        text = normalize_course_codes(text)
    
    if config.get('collapse_whitespace', True):
        text = re.sub(r'\s+', ' ', text)
    
    if config.get('dehyphenate', True):
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    if config.get('lowercase', False):
        text = text.lower()
    
    return text.strip()


def normalize_course_codes(text: str) -> str:
    # Pattern for course codes: Letters followed by optional space and numbers
    course_code_pattern = r'\b([A-Z]{2,6})\s+(\d{2,4}[A-Z]?)\b'
    
    # Replace "CPSC 350" with "CPSC350" to maintain semantic unity
    text = re.sub(course_code_pattern, r'\1\2', text, flags=re.IGNORECASE)
    
    return text


def add_embedding_prefix(text: str, task_type: str) -> str:
    if task_type == 'document':
        return f"search_document: {text}"
    elif task_type == 'query':
        return f"search_query: {text}"
    return text


def preprocess_for_embedding(texts: List[str], task_type: str, config: dict) -> List[str]:
    if not config.get('add_prefixes', True):
        return [normalize_text(text, config) for text in texts]
    
    return [
        add_embedding_prefix(normalize_text(text, config), task_type)
        for text in texts
    ]
