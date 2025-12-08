import hashlib
import os
from pathlib import Path


def generate_doc_id(file_path: str, base_dir: str = None) -> str:
    if base_dir:
        try:
            rel_path = os.path.relpath(file_path, base_dir)
        except ValueError:
            rel_path = file_path
    else:
        rel_path = file_path
    
    normalized = rel_path.replace('\\', '/').lower().strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:32]


def generate_doc_id_with_content(file_path: str, content: str, base_dir: str = None) -> str:
    if base_dir:
        try:
            rel_path = os.path.relpath(file_path, base_dir)
        except ValueError:
            rel_path = file_path
    else:
        rel_path = file_path
    
    normalized = rel_path.replace('\\', '/').lower().strip()
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    combined = f"{normalized}:{content_hash}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:32]


def get_normalized_path(file_path: str, base_dir: str = None) -> str:
    if base_dir:
        try:
            rel_path = os.path.relpath(file_path, base_dir)
        except ValueError:
            rel_path = file_path
    else:
        rel_path = file_path
    
    return rel_path.replace('\\', '/')
