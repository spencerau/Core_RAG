import os
from pathlib import Path
from datetime import datetime
from typing import Tuple
from ..utils.doc_id import generate_doc_id, get_normalized_path


def prepare_doc_metadata(file_path: str, base_dir: str = None) -> Tuple[str, str, str]:
    doc_id = generate_doc_id(file_path, base_dir)
    source_path = get_normalized_path(file_path, base_dir)
    file_stat = os.stat(file_path)
    last_modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    return doc_id, source_path, last_modified


def get_collection_name(document_type: str, config: dict) -> str:
    if 'document_type_mapping' in config.get('domain', {}):
        mapping = config['domain']['document_type_mapping']
        collection_key = mapping.get(document_type)
        if collection_key and collection_key in config['qdrant']['collections']:
            return config['qdrant']['collections'][collection_key]
    collections = config['qdrant']['collections']
    if len(collections) == 1:
        return list(collections.values())[0]
    doc_type_lower = document_type.lower().replace('_', '')
    for key, collection_name in collections.items():
        if doc_type_lower in key.lower() or doc_type_lower in collection_name.lower():
            return collection_name
    return list(collections.values())[0]


def extract_markdown_title(text: str, file_path: str) -> str:
    title = Path(file_path).stem
    for line in text.split('\n')[:5]:
        if line.startswith('# '):
            title = line[2:].strip()
            break
    return title
