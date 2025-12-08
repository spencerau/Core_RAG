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


def get_collection_name(document_type: str, config: dict, base_dir: str = None, file_path: str = None) -> str:
    if 'document_type_mapping' in config.get('domain', {}):
        mapping = config['domain']['document_type_mapping']
        collection_key = mapping.get(document_type)
        if collection_key and collection_key in config['qdrant']['collections']:
            return config['qdrant']['collections'][collection_key]
    
    collections = config['qdrant']['collections']
    
    if file_path or base_dir:
        path_to_check = file_path or base_dir
        path_str = Path(path_to_check).as_posix()
        
        sorted_collections = sorted(collections.items(), key=lambda x: len(x[1]), reverse=True)
        
        for key, collection_name in sorted_collections:
            if collection_name in path_str or collection_name.replace('_', '-') in path_str:
                return collection_name
        
        for key, collection_name in sorted_collections:
            if key in path_str or key.replace('_', '-') in path_str:
                return collection_name
    
    if document_type:
        doc_type_lower = document_type.lower().replace('_', '').replace('-', '')
        for key, collection_name in collections.items():
            key_lower = key.lower().replace('_', '').replace('-', '')
            collection_lower = collection_name.lower().replace('_', '').replace('-', '')
            if doc_type_lower in key_lower or doc_type_lower in collection_lower:
                return collection_name
    
    return list(collections.values())[0]


def extract_markdown_title(text: str, file_path: str) -> str:
    title = Path(file_path).stem
    for line in text.split('\n')[:5]:
        if line.startswith('# '):
            title = line[2:].strip()
            break
    return title
