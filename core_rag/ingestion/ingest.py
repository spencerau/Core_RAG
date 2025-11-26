import os
import re
import json
from pathlib import Path
from typing import Dict, List
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from datetime import datetime
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..utils.config_loader import load_config
from ..utils.ollama_api import get_ollama_api
from ..utils.text_preprocessing import preprocess_for_embedding
from .content_extract import extract_content
from .chunking import AdvancedChunker
from .edit_metadata import MetadataExtractor


class UnifiedIngestion:
    def __init__(self):
        self.config = load_config()
        self.client = QdrantClient(
            host=self.config['qdrant']['host'],
            port=self.config['qdrant']['port'],
            timeout=self.config['qdrant']['timeout']
        )
        self.embedding_model = self.config['embedding']['model']
        self.chunker = AdvancedChunker(self.config.get('chunker', {}))
        self.ollama_api = get_ollama_api()
        self.metadata_extractor = MetadataExtractor()
        
        self._ensure_collections_exist()
    
    def _ensure_collections_exist(self):
        collections = self.config['qdrant']['collections']
        
        for collection_name in collections.values():
            try:
                self.client.get_collection(collection_name)
                print(f"Collection '{collection_name}' already exists")
            except Exception:
                test_embedding = self._get_embedding("test")
                if test_embedding is None or len(test_embedding) == 0:
                    # Fallback: determine vector size based on embedding model
                    if 'bge-m3' in self.embedding_model:
                        vector_size = 1024
                    elif 'nomic-embed' in self.embedding_model:
                        vector_size = 768
                    else:
                        vector_size = 768
                    print(f"Could not get test embedding, using fallback vector size {vector_size} for model {self.embedding_model}")
                else:
                    vector_size = len(test_embedding)
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection '{collection_name}' with vector size {vector_size}")

    def _get_embedding(self, text: str, task_type: str = 'document') -> List[float]:
        try:
            processed_text = preprocess_for_embedding([text], task_type, self.config.get('embedding', {}))[0]
            embedding = self.ollama_api.get_embeddings(
                model=self.embedding_model,
                prompt=processed_text
            )
            return embedding if embedding is not None else []
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def _get_embeddings_batch(self, texts: List[str], task_type: str = 'document') -> List[List[float]]:
        try:
            processed_texts = preprocess_for_embedding(texts, task_type, self.config.get('embedding', {}))
            embeddings = []
            batch_size = self.config.get('embedding', {}).get('batch_size', 32)
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                for text in batch:
                    embedding = self.ollama_api.get_embeddings(
                        model=self.embedding_model,
                        prompt=text
                    )
                    embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            print(f"Error getting batch embeddings: {e}")
            return []
    
    def _chunk_text_with_metadata(self, text: str, base_metadata: Dict) -> List[tuple]:
        return self.chunker.chunk_text(text, base_metadata)
    
    def _extract_json_content_for_embedding(self, json_data, section_path: str = "") -> List[Dict]:
        content_items = []
        
        if isinstance(json_data, list):
            content_items = self._extract_json_array_content(json_data)
        
        elif isinstance(json_data, dict):
            if 'program' in json_data and 'sections' in json_data:
                content_items = self._extract_json_content(json_data)
            
            else:
                content_items = self._extract_dict_json_content(json_data)
        
        return content_items
    
    def _extract_json_array_content(self, json_array: List) -> List[Dict]:
        content_items = []
        
        for item in json_array:
            if isinstance(item, dict):
                if 'title' in item and 'url' in item:
                    text_content = f"Title: {item.get('title', '')}\n"
                    text_content += f"URL: {item.get('url', '')}\n"
                    text_content += f"Category: {item.get('category', '')}\n"
                    text_content += f"Description: {item.get('description', '')}\n"
                    
                    content_items.append({
                        'text': text_content.strip(),
                        'section_type': 'resource_link',
                        'section_name': item.get('category', 'General Resource'),
                        'section_classification': 'Resource Links'
                    })
                
                else:
                    text_parts = []
                    for key, value in item.items():
                        if isinstance(value, (str, int, float, bool)):
                            text_parts.append(f"{key.title()}: {value}")
                    
                    if text_parts:
                        content_items.append({
                            'text': '\n'.join(text_parts),
                            'section_type': 'structured_data',
                            'section_name': 'Data Entry',
                            'section_classification': 'Structured Information'
                        })
        
        return content_items
    
    def _extract_dict_json_content(self, json_dict: Dict) -> List[Dict]:
        content_items = []
        
        text_parts = []
        for key, value in json_dict.items():
            if isinstance(value, (str, int, float, bool)):
                text_parts.append(f"{key.title()}: {value}")
            elif isinstance(value, list) and value and isinstance(value[0], str):
                text_parts.append(f"{key.title()}: {', '.join(value)}")
        
        if text_parts:
            content_items.append({
                'text': '\n'.join(text_parts),
                'section_type': 'general_info',
                'section_name': 'General Information',
                'section_classification': 'Information'
            })
        
        return content_items
    
    def _extract_json_content(self, json_data: Dict) -> List[Dict]:
        content_items = []
        
        header_text = ""
        top_level_keys = self.config.get('ingestion', {}).get('json_top_level_keys', [])
        for key in top_level_keys:
            if key in json_data:
                header_text += f"{key.replace('_', ' ').title()}: {json_data[key]}\n"
        
        if 'requirements' in json_data:
            req = json_data['requirements']
            header_text += f"\nRequirements:\n"
            if isinstance(req, dict):
                for req_key, req_val in req.items():
                    if isinstance(req_val, dict):
                        for sub_key, sub_val in req_val.items():
                            header_text += f"• {sub_key.replace('_', ' ').title()}: {sub_val}\n"
                    else:
                        header_text += f"• {req_key.replace('_', ' ').title()}: {req_val}\n"
        
        if 'total_credits' in json_data:
            header_text += f"\nTotal Credits: {json_data['total_credits']}\n"
        
        content_items.append({
            'text': header_text.strip(),
            'section_type': 'overview',
            'section_name': 'Overview'
        })
        
        for section in json_data.get('sections', []):
            section_content = self._process_json_section(section)
            content_items.extend(section_content)
        
        if 'sections' in json_data:
            summary_text = f"Program Structure Summary for {json_data.get('program', 'Unknown Program')}:\n\n"
            summary_text += "This program consists of the following requirement categories:\n"
            
            for section in json_data['sections']:
                section_name = section.get('name', 'Unknown Section')
                section_credits = section.get('credits', 'N/A')
                classification = self._classify_section(section_name)
                summary_text += f"• {classification}: {section_name} ({section_credits} credits)\n"
            
            if 'total_credits' in json_data:
                summary_text += f"\nTotal Program Credits: {json_data['total_credits']}"
            
            content_items.append({
                'text': summary_text.strip(),
                'section_type': 'program_structure',
                'section_name': 'Program Structure Summary'
            })
        
        return content_items
    
    def _process_json_section(self, section: Dict) -> List[Dict]:
        content_items = []
        
        section_name = section.get('name', 'Unknown Section')
        section_credits = section.get('credits', 'N/A')
        section_classification = self._classify_section(section_name)
        
        section_text = f"Section: {section_name}\n"
        section_text += f"Classification: {section_classification}\n"
        section_text += f"Credits: {section_credits}\n"
        
        if 'notes' in section:
            section_text += f"Notes: {section['notes']}\n"
        
        if section.get('courses'):
            section_text += f"\n{section_classification} Courses:\n"
            for course in section['courses']:
                course_text = f"• {course.get('course_number', '')}: {course.get('name', '')}\n"
                if course.get('prerequisite'):
                    course_text += f"  Prerequisite: {course['prerequisite']}\n"
                if course.get('description'):
                    course_text += f"  Description: {course['description']}\n"
                section_text += course_text
        
        if 'math_sequences' in section:
            section_text += f"\nMathematics Sequence Options:\n"
            for i, seq in enumerate(section['math_sequences'], 1):
                seq_text = f"Math Sequence Option {i}:\n"
                for course in seq.get('courses', []):
                    seq_text += f"  • {course.get('course_number', '')}: {course.get('name', '')}\n"
                    if course.get('description'):
                        seq_text += f"    {course['description']}\n"
                section_text += seq_text + "\n"
        
        if 'approved_sequences' in section:
            section_text += f"\nApproved Course Sequences:\n"
            for i, seq in enumerate(section['approved_sequences'], 1):
                seq_text = f"Approved Sequence {i}:\n"
                for course in seq.get('courses', []):
                    seq_text += f"  • {course.get('course_number', '')}: {course.get('name', '')}\n"
                    if course.get('description'):
                        seq_text += f"    {course['description']}\n"
                section_text += seq_text + "\n"
        
        content_items.append({
            'text': section_text.strip(),
            'section_type': 'section',
            'section_name': section_name,
            'section_classification': section_classification
        })
        
        return content_items
    
    def _classify_section(self, section_name: str) -> str:
        return section_name
    
    def _extract_metadata_from_path(self, file_path: str) -> Dict:
        return self.metadata_extractor.extract_metadata_from_path(file_path)
    
    def _get_collection_name_from_document_type(self, document_type: str) -> str:
        if 'document_type_mapping' in self.config.get('domain', {}):
            mapping = self.config['domain']['document_type_mapping']
            collection_key = mapping.get(document_type)
            if collection_key and collection_key in self.config['qdrant']['collections']:
                return self.config['qdrant']['collections'][collection_key]
        
        collections = self.config['qdrant']['collections']
        if len(collections) == 1:
            return list(collections.values())[0]
        
        doc_type_lower = document_type.lower().replace('_', '')
        for key, collection_name in collections.items():
            if doc_type_lower in key.lower() or doc_type_lower in collection_name.lower():
                return collection_name
        
        return list(collections.values())[0]
    
    def ingest_json_file(self, file_path: str) -> bool:
        try:
            print(f"Ingesting JSON: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            file_metadata = self._extract_metadata_from_path(file_path)
            content_items = self._extract_json_content_for_embedding(json_data)
            
            if not content_items:
                print(f"  Warning: No content extracted from {file_path}")
                return False
            
            collection_name = self._get_collection_name_from_document_type(file_metadata['DocumentType'])
            
            MAX_CHUNK_SIZE = 2000
            points = []
            
            for item in content_items:
                combined_metadata = {
                    **file_metadata,
                    'section_type': item.get('section_type', 'unknown'),
                    'section_name': item.get('section_name', 'Unknown Section'),
                    'section_classification': item.get('section_classification', 'Program Requirements'),
                    'content_type': 'structured_json'
                }
                
                chunk_text = item['text']
                
                if len(chunk_text) > MAX_CHUNK_SIZE:
                    chunk_data = self._chunk_text_with_metadata(chunk_text, combined_metadata)
                    for sub_chunk_text, sub_chunk_metadata in chunk_data:
                        embedding = self._get_embedding(sub_chunk_text)
                        if not embedding:
                            continue
                        
                        sub_chunk_metadata['chunk_text'] = sub_chunk_text
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload=sub_chunk_metadata
                        )
                        points.append(point)
                else:
                    embedding = self._get_embedding(chunk_text)
                    if not embedding:
                        continue
                    
                    combined_metadata['chunk_text'] = chunk_text
                    combined_metadata['total_chunks'] = 1
                    
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=combined_metadata
                    )
                    points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"Ingested {len(points)} chunks from {len(content_items)} sections into collection '{collection_name}'")
                return True
            else:
                print(f"No valid chunks created for {file_path}")
                return False
        
        except Exception as e:
            print(f"Error ingesting JSON {file_path}: {e}")
            return False
    
    def ingest_file(self, file_path: str) -> bool:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.json':
            return self.ingest_json_file(file_path)
        elif file_extension == '.pdf':
            return self.ingest_pdf_file(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return False
    
    def ingest_pdf_file(self, file_path: str) -> bool:
        try:
            print(f"Ingesting PDF: {file_path}")
            
            text, tika_metadata = extract_content(file_path)
            if not text or len(text.strip()) < 10:
                print(f"  Warning: No content extracted from {file_path}")
                return False
            
            file_metadata = self._extract_metadata_from_path(file_path)
            combined_metadata = {**tika_metadata, **file_metadata}
            
            collection_name = self._get_collection_name_from_document_type(file_metadata['DocumentType'])
            
            chunk_data = self._chunk_text_with_metadata(text, combined_metadata)
            
            points = []
            for i, (chunk_text, chunk_metadata) in enumerate(chunk_data):
                embedding = self._get_embedding(chunk_text)
                if not embedding:
                    continue
                
                chunk_metadata['chunk_text'] = chunk_text
                chunk_metadata['total_chunks'] = len(chunk_data)
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=chunk_metadata
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"Ingested {len(points)} chunks into collection '{collection_name}'")
                return True
            else:
                print(f"No valid chunks created for {file_path}")
                return False
                
        except Exception as e:
            print(f"Error ingesting PDF {file_path}: {e}")
            return False
    
    def ingest_directory(self, directory: str, file_extensions: List[str] = None) -> Dict:
        if file_extensions is None:
            file_extensions = ['.pdf', '.json']
        
        stats = {
            'total_files': 0,
            'success_files': 0,
            'failed_files': 0,
            'collections_used': set()
        }
        
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return stats
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                if 'readme' in file_path.name.lower():
                    continue
                if file_path.name.startswith('._'):
                    continue
                    
                stats['total_files'] += 1
                
                if self.ingest_file(str(file_path)):
                    stats['success_files'] += 1
                else:
                    stats['failed_files'] += 1
        
        return stats
    
    def bulk_ingest(self, data_directories: List[str]) -> Dict:
        total_stats = {
            'total_files': 0,
            'success_files': 0,
            'failed_files': 0,
            'collections_used': set()
        }
        
        for directory in data_directories:
            print(f"\n=== Ingesting from {directory} ===")
            stats = self.ingest_directory(directory)
            
            total_stats['total_files'] += stats['total_files']
            total_stats['success_files'] += stats['success_files']
            total_stats['failed_files'] += stats['failed_files']
            total_stats['collections_used'].update(stats['collections_used'])
            
            print(f"Directory stats: {stats['success_files']}/{stats['total_files']} files successful")
        
        return total_stats
    
    def print_collection_summary(self):
        print("\n=== Collection Summary ===")
        collections = self.config['qdrant']['collections']
        
        for collection_name in collections.values():
            try:
                info = self.client.get_collection(collection_name)
                print(f"{collection_name}: {info.points_count} documents")
            except Exception as e:
                print(f"{collection_name}: Error - {e}")
    
    def clear_collections(self):
        collections = self.config['qdrant']['collections']
        
        for collection_name in collections.values():
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted collection '{collection_name}'")
            except Exception as e:
                print(f"Error deleting collection '{collection_name}': {e}")
        
        self._ensure_collections_exist()


def main():
    print("Starting unified RAG ingestion...")
    
    ingestion = UnifiedIngestion()
    config = load_config()
    data_dirs = []
    
    data_config = config.get('data', {})
    for key, path in data_config.items():
        if isinstance(path, str) and os.path.exists(path):
            data_dirs.append(path)
            print(f"Added {key} directory: {path}")
    
    if not data_dirs:
        print("No data directories found to process!")
        return
    
    stats = ingestion.bulk_ingest(data_dirs)
    
    print(f"\n=== Final Results ===")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Successfully ingested: {stats['success_files']}")
    print(f"Failed: {stats['failed_files']}")
    print(f"Collections used: {stats['collections_used']}")
    ingestion.print_collection_summary()


if __name__ == "__main__":
    main()
