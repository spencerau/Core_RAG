from pypdf import PdfReader, PdfWriter
import re
import os
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ..utils.config_loader import load_config


class MetadataExtractor:
    def __init__(self):
        self.config = load_config()
        self.metadata_config = self.config.get('metadata', {})
    
    def get_subject_mappings(self):
        mappings = {}
        domain_mappings = self.metadata_config.get('subject_mappings', {})
        
        for pattern, info in domain_mappings.items():
            if isinstance(info, dict):
                name = info.get('name', pattern)
                code = info.get('code', pattern)
                mappings[pattern] = (name, code)
            elif isinstance(info, (list, tuple)) and len(info) == 2:
                mappings[pattern] = tuple(info)
        
        return mappings
    
    def extract_metadata_from_path(self, file_path: str) -> dict:
        path = Path(file_path)
        subject_mappings = self.get_subject_mappings()
        
        metadata = {}
        
        year_pattern = self.metadata_config.get('year_pattern', r'20\d{2}')
        for part in path.parts:
            if re.match(f'^{year_pattern}$', part):
                metadata['year'] = part
                break
        
        path_mappings = self.metadata_config.get('path_mappings', {})
        path_str = str(path).lower()
        
        for path_pattern, mapping in path_mappings.items():
            if path_pattern.lower() in path_str:
                for key, value in mapping.items():
                    metadata[key] = value
                break
        
        filename_pattern = self.metadata_config.get('filename_pattern', r'(\d{4})_(.+)\.(pdf|json)')
        match = re.search(filename_pattern, path.name)
        if match:
            if 'year' not in metadata and len(match.groups()) >= 1:
                metadata['year'] = match.group(1)
            
            if len(match.groups()) >= 2:
                subject_part = match.group(2)
                
                for pattern, (full_name, code) in subject_mappings.items():
                    if pattern in subject_part:
                        metadata['subject'] = full_name
                        metadata['subject_code'] = code
                        break
                else:
                    metadata['subject'] = subject_part.replace('_', ' ')
                    metadata['subject_code'] = subject_part.lower()
        
        return metadata
    
    def add_pdf_metadata(self, pdf_path: str, metadata: dict) -> None:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        for page in reader.pages:
            writer.add_page(page)
        
        pdf_metadata = {f"/{k}": str(v) for k, v in metadata.items() if v}
        writer.add_metadata(pdf_metadata)
        
        with open(pdf_path, "wb") as f:
            writer.write(f)
    
    def add_json_metadata(self, json_path: str, metadata: dict) -> bool:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["metadata"] = {k: v for k, v in metadata.items() if v}
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            return False
    
    def process_files_in_directory(self, base_dir: str, file_extension: str) -> None:
        if not os.path.exists(base_dir):
            return
        
        for year_dir in os.listdir(base_dir):
            year_path = os.path.join(base_dir, year_dir)
            if not (os.path.isdir(year_path) and year_dir.isdigit()):
                continue
                
            print(f"Processing {file_extension} files in {base_dir} for year: {year_dir}")
            
            for filename in os.listdir(year_path):
                if not filename.endswith(file_extension):
                    continue
                if filename.startswith("backup_"):
                    continue
                    
                full_path = os.path.join(year_path, filename)
                metadata = self.extract_metadata_from_path(full_path)
                
                try:
                    if file_extension == ".pdf":
                        self.add_pdf_metadata(full_path, metadata)
                    elif file_extension == ".json":
                        self.add_json_metadata(full_path, metadata)
                    
                    print(f"Processed {filename}")
                    for key, value in metadata.items():
                        print(f"  {key}: {value}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


def main():
    print("Adding metadata to files...")
    
    extractor = MetadataExtractor()
    config = load_config()
    metadata_paths = config.get('metadata', {}).get('processing_paths', [])
    
    if not metadata_paths:
        print("No processing paths configured in metadata.processing_paths")
        return
    
    for path_config in metadata_paths:
        directory = path_config.get('directory')
        extension = path_config.get('extension')
        if directory and extension:
            extractor.process_files_in_directory(directory, extension)
    
    print("Metadata processing complete!")

if __name__ == "__main__":
    main()
