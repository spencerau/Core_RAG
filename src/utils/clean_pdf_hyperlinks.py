#!/usr/bin/env python3
import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from pypdf import PdfReader, PdfWriter


def remove_hyperlinks_from_pdf(input_path, output_path=None, safe_mode=False):
    if output_path is None:
        path_obj = Path(input_path)
        output_path = str(path_obj.parent / f"clean_{path_obj.name}")
    
    reader = PdfReader(input_path)
    writer = PdfWriter()
    
    for i, page in enumerate(reader.pages):
        writer.add_page(page)
        
        if safe_mode:
            try:
                if '/Annots' in writer.pages[i]:
                    del writer.pages[i]['/Annots']
            except:
                pass
        else:
            try:
                if '/Annots' in writer.pages[i]:
                    try:
                        annotations = writer.pages[i]['/Annots']
                        if annotations:
                            try:
                                non_link_annots = []
                                for j in range(len(annotations)):
                                    try:
                                        annot = annotations[j].get_object()
                                        if isinstance(annot, dict) and annot.get('/Subtype') != '/Link':
                                            non_link_annots.append(annotations[j])
                                    except:
                                        pass
                                
                                if len(non_link_annots) > 0:
                                    writer.pages[i]['/Annots'] = non_link_annots
                                else:
                                    del writer.pages[i]['/Annots']
                            except:
                                del writer.pages[i]['/Annots']
                    except:
                        if '/Annots' in writer.pages[i]:
                            del writer.pages[i]['/Annots']
            except:
                pass
        
    with open(output_path, "wb") as output_file:
        writer.write(output_file)
    
    return output_path


def process_file(file_path, backup=True, safe_mode=False):
    try:
        if backup:
            path_obj = Path(file_path)
            backup_path = str(path_obj.parent / f"backup_{path_obj.name}")
            if not os.path.exists(backup_path):
                shutil.copy2(file_path, backup_path)
                print(f"Created backup: {backup_path}")
                
        temp_output = file_path + ".tmp"
        try:
            remove_hyperlinks_from_pdf(file_path, temp_output, safe_mode=safe_mode)
            os.replace(temp_output, file_path)
            print(f"Successfully processed: {file_path}")
            return True
        except Exception as e:
            print(f"Processing failed for {file_path}: {e}")
            return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def process_directory_recursive(directory_path, make_backup=True, safe_mode=False):
    success = 0
    failed = 0
    
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(root, filename)
                if process_file(file_path, backup=make_backup, safe_mode=safe_mode):
                    success += 1
                else:
                    failed += 1
    
    return success, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove hyperlinks from PDF files")
    parser.add_argument("path", help="PDF file or directory to process")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backups")
    parser.add_argument("--safe", action="store_true", help="Use safe mode (remove all annotations)")
    
    args = parser.parse_args()
    
    target_path = args.path
    make_backup = not args.no_backup
    safe_mode = args.safe
    
    if os.path.isdir(target_path):
        success, failed = process_directory_recursive(target_path, make_backup, safe_mode)
        print(f"Directory processing completed: {success} successful, {failed} failed")
        
    elif os.path.isfile(target_path) and target_path.lower().endswith('.pdf'):
        if process_file(target_path, backup=make_backup, safe_mode=safe_mode):
            print("Processing completed successfully")
        else:
            print("Processing failed")
            sys.exit(1)
    else:
        print(f"Invalid path: {target_path}")
        sys.exit(1)
