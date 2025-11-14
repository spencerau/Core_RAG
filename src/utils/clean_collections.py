#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qdrant_client import QdrantClient
from utils.config_loader import load_config


def clean_all_collections(force=False):
    config = load_config()
    client = QdrantClient(
        host=config['qdrant']['host'], 
        port=config['qdrant']['port'],
        timeout=config['qdrant']['timeout']
    )
    
    try:
        collections = client.get_collections().collections
        if not collections:
            print("No collections found.")
            return
            
        print(f"Found {len(collections)} collections:")
        for collection in collections:
            print(f"  - {collection.name}")
        
        if not force:
            confirm = input("\nAre you sure you want to delete ALL collections? (y/N): ")
            if confirm.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                return
        
        for collection in collections:
            print(f"Deleting collection: {collection.name}")
            client.delete_collection(collection.name)
        print("All collections deleted successfully.")
            
    except Exception as e:
        print(f"Error cleaning collections: {e}")


if __name__ == "__main__":
    force = len(sys.argv) > 1 and sys.argv[1] == "--force"
    clean_all_collections(force=force)
