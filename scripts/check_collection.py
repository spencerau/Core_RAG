from qdrant_client import QdrantClient
from core_rag.utils.config_loader import load_config

config = load_config()
client = QdrantClient(
    host=config['qdrant']['host'],
    port=config['qdrant']['port'],
    timeout=config['qdrant']['timeout']
)

print("=== Main Collection Contents ===")
info = client.get_collection("main_collection")
print(f"Points: {info.points_count}")

# Get all points
points, _ = client.scroll(collection_name="main_collection", limit=100)
for i, point in enumerate(points, 1):
    print(f"\nPoint {i}:")
    print(f"  ID: {point.id}")
    if 'chunk_text' in point.payload:
        text = point.payload['chunk_text'][:150]
        print(f"  Text: {text}...")
    if 'source_path' in point.payload:
        print(f"  Source: {point.payload['source_path']}")
