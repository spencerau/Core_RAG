from core_rag.retrieval.unified_rag import UnifiedRAG

print("=== Testing Search ===")
rag = UnifiedRAG()

print("\n=== Direct Search Test ===")
results = rag.search_collection(
    query="Krabby Patty instructions",
    collection_name="main_collection",
    top_k=5
)

print(f"Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Keys: {list(result.keys())}")
    print(f"  Score: {result.get('score', 'N/A')}")
    print(f"  Full result: {result}"[:300])
