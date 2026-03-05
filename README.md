# Core_RAG
Core RAG pipeline, complete wiht ingestion, hybrid retrieval, and reranking.

## Testing

`docker compose up -d`: start up local services (Qdrant and PostgreSQL)

Setup an SSH tunnel in the background (not necessary for MLAT):
`ssh -fN -L 10001:localhost:10001 -L 10002:localhost:10002 dgx_cluster`

Run tests with `scripts/run_tests.sh`
