# Core_RAG
Core RAG pipeline, complete wiht ingestion, hybrid retrieval, and reranking.

## Starting up Ollama Containers (will eventualy replace with vLLM)

The DGX cluster still does not support running a docker-compose.yml file so use this:

For the Main LLM Container
```
docker run -d \
    --name spencerau-ollama \
    --restart unless-stopped \
    --gpus '"device=7"' \
    -p 10001:11434 \
    -e OLLAMA_FLASH_ATTENTION=1 \
    -e OLLAMA_KV_CACHE_TYPE=q8_0 \
    -e OLLAMA_NUM_GPU_LAYERS=999 \
    -e OLLAMA_CONTEXT_LENGTH=32768 \
    -v /models/ollama:/root/.ollama \
    ollama/ollama:latest
```

For the intermediate LLM (chat history summarization, query routing, etc)
```
docker run -d \
    --name spencerau-int-llm \
    --restart unless-stopped \
    --gpus '"device=6"' \
    -p 10002:11434 \
    -e OLLAMA_FLASH_ATTENTION=1 \
    -e OLLAMA_KV_CACHE_TYPE=q8_0 \
    -e OLLAMA_NUM_GPU_LAYERS=999 \
    -e OLLAMA_CONTEXT_LENGTH=32768 \
    -v /models/ollama:/root/.ollama \
    ollama/ollama:latest
```

## Testing

`docker compose up -d`: start up local services (Qdrant and PostgreSQL)

Setup an SSH tunnel in the background (not necessary for MLAT):
`ssh -fN -L 10001:localhost:10001 -L 10002:localhost:10002 dgx_cluster`

Run tests with `scripts/run_tests.sh`
