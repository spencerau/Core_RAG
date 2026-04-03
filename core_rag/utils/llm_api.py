import os
from typing import Dict, Iterator, List, Optional

from .config_loader import load_config
from .backends import get_backend
from .backends.base import BaseLLMBackend
from .backends.ollama import OllamaBackend


class OllamaAPI:
    """
    Public API for LLM/embedding calls throughout the pipeline.

    Delegates to the configured backend (OllamaBackend or OpenAICompatBackend).
    Switch backends in model.yaml:
        backend: ollama   # default, Ollama on localhost
        backend: vllm     # OpenAI-compatible (vLLM, mlx_lm, etc.)

    Embeddings always use OllamaBackend regardless of the configured backend,
    since mlx-lm / vLLM do not serve /v1/embeddings for the embedding model.
    """

    def __init__(self, base_url: str = None, timeout: int = 300):
        config = load_config()
        self._backend: BaseLLMBackend = get_backend(config, base_url=base_url, timeout=timeout)

        # Embeddings always go to Ollama (embedding.host/port)
        embedding_cfg = config.get('embedding', {})
        emb_host = embedding_cfg.get('host', 'localhost')
        emb_port = embedding_cfg.get('port', 11434)
        emb_url = f"http://{emb_host}:{emb_port}"
        self._embedding_backend: OllamaBackend = OllamaBackend(base_url=emb_url, timeout=timeout)

    def get_embeddings(self, model: str, prompt: str, keep_alive: str = None) -> List[float]:
        return self._embedding_backend.get_embeddings(model, prompt, keep_alive=keep_alive)

    def chat(self, model: str, messages: List[Dict], stream: bool = True,
             think: Optional[bool] = None, hide_thinking: bool = False,
             **kwargs) -> str:
        return self._backend.chat(model, messages, stream=stream, think=think,
                                  hide_thinking=hide_thinking, **kwargs)

    def chat_stream(self, model: str, messages: List[Dict],
                    think: Optional[bool] = None, hide_thinking: bool = False,
                    **kwargs) -> Iterator[str]:
        return self._backend.chat_stream(model, messages, think=think,
                                         hide_thinking=hide_thinking, **kwargs)

    def chat_with_thinking(self, model: str, messages: List[Dict],
                           stream: bool = True, think: bool = True, **kwargs) -> Dict[str, str]:
        return self._backend.chat_with_thinking(model, messages, stream=stream, think=think, **kwargs)

    def rerank(self, model: str, query: str, documents: List[str]) -> List[float]:
        return self._backend.rerank(model, query, documents)

    def check_model(self, model: str) -> bool:
        return self._backend.check_model(model)


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_ollama_api = None
_intermediate_ollama_api = None


def get_ollama_api(timeout: int = 300) -> OllamaAPI:
    global _ollama_api
    if _ollama_api is None:
        _ollama_api = OllamaAPI(timeout=timeout)
    return _ollama_api


def get_intermediate_ollama_api(timeout: int = 60) -> OllamaAPI:
    global _intermediate_ollama_api
    if _intermediate_ollama_api is None:
        host = os.environ.get("OLLAMA_INTERMEDIATE_HOST")
        port = os.environ.get("OLLAMA_INTERMEDIATE_PORT")

        if host and port:
            base_url = f"http://{host}:{port}"
        else:
            config = load_config()
            int_llm_config = config.get('intermediate_llm', {})
            host = int_llm_config.get('host', 'localhost')
            port = int_llm_config.get('port', 11434)
            base_url = f"http://{host}:{port}"

        _intermediate_ollama_api = OllamaAPI(base_url=base_url, timeout=timeout)
        print(f"Using intermediate Ollama API at {base_url}")
    return _intermediate_ollama_api
