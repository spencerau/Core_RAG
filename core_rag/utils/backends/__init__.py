from .base import BaseLLMBackend
from .ollama import OllamaBackend
from .openai_compat import OpenAICompatBackend


def get_backend(config: dict, base_url: str = None, timeout: int = 300) -> BaseLLMBackend:
    name = config.get('backend', 'ollama').lower()
    if name in ('vllm', 'openai', 'mlx'):
        return OpenAICompatBackend(base_url=base_url, timeout=timeout, config=config, backend_name=name)
    return OllamaBackend(base_url=base_url, timeout=timeout)


__all__ = ['BaseLLMBackend', 'OllamaBackend', 'OpenAICompatBackend', 'get_backend']
