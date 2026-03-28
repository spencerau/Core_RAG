import json
import re
from typing import Dict, Iterator, List, Optional

import requests

from ..config_loader import load_config
from .base import BaseLLMBackend


class OpenAICompatBackend(BaseLLMBackend):
    """
    OpenAI-compatible backend for vLLM and mlx_lm.

    Translates the Ollama-style interface to /v1/* endpoints.
    Configure in model.yaml:
        backend: vllm
        embedding.host / port  ← reused as base URL
    """

    def __init__(self, base_url: str = None, timeout: int = 300, config: dict = None):
        if config is None:
            config = load_config()

        if base_url:
            self.base_url = base_url
        else:
            embedding_config = config.get('embedding', {})
            host = embedding_config.get('host', 'localhost')
            port = embedding_config.get('port', 11434)
            self.base_url = f"http://{host}:{port}"

        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Connection': 'keep-alive'
        })

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self, model: str, prompt: str, **kwargs) -> List[float]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": model, "input": prompt}
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json().get('data', [])
            if data:
                return data[0].get('embedding', [])
            return []
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, model: str, messages: List[Dict], stream: bool = True,
             think: Optional[bool] = None, hide_thinking: bool = False,
             **kwargs) -> str:
        if stream:
            return ''.join(self.chat_stream(model, messages, think=think,
                                            hide_thinking=hide_thinking, **kwargs))

        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(model, messages, stream=False, **kwargs)

        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content'] or ''

            if hide_thinking:
                content = self._strip_thinking_tags(content)
            elif '<think>' in content:
                content = self._format_thinking_content(content)

            return content
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return ""

    def chat_stream(self, model: str, messages: List[Dict],
                    think: Optional[bool] = None, hide_thinking: bool = False,
                    **kwargs) -> Iterator[str]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(model, messages, stream=True, **kwargs)

        try:
            response = self.session.post(url, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()

            in_thinking = False

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode()
                if not decoded.startswith('data: '):
                    continue
                chunk = decoded[len('data: '):]
                if chunk.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(chunk)
                    content = data['choices'][0].get('delta', {}).get('content') or ''
                    if not content:
                        continue

                    if hide_thinking:
                        if '<think>' in content:
                            in_thinking = True
                            content = content.replace('<think>', '')
                        if '</think>' in content:
                            in_thinking = False
                            content = content.replace('</think>', '')
                        if not in_thinking:
                            yield content
                    else:
                        if '<think>' in content:
                            content = content.replace('<think>', '\n\n---\n\n**Thinking Process:**\n\n*')
                        if '</think>' in content:
                            content = content.replace('</think>', '*\n\n---\n\n')
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        except Exception as e:
            print(f"Error in streaming chat: {e}")
            yield ""

    def chat_with_thinking(self, model: str, messages: List[Dict],
                           stream: bool = True, **kwargs) -> Dict[str, str]:
        """
        Accumulates thinking (content inside <think>…</think>) separately from
        the visible response. vLLM/mlx_lm don't have a dedicated 'think' field,
        so we extract it from the content stream.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(model, messages, stream=True, **kwargs)

        thinking = ""
        content = ""
        in_thinking = False

        try:
            response = self.session.post(url, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode()
                if not decoded.startswith('data: '):
                    continue
                chunk = decoded[len('data: '):]
                if chunk.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(chunk)
                    token = data['choices'][0].get('delta', {}).get('content') or ''
                    if not token:
                        continue

                    # Route tokens into thinking vs content buckets
                    while token:
                        if not in_thinking:
                            think_start = token.find('<think>')
                            if think_start == -1:
                                content += token
                                break
                            content += token[:think_start]
                            token = token[think_start + len('<think>'):]
                            in_thinking = True
                        else:
                            think_end = token.find('</think>')
                            if think_end == -1:
                                thinking += token
                                break
                            thinking += token[:think_end]
                            token = token[think_end + len('</think>'):]
                            in_thinking = False

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            return {"thinking": thinking, "content": content}
        except Exception as e:
            print(f"Error in chat with thinking: {e}")
            return {"thinking": "", "content": ""}

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def rerank(self, model: str, query: str, documents: List[str]) -> List[float]:
        url = f"{self.base_url}/v1/rerank"
        payload = {"model": model, "query": query, "documents": documents}
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            results = response.json().get('results', [])
            scores = [0.0] * len(documents)
            for r in results:
                idx = r.get('index', -1)
                if 0 <= idx < len(scores):
                    scores[idx] = float(r.get('relevance_score', 0.0))
            return scores
        except Exception as e:
            print(f"Error in rerank: {e}")
            return [0.0] * len(documents)

    # ------------------------------------------------------------------
    # Model check
    # ------------------------------------------------------------------

    def check_model(self, model: str) -> bool:
        url = f"{self.base_url}/v1/models"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            models = [m['id'] for m in response.json().get('data', [])]
            return model in models or any(model in m for m in models)
        except Exception as e:
            print(f"Error checking models: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_payload(self, model: str, messages: List[Dict],
                       stream: bool = False, **kwargs) -> dict:
        """
        Translate Ollama-style kwargs to OpenAI payload.
        - options.num_predict → max_tokens
        - options.temperature → temperature
        - options.num_ctx    → ignored (set at server startup)
        - format (json schema dict) → extra_body.guided_json (vLLM structured output)
        """
        payload: dict = {"model": model, "messages": messages, "stream": stream}

        options = kwargs.pop('options', {})
        if 'num_predict' in options:
            payload['max_tokens'] = options['num_predict']
        if 'temperature' in options:
            payload['temperature'] = options['temperature']

        # Structured output: Ollama uses format={schema}, vLLM uses guided_json
        fmt = kwargs.pop('format', None)
        if fmt and isinstance(fmt, dict):
            payload.setdefault('extra_body', {})['guided_json'] = fmt

        # Pass through any remaining standard OpenAI kwargs
        payload.update(kwargs)
        return payload

    def _strip_thinking_tags(self, content: str) -> str:
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    def _format_thinking_content(self, content: str) -> str:
        def replace_thinking(match):
            thinking_content = match.group(1).strip()
            return f"\n\n---\n\n**Thinking Process:**\n\n*{thinking_content}*\n\n---\n\n"

        return re.sub(r'<think>(.*?)</think>', replace_thinking, content, flags=re.DOTALL)
