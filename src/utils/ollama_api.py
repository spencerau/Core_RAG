import requests
import json
from typing import List, Dict, Any, Iterator, Optional
import os

from .config_loader import load_config


class OllamaAPI:
    
    def __init__(self, base_url: str = None, timeout: int = 300):
        config = load_config()
        
        if base_url:
            self.base_url = base_url
        else:
            cluster_config = config.get('cluster', {})
            if cluster_config.get('enabled', False):
                host = cluster_config.get('ollama_host', 'localhost')
                port = cluster_config.get('ollama_port', 11434)
                self.base_url = f"http://{host}:{port}"
                print(f"Using cluster Ollama at {self.base_url}")
            else:
                embedding_config = config.get('embedding', {})
                host = embedding_config.get('ollama_host', self._get_ollama_host())
                port = embedding_config.get('ollama_port', 11434)
                self.base_url = f"http://{host}:{port}"
        
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Connection': 'keep-alive'
        })
    
    def _get_ollama_host(self) -> str:
        return os.environ.get("OLLAMA_HOST", "localhost")
    
    def get_embeddings(self, model: str, prompt: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": prompt}
        
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get('embedding', [])
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []
    
    def chat(self, model: str, messages: List[Dict], stream: bool = True, think: Optional[bool] = None, 
             hide_thinking: bool = False, **kwargs) -> str:
        if stream:
            return ''.join(self.chat_stream(model, messages, think=think, hide_thinking=hide_thinking, **kwargs))
        
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        if think is not None:
            payload["think"] = think
        
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            content = data.get('message', {}).get('content', '')
            
            if hide_thinking and content:
                content = self._strip_thinking_tags(content)
            elif not hide_thinking and content and '<think>' in content:
                content = self._format_thinking_content(content)
            
            return content
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return ""
    
    def chat_stream(self, model: str, messages: List[Dict], think: Optional[bool] = None, 
                    hide_thinking: bool = False, **kwargs) -> Iterator[str]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        if think is not None:
            payload["think"] = think
        
        try:
            response = self.session.post(url, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            accumulated_content = ""
            in_thinking = False
            
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode())
                    if 'message' in data and 'content' in data['message']:
                        content = data['message']['content']
                        accumulated_content += content
                        
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
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"Error in streaming chat: {e}")
            yield ""
    
    def chat_with_thinking(self, model: str, messages: List[Dict], stream: bool = True, **kwargs) -> Dict[str, str]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": True,
            **kwargs
        }
        
        if stream:
            thinking = ""
            content = ""
            
            try:
                response = self.session.post(url, json=payload, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode())
                        if 'message' in data:
                            if 'thinking' in data['message']:
                                thinking += data['message']['thinking']
                            if 'content' in data['message']:
                                content += data['message']['content']
                    except json.JSONDecodeError:
                        continue
                        
                return {"thinking": thinking, "content": content}
            except Exception as e:
                print(f"Error in streaming chat with thinking: {e}")
                return {"thinking": "", "content": ""}
        else:
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                message = data.get('message', {})
                return {
                    "thinking": message.get('thinking', ''),
                    "content": message.get('content', '')
                }
            except Exception as e:
                print(f"Error in chat with thinking: {e}")
                return {"thinking": "", "content": ""}
    
    def check_model(self, model: str) -> bool:
        url = f"{self.base_url}/api/tags"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            return model in models or any(model in m for m in models)
        except Exception as e:
            print(f"Error checking models: {e}")
            return False
    
    def _strip_thinking_tags(self, content: str) -> str:
        import re
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    def _format_thinking_content(self, content: str) -> str:
        import re
        
        def replace_thinking(match):
            thinking_content = match.group(1).strip()
            return f"\n\n---\n\n**Thinking Process:**\n\n*{thinking_content}*\n\n---\n\n"
        
        formatted = re.sub(r'<think>(.*?)</think>', replace_thinking, content, flags=re.DOTALL)
        return formatted


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
            llm_config = config.get('llm', {})
            host = llm_config.get('router_host', 'localhost')
            port = llm_config.get('router_port', 11435)
            base_url = f"http://{host}:{port}"
        
        _intermediate_ollama_api = OllamaAPI(base_url=base_url, timeout=timeout)
        print(f"Using intermediate Ollama API at {base_url}")
    return _intermediate_ollama_api
