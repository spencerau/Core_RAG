from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional


class BaseLLMBackend(ABC):

    @abstractmethod
    def get_embeddings(self, model: str, prompt: str, **kwargs) -> List[float]:
        ...

    @abstractmethod
    def chat(self, model: str, messages: List[Dict], stream: bool = True,
             think: Optional[bool] = None, hide_thinking: bool = False,
             **kwargs) -> str:
        ...

    @abstractmethod
    def chat_stream(self, model: str, messages: List[Dict],
                    think: Optional[bool] = None, hide_thinking: bool = False,
                    **kwargs) -> Iterator[str]:
        ...

    @abstractmethod
    def chat_with_thinking(self, model: str, messages: List[Dict],
                           stream: bool = True, think: bool = True, **kwargs) -> Dict[str, str]:
        ...

    @abstractmethod
    def rerank(self, model: str, query: str, documents: List[str]) -> List[float]:
        ...

    @abstractmethod
    def check_model(self, model: str) -> bool:
        ...
