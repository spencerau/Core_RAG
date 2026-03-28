from typing import List, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ..utils.config_loader import load_config
from .schemas import RouterOutput


class QueryRouter:
    def __init__(self, ollama_api=None):
        self.config = load_config()
        self.ollama_api = ollama_api
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        try:
            router_yaml = load_config('query_router.yaml')
            return router_yaml.get('prompt_template', '').strip()
        except FileNotFoundError:
            return ''

    def route_query(self, query: str, conversation_history: List[Dict] = None,
                   user_context: Dict = None, method: str = 'llm') -> Dict:
        if not self.ollama_api or method == 'simple':
            return self.route_simple(query)

        return self.route_with_llm_analysis(query, conversation_history, user_context)

    def route_simple(self, query: str) -> Dict:
        query_lower = query.lower()
        collections = []

        router_config = self.config.get('query_router', {})
        collection_keywords = router_config.get('collection_keywords', {})
        collection_descriptions = router_config.get('collection_descriptions', {})

        default_collections = router_config.get('default_collections',
                                                list(collection_descriptions.keys())[:1] if collection_descriptions else [])

        for collection_name, keywords in collection_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                collections.append(collection_name)

        if not collections:
            collections = default_collections

        base_tokens = self.config.get('llm', {}).get('max_tokens', 15000)
        token_allocation = base_tokens
        if len(collections) > 2:
            token_allocation = int(base_tokens * 1.3)
        elif 'plan' in query_lower or 'schedule' in query_lower:
            token_allocation = int(base_tokens * 1.6)

        return RouterOutput(
            collections=list(set(collections)),
            token_allocation=min(max(token_allocation, 150), 2000),
            reasoning=f'Keyword-based routing: {", ".join(collections)}',
            confidence=0.6
        ).model_dump()

    def route_with_llm_analysis(self, query: str, conversation_history: List[Dict] = None,
                               user_context: Dict = None) -> Dict:
        router_config = self.config.get('query_router', {})
        default_collections = router_config.get('default_collections', list(router_config.get('collection_descriptions', {}).keys()))

        int_llm_config = self.config.get('intermediate_llm', {})

        if not self.ollama_api:
            return RouterOutput(
                collections=default_collections,
                token_allocation=min(int_llm_config.get('max_tokens', 2000), 2000),
                reasoning='No LLM available for routing',
                confidence=0.0
            ).model_dump()

        collection_descriptions = router_config.get('collection_descriptions', {})
        min_tokens = router_config.get('min_tokens', 150)
        max_tokens = router_config.get('max_tokens', 2000)

        collections_desc = "\n".join([
            f"- {name}: {desc}"
            for name, desc in collection_descriptions.items()
        ])

        context_parts = []
        if user_context:
            for key, value in user_context.items():
                if value:
                    context_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        context = "\n".join(context_parts) if context_parts else "No user context provided"

        conversation_context = ""
        if conversation_history:
            last_n = router_config.get('last_n_messages', 3)
            recent = conversation_history[-last_n:] if len(conversation_history) > last_n else conversation_history
            for msg in recent:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation_context += f"{role.capitalize()}: {content}\n"

        prompt_template = self._prompt_template or router_config.get('prompt_template', '')
        if not prompt_template:
            raise RuntimeError("No prompt template found. Ensure configs/query_router.yaml exists with a prompt_template key.")

        prompt = prompt_template.format(
            router_context=self.config.get('domain', {}).get('router_context', 'a chatbot'),
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            collections_desc=collections_desc,
            context=context,
            conversation_context=conversation_context if conversation_context else "No previous conversation",
            query=query
        )

        router_think = int_llm_config.get('think', router_config.get('think', False))
        sampling = int_llm_config.get('sampling', router_config.get('sampling', {}))
        options = {
            'temperature': int_llm_config.get('temperature', router_config.get('temperature', 0.1)),
            'num_predict': int_llm_config.get('max_tokens', router_config.get('max_tokens', 500)),
            **sampling
        }
        try:
            result_raw = self.ollama_api.chat_with_thinking(
                model=int_llm_config.get('model', router_config.get('model', 'qwen3.5:9b')),
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
                think=router_think,
                format=RouterOutput.model_json_schema(),
                options=options
            )
            if result_raw.get('thinking'):
                print(f"Router thinking: {result_raw['thinking']}")
            response = result_raw.get('content', '')

            if not response or not response.strip():
                return RouterOutput(
                    collections=default_collections,
                    token_allocation=min(llm_config.get('max_tokens', 2000), 2000),
                    reasoning='Empty response from router LLM',
                    confidence=0.0
                ).model_dump()

            result = RouterOutput.model_validate_json(response)

            # Clamp token allocation to configured range and filter invalid collections
            clamped_tokens = max(min_tokens, min(result.token_allocation, max_tokens))
            valid_collections = [c for c in result.collections if c in collection_descriptions]
            if not valid_collections:
                valid_collections = default_collections

            return RouterOutput(
                collections=valid_collections,
                token_allocation=clamped_tokens,
                reasoning=result.reasoning,
                confidence=result.confidence
            ).model_dump()

        except Exception as e:
            print(f"LLM routing error: {e}")
            return RouterOutput(
                collections=default_collections,
                token_allocation=min(llm_config.get('max_tokens', 2000), 2000),
                reasoning=f'Error: {str(e)}',
                confidence=0.0
            ).model_dump()


def create_router(ollama_api=None):
    try:
        return QueryRouter(ollama_api)
    except Exception as e:
        print(f"Error creating query router: {e}")
        return None
