import re
import json
from typing import List, Dict
from textwrap import dedent
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ..utils.config_loader import load_config


class QueryRouter:
    def __init__(self, ollama_api=None):
        self.config = load_config()
        self.ollama_api = ollama_api
    
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
        
        return {
            'collections': list(set(collections)),
            'token_allocation': token_allocation,
            'reasoning': f'Keyword-based routing: {", ".join(collections)}'
        }

    def route_with_llm_analysis(self, query: str, conversation_history: List[Dict] = None,
                               user_context: Dict = None) -> Dict:
        router_config = self.config.get('query_router', {})
        default_collections = router_config.get('default_collections', list(router_config.get('collection_descriptions', {}).keys()))
        
        if not self.ollama_api:
            return {
                'collections': default_collections,
                'token_allocation': llm_config.get('max_tokens', 15000),
                'reasoning': 'No LLM available for routing'
            }
        
        llm_config = self.config.get('llm', {})
        
        collection_descriptions = router_config.get('collection_descriptions', {})
        min_tokens = router_config.get('min_tokens', 150)
        max_tokens = router_config.get('max_tokens', 1250)
        
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
        
        prompt_template = router_config.get('prompt_template', dedent("""
            You are a routing system for {router_context}.

            Analyze the query and determine:
              1. Which knowledge collections are needed to answer it (select one or more)
              2. How many tokens the response should use (between {min_tokens} and {max_tokens})

            Available Collections:
            {collections_desc}

            User Context:
            {context}

            Recent Conversation:
            {conversation_context}

            Current Query: "{query}"

            Token Allocation Guidelines:
              - Simple lookups: {min_tokens}-300 tokens
              - Detailed information: 300-500 tokens
              - Multiple topics: 500-800 tokens
              - Complex requests: 800-{max_tokens} tokens

            Respond ONLY with valid JSON in this exact format:
            {{"collections": ["collection1", "collection2"], "token_allocation": 500, "reasoning": "brief explanation"}}
        """).strip())
        
        prompt = prompt_template.format(
            router_context=self.config.get('domain', {}).get('router_context', 'a chatbot'),
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            collections_desc=collections_desc,
            context=context,
            conversation_context=conversation_context if conversation_context else "No previous conversation",
            query=query
        )

        try:
            response = self.ollama_api.chat(
                model=router_config.get('router_model', 'gemma3:4b'),
                messages=[{'role': 'user', 'content': prompt}],
                stream=False,
                format='json',
                options={
                    'temperature': router_config.get('router_temperature', 0.1),
                    'num_predict': router_config.get('router_max_tokens', 500)
                }
            )
            
            if not response or not response.strip():
                return {
                    'collections': default_collections,
                    'token_allocation': llm_config.get('max_tokens', 15000),
                    'reasoning': 'Empty response from router LLM'
                }
            
            cleaned_response = ' '.join(response.split())
            
            try:
                result = json.loads(cleaned_response)
            except json.JSONDecodeError:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        return {
                            'collections': default_collections,
                            'token_allocation': llm_config.get('max_tokens', 15000),
                            'reasoning': 'Failed to parse router response'
                        }
                else:
                    return {
                        'collections': default_collections,
                        'token_allocation': llm_config.get('max_tokens', 15000),
                        'reasoning': 'No valid JSON found in response'
                    }
            collections = result.get('collections', default_collections)
            token_allocation = result.get('token_allocation', llm_config.get('max_tokens', 15000))
            reasoning = result.get('reasoning', 'LLM analysis')
            
            token_allocation = max(min_tokens, min(token_allocation, max_tokens))
            
            valid_collections = [c for c in collections if c in collection_descriptions]
            if not valid_collections:
                valid_collections = default_collections
            
            return {
                'collections': valid_collections,
                'token_allocation': token_allocation,
                'reasoning': reasoning
            }
                
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            print(f"Response was: {response[:500] if 'response' in locals() else 'No response'}")
            return {
                'collections': default_collections,
                'token_allocation': llm_config.get('max_tokens', 15000),
                'reasoning': f'JSON decode failed: {str(je)}'
            }
        except Exception as e:
            print(f"LLM routing error: {e}")
            return {
                'collections': default_collections,
                'token_allocation': llm_config.get('max_tokens', 15000),
                'reasoning': f'Error: {str(e)}'
            }


def create_router(ollama_api=None):
    try:
        return QueryRouter(ollama_api)
    except Exception as e:
        print(f"Error creating query router: {e}")
        return None
