from typing import Dict, List, Generator


class LLMHandler:
    def __init__(self, config: dict, ollama_api, system_prompt: str):
        self.config = config
        self.ollama_api = ollama_api
        self.system_prompt = system_prompt
    
    def get_response(self, prompt: str, history: List[Dict] = None, tokens: int = 600) -> str:
        messages = [{'role': 'system', 'content': self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({'role': 'user', 'content': prompt})
        try:
            options = {'temperature': self.config['llm']['temperature'], 'num_predict': tokens}
            if 'num_ctx' in self.config['llm']:
                options['num_ctx'] = self.config['llm']['num_ctx']
            resp = self.ollama_api.chat(
                model=self.config['llm']['primary_model'], messages=messages, stream=False,
                options=options
            )
            return resp.strip() if resp else "I couldn't generate a response."
        except Exception as e:
            print(f"LLM error: {e}")
            return f"Error: {str(e)}"
    
    def get_response_stream(self, prompt: str, history: List[Dict] = None, 
                            tokens: int = 600) -> Generator[str, None, None]:
        messages = [{'role': 'system', 'content': self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({'role': 'user', 'content': prompt})
        try:
            options = {'temperature': self.config['llm']['temperature'], 'num_predict': tokens}
            if 'num_ctx' in self.config['llm']:
                options['num_ctx'] = self.config['llm']['num_ctx']
            for chunk in self.ollama_api.chat(
                model=self.config['llm']['primary_model'], messages=messages, stream=True,
                options=options
            ):
                yield chunk
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"Error: {str(e)}"


def format_system_prompt(config: dict) -> str:
    template = config['llm']['system_prompt']
    domain = config.get('domain', {})
    return template.format(
        role=domain.get('role', 'assistant'),
        department=domain.get('department', 'organization'),
        contact_email=domain.get('contact_email', 'support')
    )
