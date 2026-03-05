"""
Compresses a batch of conversation messages into a single summary using the
intermediate LLM (spencerau-intermediate-llm on the DGX cluster).
"""

import logging
from typing import List, Dict

from core_rag.utils.ollama_api import get_intermediate_ollama_api
from core_rag.utils.config_loader import load_config

logger = logging.getLogger(__name__)

_COMPRESSION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Given a set of conversation turns between a user and an AI assistant, "
    "produce a concise factual summary that captures: "
    "the key topics discussed, any important facts or answers provided, "
    "and any context that would be relevant to understanding future messages. "
    "Be brief but complete. Do not editorialize."
)

_COMPRESSION_USER_TEMPLATE = (
    "Summarize the following conversation turns into a single concise paragraph "
    "that can be used as context for future messages:\n\n{turns}"
)


def compress_messages(messages: List[Dict], config: dict = None) -> str:
    """
    Call the intermediate LLM to produce a summary of the given message list.

    Parameters
    ----------
    messages : list of dicts with keys 'role' and 'content', ordered by message_index.
    config   : loaded config dict; loads from file if None.

    Returns
    -------
    The summary string, or a plain concatenation fallback if the LLM call fails.
    """
    if config is None:
        config = load_config()

    # Format messages as readable turns
    lines = []
    for msg in messages:
        role_label = "User" if msg['role'] == 'user' else "Assistant"
        lines.append(f"{role_label}: {msg['content']}")
    turns_text = "\n".join(lines)

    user_prompt = _COMPRESSION_USER_TEMPLATE.format(turns=turns_text)

    api = get_intermediate_ollama_api()
    router_model = config.get('query_router', {}).get('router_model', 'gpt-oss:20b')
    router_timeout = config.get('llm', {}).get('router_timeout', 120)

    llm_messages = [
        {'role': 'system', 'content': _COMPRESSION_SYSTEM_PROMPT},
        {'role': 'user',   'content': user_prompt},
    ]

    options = {
        'temperature': 0.1,
        'num_predict': 400,
    }

    try:
        summary = api.chat(
            model=router_model,
            messages=llm_messages,
            stream=False,
            options=options,
        )
        return summary.strip()
    except Exception as exc:
        logger.warning("Compression LLM call failed (%s); falling back to truncation.", exc)
        # Fallback: return a plain truncated version of the turns
        fallback = turns_text[:1000]
        if len(turns_text) > 1000:
            fallback += " [truncated]"
        return fallback
