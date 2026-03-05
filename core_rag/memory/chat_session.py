"""
High-level chat session manager.

Usage
-----
from core_rag.memory import ChatSession

session = ChatSession(user_id="alice")
answer = session.chat("What is a Krabby Patty?")

# Resume an existing session
session2 = ChatSession(user_id="alice", session_id=session.session_id)
"""

import logging
from typing import Generator, List, Dict, Optional

from core_rag.utils.config_loader import load_config
from core_rag.retrieval.unified_rag import UnifiedRAG
from core_rag.memory.db import init_db
from core_rag.memory import session_store, compressor

logger = logging.getLogger(__name__)


class ChatSession:
    """
    Wraps UnifiedRAG with persistent chat history stored in PostgreSQL.

    Every `compression_trigger` user messages, the oldest uncompressed batch is
    summarised by the intermediate LLM, the raw messages are moved to
    archived_messages for auditing, and the summary is prepended to context on
    subsequent calls.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str = None,
        config: dict = None,
    ):
        self.config = config or load_config()
        self.user_id = user_id
        self.compression_trigger = (
            self.config.get('memory', {}).get('compression_trigger', 5)
        )

        # Ensure schema exists
        init_db(self.config)

        # Get or create the session row
        self.session_id = session_store.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            config=self.config,
        )

        # Lazy-init RAG (heavy — loads models, connects to Qdrant/Ollama)
        self._rag: Optional[UnifiedRAG] = None

    @property
    def rag(self) -> UnifiedRAG:
        if self._rag is None:
            self._rag = UnifiedRAG()
        return self._rag

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        stream: bool = False,
        **kwargs,
    ):
        """
        Send a query and return the assistant's answer.

        Parameters
        ----------
        query  : The user's message.
        stream : If True, returns a generator of string tokens.
        **kwargs: Forwarded to UnifiedRAG.answer_question (e.g. return_debug_info,
                  selected_collections, enable_thinking, …).

        Returns
        -------
        str (stream=False) or Generator[str] (stream=True).
        """
        # 1. Persist the user message
        current_index = session_store.add_message(
            session_id=self.session_id,
            user_id=self.user_id,
            role='user',
            content=query,
            config=self.config,
        )
        session_store.touch_session(self.session_id, self.config)

        # 2. Compression check
        active_user_count = session_store.count_active_user_messages(
            self.session_id, self.config
        )
        if active_user_count >= self.compression_trigger:
            self._compress_and_archive(exclude_index=current_index)

        # 3. Build conversation_history for the LLM
        history = self._build_history(current_user_index=current_index)

        # 4. Call the RAG pipeline
        if stream:
            return self._stream_with_storage(query, history, **kwargs)

        answer = self.rag.answer_question(
            query=query,
            conversation_history=history if history else None,
            stream=False,
            **kwargs,
        )

        # 5. Persist the assistant reply
        session_store.add_message(
            session_id=self.session_id,
            user_id=self.user_id,
            role='assistant',
            content=answer,
            config=self.config,
        )
        return answer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_and_archive(self, exclude_index: int):
        """
        Compress ALL current active messages except the one just added
        (the current user turn), store the summary, and archive the raws.
        """
        active = session_store.get_active_messages(self.session_id, self.config)
        # Exclude the current user message — it hasn't been answered yet
        batch = [m for m in active if m['message_index'] != exclude_index]
        if not batch:
            return

        logger.debug(
            "Compressing %d messages for session %s", len(batch), self.session_id
        )

        summary = compressor.compress_messages(batch, self.config)

        # Add user_id to each row for archiving (session_store omits it from SELECT)
        for m in batch:
            m.setdefault('user_id', self.user_id)

        compression_id = session_store.insert_compression(
            session_id=self.session_id,
            user_id=self.user_id,
            summary=summary,
            message_index_start=batch[0]['message_index'],
            message_index_end=batch[-1]['message_index'],
            config=self.config,
        )
        session_store.archive_messages(
            session_id=self.session_id,
            message_rows=batch,
            compression_id=compression_id,
            config=self.config,
        )
        logger.info(
            "Archived %d messages into compression %d for session %s",
            len(batch), compression_id, self.session_id,
        )

    def _build_history(self, current_user_index: int) -> List[Dict]:
        """
        Build the conversation_history list passed to UnifiedRAG.answer_question.

        Format:
          [ {role: "user", content: "[Conversation summary]"},
            {role: "assistant", content: <summary text>},
            ...  (one pair per stored compression)
            {role: "user",      content: <prev user msg>},
            {role: "assistant", content: <prev answer>},
            ...  (uncompressed active messages, excluding current user turn)
          ]
        """
        history: List[Dict] = []

        # Compressions as pseudo-turns
        for comp in session_store.get_compressions(self.session_id, self.config):
            history.append({'role': 'user', 'content': '[Previous conversation summary]'})
            history.append({'role': 'assistant', 'content': comp['summary']})

        # Active messages, excluding the current (unanswered) user turn
        active = session_store.get_active_messages(self.session_id, self.config)
        for msg in active:
            if msg['message_index'] == current_user_index:
                continue
            history.append({'role': msg['role'], 'content': msg['content']})

        return history

    def _stream_with_storage(self, query: str, history: List[Dict], **kwargs):
        """
        Wraps the streaming generator so the assistant response is stored in
        PostgreSQL after the caller has consumed all tokens.
        """
        raw_gen = self.rag.answer_question(
            query=query,
            conversation_history=history if history else None,
            stream=True,
            **kwargs,
        )

        def _generator():
            chunks = []
            for token in raw_gen:
                chunks.append(token)
                yield token
            full_response = ''.join(chunks)
            session_store.add_message(
                session_id=self.session_id,
                user_id=self.user_id,
                role='assistant',
                content=full_response,
                config=self.config,
            )

        return _generator()
