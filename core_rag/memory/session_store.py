"""
CRUD operations for sessions, messages, archived_messages, and compressions tables.
"""

import uuid
from typing import List, Dict, Optional
from .db import get_connection


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(user_id: str, session_id: str = None, config: dict = None) -> str:
    """Insert a new session row and return its UUID as a string."""
    if session_id is None:
        session_id = str(uuid.uuid4())
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, user_id) VALUES (%s::uuid, %s)",
                (session_id, user_id),
            )
    return session_id


def session_exists(session_id: str, config: dict = None) -> bool:
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM sessions WHERE session_id = %s::uuid",
                (session_id,),
            )
            return cur.fetchone() is not None


def get_or_create_session(
    user_id: str, session_id: str = None, config: dict = None
) -> str:
    """
    Return session_id if it exists and belongs to user_id, otherwise create one.
    Raises ValueError if session_id is given but belongs to a different user.
    """
    if session_id:
        with get_connection(config) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id FROM sessions WHERE session_id = %s::uuid",
                    (session_id,),
                )
                row = cur.fetchone()
        if row is None:
            return create_session(user_id, session_id, config)
        if row[0] != user_id:
            raise ValueError(f"Session {session_id} belongs to a different user")
        return session_id
    return create_session(user_id, config=config)


def touch_session(session_id: str, config: dict = None):
    """Update sessions.updated_at to now."""
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET updated_at = NOW() WHERE session_id = %s::uuid",
                (session_id,),
            )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def _next_message_index(cur, session_id: str) -> int:
    """Return the next message_index for a session (across active + archived)."""
    cur.execute(
        """
        SELECT COALESCE(MAX(message_index), -1) + 1
        FROM (
            SELECT message_index FROM messages WHERE session_id = %s::uuid
            UNION ALL
            SELECT message_index FROM archived_messages WHERE session_id = %s::uuid
        ) AS all_msgs
        """,
        (session_id, session_id),
    )
    return cur.fetchone()[0]


def add_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    config: dict = None,
) -> int:
    """Insert a message and return its message_index."""
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            idx = _next_message_index(cur, session_id)
            cur.execute(
                """
                INSERT INTO messages (session_id, user_id, role, content, message_index)
                VALUES (%s::uuid, %s, %s, %s, %s)
                RETURNING message_index
                """,
                (session_id, user_id, role, content, idx),
            )
            return cur.fetchone()[0]


def get_active_messages(session_id: str, config: dict = None) -> List[Dict]:
    """Return all active (unarchived) messages ordered by message_index."""
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, role, content, message_index
                FROM messages
                WHERE session_id = %s::uuid
                ORDER BY message_index ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
    return [
        {"id": r[0], "role": r[1], "content": r[2], "message_index": r[3]}
        for r in rows
    ]


def count_active_user_messages(session_id: str, config: dict = None) -> int:
    """Count unarchived user-role messages for a session."""
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM messages
                WHERE session_id = %s::uuid AND role = 'user'
                """,
                (session_id,),
            )
            return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Compressions + archiving
# ---------------------------------------------------------------------------

def insert_compression(
    session_id: str,
    user_id: str,
    summary: str,
    message_index_start: int,
    message_index_end: int,
    config: dict = None,
) -> int:
    """Insert a compression row and return its id."""
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO compressions
                    (session_id, user_id, summary, message_index_start, message_index_end)
                VALUES (%s::uuid, %s, %s, %s, %s)
                RETURNING id
                """,
                (session_id, user_id, summary, message_index_start, message_index_end),
            )
            return cur.fetchone()[0]


def archive_messages(
    session_id: str,
    message_rows: List[Dict],
    compression_id: int,
    config: dict = None,
):
    """
    Move the given message rows from messages → archived_messages.
    message_rows should be dicts with keys: id, role, content, message_index, user_id (optional).
    """
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            for row in message_rows:
                cur.execute(
                    """
                    INSERT INTO archived_messages
                        (session_id, user_id, role, content, message_index, compression_id)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        row.get('user_id', ''),
                        row['role'],
                        row['content'],
                        row['message_index'],
                        compression_id,
                    ),
                )
            ids = [r['id'] for r in message_rows]
            cur.execute(
                "DELETE FROM messages WHERE id = ANY(%s)",
                (ids,),
            )


def get_compressions(session_id: str, config: dict = None) -> List[Dict]:
    """Return all compressions for a session ordered chronologically."""
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, summary, message_index_start, message_index_end, created_at
                FROM compressions
                WHERE session_id = %s::uuid
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "summary": r[1],
            "message_index_start": r[2],
            "message_index_end": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]
