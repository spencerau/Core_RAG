"""
Integration tests for the memory module (ChatSession, session_store, compressor).

Prerequisites
-------------
- PostgreSQL running locally (see scripts/schema.sql or run via Docker)
- Qdrant running locally at localhost:6333
- Ollama containers on the cluster reachable at configured host/ports

Run with:
    pytest tests/test_memory.py -v -s
"""

import pytest
from core_rag.memory.db import init_db, get_connection
from core_rag.memory import session_store
from core_rag.memory import ChatSession
from core_rag.utils.config_loader import load_config

TEST_USER = "test_memory_user"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    cfg = load_config()
    # Use a lower compression trigger so tests don't need 5 full RAG calls
    cfg.setdefault('memory', {})['compression_trigger'] = 2
    return cfg


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown(config):
    """Ensure DB schema exists, then clean up all test data after the module runs."""
    init_db(config)
    yield
    # Teardown: delete all sessions (cascades to messages, archived_messages, compressions)
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE user_id = %s", (TEST_USER,))
    print(f"\n[teardown] Cleaned up all sessions for user '{TEST_USER}'")


@pytest.fixture(scope="module")
def session(config):
    """A single ChatSession shared across most tests."""
    s = ChatSession(user_id=TEST_USER, config=config)
    print(f"\n[fixture] Created session: {s.session_id}")
    return s


# ---------------------------------------------------------------------------
# DB / session_store layer (no RAG required)
# ---------------------------------------------------------------------------

def test_init_db(config):
    """Schema creation is idempotent."""
    init_db(config)  # second call should not raise


def test_create_session(config):
    sid = session_store.create_session(TEST_USER, config=config)
    assert isinstance(sid, str) and len(sid) == 36
    print(f"\nCreated session {sid}")


def test_get_or_create_session_new(config):
    sid = session_store.get_or_create_session(TEST_USER, config=config)
    assert sid is not None
    print(f"\nget_or_create_session (new): {sid}")


def test_get_or_create_session_existing(config):
    sid1 = session_store.create_session(TEST_USER, config=config)
    sid2 = session_store.get_or_create_session(TEST_USER, session_id=sid1, config=config)
    assert sid1 == sid2
    print(f"\nget_or_create_session (existing) returned same id: {sid1}")


def test_get_or_create_session_wrong_user(config):
    sid = session_store.create_session(TEST_USER, config=config)
    with pytest.raises(ValueError, match="different user"):
        session_store.get_or_create_session("other_user", session_id=sid, config=config)


def test_add_and_get_messages(config):
    sid = session_store.create_session(TEST_USER, config=config)
    session_store.add_message(sid, TEST_USER, 'user',      'Hello', config)
    session_store.add_message(sid, TEST_USER, 'assistant', 'Hi!',   config)
    session_store.add_message(sid, TEST_USER, 'user',      'Bye',   config)

    msgs = session_store.get_active_messages(sid, config)
    assert len(msgs) == 3
    assert msgs[0]['role'] == 'user'
    assert msgs[0]['content'] == 'Hello'
    assert msgs[1]['role'] == 'assistant'
    assert msgs[2]['role'] == 'user'

    count = session_store.count_active_user_messages(sid, config)
    assert count == 2
    print(f"\nadd/get messages: 3 messages stored, 2 user messages counted")


def test_insert_compression_and_archive(config):
    sid = session_store.create_session(TEST_USER, config=config)
    session_store.add_message(sid, TEST_USER, 'user',      'Q1', config)
    session_store.add_message(sid, TEST_USER, 'assistant', 'A1', config)

    active = session_store.get_active_messages(sid, config)
    for m in active:
        m.setdefault('user_id', TEST_USER)

    comp_id = session_store.insert_compression(
        session_id=sid,
        user_id=TEST_USER,
        summary="The user asked Q1 and received A1.",
        message_index_start=active[0]['message_index'],
        message_index_end=active[-1]['message_index'],
        config=config,
    )
    assert isinstance(comp_id, int) and comp_id > 0

    session_store.archive_messages(sid, active, comp_id, config)

    remaining = session_store.get_active_messages(sid, config)
    assert len(remaining) == 0, "All messages should be archived"

    compressions = session_store.get_compressions(sid, config)
    assert len(compressions) == 1
    assert "Q1" in compressions[0]['summary'] or "user" in compressions[0]['summary']
    print(f"\nCompression {comp_id} stored, {len(active)} messages archived")


# ---------------------------------------------------------------------------
# Full ChatSession integration (requires Qdrant + Ollama)
# ---------------------------------------------------------------------------

def test_chat_session_init(session):
    assert session.session_id is not None
    assert len(session.session_id) == 36
    print(f"\nChatSession initialised with session_id {session.session_id}")


def test_first_chat(session, config):
    """Single chat turn — user and assistant messages should be stored."""
    answer = session.chat("What resources are available for job coaching?")

    assert isinstance(answer, str)
    assert len(answer) > 0

    msgs = session_store.get_active_messages(session.session_id, config)
    roles = [m['role'] for m in msgs]
    assert 'user' in roles
    assert 'assistant' in roles
    print(f"\nFirst chat answer ({len(answer)} chars), {len(msgs)} messages stored")


def test_second_chat_triggers_compression(session, config):
    """
    Second user message should meet the compression_trigger=2 threshold,
    causing the first 2 messages (user+assistant) to be archived and a
    compression summary to be stored.
    """
    answer = session.chat("What specific techniques does it recommend?")

    assert isinstance(answer, str)
    assert len(answer) > 0

    compressions = session_store.get_compressions(session.session_id, config)
    assert len(compressions) >= 1, "Expected at least one compression after 2 user messages"

    # The archived table should have rows
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM archived_messages WHERE session_id = %s::uuid",
                (session.session_id,),
            )
            archived_count = cur.fetchone()[0]

    assert archived_count >= 2, f"Expected archived messages, got {archived_count}"
    print(
        f"\nCompression triggered: {len(compressions)} compression(s), "
        f"{archived_count} archived message(s)"
    )


def test_history_includes_compression(session, config):
    """
    After compression, _build_history should include the summary as a
    pseudo-turn and only recent active messages.
    """
    # Add a dummy user message to get the current index, then inspect history
    dummy_idx = session_store.add_message(
        session.session_id, TEST_USER, 'user', '__probe__', config
    )
    history = session._build_history(current_user_index=dummy_idx)

    # History should contain compression pseudo-turns
    assert any(m['content'] == '[Previous conversation summary]' for m in history), \
        "Expected compression pseudo-turn in history"

    # Remove the probe message so it doesn't affect later tests
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM messages WHERE session_id = %s::uuid AND content = '__probe__'",
                (session.session_id,),
            )
    print(f"\nHistory contains compression pseudo-turn ({len(history)} entries)")


def test_session_resume(config):
    """Creating a new ChatSession with an existing session_id should resume it."""
    original = ChatSession(user_id=TEST_USER, config=config)
    original.chat("What courses are required for a CS degree?")

    resumed = ChatSession(
        user_id=TEST_USER,
        session_id=original.session_id,
        config=config,
    )
    assert resumed.session_id == original.session_id
    answer = resumed.chat("Can you summarise what we discussed?")
    assert isinstance(answer, str) and len(answer) > 0
    print(f"\nResumed session {resumed.session_id}, got answer ({len(answer)} chars)")


def test_streaming_chat(session, config):
    """Streaming response should store the full assistant message after iteration."""
    def _total_messages(sid):
        with get_connection(config) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT id FROM messages         WHERE session_id = %s::uuid
                        UNION ALL
                        SELECT id FROM archived_messages WHERE session_id = %s::uuid
                    ) AS all_msgs
                    """,
                    (sid, sid),
                )
                return cur.fetchone()[0]

    count_before = _total_messages(session.session_id)

    gen = session.chat("What are some good recipes for a beginner cook?", stream=True)
    full = ""
    for token in gen:
        full += token

    assert isinstance(full, str) and len(full) > 0

    count_after = _total_messages(session.session_id)
    # Should have at least 2 new rows: the user message + the assistant reply
    assert count_after >= count_before + 2, \
        f"Expected at least 2 new messages, got {count_after - count_before}"
    print(f"\nStreaming chat: {len(full)} chars received, {count_after - count_before} new rows stored")
