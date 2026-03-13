-- Core RAG chat history and session management schema
-- Run once against your local PostgreSQL instance:
--   psql -U <user> -d <database> -f scripts/schema.sql

CREATE TABLE IF NOT EXISTS sessions (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Active (uncompressed) messages for a session
CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    message_index   INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Raw messages moved here after compression (audit trail)
CREATE TABLE IF NOT EXISTS archived_messages (
    id              SERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    message_index   INTEGER NOT NULL,
    compression_id  INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- LLM-generated summaries of archived message batches
CREATE TABLE IF NOT EXISTS compressions (
    id                   SERIAL PRIMARY KEY,
    session_id           UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id              TEXT NOT NULL,
    summary              TEXT NOT NULL,
    message_index_start  INTEGER NOT NULL,
    message_index_end    INTEGER NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, message_index);

CREATE INDEX IF NOT EXISTS idx_archived_messages_session
    ON archived_messages(session_id, message_index);

CREATE INDEX IF NOT EXISTS idx_compressions_session
    ON compressions(session_id, created_at);
