import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv

from core_rag.utils.config_loader import load_config

load_dotenv()  # loads .env into os.environ if present

_pool = None


def _get_pool(config: dict = None):
    global _pool
    if _pool is None:
        if config is None:
            config = load_config()
        pg = config.get('postgresql', {})

        # Env vars override config file values, consistent with QDRANT_HOST / OLLAMA_HOST
        host     = os.environ.get('POSTGRES_HOST',     pg.get('host',     'localhost'))
        port     = int(os.environ.get('POSTGRES_PORT', pg.get('port',     5432)))
        dbname   = os.environ.get('POSTGRES_DB',       pg.get('database', 'core_rag'))
        user     = os.environ.get('POSTGRES_USER',     pg.get('user',     'postgres'))
        password = os.environ.get('POSTGRES_PASSWORD', pg.get('password', ''))

        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=pg.get('pool_max', 10),
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
    return _pool


@contextmanager
def get_connection(config: dict = None):
    p = _get_pool(config)
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def init_db(config: dict = None):
    """Create tables if they don't exist by running schema.sql."""
    schema_path = Path(__file__).parent.parent.parent / 'scripts' / 'schema.sql'
    sql = schema_path.read_text()
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
