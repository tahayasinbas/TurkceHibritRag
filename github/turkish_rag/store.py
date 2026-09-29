"""An isolated schema for the reference implementation; no legacy-table deletion."""
from contextlib import contextmanager
from dataclasses import asdict


@contextmanager
def connect():
    import psycopg
    from pgvector.psycopg import register_vector
    from .config import database_kwargs
    with psycopg.connect(**database_kwargs()) as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)
        yield conn


def initialize(conn, settings):
    from psycopg.types.json import Jsonb
    conn.execute('CREATE SCHEMA IF NOT EXISTS rag_reference')
    conn.execute('''CREATE TABLE IF NOT EXISTS rag_reference.config (
        id integer PRIMARY KEY CHECK (id = 1), settings jsonb NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS rag_reference.chunks (
        id text PRIMARY KEY, content text NOT NULL,
        metadata jsonb NOT NULL, embedding vector(1024) NOT NULL)''')
    # Changing settings silently would mix incompatible corpora.
    conn.execute('INSERT INTO rag_reference.config VALUES (1, %s) ON CONFLICT DO NOTHING',
                 (Jsonb(asdict(settings)),))
    saved = conn.execute('SELECT settings FROM rag_reference.config WHERE id = 1').fetchone()[0]
    if saved != asdict(settings):
        raise ValueError('Stored settings differ. Use a new reference database.')


def load_settings(conn):
    from .config import Settings
    exists = conn.execute("SELECT to_regclass('rag_reference.config')").fetchone()[0]
    if not exists:
        raise ValueError('No reference corpus found. Run ingest first.')
    row = conn.execute('SELECT settings FROM rag_reference.config WHERE id = 1').fetchone()
    if row is None:
        raise ValueError('No corpus settings found. Run ingest first.')
    return Settings(**row[0])


def make_index(conn, settings):
    from psycopg import sql
    conn.execute(sql.SQL('''CREATE INDEX IF NOT EXISTS rag_reference_hnsw
        ON rag_reference.chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = {}, ef_construction = {})''').format(
        sql.Literal(settings.hnsw_m), sql.Literal(settings.hnsw_ef_construction)))
    conn.execute('ANALYZE rag_reference.chunks')


def dense_candidates(conn, vector, settings):
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)",
                 (str(settings.hnsw_ef_search),))
    return conn.execute('''SELECT id FROM rag_reference.chunks
        ORDER BY embedding <=> %s LIMIT %s''', (vector, settings.candidate_k)).fetchall()
