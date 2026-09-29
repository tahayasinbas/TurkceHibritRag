"""Explicit reconstruction defaults, not recovered historical experiment settings."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    embedding_model: str = 'BAAI/bge-m3'
    chunk_size: int = 2000
    chunk_overlap: int = 200
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_weight: float = 0.4
    candidate_k: int = 5
    top_k: int = 5
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    temperature: float = 0.4
    batch_size: int = 16

    def __post_init__(self):
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError('Invalid chunk size or overlap.')
        if not 0 <= self.bm25_weight <= 1:
            raise ValueError('Invalid BM25 weight.')
        if self.top_k < 1 or self.candidate_k < self.top_k or self.batch_size < 1:
            raise ValueError('Require candidate_k >= top_k >= 1 and batch_size >= 1.')
        if self.hnsw_ef_search < self.candidate_k:
            raise ValueError('ef_search must cover candidate_k.')


def database_kwargs() -> dict:
    from dotenv import load_dotenv
    load_dotenv()
    password = os.getenv('POSTGRES_PASSWORD')
    if not password:
        raise ValueError('Set POSTGRES_PASSWORD in .env first.')
    return dict(host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5434')),
                dbname=os.getenv('POSTGRES_DB', 'turkish_rag_reference'),
                user=os.getenv('POSTGRES_USER', 'rag'), password=password)
