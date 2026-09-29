"""Dependency-free fusion and retrieval evaluation routines."""
import math
import re


def tokenize(text: str) -> list[str]:
    """Turkish-aware lowercasing followed by word tokenization; no stemming."""
    return re.findall(r"\w+", text.translate(str.maketrans({'I': 'ı', 'İ': 'i'})).lower())


def token_windows(token_ids: list[int], size: int, overlap: int):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require size > 0 and 0 <= overlap < size.')
    for start in range(0, len(token_ids), size - overlap):
        yield start, token_ids[start:start + size]
        if start + size >= len(token_ids):
            break


def hybrid_scores(dense: dict[str, float], sparse: dict[str, float],
                  bm25_weight: float = 0.4) -> list[tuple[str, float]]:
    """Score the union of candidates with cosine + min-max BM25.

    Both inputs must contain scores for every candidate, including cross-channel
    candidates. Normalize BM25 over this union. A constant BM25 range contributes
    zero. Cosine values retain their original [-1, 1] scale.
    """
    if not 0 <= bm25_weight <= 1:
        raise ValueError('bm25_weight must be between 0 and 1.')
    if dense.keys() != sparse.keys():
        raise ValueError('Both channels must score the same candidate IDs.')
    if not dense:
        return []
    if not all(math.isfinite(x) for x in [*dense.values(), *sparse.values()]):
        raise ValueError('Scores must be finite.')
    low, high = min(sparse.values()), max(sparse.values())
    scores = {}
    for key in dense:
        normalized = (sparse[key] - low) / (high - low) if high > low else 0.0
        scores[key] = (1 - bm25_weight) * dense[key] + bm25_weight * normalized
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def retrieval_metrics(retrieved: list[str], relevant: list[str], k: int = 5) -> dict:
    """Binary relevance metrics; requires manually/reference-labelled chunk IDs."""
    if k < 1:
        raise ValueError('k must be positive.')
    gold = set(relevant)
    if not gold:
        raise ValueError('At least one relevant chunk ID is required.')
    ranked = list(dict.fromkeys(retrieved))[:k]
    hits = sum(x in gold for x in ranked)
    precision = hits / k
    recall = hits / len(gold)
    return {
        'precision_at_k': precision,
        'recall_at_k': recall,
        'f1_at_k': 2 * precision * recall / (precision + recall) if hits else 0.0,
        'hit_rate_at_k': float(hits > 0),
        'mrr_at_k': next((1 / rank for rank, x in enumerate(ranked, 1) if x in gold), 0.0),
    }
