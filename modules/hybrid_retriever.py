from sentence_transformers import CrossEncoder
from modules.vector_store import search as faiss_search
from modules.bm25_retriever import search_bm25
from modules.embeddings import embed_query

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _reciprocal_rank_fusion(faiss_results: list, bm25_results: list, k: int = 60) -> list:
    """
    Merge FAISS and BM25 results using Reciprocal Rank Fusion (RRF).
    RRF score = sum of 1 / (k + rank) across all result lists.
    Chunks appearing in both lists get a higher combined score.
    """
    scores = {}
    chunk_map = {}

    for rank, chunk in enumerate(faiss_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[cid] for cid, _ in ranked]


def _rerank(query: str, chunks: list, top_k: int) -> list:
    cross_encoder = get_cross_encoder()
    pairs = [[query, c["text"]] for c in chunks]
    ce_scores = cross_encoder.predict(pairs)

    for chunk, score in zip(chunks, ce_scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def hybrid_search(query: str, top_k: int = 5, fetch_k: int = 20) -> list:
    """
    Full hybrid retrieval pipeline:
    1. FAISS semantic search (fetch_k candidates)
    2. BM25 keyword search (fetch_k candidates) — falls back gracefully if index missing
    3. Reciprocal Rank Fusion to merge results
    4. Cross-encoder reranking to pick best top_k
    """
    query_embedding = embed_query(query)
    faiss_results = faiss_search(query_embedding, top_k=fetch_k)

    try:
        bm25_results = search_bm25(query, top_k=fetch_k)
    except FileNotFoundError:
        bm25_results = []  # graceful fallback to FAISS-only if BM25 index missing

    if not bm25_results:
        # No BM25 — skip RRF, just rerank FAISS results directly
        return _rerank(query, faiss_results, top_k=top_k)

    merged = _reciprocal_rank_fusion(faiss_results, bm25_results)
    reranked = _rerank(query, merged, top_k=top_k)
    return reranked
