import pickle
import os
from rank_bm25 import BM25Okapi

BM25_PATH = "data/processed/bm25.pkl"


def _tokenize(text: str) -> list:
    return text.lower().split()


def build_bm25(chunks: list) -> BM25Okapi:
    corpus = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def save_bm25(bm25: BM25Okapi, metadata: list):
    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": metadata}, f)


def load_bm25() -> tuple:
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError("No BM25 index found. Please process and index papers first.")
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["metadata"]


def add_to_bm25(new_chunks: list):
    """Rebuild BM25 index with new chunks appended (BM25 doesn't support incremental updates)."""
    try:
        _, existing_metadata = load_bm25()
    except FileNotFoundError:
        existing_metadata = []

    all_metadata = existing_metadata + [{k: v for k, v in c.items() if k != "embedding"} for c in new_chunks]
    bm25 = build_bm25(all_metadata)
    save_bm25(bm25, all_metadata)


def clear_bm25():
    """Delete BM25 index from disk."""
    if os.path.exists(BM25_PATH):
        os.remove(BM25_PATH)


def remove_paper_from_bm25(paper_name: str):
    """Rebuild BM25 index excluding all chunks belonging to paper_name."""
    try:
        _, metadata = load_bm25()
    except FileNotFoundError:
        return
    remaining = [c for c in metadata if c.get("paper_name") != paper_name]
    if not remaining:
        clear_bm25()
        return
    bm25 = build_bm25(remaining)
    save_bm25(bm25, remaining)


def search_bm25(query: str, top_k: int = 5) -> list:
    bm25, metadata = load_bm25()
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    # Pair scores with metadata and sort descending
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in scored:
        if score == 0:
            continue
        result = metadata[idx].copy()
        result["bm25_score"] = round(float(score), 4)
        results.append(result)

    return results
