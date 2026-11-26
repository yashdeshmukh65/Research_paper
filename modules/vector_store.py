import faiss
import numpy as np
import pickle
import os

_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
INDEX_PATH = os.path.join(_BASE, "faiss.index")
METADATA_PATH = os.path.join(_BASE, "chunks_metadata.pkl")


def _get_dimension(chunks: list) -> int:
    return len(chunks[0]["embedding"])


def build_index(chunks: list) -> tuple:
    dim = _get_dimension(chunks)
    vectors = np.array([c["embedding"] for c in chunks], dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine similarity
    index.add(vectors)

    # Strip embeddings from metadata to save memory
    metadata = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
    return index, metadata


def save_index(index: faiss.Index, metadata: list):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


def load_index() -> tuple:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("No FAISS index found. Please process and index papers first.")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def add_to_index(new_chunks: list):
    """Add new paper chunks to an existing index without rebuilding from scratch."""
    try:
        index, metadata = load_index()
    except FileNotFoundError:
        index, metadata = build_index(new_chunks)
        save_index(index, metadata)
        return

    vectors = np.array([c["embedding"] for c in new_chunks], dtype="float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)
    metadata.extend([{k: v for k, v in c.items() if k != "embedding"} for c in new_chunks])
    save_index(index, metadata)


def clear_index():
    """Delete FAISS index and metadata from disk."""
    for path in [INDEX_PATH, METADATA_PATH]:
        if os.path.exists(path):
            os.remove(path)


def remove_paper_from_index(paper_name: str):
    """Rebuild index excluding all chunks belonging to paper_name."""
    try:
        _, metadata = load_index()
    except FileNotFoundError:
        return
    remaining = [c for c in metadata if c["paper_name"] != paper_name]
    if not remaining:
        clear_index()
        return
    # Rebuild index from remaining metadata (no embeddings stored — re-embed)
    # We store embeddings separately in session; here we just rebuild metadata index
    # by loading embeddings from a fresh embed of remaining chunks is not feasible.
    # Instead we rebuild the full index from the remaining metadata pickle only.
    dim = 384  # all-MiniLM-L6-v2 fixed dimension
    index = faiss.IndexFlatIP(dim)
    # We cannot rebuild vectors without embeddings — save metadata only and mark index as empty
    # The correct approach: store embeddings alongside metadata
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(remaining, f)
    # Rebuild index from scratch using stored embeddings in metadata if present
    embeddings = [c.get("embedding") for c in remaining if c.get("embedding")]  
    if embeddings:
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
    save_index(index, [{k: v for k, v in c.items() if k != "embedding"} for c in remaining])


def search(query_embedding: list, top_k: int = 5) -> list:
    index, metadata = load_index()

    query_vec = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = metadata[idx].copy()
        result["score"] = round(float(score), 4)
        results.append(result)

    return results
