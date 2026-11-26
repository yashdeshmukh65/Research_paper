import os
import numpy as np
import google.generativeai as genai
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from agents.paper_analysis import analyze_multiple_papers, PaperAnalysis
from modules.embeddings import embed_texts
from utils.prompts import build_research_gap_prompt
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

NOT_MENTIONED = "Not mentioned in the paper"

# Cosine distance threshold for agglomerative clustering
# Lower = tighter clusters (more specific themes)
# Higher = broader clusters (more general themes)
CLUSTER_DISTANCE_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Step 1: Extract statements
# ---------------------------------------------------------------------------

def _extract_statements(analyses: list[PaperAnalysis]) -> list[dict]:
    """
    Pull limitations and future_work from each paper analysis into
    individual statement objects tagged with source and type.
    """
    statements = []
    for paper in analyses:
        for field, stype in [("limitations", "limitation"), ("future_work", "future_work")]:
            value = getattr(paper, field)
            if value == NOT_MENTIONED or not value.strip():
                continue
            # Split on sentence boundaries or semicolons for finer granularity
            for sentence in _split_sentences(value):
                sentence = sentence.strip()
                if len(sentence) > 20:  # filter out noise/fragments
                    statements.append({
                        "text": sentence,
                        "paper_name": paper.paper_name,
                        "type": stype
                    })
    return statements


def _split_sentences(text: str) -> list:
    import re
    # Split on ". ", "; " or newlines while keeping content
    parts = re.split(r"(?<=[.;])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Step 2: Embed statements
# ---------------------------------------------------------------------------

def _embed_statements(statements: list[dict]) -> np.ndarray:
    texts = [s["text"] for s in statements]
    vectors = embed_texts(texts)
    return np.array(vectors, dtype="float32")


# ---------------------------------------------------------------------------
# Step 3: Cluster semantically similar statements
# ---------------------------------------------------------------------------

def _cluster_statements(statements: list[dict], vectors: np.ndarray) -> dict[int, list[dict]]:
    """
    Use Agglomerative Clustering with cosine distance.
    Returns a dict mapping cluster_id -> list of statement dicts.
    Falls back to a single cluster if too few statements to cluster.
    """
    n = len(statements)
    if n < 2:
        return {0: statements}

    # Compute cosine distance matrix (1 - cosine_similarity)
    sim_matrix = cosine_similarity(vectors)
    distance_matrix = 1 - sim_matrix
    np.fill_diagonal(distance_matrix, 0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD
    )
    labels = clustering.fit_predict(distance_matrix)

    clusters: dict[int, list[dict]] = {}
    for label, statement in zip(labels, statements):
        clusters.setdefault(int(label), []).append(statement)

    # Sort clusters by size descending (most recurring themes first)
    return dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))


# ---------------------------------------------------------------------------
# Step 4: Format clusters for the prompt
# ---------------------------------------------------------------------------

def _format_clusters_for_prompt(clusters: dict[int, list[dict]]) -> str:
    parts = []
    for cluster_id, statements in clusters.items():
        papers_in_cluster = list({s["paper_name"] for s in statements})
        evidence_lines = [
            f"  - [{s['type'].upper()}] \"{s['text']}\" — {s['paper_name']}"
            for s in statements
        ]
        parts.append(
            f"CLUSTER {cluster_id + 1} (appears across {len(papers_in_cluster)} paper(s): "
            f"{', '.join(papers_in_cluster)}):\n" + "\n".join(evidence_lines)
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 5: Build structured output
# ---------------------------------------------------------------------------

def _build_cluster_summary(clusters: dict[int, list[dict]]) -> list[dict]:
    """Structured cluster data returned alongside the LLM narrative."""
    summary = []
    for cluster_id, statements in clusters.items():
        papers = list({s["paper_name"] for s in statements})
        limitations = [s for s in statements if s["type"] == "limitation"]
        future_work = [s for s in statements if s["type"] == "future_work"]
        summary.append({
            "cluster_id": cluster_id + 1,
            "papers_involved": papers,
            "paper_count": len(papers),
            "statement_count": len(statements),
            "limitations": [s["text"] for s in limitations],
            "future_work_statements": [s["text"] for s in future_work],
        })
    return summary


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def find_research_gaps(paper_names: list) -> dict:
    if len(paper_names) < 2:
        raise ValueError("Research gap analysis requires at least 2 papers.")

    # Step 1: Analyze all papers
    analyses = analyze_multiple_papers(paper_names)

    # Step 2: Extract limitation + future work statements
    statements = _extract_statements(analyses)
    if not statements:
        return {
            "paper_names": paper_names,
            "clusters": [],
            "research_directions": "No limitations or future work statements could be extracted from the provided papers.",
            "total_statements": 0
        }

    # Step 3: Embed statements
    vectors = _embed_statements(statements)

    # Step 4: Cluster by semantic similarity
    clusters = _cluster_statements(statements, vectors)

    # Step 5: Format for LLM prompt
    clusters_text = _format_clusters_for_prompt(clusters)
    prompt = build_research_gap_prompt(clusters_text)

    # Step 6: Gemini synthesizes research directions
    response = _gemini_model.generate_content(prompt)

    return {
        "paper_names": paper_names,
        "total_statements": len(statements),
        "total_clusters": len(clusters),
        "clusters": _build_cluster_summary(clusters),
        "research_directions": response.text.strip()
    }
