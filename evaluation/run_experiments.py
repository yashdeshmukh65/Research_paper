"""
Experiment runner for all 6 ANNA evaluation experiments.

Experiments:
  1. LLM without RAG vs LLM with RAG
  2. Vector search vs BM25 vs Hybrid search
  3. Different chunk sizes and overlap
  4. Different embedding models
  5. RAG without reranking vs with reranking
  6. Basic RAG vs agentic research workflow

Results saved to evaluation/results/ as JSON files.
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from evaluation.eval_dataset import EVAL_DATASET, CHUNK_SIZE_CONFIGS, EMBEDDING_MODELS
from modules.evaluator import (
    evaluate_query, precision_at_k, recall_at_k, mrr,
    faithfulness_score, answer_relevance_score, hallucination_analysis,
    citation_accuracy, build_context_string, build_relevant_ids,
    measure_latency, limitation_f1, limitation_precision, limitation_recall
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(name: str, data: dict):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


def _avg(values: list) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _aggregate(query_results: list) -> dict:
    keys = [
        "precision_at_5", "recall_at_5", "mrr", "faithfulness",
        "answer_relevance", "citation_accuracy", "hallucination_rate",
        "retrieval_latency_s", "generation_latency_s", "total_latency_s"
    ]
    return {k: _avg([r[k] for r in query_results if k in r]) for k in keys}


def _get_indexed_papers() -> list:
    from modules.vector_store import load_index
    _, metadata = load_index()
    return list({c["paper_name"] for c in metadata})


# ---------------------------------------------------------------------------
# Retriever + generator functions for each experiment variant
# ---------------------------------------------------------------------------

def _faiss_retriever(question: str, top_k: int = 5) -> list:
    from modules.embeddings import embed_query
    from modules.vector_store import search
    return search(embed_query(question), top_k=top_k)


def _bm25_retriever(question: str, top_k: int = 5) -> list:
    from modules.bm25_retriever import search_bm25
    return search_bm25(question, top_k=top_k)


def _hybrid_retriever(question: str, top_k: int = 5) -> list:
    from modules.hybrid_retriever import hybrid_search
    return hybrid_search(question, top_k=top_k)


def _hybrid_no_rerank_retriever(question: str, top_k: int = 5) -> list:
    """Hybrid without cross-encoder reranking — RRF only."""
    from modules.embeddings import embed_query
    from modules.vector_store import search as faiss_search
    from modules.bm25_retriever import search_bm25
    from modules.hybrid_retriever import _reciprocal_rank_fusion
    qe = embed_query(question)
    faiss_r = faiss_search(qe, top_k=20)
    bm25_r = search_bm25(question, top_k=20)
    merged = _reciprocal_rank_fusion(faiss_r, bm25_r)
    return merged[:top_k]


def _rag_generator(question: str, chunks: list) -> str:
    import google.generativeai as genai
    from utils.prompts import build_rag_prompt
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = build_rag_prompt(question, chunks)
    return model.generate_content(prompt).text.strip()


def _no_rag_generator(question: str, chunks: list = None) -> str:
    """LLM answer with no retrieved context."""
    import google.generativeai as genai
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"You are a research assistant. Answer the following question based on your general knowledge.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return model.generate_content(prompt).text.strip()


# ---------------------------------------------------------------------------
# Experiment 1: LLM without RAG vs LLM with RAG
# ---------------------------------------------------------------------------

def experiment_1_rag_vs_no_rag():
    print("\n[Experiment 1] LLM without RAG vs LLM with RAG")
    no_rag_results, rag_results = [], []

    for item in EVAL_DATASET:
        q = item["question"]
        kw = item["expected_keywords"]
        print(f"  Q: {q[:60]}...")

        # No RAG
        answer_no_rag, lat = measure_latency(_no_rag_generator, q)
        no_rag_results.append({
            "question": q,
            "answer": answer_no_rag,
            "faithfulness": faithfulness_score(q, "", answer_no_rag),
            "answer_relevance": answer_relevance_score(q, answer_no_rag),
            "hallucination_rate": hallucination_analysis("", answer_no_rag)["hallucination_rate"],
            "total_latency_s": lat,
            "citation_accuracy": 0.0,
            "precision_at_5": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
        })

        # With RAG
        rag_result = evaluate_query(q, kw, _hybrid_retriever, _rag_generator)
        rag_results.append(rag_result)

    result = {
        "experiment": "1_rag_vs_no_rag",
        "timestamp": datetime.now().isoformat(),
        "no_rag": {"per_query": no_rag_results, "aggregate": _aggregate(no_rag_results)},
        "rag":    {"per_query": rag_results,    "aggregate": _aggregate(rag_results)},
    }
    _save("exp1_rag_vs_no_rag", result)
    return result


# ---------------------------------------------------------------------------
# Experiment 2: Vector search vs BM25 vs Hybrid
# ---------------------------------------------------------------------------

def experiment_2_retrieval_methods():
    print("\n[Experiment 2] Vector vs BM25 vs Hybrid retrieval")
    variants = {
        "faiss_vector": _faiss_retriever,
        "bm25_keyword": _bm25_retriever,
        "hybrid":       _hybrid_retriever,
    }
    all_results = {}

    for variant_name, retriever_fn in variants.items():
        print(f"  Variant: {variant_name}")
        query_results = []
        for item in EVAL_DATASET:
            q = item["question"]
            kw = item["expected_keywords"]
            retrieved, lat = measure_latency(retriever_fn, q)
            relevant_ids = build_relevant_ids(retrieved, kw)
            query_results.append({
                "question": q,
                "precision_at_5": precision_at_k(retrieved, relevant_ids, 5),
                "recall_at_5": recall_at_k(retrieved, relevant_ids, 5),
                "mrr": mrr(retrieved, relevant_ids),
                "retrieval_latency_s": lat,
                "retrieved_count": len(retrieved),
            })
        all_results[variant_name] = {
            "per_query": query_results,
            "aggregate": _aggregate(query_results)
        }

    result = {
        "experiment": "2_retrieval_methods",
        "timestamp": datetime.now().isoformat(),
        **all_results
    }
    _save("exp2_retrieval_methods", result)
    return result


# ---------------------------------------------------------------------------
# Experiment 3: Chunk size and overlap
# ---------------------------------------------------------------------------

def experiment_3_chunk_sizes():
    print("\n[Experiment 3] Chunk size and overlap configurations")
    from modules.pdf_processor import process_multiple_pdfs
    from modules.text_cleaner import clean_paper
    from modules.embeddings import embed_chunks, get_model
    import faiss

    paper_names = _get_indexed_papers()
    if not paper_names:
        print("  No papers indexed. Skipping experiment 3.")
        return {}

    # Load raw paper files
    upload_dir = "data/uploaded_papers"
    pdf_files = [
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir) if f.endswith(".pdf")
    ] if os.path.exists(upload_dir) else []

    if not pdf_files:
        print("  No PDF files found in data/uploaded_papers. Skipping experiment 3.")
        return {}

    all_results = {}
    papers_raw = process_multiple_pdfs(pdf_files)

    for config in CHUNK_SIZE_CONFIGS:
        cs = config["chunk_size_tokens"]
        ov = config["overlap_tokens"]
        label = f"chunk_{cs}_overlap_{ov}"
        print(f"  Config: chunk_size={cs}, overlap={ov}")

        # Re-chunk with this config
        chunk_size_chars = cs * 4
        overlap_chars = ov * 4
        chunks = []
        chunk_id = 0
        for paper in papers_raw:
            if "error" in paper:
                continue
            cleaned = clean_paper(paper)
            for page in cleaned["pages"]:
                text = page["text"]
                if len(text) <= chunk_size_chars:
                    chunks.append({
                        "chunk_id": chunk_id, "text": text,
                        "paper_name": cleaned["paper_name"],
                        "page_number": page["page_number"],
                        "section": page["section"], "author": cleaned["author"]
                    })
                    chunk_id += 1
                else:
                    start = 0
                    while start < len(text):
                        part = text[start:start + chunk_size_chars].strip()
                        if part:
                            chunks.append({
                                "chunk_id": chunk_id, "text": part,
                                "paper_name": cleaned["paper_name"],
                                "page_number": page["page_number"],
                                "section": page["section"], "author": cleaned["author"]
                            })
                            chunk_id += 1
                        start += chunk_size_chars - overlap_chars

        # Embed and build temp FAISS index
        chunks_embedded = embed_chunks(chunks)
        model = get_model()
        dim = len(chunks_embedded[0]["embedding"])
        vectors = np.array([c["embedding"] for c in chunks_embedded], dtype="float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        metadata = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks_embedded]

        def temp_retriever(question: str, _index=index, _meta=metadata, _model=model) -> list:
            qv = np.array([_model.encode(question, convert_to_numpy=True)], dtype="float32")
            faiss.normalize_L2(qv)
            scores, indices = _index.search(qv, 5)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    r = _meta[idx].copy()
                    r["score"] = round(float(score), 4)
                    results.append(r)
            return results

        query_results = []
        for item in EVAL_DATASET:
            q = item["question"]
            kw = item["expected_keywords"]
            retrieved, lat = measure_latency(temp_retriever, q)
            relevant_ids = build_relevant_ids(retrieved, kw)
            query_results.append({
                "question": q,
                "precision_at_5": precision_at_k(retrieved, relevant_ids, 5),
                "recall_at_5": recall_at_k(retrieved, relevant_ids, 5),
                "mrr": mrr(retrieved, relevant_ids),
                "retrieval_latency_s": lat,
                "chunk_count": len(chunks),
            })

        all_results[label] = {
            "config": config,
            "total_chunks": len(chunks),
            "per_query": query_results,
            "aggregate": _aggregate(query_results)
        }

    result = {
        "experiment": "3_chunk_sizes",
        "timestamp": datetime.now().isoformat(),
        **all_results
    }
    _save("exp3_chunk_sizes", result)
    return result


# ---------------------------------------------------------------------------
# Experiment 4: Embedding models
# ---------------------------------------------------------------------------

def experiment_4_embedding_models():
    print("\n[Experiment 4] Embedding model comparison")
    from sentence_transformers import SentenceTransformer
    from modules.pdf_processor import process_multiple_pdfs
    from modules.text_cleaner import clean_paper
    from modules.chunking import chunk_paper
    import faiss

    upload_dir = "data/uploaded_papers"
    pdf_files = [
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir) if f.endswith(".pdf")
    ] if os.path.exists(upload_dir) else []

    if not pdf_files:
        print("  No PDF files found. Skipping experiment 4.")
        return {}

    papers_raw = process_multiple_pdfs(pdf_files)
    base_chunks = []
    for paper in papers_raw:
        if "error" not in paper:
            base_chunks.extend(chunk_paper(paper))

    all_results = {}

    for model_name in EMBEDDING_MODELS:
        print(f"  Model: {model_name}")
        model = SentenceTransformer(model_name)

        texts = [c["text"] for c in base_chunks]
        embed_start = time.perf_counter()
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")
        embed_time = round(time.perf_counter() - embed_start, 3)
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        def temp_retriever(question: str, _index=index, _meta=base_chunks, _model=model) -> list:
            qv = np.array([_model.encode(question, convert_to_numpy=True)], dtype="float32")
            faiss.normalize_L2(qv)
            scores, indices = _index.search(qv, 5)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    r = _meta[idx].copy()
                    r["score"] = round(float(score), 4)
                    results.append(r)
            return results

        query_results = []
        for item in EVAL_DATASET:
            q = item["question"]
            kw = item["expected_keywords"]
            retrieved, lat = measure_latency(temp_retriever, q)
            relevant_ids = build_relevant_ids(retrieved, kw)
            query_results.append({
                "question": q,
                "precision_at_5": precision_at_k(retrieved, relevant_ids, 5),
                "recall_at_5": recall_at_k(retrieved, relevant_ids, 5),
                "mrr": mrr(retrieved, relevant_ids),
                "retrieval_latency_s": lat,
            })

        all_results[model_name] = {
            "embedding_time_s": embed_time,
            "vector_dim": dim,
            "per_query": query_results,
            "aggregate": _aggregate(query_results)
        }

    result = {
        "experiment": "4_embedding_models",
        "timestamp": datetime.now().isoformat(),
        **all_results
    }
    _save("exp4_embedding_models", result)
    return result


# ---------------------------------------------------------------------------
# Experiment 5: RAG without reranking vs with reranking
# ---------------------------------------------------------------------------

def experiment_5_reranking():
    print("\n[Experiment 5] Without reranking vs with reranking")
    variants = {
        "hybrid_no_rerank": _hybrid_no_rerank_retriever,
        "hybrid_with_rerank": _hybrid_retriever,
    }
    all_results = {}

    for variant_name, retriever_fn in variants.items():
        print(f"  Variant: {variant_name}")
        query_results = []
        for item in EVAL_DATASET:
            q = item["question"]
            kw = item["expected_keywords"]
            result = evaluate_query(q, kw, retriever_fn, _rag_generator)
            query_results.append(result)

        all_results[variant_name] = {
            "per_query": query_results,
            "aggregate": _aggregate(query_results)
        }

    result = {
        "experiment": "5_reranking",
        "timestamp": datetime.now().isoformat(),
        **all_results
    }
    _save("exp5_reranking", result)
    return result


# ---------------------------------------------------------------------------
# Experiment 6: Basic RAG vs Agentic workflow
# ---------------------------------------------------------------------------

def experiment_6_agentic_vs_basic():
    print("\n[Experiment 6] Basic RAG vs Agentic research workflow")
    from workflows.research_workflow import run as run_workflow

    paper_names = _get_indexed_papers()
    if not paper_names:
        print("  No papers indexed. Skipping experiment 6.")
        return {}

    basic_results, agentic_results = [], []

    for item in EVAL_DATASET[:4]:  # limit to 4 queries — agentic is expensive
        q = item["question"]
        kw = item["expected_keywords"]
        print(f"  Q: {q[:60]}...")

        # Basic RAG
        basic = evaluate_query(q, kw, _hybrid_retriever, _rag_generator)
        basic_results.append(basic)

        # Agentic workflow
        start = time.perf_counter()
        try:
            state = run_workflow(paper_names=paper_names, question=q, tasks=["analyze", "qa"])
            qa = state.get("qa_results", [{}])[0]
            answer = qa.get("answer", "")
            retrieved = _hybrid_retriever(q)
            context = build_context_string(retrieved)
            lat = round(time.perf_counter() - start, 3)

            agentic_results.append({
                "question": q,
                "answer": answer,
                "faithfulness": faithfulness_score(q, context, answer),
                "answer_relevance": answer_relevance_score(q, answer),
                "hallucination_rate": hallucination_analysis(context, answer)["hallucination_rate"],
                "citation_accuracy": citation_accuracy(answer, retrieved),
                "total_latency_s": lat,
                "precision_at_5": basic["precision_at_5"],
                "recall_at_5": basic["recall_at_5"],
                "mrr": basic["mrr"],
            })
        except Exception as e:
            agentic_results.append({"question": q, "error": str(e), "total_latency_s": 0})

    result = {
        "experiment": "6_agentic_vs_basic",
        "timestamp": datetime.now().isoformat(),
        "basic_rag":       {"per_query": basic_results,   "aggregate": _aggregate(basic_results)},
        "agentic_workflow": {"per_query": agentic_results, "aggregate": _aggregate(agentic_results)},
    }
    _save("exp6_agentic_vs_basic", result)
    return result


# ---------------------------------------------------------------------------
# Limitation extraction evaluation
# ---------------------------------------------------------------------------

def experiment_limitation_extraction():
    print("\n[Experiment L] Limitation extraction precision/recall/F1")
    from agents.paper_analysis import analyze_multiple_papers
    from evaluation.eval_dataset import LIMITATION_GROUND_TRUTH
    import re

    paper_names = _get_indexed_papers()
    if not paper_names:
        print("  No papers indexed. Skipping.")
        return {}

    analyses = analyze_multiple_papers(paper_names)
    query_results = []

    for analysis in analyses:
        pname = analysis.paper_name
        raw_lim = analysis.limitations
        if raw_lim == "Not mentioned in the paper":
            predicted = []
        else:
            predicted = [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+", raw_lim) if len(s.strip()) > 20]

        ground_truth = LIMITATION_GROUND_TRUTH.get(pname, [])
        if not ground_truth:
            print(f"  No ground truth for '{pname}' — skipping F1 calculation.")
            query_results.append({
                "paper": pname,
                "predicted_count": len(predicted),
                "ground_truth_count": 0,
                "note": "No ground truth available"
            })
            continue

        query_results.append({
            "paper": pname,
            "predicted_count": len(predicted),
            "ground_truth_count": len(ground_truth),
            "precision": limitation_precision(predicted, ground_truth),
            "recall": limitation_recall(predicted, ground_truth),
            "f1": limitation_f1(predicted, ground_truth),
            "predicted": predicted,
            "ground_truth": ground_truth
        })

    result = {
        "experiment": "L_limitation_extraction",
        "timestamp": datetime.now().isoformat(),
        "per_paper": query_results,
        "aggregate": {
            "avg_precision": _avg([r["precision"] for r in query_results if "precision" in r]),
            "avg_recall":    _avg([r["recall"]    for r in query_results if "recall"    in r]),
            "avg_f1":        _avg([r["f1"]        for r in query_results if "f1"        in r]),
        }
    }
    _save("expL_limitation_extraction", result)
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ANNA evaluation experiments")
    parser.add_argument("--exp", type=str, default="all",
        help="Which experiment to run: 1,2,3,4,5,6,L or 'all'")
    args = parser.parse_args()

    exp_map = {
        "1": experiment_1_rag_vs_no_rag,
        "2": experiment_2_retrieval_methods,
        "3": experiment_3_chunk_sizes,
        "4": experiment_4_embedding_models,
        "5": experiment_5_reranking,
        "6": experiment_6_agentic_vs_basic,
        "L": experiment_limitation_extraction,
    }

    if args.exp == "all":
        for name, fn in exp_map.items():
            fn()
    elif args.exp in exp_map:
        exp_map[args.exp]()
    else:
        print(f"Unknown experiment: {args.exp}. Choose from: {list(exp_map.keys())} or 'all'")
