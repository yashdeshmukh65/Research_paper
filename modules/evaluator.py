"""
Core evaluation metrics for ANNA's RAG pipeline.

Metrics implemented:
- Precision@K        : fraction of retrieved chunks that are relevant
- Recall@K           : fraction of relevant chunks that were retrieved
- MRR                : Mean Reciprocal Rank of first relevant result
- Faithfulness       : LLM-judged — is the answer supported by retrieved context?
- Answer Relevance   : LLM-judged — does the answer address the question?
- Citation Accuracy  : do cited page numbers appear in retrieved chunks?
- Latency            : wall-clock time for retrieval + generation
- Hallucination Score: fraction of answer sentences unsupported by context
"""

import time
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_judge = genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Retrieval metrics (reference-free — based on chunk metadata)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list, relevant_ids: set, k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    if not retrieved or not relevant_ids:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if _chunk_id(c) in relevant_ids)
    return round(hits / k, 4)


def recall_at_k(retrieved: list, relevant_ids: set, k: int) -> float:
    """Fraction of all relevant chunks that appear in top-k retrieved."""
    if not retrieved or not relevant_ids:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if _chunk_id(c) in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def mrr(retrieved: list, relevant_ids: set) -> float:
    """Mean Reciprocal Rank — position of first relevant result."""
    for rank, chunk in enumerate(retrieved, 1):
        if _chunk_id(chunk) in relevant_ids:
            return round(1.0 / rank, 4)
    return 0.0


def _chunk_id(chunk: dict) -> str:
    """Stable identifier for a chunk using paper + page + chunk_id."""
    return f"{chunk.get('paper_name', '')}__p{chunk.get('page_number', '')}__c{chunk.get('chunk_id', '')}"


def build_relevant_ids(retrieved: list, question_keywords: list) -> set:
    """
    When no ground truth page labels exist, approximate relevance by
    checking if a chunk's text contains any of the expected keywords.
    This is a keyword-overlap proxy for relevance labelling.
    """
    relevant = set()
    for chunk in retrieved:
        text_lower = chunk.get("text", "").lower()
        if any(kw.lower() in text_lower for kw in question_keywords):
            relevant.add(_chunk_id(chunk))
    return relevant


# ---------------------------------------------------------------------------
# Citation accuracy
# ---------------------------------------------------------------------------

def citation_accuracy(answer: str, retrieved: list) -> float:
    """
    Check what fraction of page numbers cited in the answer (Page: N)
    actually appear in the retrieved chunks.
    """
    cited_pages = set(int(p) for p in re.findall(r"[Pp]age[:\s]+(\d+)", answer))
    if not cited_pages:
        return 1.0  # no citations made — not penalised

    retrieved_pages = {c.get("page_number") for c in retrieved}
    valid = cited_pages & retrieved_pages
    return round(len(valid) / len(cited_pages), 4)


# ---------------------------------------------------------------------------
# LLM-as-judge metrics
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """You are an evaluation judge. Given a question, a retrieved context, and an answer, 
assess whether every factual claim in the answer is supported by the retrieved context.

Question: {question}

Retrieved Context:
{context}

Answer:
{answer}

Score the faithfulness from 0.0 to 1.0:
- 1.0 = every claim is directly supported by the context
- 0.5 = some claims supported, some not verifiable
- 0.0 = answer contains claims not present in the context (hallucination)

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

_RELEVANCE_PROMPT = """You are an evaluation judge. Given a question and an answer, 
assess whether the answer directly and completely addresses the question.

Question: {question}

Answer:
{answer}

Score the answer relevance from 0.0 to 1.0:
- 1.0 = answer fully and directly addresses the question
- 0.5 = answer partially addresses the question
- 0.0 = answer does not address the question

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

_HALLUCINATION_PROMPT = """You are an evaluation judge. Given a retrieved context and an answer,
identify sentences in the answer that contain claims NOT supported by the context.

Retrieved Context:
{context}

Answer:
{answer}

Count the total sentences in the answer and how many are unsupported.
Respond in this exact format:
TOTAL_SENTENCES: <number>
UNSUPPORTED_SENTENCES: <number>
UNSUPPORTED_CONTENT: <list the unsupported sentences, or "None">"""


def _call_judge(prompt: str) -> str:
    try:
        response = _judge.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "0.5"


def faithfulness_score(question: str, context: str, answer: str) -> float:
    prompt = _FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
    raw = _call_judge(prompt)
    try:
        return round(float(re.search(r"[\d.]+", raw).group()), 4)
    except Exception:
        return 0.5


def answer_relevance_score(question: str, answer: str) -> float:
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    raw = _call_judge(prompt)
    try:
        return round(float(re.search(r"[\d.]+", raw).group()), 4)
    except Exception:
        return 0.5


def hallucination_analysis(context: str, answer: str) -> dict:
    prompt = _HALLUCINATION_PROMPT.format(context=context, answer=answer)
    raw = _call_judge(prompt)

    total = re.search(r"TOTAL_SENTENCES:\s*(\d+)", raw)
    unsupported = re.search(r"UNSUPPORTED_SENTENCES:\s*(\d+)", raw)
    content = re.search(r"UNSUPPORTED_CONTENT:\s*(.+)", raw, re.DOTALL)

    total_n = int(total.group(1)) if total else 1
    unsupported_n = int(unsupported.group(1)) if unsupported else 0
    hallucination_rate = round(unsupported_n / max(total_n, 1), 4)

    return {
        "total_sentences": total_n,
        "unsupported_sentences": unsupported_n,
        "hallucination_rate": hallucination_rate,
        "unsupported_content": content.group(1).strip() if content else "None"
    }


# ---------------------------------------------------------------------------
# Limitation extraction metrics (precision / recall / F1)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_overlap(pred: str, ref: str) -> float:
    pred_tokens = set(_normalize(pred).split())
    ref_tokens = set(_normalize(ref).split())
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def limitation_precision(predicted: list, ground_truth: list, threshold: float = 0.4) -> float:
    if not predicted:
        return 0.0
    hits = sum(
        1 for p in predicted
        if any(_token_overlap(p, g) >= threshold for g in ground_truth)
    )
    return round(hits / len(predicted), 4)


def limitation_recall(predicted: list, ground_truth: list, threshold: float = 0.4) -> float:
    if not ground_truth:
        return 0.0
    hits = sum(
        1 for g in ground_truth
        if any(_token_overlap(p, g) >= threshold for p in predicted)
    )
    return round(hits / len(ground_truth), 4)


def limitation_f1(predicted: list, ground_truth: list, threshold: float = 0.4) -> float:
    p = limitation_precision(predicted, ground_truth, threshold)
    r = limitation_recall(predicted, ground_truth, threshold)
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(fn, *args, **kwargs) -> tuple:
    """Returns (result, latency_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    latency = round(time.perf_counter() - start, 3)
    return result, latency


# ---------------------------------------------------------------------------
# Context builder (for LLM judge prompts)
# ---------------------------------------------------------------------------

def build_context_string(retrieved: list) -> str:
    return "\n\n".join(
        f"[{c.get('paper_name')} | Page {c.get('page_number')}]\n{c.get('text', '')}"
        for c in retrieved
    )


# ---------------------------------------------------------------------------
# Full single-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(question: str, expected_keywords: list, retriever_fn, generator_fn) -> dict:
    """
    Run one full evaluation cycle for a single query.
    retriever_fn(question) -> list of chunks
    generator_fn(question, chunks) -> answer string
    """
    retrieved, retrieval_latency = measure_latency(retriever_fn, question)
    relevant_ids = build_relevant_ids(retrieved, expected_keywords)

    context = build_context_string(retrieved)
    answer, generation_latency = measure_latency(generator_fn, question, retrieved)

    faith = faithfulness_score(question, context, answer)
    relevance = answer_relevance_score(question, answer)
    hallucination = hallucination_analysis(context, answer)
    citation = citation_accuracy(answer, retrieved)

    return {
        "question": question,
        "retrieved_count": len(retrieved),
        "relevant_count": len(relevant_ids),
        "precision_at_5": precision_at_k(retrieved, relevant_ids, 5),
        "recall_at_5": recall_at_k(retrieved, relevant_ids, 5),
        "mrr": mrr(retrieved, relevant_ids),
        "faithfulness": faith,
        "answer_relevance": relevance,
        "citation_accuracy": citation,
        "hallucination_rate": hallucination["hallucination_rate"],
        "unsupported_sentences": hallucination["unsupported_sentences"],
        "retrieval_latency_s": retrieval_latency,
        "generation_latency_s": generation_latency,
        "total_latency_s": round(retrieval_latency + generation_latency, 3),
        "answer": answer,
        "unsupported_content": hallucination["unsupported_content"]
    }
