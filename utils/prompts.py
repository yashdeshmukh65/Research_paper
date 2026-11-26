# ---------------------------------------------------------------------------
# RAG Q&A Prompt
# ---------------------------------------------------------------------------

RAG_QA_PROMPT = """You are an expert research assistant that helps researchers analyze academic papers.

You are given a set of retrieved passages from research papers uploaded by the user.
Your job is to answer the user's question strictly based on the provided context.

STRICT RULES:
- Only use information present in the retrieved context below.
- Do NOT invent, assume, or hallucinate any research findings, statistics, or claims.
- If the retrieved context does not contain enough information to answer the question, clearly say:
  "The uploaded papers do not contain sufficient information to answer this question."
- For every factual claim you make, cite the source using this format: (Paper: <paper_name>, Page: <page_number>)
- If multiple papers discuss the same point, cite all relevant sources.

---

RETRIEVED CONTEXT:
{context}

---

USER QUESTION:
{question}

---

ANSWER:
"""


def format_context(retrieved_chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(
            f"[{i}] Paper: {chunk['paper_name']} | Page: {chunk['page_number']} | Section: {chunk['section']}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def build_rag_prompt(question: str, retrieved_chunks: list) -> str:
    context = format_context(retrieved_chunks)
    return RAG_QA_PROMPT.format(context=context, question=question)


# ---------------------------------------------------------------------------
# Paper Analysis Prompt
# ---------------------------------------------------------------------------

PAPER_ANALYSIS_PROMPT = """You are ANNA, an expert research assistant. Analyze the following research paper content and extract structured information.

STRICT RULES:
- Extract only what is explicitly stated in the paper content below.
- If a field is not mentioned, return exactly: "Not mentioned in the paper".
- Do NOT infer, assume, or hallucinate any information.
- Be concise but complete.

PAPER CONTENT:
{paper_content}

---

Return your response as a valid JSON object with exactly these keys:
{{
  "research_problem": "What problem is being addressed?",
  "methodology": "What approach or method is used?",
  "model": "Which architecture or algorithm is proposed?",
  "dataset": "Which datasets are used?",
  "evaluation_metrics": "Which metrics are reported (e.g. accuracy, F1, mAP)?",
  "results": "What were the key findings or performance numbers?",
  "limitations": "What limitations do the authors state?",
  "future_work": "What future directions do the authors propose?"
}}

JSON RESPONSE:
"""


# ---------------------------------------------------------------------------
# Paper Comparison Prompt
# ---------------------------------------------------------------------------

COMPARISON_PROMPT = """You are ANNA, an expert research assistant. Compare the following research papers based on their extracted structured information.

STRICT RULES:
- Only use the information provided below for each paper.
- Do NOT invent or assume any values.
- If a field is missing for a paper, write "Not reported".
- Be objective and factual.

PAPERS DATA:
{papers_data}

---

Provide:
1. A structured comparison table covering: Model, Dataset, Evaluation Metrics, Key Results, Limitations.
2. A brief paragraph (3-5 sentences) summarizing the key differences and similarities across the papers.
3. A recommendation on which paper's approach seems most promising and why, based strictly on the provided data.

RESPONSE:
"""


def build_analysis_prompt(paper_content: str) -> str:
    return PAPER_ANALYSIS_PROMPT.format(paper_content=paper_content)


def build_comparison_prompt(papers_data: str) -> str:
    return COMPARISON_PROMPT.format(papers_data=papers_data)


# ---------------------------------------------------------------------------
# Research Gap Prompt
# ---------------------------------------------------------------------------

RESEARCH_GAP_PROMPT = """You are ANNA, an expert research assistant helping researchers identify potential research gaps.

You are given clusters of semantically similar limitations and future-work statements extracted from multiple research papers.
Your task is to synthesize these into candidate research directions.

CRITICAL RULES:
- Clearly distinguish between what authors EXPLICITLY stated vs. what you are inferring.
- Use phrases like "The authors explicitly note..." for direct evidence.
- Use phrases like "This suggests a candidate research direction..." for inferred gaps.
- Do NOT claim these gaps are scientifically proven to be unexplored — they are candidates based on the provided papers only.
- Do NOT hallucinate limitations or future work not present in the evidence below.
- Every suggested direction must be traceable to at least one piece of evidence.

LIMITATION & FUTURE WORK CLUSTERS:
{clusters}

---

For each cluster, provide:
1. THEME NAME: A concise label for the recurring theme (e.g., "Data Efficiency and Generalization").
2. EVIDENCE: List the specific paper statements that form this cluster, with paper names.
3. CANDIDATE RESEARCH DIRECTION: A concrete, actionable research direction suggested by this cluster.
4. CONFIDENCE: Low / Medium / High — based on how many papers independently raise this theme.

End with an OVERALL SUMMARY of the top 3 most promising research directions across all clusters.

RESPONSE:
"""


def build_research_gap_prompt(clusters: str) -> str:
    return RESEARCH_GAP_PROMPT.format(clusters=clusters)


# ---------------------------------------------------------------------------
# Report Generation Prompt
# ---------------------------------------------------------------------------

REPORT_PROMPT = """You are ANNA, an expert research assistant. Generate a structured literature review report based on the research analysis data provided below.

STRICT RULES:
- Only use the information provided. Do NOT hallucinate findings.
- Write in formal academic style.
- Every claim must be traceable to a specific paper.
- Clearly separate what papers explicitly state from synthesized observations.

ANALYSIS DATA:
{analysis_data}

---

Generate a report with the following sections:

# Literature Review Report

## 1. Overview
Brief summary of the papers covered and the research domain.

## 2. Research Problems Addressed
What problems do these papers collectively address?

## 3. Methodologies and Models
Summarize the approaches used across papers. Note similarities and differences.

## 4. Datasets and Evaluation
What datasets and metrics are used? Are there common benchmarks?

## 5. Key Results and Findings
Summarize the most significant results. Cite paper names.

## 6. Limitations Across Papers
What limitations are commonly reported?

## 7. Research Gaps and Future Directions
Based on the gap analysis, what are the most promising directions?

## 8. Conclusion
A concise closing synthesis of the literature.

REPORT:
"""


def build_report_prompt(analysis_data: str) -> str:
    return REPORT_PROMPT.format(analysis_data=analysis_data)
