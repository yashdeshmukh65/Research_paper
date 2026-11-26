import os
import google.generativeai as genai
from agents.paper_analysis import analyze_multiple_papers, PaperAnalysis
from utils.prompts import build_comparison_prompt
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_papers_for_prompt(analyses: list[PaperAnalysis]) -> str:
    parts = []
    for i, paper in enumerate(analyses, 1):
        parts.append(
            f"PAPER {i}: {paper.paper_name}\n"
            f"  Research Problem   : {paper.research_problem}\n"
            f"  Methodology        : {paper.methodology}\n"
            f"  Model/Algorithm    : {paper.model}\n"
            f"  Dataset            : {paper.dataset}\n"
            f"  Evaluation Metrics : {paper.evaluation_metrics}\n"
            f"  Results            : {paper.results}\n"
            f"  Limitations        : {paper.limitations}\n"
            f"  Future Work        : {paper.future_work}"
        )
    return "\n\n".join(parts)


def _build_comparison_table(analyses: list[PaperAnalysis]) -> dict:
    """Build a structured dict comparison table directly from Pydantic objects."""
    fields = ["model", "dataset", "evaluation_metrics", "results", "limitations", "future_work"]
    table = {}
    for field in fields:
        table[field] = {
            paper.paper_name: getattr(paper, field)
            for paper in analyses
        }
    return table


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------

def compare_papers(paper_names: list) -> dict:
    if len(paper_names) < 2:
        raise ValueError("At least 2 papers are required for comparison.")

    analyses = analyze_multiple_papers(paper_names)

    # Structured table built directly from Pydantic objects (no LLM needed)
    comparison_table = _build_comparison_table(analyses)

    # LLM generates the narrative summary + recommendation
    papers_data = _format_papers_for_prompt(analyses)
    prompt = build_comparison_prompt(papers_data)
    response = _gemini_model.generate_content(prompt)

    return {
        "papers_compared": paper_names,
        "comparison_table": comparison_table,
        "narrative": response.text.strip(),
        "individual_analyses": [a.model_dump() for a in analyses]
    }
