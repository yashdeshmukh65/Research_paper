import os
import json
from datetime import datetime
import google.generativeai as genai
from utils.prompts import build_report_prompt
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

REPORTS_DIR = "reports"


def _format_analysis_data(workflow_state: dict) -> str:
    """Serialize the full workflow state into a readable string for the report prompt."""
    parts = []

    paper_names = workflow_state.get("paper_names", [])
    parts.append(f"PAPERS ANALYZED: {', '.join(paper_names)}\n")

    # Individual analyses
    analyses = workflow_state.get("analyses", [])
    if analyses:
        parts.append("=== INDIVIDUAL PAPER ANALYSES ===")
        for a in analyses:
            parts.append(
                f"\nPaper: {a['paper_name']}\n"
                f"  Problem    : {a['research_problem']}\n"
                f"  Methodology: {a['methodology']}\n"
                f"  Model      : {a['model']}\n"
                f"  Dataset    : {a['dataset']}\n"
                f"  Metrics    : {a['evaluation_metrics']}\n"
                f"  Results    : {a['results']}\n"
                f"  Limitations: {a['limitations']}\n"
                f"  Future Work: {a['future_work']}"
            )

    # Comparison narrative
    comparison = workflow_state.get("comparison")
    if comparison and comparison.get("narrative"):
        parts.append("\n=== COMPARISON NARRATIVE ===")
        parts.append(comparison["narrative"])

    # Research gaps
    gaps = workflow_state.get("research_gaps")
    if gaps and gaps.get("research_directions"):
        parts.append("\n=== RESEARCH GAP ANALYSIS ===")
        parts.append(gaps["research_directions"])

    # Q&A results if any
    qa_results = workflow_state.get("qa_results", [])
    if qa_results:
        parts.append("\n=== QUESTION & ANSWER RESULTS ===")
        for qa in qa_results:
            parts.append(f"Q: {qa['question']}\nA: {qa['answer']}")

    return "\n".join(parts)


def generate_report(workflow_state: dict) -> dict:
    analysis_data = _format_analysis_data(workflow_state)
    prompt = build_report_prompt(analysis_data)
    response = _gemini_model.generate_content(prompt)
    report_text = response.text.strip()

    # Save report to disk
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_slug = "_".join(workflow_state.get("paper_names", ["report"]))[:50]
    filename = f"{REPORTS_DIR}/report_{paper_slug}_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_text)

    return {
        "report": report_text,
        "saved_to": filename
    }
