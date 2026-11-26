import os
import json
import google.generativeai as genai
from pydantic import BaseModel, field_validator
from modules.vector_store import load_index
from utils.prompts import build_analysis_prompt
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

NOT_MENTIONED = "Not mentioned in the paper"


# ---------------------------------------------------------------------------
# Pydantic schema for structured output validation
# ---------------------------------------------------------------------------

class PaperAnalysis(BaseModel):
    paper_name: str
    research_problem: str
    methodology: str
    model: str
    dataset: str
    evaluation_metrics: str
    results: str
    limitations: str
    future_work: str

    @field_validator("*", mode="before")
    @classmethod
    def empty_to_not_mentioned(cls, v):
        if not v or str(v).strip() == "":
            return NOT_MENTIONED
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_paper_chunks(paper_name: str) -> list:
    """Load all chunks belonging to a specific paper from the FAISS metadata."""
    _, metadata = load_index()
    return [c for c in metadata if c["paper_name"].lower() == paper_name.lower()]


def _build_paper_content(chunks: list) -> str:
    """Reconstruct a readable version of the paper from its chunks, ordered by page."""
    sorted_chunks = sorted(chunks, key=lambda c: (c["page_number"], c["chunk_id"]))
    parts = []
    seen_pages = set()
    for chunk in sorted_chunks:
        page = chunk["page_number"]
        header = f"[Page {page} | Section: {chunk['section']}]" if page not in seen_pages else ""
        seen_pages.add(page)
        parts.append(f"{header}\n{chunk['text']}" if header else chunk["text"])
    return "\n\n".join(parts)


def _parse_gemini_json(raw: str) -> dict:
    """Extract JSON from Gemini response, stripping markdown code fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_paper(paper_name: str) -> PaperAnalysis:
    chunks = _get_paper_chunks(paper_name)
    if not chunks:
        raise ValueError(f"No indexed chunks found for paper: '{paper_name}'. Please index the paper first.")

    paper_content = _build_paper_content(chunks)

    # Truncate to ~12000 chars to stay within Gemini context safely
    if len(paper_content) > 12000:
        paper_content = paper_content[:12000] + "\n\n[Content truncated for analysis...]"

    prompt = build_analysis_prompt(paper_content)
    response = _gemini_model.generate_content(prompt)

    parsed = _parse_gemini_json(response.text)
    parsed["paper_name"] = paper_name

    return PaperAnalysis(**parsed)


def analyze_multiple_papers(paper_names: list) -> list[PaperAnalysis]:
    results = []
    for name in paper_names:
        try:
            results.append(analyze_paper(name))
        except Exception as e:
            results.append(PaperAnalysis(
                paper_name=name,
                research_problem=f"Analysis failed: {str(e)}",
                methodology=NOT_MENTIONED,
                model=NOT_MENTIONED,
                dataset=NOT_MENTIONED,
                evaluation_metrics=NOT_MENTIONED,
                results=NOT_MENTIONED,
                limitations=NOT_MENTIONED,
                future_work=NOT_MENTIONED
            ))
    return results
