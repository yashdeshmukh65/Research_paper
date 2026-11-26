import re


# Common research paper section headers
SECTION_HEADERS = [
    "abstract", "introduction", "background", "related work",
    "literature review", "methodology", "methods", "approach",
    "proposed method", "experiments", "experimental setup",
    "results", "evaluation", "discussion", "conclusion",
    "future work", "references", "acknowledgements", "appendix"
]

_HEADER_PATTERN = re.compile(
    r"(?i)^\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*$",
    re.MULTILINE
)


def detect_section(text: str) -> str:
    matches = _HEADER_PATTERN.findall(text)
    return matches[-1].title() if matches else "Unknown"


def clean_text(text: str) -> str:
    if not text or text.startswith("[SCANNED"):
        return ""

    # Remove hyphenation at line breaks (e.g., "meth-\nod" -> "method")
    text = re.sub(r"-\n", "", text)

    # Collapse multiple newlines into a single newline
    text = re.sub(r"\n{2,}", "\n", text)

    # Replace remaining single newlines with space
    text = re.sub(r"\n", " ", text)

    # Remove non-printable / special characters
    text = re.sub(r"[^\x20-\x7E]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_paper(paper: dict) -> dict:
    cleaned_pages = []
    for page in paper.get("pages", []):
        cleaned = clean_text(page["text"])
        if cleaned:
            cleaned_pages.append({
                "page_number": page["page_number"],
                "text": cleaned,
                "section": detect_section(page["text"])
            })
    return {
        "paper_name": paper["paper_name"],
        "author": paper.get("author", "Unknown"),
        "total_pages": paper.get("total_pages", 0),
        "pages": cleaned_pages
    }
