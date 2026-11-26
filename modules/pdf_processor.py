import pymupdf as fitz
import os
from typing import Union


def extract_metadata(doc: fitz.Document, filename: str) -> dict:
    meta = doc.metadata
    return {
        "paper_name": meta.get("title") or os.path.splitext(filename)[0],
        "author": meta.get("author") or "Unknown",
        "total_pages": doc.page_count,
    }


def extract_pages(doc: fitz.Document) -> list:
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            text = "[SCANNED OR EMPTY PAGE - No extractable text]"
        pages.append({
            "page_number": i + 1,
            "text": text
        })
    return pages


def process_pdf(file_path: str, original_filename: str = None) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {file_path}")

    # Use original_filename if provided, otherwise fall back to file_path basename
    filename = original_filename if original_filename else os.path.basename(file_path)
    doc = fitz.open(file_path)

    result = extract_metadata(doc, filename)
    result["pages"] = extract_pages(doc)

    doc.close()
    return result


def process_multiple_pdfs(file_paths: list) -> list:
    results = []
    for path in file_paths:
        try:
            results.append(process_pdf(path))
        except Exception as e:
            results.append({
                "paper_name": os.path.basename(path),
                "error": str(e)
            })
    return results
