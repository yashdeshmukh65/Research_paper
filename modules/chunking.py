from modules.text_cleaner import clean_paper

# ~4 characters per token is a standard approximation
CHARS_PER_TOKEN = 4
CHUNK_SIZE_TOKENS = 1000
CHUNK_OVERLAP_TOKENS = 150

CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN      # 4000 chars
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN  # 600 chars


def _split_text(text: str, chunk_size: int, overlap: int) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def chunk_paper(paper: dict) -> list:
    cleaned = clean_paper(paper)
    chunks = []
    chunk_id = 0

    for page in cleaned["pages"]:
        page_text = page["text"]
        page_number = page["page_number"]
        section = page["section"]

        # If page text fits in one chunk, keep it as-is
        if len(page_text) <= CHUNK_SIZE_CHARS:
            chunks.append({
                "chunk_id": chunk_id,
                "text": page_text,
                "paper_name": cleaned["paper_name"],
                "author": cleaned["author"],
                "page_number": page_number,
                "section": section,
            })
            chunk_id += 1
        else:
            for part in _split_text(page_text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS):
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": part,
                    "paper_name": cleaned["paper_name"],
                    "author": cleaned["author"],
                    "page_number": page_number,
                    "section": section,
                })
                chunk_id += 1

    return chunks


def chunk_multiple_papers(papers: list) -> list:
    all_chunks = []
    for paper in papers:
        if "error" in paper:
            continue
        all_chunks.extend(chunk_paper(paper))
    return all_chunks
