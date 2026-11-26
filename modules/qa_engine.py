import os
import google.generativeai as genai
from modules.hybrid_retriever import hybrid_search
from utils.prompts import build_rag_prompt
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")


def retrieve(question: str, top_k: int = 5) -> list:
    return hybrid_search(question, top_k=top_k)


def ask(question: str, top_k: int = 5) -> dict:
    retrieved_chunks = retrieve(question, top_k=top_k)

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "No relevant content found in the uploaded papers.",
            "sources": []
        }

    prompt = build_rag_prompt(question, retrieved_chunks)
    response = _gemini_model.generate_content(prompt)

    sources = [
        {
            "paper_name": c["paper_name"],
            "page_number": c["page_number"],
            "section": c.get("section", "Unknown"),
            # normalize: reranked chunks use rerank_score; FAISS uses score; BM25 uses bm25_score
            "relevance_score": c.get("rerank_score") or c.get("score") or c.get("bm25_score") or 0.0
        }
        for c in retrieved_chunks
    ]

    return {
        "question": question,
        "answer": response.text.strip(),
        "sources": sources
    }
