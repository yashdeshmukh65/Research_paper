import os
import sys
import json
import logging
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SAGE")

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SAGE – Scholarly Analysis and Gap Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Lazy imports — cached so heavy models load only once per server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline_modules():
    from modules.pdf_processor import process_pdf
    from modules.chunking import chunk_paper
    from modules.embeddings import embed_chunks
    from modules.vector_store import add_to_index, load_index
    from modules.bm25_retriever import add_to_bm25
    return process_pdf, chunk_paper, embed_chunks, add_to_index, load_index, add_to_bm25


@st.cache_resource(show_spinner=False)
def get_agents():
    from modules.qa_engine import ask
    from agents.paper_analysis import analyze_multiple_papers
    from agents.comparison_agent import compare_papers
    from agents.research_gap_agent import find_research_gaps
    return ask, analyze_multiple_papers, compare_papers, find_research_gaps


# ---------------------------------------------------------------------------
# Fresh-session workspace management
# ---------------------------------------------------------------------------

def _clear_disk_indexes():
    """Delete FAISS and BM25 index files from disk — called once per new session."""
    from modules.vector_store import clear_index
    from modules.bm25_retriever import clear_bm25
    try:
        clear_index()
        clear_bm25()
        logger.info("Disk indexes cleared for new session.")
    except Exception as e:
        logger.warning(f"Could not clear disk indexes: {e}")


def _clear_uploaded_files():
    """Remove all PDFs from data/uploaded_papers/."""
    upload_dir = os.path.join("data", "uploaded_papers")
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.endswith(".pdf"):
                try:
                    os.remove(os.path.join(upload_dir, f))
                except Exception as e:
                    logger.warning(f"Could not delete {f}: {e}")


def init_state():
    """
    Called once per browser session.
    On first run: clears disk indexes so old papers never bleed into new sessions.
    """
    if "session_initialized" not in st.session_state:
        _clear_disk_indexes()
        _clear_uploaded_files()
        st.session_state["session_initialized"] = True
        st.session_state["indexed_papers"] = []      # {name, pages, chunks}
        st.session_state["chat_history"] = []
        st.session_state["last_analysis"] = None
        st.session_state["last_comparison"] = None
        st.session_state["last_gaps"] = None
        logger.info("New session initialized — fresh workspace.")

    # Ensure all keys exist even if session was partially initialized
    defaults = {
        "indexed_papers": [],
        "chat_history": [],
        "last_analysis": None,
        "last_comparison": None,
        "last_gaps": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_paper_names() -> list:
    """Return list of paper name strings from session state."""
    return [p["name"] for p in st.session_state.get("indexed_papers", [])]


def _remove_paper(paper_name: str):
    """Remove a single paper from indexes, disk, and session state."""
    from modules.vector_store import remove_paper_from_index
    from modules.bm25_retriever import remove_paper_from_bm25

    try:
        remove_paper_from_index(paper_name)
        remove_paper_from_bm25(paper_name)
        logger.info(f"Removed '{paper_name}' from indexes.")
    except Exception as e:
        logger.error(f"Error removing '{paper_name}' from indexes: {e}")

    # Remove PDF from disk
    upload_dir = os.path.join("data", "uploaded_papers")
    for f in os.listdir(upload_dir) if os.path.exists(upload_dir) else []:
        if os.path.splitext(f)[0] == paper_name or f == paper_name + ".pdf":
            try:
                os.remove(os.path.join(upload_dir, f))
            except Exception:
                pass

    # Remove from session
    st.session_state["indexed_papers"] = [
        p for p in st.session_state["indexed_papers"] if p["name"] != paper_name
    ]
    # Invalidate cached results that referenced this paper
    st.session_state["last_analysis"] = None
    st.session_state["last_comparison"] = None
    st.session_state["last_gaps"] = None
    st.session_state["last_report"] = None
    logger.info(f"'{paper_name}' removed from session.")


def _clear_all_papers():
    """Remove all papers from indexes, disk, and session state."""
    from modules.vector_store import clear_index
    from modules.bm25_retriever import clear_bm25
    try:
        clear_index()
        clear_bm25()
    except Exception as e:
        logger.error(f"Error clearing indexes: {e}")
    _clear_uploaded_files()
    st.session_state["indexed_papers"] = []
    st.session_state["last_analysis"] = None
    st.session_state["last_comparison"] = None
    st.session_state["last_gaps"] = None
    st.session_state["chat_history"] = []
    logger.info("All papers cleared.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/microscope.png", width=60)
        st.title("SAGE")
        st.caption("Scholarly Analysis and Gap Explorer")
        st.divider()

        page = st.radio(
            "Navigate",
            options=[
                "📤  Upload Papers",
                "💬  Research Chat",
                "🔍  Paper Analysis",
                "⚖️  Paper Comparison",
                "🕳️  Research Gap Finder",
                "📥  Export Reports",
            ],
            label_visibility="collapsed"
        )

        st.divider()
        papers = st.session_state.get("indexed_papers", [])
        st.caption(f"**{len(papers)} paper(s) in session**")

        for p in papers:
            col_name, col_btn = st.columns([4, 1])
            col_name.caption(f"• {p['name']}")
            if col_btn.button("✕", key=f"remove_{p['name']}", help=f"Remove {p['name']}"):
                _remove_paper(p["name"])
                st.rerun()

        if papers:
            if st.button("🗑️ Clear All Papers", use_container_width=True):
                _clear_all_papers()
                st.rerun()

    return page


# ---------------------------------------------------------------------------
# Page 1: Upload Research Papers
# ---------------------------------------------------------------------------
def page_upload():
    st.header("📤 Upload Research Papers")
    st.caption("Upload PDF files to extract, chunk, embed and index them for analysis.")

    already = get_paper_names()
    if already:
        st.info(f"**{len(already)} paper(s) already indexed this session:** {', '.join(already)}")

    uploaded_files = st.file_uploader(
        "Drop your PDF files here",
        type=["pdf"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        if not already:
            st.info("Upload one or more PDF research papers to get started.")
        return

    # Filter out already-indexed papers
    new_files = [f for f in uploaded_files if f.name.replace(".pdf", "") not in already
                 and os.path.splitext(f.name)[0] not in already]

    if not new_files:
        st.warning("All uploaded files are already indexed in this session.")
        return

    if st.button("Process & Index Papers", type="primary", use_container_width=True):
        process_pdf, chunk_paper, embed_chunks, add_to_index, _, add_to_bm25 = get_pipeline_modules()

        progress = st.progress(0, text="Starting...")
        results = []

        for i, uploaded_file in enumerate(new_files):
            name = uploaded_file.name
            progress.progress(i / len(new_files), text=f"Processing: {name}")
            tmp_path = None

            try:
                # Validate file size (max 50MB)
                file_bytes = uploaded_file.read()
                if len(file_bytes) > 50 * 1024 * 1024:
                    raise ValueError("File exceeds 50MB limit.")

                # Write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                # Save to uploads folder
                save_path = os.path.join("data", "uploaded_papers", name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(file_bytes)

                # Pipeline
                logger.info(f"Processing: {name}")
                paper_data = process_pdf(tmp_path, original_filename=name)

                if not paper_data.get("pages"):
                    raise ValueError("No pages could be extracted. File may be scanned or corrupted.")

                chunks = chunk_paper(paper_data)
                if not chunks:
                    raise ValueError("No text chunks produced. Paper may be empty after cleaning.")

                chunks_with_embeddings = embed_chunks(chunks)
                add_to_index(chunks_with_embeddings)
                add_to_bm25(chunks_with_embeddings)

                paper_name = paper_data["paper_name"]

                # Avoid duplicates in session
                if paper_name not in get_paper_names():
                    st.session_state["indexed_papers"].append({
                        "name": paper_name,
                        "pages": paper_data["total_pages"],
                        "chunks": len(chunks),
                        "filename": name
                    })

                results.append({
                    "name": paper_name,
                    "pages": paper_data["total_pages"],
                    "chunks": len(chunks),
                    "status": "✅ Success"
                })
                logger.info(f"Indexed '{paper_name}' — {len(chunks)} chunks.")

            except Exception as e:
                logger.error(f"Failed to process '{name}': {e}")
                results.append({"name": name, "pages": "-", "chunks": "-", "status": f"❌ {str(e)}"})

            finally:
                # Always clean up temp file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        progress.progress(1.0, text="Done!")

        successes = [r for r in results if "✅" in r["status"]]
        failures  = [r for r in results if "❌" in r["status"]]

        if successes:
            st.success(f"Successfully indexed {len(successes)} paper(s).")
        if failures:
            st.error(f"{len(failures)} file(s) failed to process.")

        st.dataframe(
            results,
            column_config={
                "name": "Paper", "pages": "Pages",
                "chunks": "Chunks Created", "status": "Status"
            },
            use_container_width=True,
            hide_index=True
        )


# ---------------------------------------------------------------------------
# Page 2: Research Chat
# ---------------------------------------------------------------------------
def page_chat():
    st.header("💬 Research Chat")
    st.caption("Ask questions across all your uploaded papers. Answers include source citations.")

    papers = get_paper_names()
    if not papers:
        st.warning("No papers indexed yet. Please upload papers first.")
        return

    st.info(f"Searching across **{len(papers)}** paper(s): {', '.join(papers)}")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.caption(
                            f"**{src['paper_name']}** — Page {src['page_number']} "
                            f"| Section: {src['section']} | Score: {src['relevance_score']}"
                        )

    question = st.chat_input("Ask a question about your research papers...")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching papers and generating answer..."):
                try:
                    ask, *_ = get_agents()
                    result = ask(question)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                except Exception as e:
                    logger.error(f"Q&A failed: {e}")
                    answer = f"⚠️ An error occurred while generating the answer: {str(e)}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("📎 Sources"):
                    for src in sources:
                        st.caption(
                            f"**{src['paper_name']}** — Page {src['page_number']} "
                            f"| Section: {src['section']} | Score: {src['relevance_score']}"
                        )

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    if st.session_state["chat_history"]:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Page 3: Paper Analysis
# ---------------------------------------------------------------------------
def page_analysis():
    st.header("🔍 Paper Analysis")
    st.caption("Extract structured information from each paper automatically.")

    papers = get_paper_names()
    if not papers:
        st.warning("No papers indexed yet. Please upload papers first.")
        return

    selected = st.multiselect("Select papers to analyze", papers, default=papers)
    if not selected:
        st.info("Select at least one paper to analyze.")
        return

    if st.button("Run Analysis", type="primary", use_container_width=True):
        _, analyze_multiple_papers, *_ = get_agents()
        with st.spinner("Analyzing papers with Gemini..."):
            try:
                analyses = analyze_multiple_papers(selected)
                st.session_state["last_analysis"] = [a.model_dump() for a in analyses]
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                st.error(f"Analysis failed: {str(e)}")

    if st.session_state["last_analysis"]:
        analyses = st.session_state["last_analysis"]

        view = st.radio(
            "View mode",
            ["📊 Side-by-Side Comparison Table", "📄 Individual Paper Cards"],
            horizontal=True
        )

        fields = [
            ("Research Problem",  "research_problem"),
            ("Methodology",       "methodology"),
            ("Model / Algorithm", "model"),
            ("Dataset",           "dataset"),
            ("Evaluation Metrics","evaluation_metrics"),
            ("Results",           "results"),
            ("Limitations",       "limitations"),
            ("Future Work",       "future_work"),
        ]

        if "Side-by-Side" in view:
            # Build comparison table: rows = fields, columns = papers
            header_cols = st.columns([2] + [3] * len(analyses))
            header_cols[0].markdown("**Field**")
            for i, a in enumerate(analyses):
                header_cols[i + 1].markdown(f"**{a['paper_name']}**")
            st.divider()
            for label, key in fields:
                row_cols = st.columns([2] + [3] * len(analyses))
                row_cols[0].markdown(f"**{label}**")
                for i, a in enumerate(analyses):
                    row_cols[i + 1].caption(a.get(key, "—"))
                st.divider()
        else:
            for analysis in analyses:
                with st.expander(f"📄 {analysis['paper_name']}", expanded=True):
                    cols = st.columns(2)
                    for i, (label, key) in enumerate(fields):
                        with cols[i % 2]:
                            st.markdown(f"**{label}**")
                            st.caption(analysis.get(key, "—"))
                            st.divider()


# ---------------------------------------------------------------------------
# Page 4: Paper Comparison
# ---------------------------------------------------------------------------
def page_comparison():
    st.header("⚖️ Paper Comparison")
    st.caption("Compare methodologies, datasets, results and limitations across papers.")

    papers = get_paper_names()
    if len(papers) < 2:
        st.warning("At least 2 indexed papers are required for comparison.")
        return

    selected = st.multiselect("Select papers to compare (min 2)", papers, default=papers[:2])
    if len(selected) < 2:
        st.info("Select at least 2 papers.")
        return

    if st.button("Compare Papers", type="primary", use_container_width=True):
        _, _, compare_papers, *_ = get_agents()
        with st.spinner("Comparing papers..."):
            try:
                result = compare_papers(selected)
                st.session_state["last_comparison"] = result
            except Exception as e:
                logger.error(f"Comparison failed: {e}")
                st.error(f"Comparison failed: {str(e)}")

    if st.session_state["last_comparison"]:
        result = st.session_state["last_comparison"]
        table = result["comparison_table"]

        st.subheader("Comparison Table")
        rows = []
        for field, paper_values in table.items():
            row = {"Field": field.replace("_", " ").title()}
            row.update(paper_values)
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.subheader("Narrative Summary")
        st.markdown(result["narrative"])


# ---------------------------------------------------------------------------
# Page 5: Research Gap Finder
# ---------------------------------------------------------------------------
def page_gaps():
    st.header("🕳️ Research Gap Finder")
    st.caption("Identify recurring limitations and candidate research directions across papers.")

    papers = get_paper_names()
    if len(papers) < 2:
        st.warning("At least 2 indexed papers are required for gap analysis.")
        return

    selected = st.multiselect("Select papers for gap analysis", papers, default=papers)
    if len(selected) < 2:
        st.info("Select at least 2 papers.")
        return

    if st.button("Find Research Gaps", type="primary", use_container_width=True):
        _, _, _, find_research_gaps = get_agents()
        with st.spinner("Clustering limitations and synthesizing research directions..."):
            try:
                result = find_research_gaps(selected)
                st.session_state["last_gaps"] = result
            except Exception as e:
                logger.error(f"Gap analysis failed: {e}")
                st.error(f"Gap analysis failed: {str(e)}")

    if st.session_state["last_gaps"]:
        result = st.session_state["last_gaps"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Papers Analyzed", len(result["paper_names"]))
        col2.metric("Statements Extracted", result["total_statements"])
        col3.metric("Clusters Found", result["total_clusters"])

        st.subheader("Limitation Clusters")
        for cluster in result["clusters"]:
            with st.expander(
                f"Cluster {cluster['cluster_id']} — {cluster['paper_count']} paper(s), "
                f"{cluster['statement_count']} statement(s)",
                expanded=cluster["cluster_id"] == 1
            ):
                if cluster["limitations"]:
                    st.markdown("**Limitations:**")
                    for lim in cluster["limitations"]:
                        st.caption(f"• {lim}")
                if cluster["future_work_statements"]:
                    st.markdown("**Future Work Statements:**")
                    for fw in cluster["future_work_statements"]:
                        st.caption(f"• {fw}")
                st.caption(f"Papers: {', '.join(cluster['papers_involved'])}")

        st.subheader("Candidate Research Directions")
        st.markdown(result["research_directions"])


# ---------------------------------------------------------------------------
# Page 6: Export Reports
# ---------------------------------------------------------------------------
def page_export():
    st.header("📥 Export Reports")
    st.caption("Download your analysis, comparison, gap analysis and literature review reports.")

    has_content = any([
        st.session_state.get("last_analysis"),
        st.session_state.get("last_comparison"),
        st.session_state.get("last_gaps"),
    ])

    if not has_content:
        st.info("No results to export yet. Run analysis, comparison, or gap finder first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.get("last_analysis"):
            st.subheader("Paper Analysis")
            st.download_button(
                "⬇️ Download Analysis (JSON)",
                data=json.dumps(st.session_state["last_analysis"], indent=2),
                file_name="paper_analysis.json",
                mime="application/json",
                use_container_width=True
            )

        if st.session_state.get("last_gaps"):
            st.subheader("Research Gaps")
            gaps = st.session_state["last_gaps"]
            export = {
                "paper_names": gaps["paper_names"],
                "total_clusters": gaps["total_clusters"],
                "clusters": gaps["clusters"],
                "research_directions": gaps["research_directions"]
            }
            st.download_button(
                "⬇️ Download Gap Analysis (JSON)",
                data=json.dumps(export, indent=2),
                file_name="research_gaps.json",
                mime="application/json",
                use_container_width=True
            )

    with col2:
        if st.session_state.get("last_comparison"):
            st.subheader("Paper Comparison")
            st.download_button(
                "⬇️ Download Comparison (JSON)",
                data=json.dumps(st.session_state["last_comparison"], indent=2),
                file_name="paper_comparison.json",
                mime="application/json",
                use_container_width=True
            )

    if st.session_state.get("chat_history"):
        st.subheader("Chat History")
        chat_export = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["chat_history"]
        ]
        st.download_button(
            "⬇️ Download Chat History (JSON)",
            data=json.dumps(chat_export, indent=2),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True
        )

    reports_dir = "reports"
    if os.path.exists(reports_dir):
        saved = [f for f in os.listdir(reports_dir) if f.endswith(".md")]
        if saved:
            st.subheader("Previously Saved Reports")
            for fname in sorted(saved, reverse=True):
                fpath = os.path.join(reports_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    st.download_button(
                        f"⬇️ {fname}",
                        data=content,
                        file_name=fname,
                        mime="text/markdown",
                        use_container_width=True,
                        key=f"dl_{fname}"
                    )
                except Exception as e:
                    st.caption(f"Could not read {fname}: {e}")


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
def main():
    init_state()
    page = render_sidebar()

    if   "Upload"     in page: page_upload()
    elif "Chat"       in page: page_chat()
    elif "Analysis"   in page: page_analysis()
    elif "Comparison" in page: page_comparison()
    elif "Gap"        in page: page_gaps()
    elif "Export"     in page: page_export()


if __name__ == "__main__":
    main()
