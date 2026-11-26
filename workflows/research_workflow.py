from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from agents.paper_analysis import analyze_multiple_papers
from agents.comparison_agent import compare_papers
from agents.research_gap_agent import find_research_gaps
from agents.report_agent import generate_report
from modules.qa_engine import ask


# ---------------------------------------------------------------------------
# Workflow State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    # Inputs
    paper_names: list[str]
    question: Optional[str]
    tasks: list[str]               # e.g. ["analyze", "compare", "gaps", "qa", "report"]

    # Outputs populated by each node
    analyses: Optional[list[dict]]
    comparison: Optional[dict]
    research_gaps: Optional[dict]
    qa_results: Optional[list[dict]]
    report: Optional[dict]

    # Control
    errors: Annotated[list[str], lambda a, b: a + b]
    completed_tasks: Annotated[list[str], lambda a, b: a + b]


# ---------------------------------------------------------------------------
# Node: Task Router
# Decides which tasks to run based on the user request.
# This is the only node that does "reasoning" — everything else executes.
# ---------------------------------------------------------------------------

def route_tasks(state: ResearchState) -> ResearchState:
    tasks = state.get("tasks", [])

    # If no explicit tasks given, infer from question + paper count
    if not tasks:
        question = state.get("question", "").lower()
        paper_names = state.get("paper_names", [])
        inferred = ["analyze"]

        if len(paper_names) >= 2:
            inferred.append("compare")

        if any(w in question for w in ["gap", "limitation", "future", "missing", "unexplored"]):
            inferred.append("gaps")

        if question:
            inferred.append("qa")

        if any(w in question for w in ["report", "review", "summarize", "literature"]):
            inferred.append("report")

        tasks = inferred

    return {**state, "tasks": tasks, "completed_tasks": [], "errors": []}


# ---------------------------------------------------------------------------
# Node: Paper Analysis
# ---------------------------------------------------------------------------

def run_paper_analysis(state: ResearchState) -> ResearchState:
    if "analyze" not in state["tasks"]:
        return state
    try:
        analyses = analyze_multiple_papers(state["paper_names"])
        return {
            **state,
            "analyses": [a.model_dump() for a in analyses],
            "completed_tasks": ["analyze"]
        }
    except Exception as e:
        return {**state, "errors": [f"Paper analysis failed: {str(e)}"]}


# ---------------------------------------------------------------------------
# Node: Comparison
# ---------------------------------------------------------------------------

def run_comparison(state: ResearchState) -> ResearchState:
    if "compare" not in state["tasks"]:
        return state
    if len(state["paper_names"]) < 2:
        return {**state, "errors": ["Comparison requires at least 2 papers."]}
    try:
        result = compare_papers(state["paper_names"])
        return {**state, "comparison": result, "completed_tasks": ["compare"]}
    except Exception as e:
        return {**state, "errors": [f"Comparison failed: {str(e)}"]}


# ---------------------------------------------------------------------------
# Node: Research Gap Analysis
# ---------------------------------------------------------------------------

def run_research_gaps(state: ResearchState) -> ResearchState:
    if "gaps" not in state["tasks"]:
        return state
    if len(state["paper_names"]) < 2:
        return {**state, "errors": ["Gap analysis requires at least 2 papers."]}
    try:
        result = find_research_gaps(state["paper_names"])
        return {**state, "research_gaps": result, "completed_tasks": ["gaps"]}
    except Exception as e:
        return {**state, "errors": [f"Research gap analysis failed: {str(e)}"]}


# ---------------------------------------------------------------------------
# Node: RAG Q&A
# ---------------------------------------------------------------------------

def run_qa(state: ResearchState) -> ResearchState:
    if "qa" not in state["tasks"]:
        return state
    question = state.get("question")
    if not question:
        return state
    try:
        result = ask(question)
        return {**state, "qa_results": [result], "completed_tasks": ["qa"]}
    except Exception as e:
        return {**state, "errors": [f"Q&A failed: {str(e)}"]}


# ---------------------------------------------------------------------------
# Node: Report Generation
# ---------------------------------------------------------------------------

def run_report(state: ResearchState) -> ResearchState:
    if "report" not in state["tasks"]:
        return state
    try:
        result = generate_report(state)
        return {**state, "report": result, "completed_tasks": ["report"]}
    except Exception as e:
        return {**state, "errors": [f"Report generation failed: {str(e)}"]}


# ---------------------------------------------------------------------------
# Conditional edge: should we run report?
# Report runs last, only after other tasks have produced output.
# ---------------------------------------------------------------------------

def should_generate_report(state: ResearchState) -> str:
    if "report" in state["tasks"]:
        return "report_node"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("router",      route_tasks)
    graph.add_node("analyze",     run_paper_analysis)
    graph.add_node("compare",     run_comparison)
    graph.add_node("gaps",        run_research_gaps)
    graph.add_node("qa",          run_qa)
    graph.add_node("report_node", run_report)

    # Linear flow: router → analyze → compare → gaps → qa → conditional report
    graph.set_entry_point("router")
    graph.add_edge("router",  "analyze")
    graph.add_edge("analyze", "compare")
    graph.add_edge("compare", "gaps")
    graph.add_edge("gaps",    "qa")
    graph.add_conditional_edges("qa", should_generate_report, {"report_node": "report_node", END: END})
    graph.add_edge("report_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def run(
    paper_names: list[str],
    question: str = "",
    tasks: list[str] = None
) -> ResearchState:
    """
    Main entry point for the research workflow.

    Args:
        paper_names: List of paper names already indexed in FAISS.
        question:    Optional user question for RAG Q&A.
        tasks:       Explicit list of tasks to run. If empty, auto-inferred.
                     Options: "analyze", "compare", "gaps", "qa", "report"

    Returns:
        Final ResearchState with all results populated.
    """
    initial_state: ResearchState = {
        "paper_names": paper_names,
        "question": question,
        "tasks": tasks or [],
        "analyses": None,
        "comparison": None,
        "research_gaps": None,
        "qa_results": None,
        "report": None,
        "errors": [],
        "completed_tasks": []
    }
    workflow = get_workflow()
    return workflow.invoke(initial_state)
