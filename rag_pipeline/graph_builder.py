from __future__ import annotations
from pathlib import Path
from langgraph.graph import StateGraph, END
from rag_pipeline.graph_state import GraphState
from rag_pipeline import nodes
from typing import List, Optional
import nest_asyncio
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles


def build_graph(
    pdf_path: Path | None = None,
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
    session_id: Optional[str] = None,
    is_new_chat: bool = True,
):
    g = StateGraph(GraphState)

    init_state = {
        "is_new_chat": is_new_chat,
    }

    if session_id:
        init_state["session_id"] = session_id
        init_state["is_new_chat"] = False

    if hybrid_weights:
        init_state["hybrid_weights"] = hybrid_weights

    # Nodes
    g.add_node("initialize_session", nodes.node_initialize_session)

    # Add complexity determination node
    g.add_node("determine_complexity", nodes.node_simple_or_not)

    # Standard RAG nodes
    if pdf_path:
        g.add_node(
            "retrieve", lambda s: nodes.node_retrieve_file_embedding(s, pdf_path)
        )
    elif retrieval_type == "hyde" and hybrid_weights:
        g.add_node("retrieve", nodes.node_retrieve_hyde_hybrid)
    elif retrieval_type == "hyde":
        g.add_node("retrieve", nodes.node_retrieve_hyde)
    elif retrieval_type == "summary" and hybrid_weights:
        g.add_node("retrieve", nodes.node_retrieve_summary_hybrid)
    elif retrieval_type == "summary":
        g.add_node("retrieve", nodes.node_retrieve_summary)
    elif hybrid_weights:
        g.add_node("retrieve", nodes.node_retrieve_hybrid)
    else:
        g.add_node("retrieve", nodes.node_retrieve)

    g.add_node("relevance_check", nodes.node_relevance_check)
    g.add_node("llm_answer", nodes.node_llm_answer)

    # IRCoT nodes
    g.add_node("initialize_ircot", nodes.node_initialize_ircot)
    g.add_node("generate_next_cot", nodes.node_generate_next_cot)
    g.add_node("retrieve_with_cot", nodes.node_retrieve_with_cot)
    g.add_node("check_ircot_complete", nodes.node_check_ircot_complete)
    g.add_node("compile_ircot_answer", nodes.node_compile_ircot_answer)

    # Edges for initialization
    g.set_entry_point("initialize_session")
    g.add_edge("initialize_session", "determine_complexity")

    # Standard RAG flow
    g.add_conditional_edges(
        "determine_complexity",
        lambda x: x["next"],  # Routing function that extracts decision from the state
        {"simple": "retrieve", "complex": "initialize_ircot"},
    )

    # Standard RAG path
    g.add_edge("retrieve", "relevance_check")
    g.add_edge("relevance_check", "llm_answer")
    g.add_edge("llm_answer", END)

    # IRCoT path
    g.add_edge("initialize_ircot", "generate_next_cot")
    g.add_edge("generate_next_cot", "check_ircot_complete")

    # Conditional edges for IRCoT process
    g.add_conditional_edges(
        "check_ircot_complete",
        lambda x: x["next"],  # Routing function that extracts decision from the state
        {
            "continue": "retrieve_with_cot",
            "complete": "compile_ircot_answer",
            "max_iterations_reached": "compile_ircot_answer",
        },
    )

    g.add_edge("retrieve_with_cot", "generate_next_cot")
    g.add_edge("compile_ircot_answer", END)

    return g.compile(), init_state


# graph visualization
def visualize_graph(
    graph: StateGraph, output_path: Path = Path("./workflow_graph.png")
):
    nest_asyncio.apply()

    display(
        Image(
            graph.get_graph().draw_mermaid_png(
                curve_style=CurveStyle.LINEAR,
                node_colors=NodeStyles(
                    first="#ffdfba", last="#baffc9", default="#fad7de"
                ),
                wrap_label_n_words=9,
                output_file_path=None,
                draw_method=MermaidDrawMethod.PYPPETEER,
                background_color="white",
                padding=10,
            )
        )
    )
