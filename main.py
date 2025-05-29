from __future__ import annotations
import argparse, json, sys, ast
from pathlib import Path
from rag_pipeline.graph_builder import build_graph, visualize_graph
from rag_pipeline.graph_state import GraphState
from langchain.schema import Document
from typing import Any, List, Dict
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# final_state에서 HumanMessage / AIMessage 객체를 문자열로 변환
def convert_to_string(value):

    if isinstance(value, (HumanMessage, AIMessage)):
        return value.content
    elif isinstance(value, Document):
        return value.page_content
    elif isinstance(value, list):
        return [convert_to_string(item) for item in value]
    elif isinstance(value, dict):
        return {key: convert_to_string(val) for key, val in value.items()}
    else:
        # For basic types like str, int, float, bool, etc.
        return value


def run(
    query: str,
    pdf_path: str | None = None,
    query_type: str | None = None,
    hybrid_weights: List[float] | None = None,
    session_id: str | None = None,
):
    # The build_graph function returns a tuple of (graph, init_state)
    # Unpack it properly
    graph, builder_init_state = build_graph(
        Path(pdf_path) if pdf_path else None,
        query_type if query_type else None,
        hybrid_weights,
        session_id if session_id else None,
    )

    # Create our query-specific initial state
    init_state: GraphState = {"question": [query], "messages": [("user", query)]}

    # Merge with the initial state from build_graph
    init_state.update(builder_init_state)

    # Now invoke the graph with the combined initial state
    final_state = graph.invoke(init_state)

    # Convert final_state to JSON serializable format before printing or saving
    final_state_converted = convert_to_string(final_state)

    # graph visualization
    visualize_graph(graph)

    # for debugging
    print("\n===== 최종 답변 =====\n")
    print(final_state["answer"])
    print("\n===== 내부 상태 (디버그) =====\n")
    # Use the converted state for JSON serialization
    print(json.dumps(final_state_converted, indent=2, ensure_ascii=False))

    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # save json file - use the converted state
    output_data = {
        "answer": final_state["answer"],
        "debug_state": final_state_converted,
    }

    output_filename = f"./output/output_{uuid.uuid4()}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved {output_filename}! \n")

    # # for eval
    # serializable = {
    #     "question": [
    #         m.content if hasattr(m, "content") else str(m)
    #         for m in final_state.get("question", [])
    #     ],
    #     "explanation": final_state.get("explanation", ""),
    #     "context": final_state.get("context", ""),
    #     "answer": final_state.get("answer", ""),
    #     "scores": final_state.get("scores", ""),
    # }
    # return serializable


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="question")
    p.add_argument("--pdf", help="pdf file path", default=None)
    p.add_argument("--type", help="query type (hyde, summary)", default=None)
    p.add_argument(
        "--hybrid", help="hybrid retriever weights [float1,float2]", default=None
    )
    p.add_argument("--session_id", help="session ID for state management", default=None)
    args = p.parse_args()

    hybrid_weights = None
    if args.hybrid:
        try:
            hybrid_weights = ast.literal_eval(args.hybrid)
            if not isinstance(hybrid_weights, list) or len(hybrid_weights) != 2:
                print(
                    "Warning: hybrid weights should be a list of two floats [float1,float2]"
                )
                hybrid_weights = None
        except (SyntaxError, ValueError):
            print(
                "Warning: Could not parse hybrid weights. Format should be [float1,float2]"
            )

    print(f"Query received: {args.query} (Type: {type(args.query)})")
    if hybrid_weights:
        print(f"Using hybrid retrieval with weights: {hybrid_weights}")

    run(args.query, args.pdf, args.type, hybrid_weights, args.session_id)
