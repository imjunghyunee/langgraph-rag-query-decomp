from __future__ import annotations
import json
import requests
import os
from typing import List, Dict, Any, Tuple
from rag_pipeline.graph_state import GraphState
from rag_pipeline import retrievers, config, utils


# Initialize session management
def node_initialize_session(state: GraphState) -> GraphState:
    """새 채팅 세션인지 기존 세션인지 초기화"""
    is_new_chat = state.get("is_new_chat", True)

    if is_new_chat:
        # 새 채팅 세션 생성
        session_id = utils.generate_session_id()
        chat_history = []
        compressed_context = ""
    else:
        # 기존 채팅 세션 계속
        session_id = state.get("session_id", "")
        chat_history = utils.load_chat_history(session_id) if session_id else []
        compressed_context = utils.compress_chat_history(chat_history)

    return {
        "session_id": session_id,
        "chat_history": chat_history,
        "compressed_context": compressed_context,
    }


def node_retrieve_file_embedding(state: GraphState, pdf_path: str) -> GraphState:
    query = state["question"][-1]
    context = retrievers.retrieve_from_file_embedding(query, pdf_path)
    return {"context": context}


def node_retrieve(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 압축된 대화 컨텍스트가 존재하는 경우 질문에 통합
    compressed_context = state.get("compressed_context", "")
    if compressed_context and len(state.get("chat_history", [])) > 0:
        enhanced_query = (
            f"{query} (Previous conversation context: {compressed_context})"
        )
    else:
        enhanced_query = query

    context = retrievers.vectordb_retrieve(enhanced_query)
    return {"context": context}


def node_retrieve_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    # 압축된 대화 컨텍스트가 존재하는 경우 질문에 통합
    compressed_context = state.get("compressed_context", "")
    if compressed_context and len(state.get("chat_history", [])) > 0:
        enhanced_query = (
            f"{query} (Previous conversation context: {compressed_context})"
        )
    else:
        enhanced_query = query

    context = retrievers.vectordb_hybrid_retrieve(
        enhanced_query, weights=hybrid_weights
    )
    return {"context": context}


def node_retrieve_summary(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context, explanation = retrievers.summary_retrieve(query)
    return {"context": context, "explanation": explanation}


def node_retrieve_summary_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    context, explanation = retrievers.summary_hybrid_retrieve(
        query, weights=hybrid_weights
    )
    return {"context": context, "explanation": explanation}


def node_retrieve_hyde(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context, explanation = retrievers.hyde_retrieve(query)
    return {"context": context, "explanation": explanation}


def node_retrieve_hyde_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    context, explanation = retrievers.hyde_hybrid_retrieve(
        query, weights=hybrid_weights
    )
    return {"context": context, "explanation": explanation}


def node_relevance_check(state: GraphState) -> GraphState:
    with open(config.SCORE_PATH, "r", encoding="utf-8") as f:
        scores: List[float] = json.load(f)
        print(scores)

    context_docs = state["context"]  # List[Document]
    contents = [d.page_content for d in context_docs]

    filtered_scores: List[float] = []
    filtered_context: List[str] = []

    for i, score in enumerate(scores):
        if score >= config.SIM_THRESHOLD:
            filtered_scores.append(score)
            filtered_context.append(contents[i])

    return {
        "filtered_context": filtered_context,
        "scores": scores,
        "filtered_scores": filtered_scores,
    }


def node_llm_answer(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    chat_history = state.get("chat_history", [])

    context_docs = state["context"]
    contents = [d.page_content for d in context_docs]
    context_str = "\n\n---\n\n".join(contents)

    # 대화 기록이 있는 경우 다중 턴 페이로드 사용
    if chat_history:
        messages = utils.build_payload_for_multiturn_answer(
            query, context_str, chat_history
        )
    else:
        messages = utils.build_payload_for_llm_answer(query, context_str)

    # OpenAI API 호출로 변경
    res = utils.call_openai_api(messages)
    answer = res["choices"][0]["message"]["content"].strip()
    if not isinstance(answer, str):
        answer = str(answer)

    # 대화 기록 업데이트
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    # 대화 기록 저장
    session_id = state.get("session_id", "")
    if session_id:
        utils.save_chat_history(session_id, chat_history)

    return {
        "answer": answer,
        "messages": [("assistant", answer)],
        "chat_history": chat_history,
    }


def node_simple_or_not(state: GraphState) -> dict:
    """Determine if the question is simple or requires complex multi-hop reasoning."""
    query: str = state["question"][-1]

    messages = utils.build_payload_for_complexity_check(query)
    response = utils.call_openai_api(messages, max_tokens=10, temperature=0.1)
    decision = response["choices"][0]["message"]["content"].strip().lower()

    # Ensure response is valid
    if decision not in ["simple", "complex"]:
        print(f"Invalid complexity decision: '{decision}', defaulting to 'simple'")
        decision = "simple"

    print(f"Question complexity determined as: {decision}")
    # Return as a dictionary with a routing key
    return {"next": decision}


def node_initialize_ircot(state: GraphState) -> GraphState:
    """Initialize the IRCoT process with first retrieval."""
    query: str = state["question"][-1]

    # Initial retrieval based on the question
    context_docs = retrievers.vectordb_retrieve(query)
    paragraphs = [doc.page_content for doc in context_docs]

    return {
        "ircot_paragraphs": paragraphs,
        "ircot_sentences": [],
        "ircot_iterations": 0,
        "context": context_docs,  # Keep the original context field updated
    }


def node_generate_next_cot(state: GraphState) -> GraphState:
    """Generate the next sentence in the chain-of-thought reasoning."""
    query: str = state["question"][-1]
    paragraphs = state["ircot_paragraphs"]
    cot_sentences = state["ircot_sentences"]

    # Increment iteration counter
    iterations = state["ircot_iterations"] + 1

    messages = utils.build_payload_for_cot_generation(query, paragraphs, cot_sentences)
    response = utils.call_openai_api(messages, max_tokens=200, temperature=0.7)
    next_sentence = response["choices"][0]["message"]["content"].strip()

    # Add the new sentence to the CoT reasoning
    updated_sentences = cot_sentences + [next_sentence]

    # Check if this sentence contains the final answer
    answer_found = utils.check_answer_found(next_sentence)

    return {
        "ircot_sentences": updated_sentences,
        "ircot_iterations": iterations,
        "ircot_answer_found": answer_found,
        "ircot_latest_sentence": next_sentence,
    }


def node_retrieve_with_cot(state: GraphState) -> GraphState:
    """Retrieve more paragraphs based on the latest CoT sentence."""
    latest_sentence = state["ircot_latest_sentence"]

    # Extract a search query from the latest reasoning step
    search_query = utils.extract_query_from_cot(latest_sentence)

    # Retrieve new documents
    new_docs = retrievers.vectordb_retrieve(search_query)
    new_paragraphs = [doc.page_content for doc in new_docs]

    # Add new paragraphs to existing ones, removing duplicates
    existing_paragraphs = set(state["ircot_paragraphs"])
    for paragraph in new_paragraphs:
        existing_paragraphs.add(paragraph)

    # Update paragraphs in state
    updated_paragraphs = list(existing_paragraphs)

    return {
        "ircot_paragraphs": updated_paragraphs,
    }


def node_check_ircot_complete(state: GraphState) -> dict:
    """Determine if IRCoT process should continue or finish."""
    if state["ircot_answer_found"]:
        return {"next": "complete"}

    if state["ircot_iterations"] >= config.IRCOT_MAX_ITERATIONS:
        return {"next": "max_iterations_reached"}

    return {"next": "continue"}


def node_compile_ircot_answer(state: GraphState) -> GraphState:
    """Compile the final answer from the CoT reasoning."""
    query: str = state["question"][-1]
    cot_sentences = state["ircot_sentences"]

    final_answer = utils.compile_final_ircot_answer(query, cot_sentences)

    # Format the answer to include reasoning if needed
    if config.INCLUDE_IRCOT_REASONING:
        reasoning = "\n".join(cot_sentences)
        formatted_answer = f"{final_answer}\n\nReasoning:\n{reasoning}"
    else:
        formatted_answer = final_answer

    # Update chat history
    chat_history = state.get("chat_history", [])
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": formatted_answer})

    # Save chat history if session exists
    session_id = state.get("session_id", "")
    if session_id:
        utils.save_chat_history(session_id, chat_history)

    return {
        "answer": formatted_answer,
        "messages": [("assistant", formatted_answer)],
        "chat_history": chat_history,
    }
