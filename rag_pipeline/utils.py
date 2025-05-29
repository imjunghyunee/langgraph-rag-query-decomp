from __future__ import annotations
from typing import List, Dict, Any
from langchain.schema import Document
from rag_pipeline import config
import os
import base64
import cv2
from pathlib import Path
from pdf2image import convert_from_path
import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter
import uuid
import json
import time
import openai


def encode_image(image_path, image_size=(837, 1012)):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=image_size, interpolation=cv2.INTER_CUBIC)

    except Exception as _:
        print(f"Error when encoding image: {image_path}")
        return ""

    _, ext = os.path.splitext(image_path)
    _, encoded_image = cv2.imencode(ext, img)
    encoded_string = base64.b64encode(encoded_image).decode("utf-8")
    return encoded_string


def get_page_number(filename):
    if filename.startswith("page_") and filename.endswith(".png"):
        try:
            return int(filename[5:-4])  # "page_X.png" -> X 추출
        except ValueError:
            return float("inf")
    return float("inf")


def pdf_to_docs(file_path: Path) -> List[Document]:
    temp_img_dir = Path("./data/temp_img")
    os.makedirs(temp_img_dir, exist_ok=True)

    pdf_name = file_path.stem  # 확장자 없이 파일명 추출
    images = convert_from_path(str(file_path))

    # 각 페이지를 png로 저장
    for idx, image in enumerate(images):
        output_path = temp_img_dir / f"page_{idx+1}.png"
        image.save(output_path, "PNG")

    print(f"PDF successfully converted: {pdf_name} -> {len(images)} pages")

    # Text Extraction
    all_texts = []

    for filename in sorted(os.listdir(temp_img_dir), key=get_page_number):
        img_path = os.path.join(temp_img_dir, filename)
        if not os.path.isfile(img_path):
            continue

        # Encode image
        image_url = encode_image(img_path, image_size=(837, 1012))
        _, image_ext = os.path.splitext(filename)
        image_ext = image_ext.lstrip(".")  # e.g. png, jpg
        image_url = f"data:image/{image_ext};base64,{image_url}"

        messages = [
            {
                "role": "user",
                "content": f"[Question]:{query_text}, [Context]:{image_url}",
            }
        ]

        response = call_openai_api(messages)
        text = response["choices"][0]["message"]["content"]

        print(f"Successfully extracted text from: {filename}\n")
        all_texts.append(f"{text.strip()}\n")

    combined_texts = "\n".join(all_texts)

    # Split extracted text
    headers_to_split_on = [("#", "Header1"), ...]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split=headers_to_split_on, strip_headers=False
    )

    split_contents = md_splitter.split_text(combined_texts)
    print("\nSuccesfully split text!")

    return split_contents


def request_llm(messages, max_tokens=5000, temperature=0.65, top_p=0.95):
    """LLM API를 호출하는 통합 함수"""
    try:
        # Use requests to call local LLM instead of OpenAI client
        payload = {
            "model": config.REMOTE_LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "n": 1,
        }

        response = requests.post(config.REMOTE_LLM_URL, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except Exception as e:
        print(f"LLM API 호출 중 오류 발생: {e}")
        return {
            "choices": [
                {
                    "message": {
                        "content": f"Error: {str(e)}",
                    }
                }
            ]
        }


def build_payload_for_summary_generation(query_text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that generates summaries based on the provided question.""",
        },
        {"role": "user", "content": f"[Question]:{query_text}"},
    ]
    return messages


def build_payload_for_hyde(query_text: str) -> dict:
    """OpenAI API용 페이로드 함수로 대체"""
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that generates answers based on the provided question and context.""",
        },
        {"role": "user", "content": f"[Question]:{query_text}"},
    ]
    return messages


def build_payload_for_llm_answer(query_text: str, context: str) -> dict:
    """OpenAI API용 페이로드 함수로 대체"""
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that generates answers based on the provided question and context.""",
        },
        {
            "role": "user",
            "content": f"[Question]:{query_text}, [Context]:{context}",
        },
    ]
    return messages


# Multi-turn chat related functions
def generate_session_id() -> str:
    """새로운 채팅 세션 ID 생성"""
    return str(uuid.uuid4())


def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable format."""
    from langchain.schema.messages import HumanMessage, AIMessage
    from langchain.schema import Document

    if isinstance(obj, (HumanMessage, AIMessage)):
        return {"role": obj.type, "content": obj.content}
    elif isinstance(obj, Document):
        return {"page_content": obj.page_content, "metadata": obj.metadata}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(val) for key, val in obj.items()}
    else:
        # For basic types like str, int, float, bool, etc.
        return obj


def save_chat_history(session_id: str, chat_history: List[Dict[str, str]]) -> None:
    """채팅 기록 저장"""
    history_dir = os.path.join(config.OUTPUT_DIR, "chat_history")
    os.makedirs(history_dir, exist_ok=True)

    file_path = os.path.join(history_dir, f"{session_id}.json")

    # Convert chat history to serializable format
    serializable_history = convert_to_serializable(chat_history)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, ensure_ascii=False, indent=2)


def load_chat_history(session_id: str) -> List[Dict[str, str]]:
    """저장된 채팅 기록 불러오기"""
    file_path = os.path.join(config.OUTPUT_DIR, "chat_history", f"{session_id}.json")
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compress_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """채팅 기록 압축하여 컨텍스트 생성

    최근 N개 대화만 유지하고, LLM을 통해 중요 정보를 압축
    """
    if not chat_history:
        return ""

    # 최근 N개 대화만 사용
    recent_history = chat_history[-config.MAX_HISTORY_LENGTH :]

    # 압축이 필요 없을 만큼 적은 경우
    if len(recent_history) < 3:
        formatted_history = []
        for msg in recent_history:
            role = msg["role"]
            content = msg["content"]
            formatted_history.append(f"{role}: {content}")
        return "\n".join(formatted_history)

    # LLM을 사용한 히스토리 압축 (대화가 많은 경우)
    if config.COMPRESS_HISTORY:
        history_text = ""
        for msg in recent_history:
            role = msg["role"]
            content = msg["content"]
            history_text += f"{role}: {content}\n"

        payload = {
            "model": config.REMOTE_LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that summarizes conversation history. 
                    Extract and summarize key information from the conversation while preserving important context.""",
                },
                {
                    "role": "user",
                    "content": f"Summarize this conversation:\n{history_text}",
                },
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
        }

        response = requests.post(config.REMOTE_LLM_URL, json=payload).json()
        return response["choices"][0]["message"]["content"]

    # 압축없이 최근 대화 반환
    formatted_history = []
    for msg in recent_history:
        role = msg["role"]
        content = msg["content"]
        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history)


def build_payload_for_multiturn_answer(
    query_text: str, context: str, chat_history: List[Dict[str, str]]
) -> dict:
    """다중 턴 대화를 위한 페이로드 생성 (OpenAI API 용)"""
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that generates answers based on the provided question, 
            conversation history, and context. Maintain consistency with previous responses and build upon the conversation.""",
        }
    ]

    # 최근 몇 개의 대화 기록을 추가
    recent_history = chat_history[-min(len(chat_history), 3) :]
    for msg in recent_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 현재 질문과 컨텍스트 추가
    messages.append(
        {"role": "user", "content": f"[Question]:{query_text}, [Context]:{context}"}
    )

    return messages


def build_payload_for_complexity_check(query_text: str) -> dict:
    """Determine if the question requires simple retrieval or complex multi-hop reasoning."""
    messages = [
        {
            "role": "system",
            "content": """Determine if the user's question requires simple retrieval or complex multi-hop reasoning.
            
            Simple questions can be answered with a single retrieval operation and direct generation of an answer.
            Complex questions require multiple reasoning steps and retrievals to reach a final answer.
            
            Reply with ONLY one of two options:
            - "simple": If the question can be answered directly with a single retrieval.
            - "complex": If the question requires multi-hop reasoning and multiple retrievals.
            
            Respond with only the word "simple" or "complex".
            """,
        },
        {"role": "user", "content": query_text},
    ]
    return messages


def build_payload_for_cot_generation(
    query_text: str, paragraphs: List[str], cot_sentences: List[str]
) -> dict:
    """Generate the next sentence in the chain-of-thought reasoning."""
    # Format the context and previous reasoning
    context = "\n\n".join(paragraphs)
    previous_reasoning = (
        "\n".join(cot_sentences) if cot_sentences else "No previous reasoning yet."
    )

    messages = [
        {
            "role": "system",
            "content": """You are generating the next step in a chain-of-thought reasoning process to answer a complex question.
            
            Follow these guidelines:
            1. Analyze the question, available context, and previous reasoning.
            2. Generate only ONE logical next sentence that advances the reasoning.
            3. If you can determine the final answer, include "The final answer is:" followed by the answer.
            4. Focus on extracting insights from the context and connecting information.
            5. Keep your response brief and targeted - just one clear reasoning step.
            """,
        },
        {
            "role": "user",
            "content": f"Question: {query_text}\n\nAvailable Context:\n{context}\n\nPrevious Reasoning Steps:\n{previous_reasoning}\n\nNext reasoning step:",
        },
    ]
    return messages


def check_answer_found(cot_sentence: str) -> bool:
    """Check if the final answer has been found in the CoT reasoning."""
    return "the final answer is:" in cot_sentence.lower()


def extract_query_from_cot(cot_sentence: str) -> str:
    """Extract a search query from the latest CoT sentence for next retrieval."""
    messages = [
        {
            "role": "system",
            "content": """Extract a concise search query from the reasoning step that can be used to retrieve more relevant information.
            Create a search query that will help answer the implicit question in this reasoning step.
            Keep the query focused, specific, and under 15 words.
            Respond with ONLY the search query, no explanation or additional text.
            """,
        },
        {"role": "user", "content": f"Reasoning step: {cot_sentence}"},
    ]

    response = call_openai_api(messages, max_tokens=50, temperature=0.3)
    return response["choices"][0]["message"]["content"].strip()


def compile_final_ircot_answer(question: str, cot_sentences: List[str]) -> str:
    """Compile the final answer from the CoT reasoning process."""
    reasoning = "\n".join(cot_sentences)

    messages = [
        {
            "role": "system",
            "content": """Generate a final, complete answer to the original question based on the chain-of-thought reasoning provided.
            Your answer should be comprehensive, well-structured, and directly address the question.
            Incorporate the key insights from the reasoning but present them in a coherent, reader-friendly way.
            """,
        },
        {
            "role": "user",
            "content": f"Original Question: {question}\n\nReasoning Process:\n{reasoning}\n\nPlease provide the final answer:",
        },
    ]

    response = call_openai_api(messages, max_tokens=500, temperature=0.5)
    return response["choices"][0]["message"]["content"].strip()
