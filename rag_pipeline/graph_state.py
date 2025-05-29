from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    question: Annotated[List[str], add_messages]
    explanation: Annotated[str, "Explanation"]
    context: Annotated[str, "Context"]
    filtered_context: Annotated[str, "Filtered_Context"]
    examples: Annotated[str, "Examples"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    # relevance: Annotated[str, "Relevance"]
    scores: Annotated[List[float], "Scores"]
    filtered_scores: Annotated[List[float], "Filtered Scores"]
    
    # Multi-turn chat state
    chat_history: Annotated[List[Dict[str, str]], "Chat History"]  # 전체 대화 기록
    session_id: Annotated[str, "Session ID"]  # 현재 채팅 세션 ID
    is_new_chat: Annotated[bool, "Is New Chat"]  # 새로운 채팅인지 여부
    compressed_context: Annotated[str, "Compressed Context"]  # 압축된 이전 대화 컨텍스트
    
    # IRCoT state
    ircot_paragraphs: Annotated[List[str], "IRCoT Retrieved Paragraphs"]
    ircot_sentences: Annotated[List[str], "IRCoT Reasoning Steps"]
    ircot_iterations: Annotated[int, "IRCoT Iteration Counter"]
    ircot_answer_found: Annotated[bool, "IRCoT Answer Found Flag"]
    ircot_latest_sentence: Annotated[str, "IRCoT Latest Reasoning Sentence"]
