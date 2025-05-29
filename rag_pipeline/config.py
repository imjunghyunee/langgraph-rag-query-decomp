from pathlib import Path
import os

# import openai


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val == "True"


# ----- 벡터 DB 및 임베딩 설정 -----
EMBED_MODEL_NAME: str = "jinaai/jina-embeddings-v3"
RERANKER_NAME: str = "BAAI/bge-reranker-v2-m3"
CONTENT_DB_PATH: Path = Path("./vectordb/faiss")

# ----- Local LLM Settings -----
REMOTE_LLM_URL: str = "http://localhost:8000/v1/chat/completions"
REMOTE_LLM_MODEL: str = "agent:llama-4-scout-17B-16E-instruct"

# ----- OpenAI API Settings (commented out) -----
# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL: str = "gpt-4o"

# Configure OpenAI client (commented out)
# openai.api_key = OPENAI_API_KEY

# ----- 검색 파라미터 -----
TOP_K: int = int(os.getenv("TOP_K", 3))
SIM_THRESHOLD: float = float(os.getenv("SIM_THRESHOLD", 0.70))
RERANK: bool = _get_bool("RERANK", False)

# ----- Multi-turn 설정 -----
MAX_HISTORY_LENGTH: int = int(
    os.getenv("MAX_HISTORY_LENGTH", 5)
)  # 대화 히스토리 최대 길이
COMPRESS_HISTORY: bool = _get_bool("COMPRESS_HISTORY", True)  # 히스토리 압축 여부
HISTORY_WEIGHT: float = float(os.getenv("HISTORY_WEIGHT", 0.3))  # 히스토리 가중치

OUTPUT_DIR: str = "./output"
SCORE_PATH: str = "./output/similarity_score.json"

# ----- IRCoT Settings -----
IRCOT_MAX_ITERATIONS: int = int(os.getenv("IRCOT_MAX_ITERATIONS", 5))
INCLUDE_IRCOT_REASONING: bool = _get_bool("INCLUDE_IRCOT_REASONING", False)
