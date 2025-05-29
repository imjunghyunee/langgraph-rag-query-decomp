from __future__ import annotations
import json, requests
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.schema.messages import HumanMessage

# from pdf2image import convert_from_path
from rag_pipeline import config, utils
import torch.nn.functional as F
import logging
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# 기본 임베딩 모델 로드
model = SentenceTransformer(config.EMBED_MODEL_NAME, trust_remote_code=True)
model.to(device)

# LangChain용 임베딩 래퍼
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBED_MODEL_NAME,
    model_kwargs={"device": device, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

# Cross-Encoder Reranker
reranker = HuggingFaceCrossEncoder(model_name=config.RERANKER_NAME)


def load_parent_store(jsonl_path: Path) -> InMemoryStore:
    """JSONL 파일을 읽어 InMemoryStore에 적재"""
    with jsonl_path.open("r", encoding="utf-8") as f:
        docs = [
            Document(
                id=rec["id"],
                page_content=rec["page_content"],
                metadata=rec["metadata"],
            )
            for rec in map(json.loads, f)
        ]
    store = InMemoryStore()
    store.mset([(d.id, d) for d in docs])
    return store


def _rerank(query: str, docs: List[Document]) -> Tuple[List[Document], List[float]]:
    """Cross-Encoder 점수로 재정렬"""
    passages = [d.page_content for d in docs]
    pairs = [[query, passage] for passage in passages]
    scores = reranker.score(pairs)
    ranked = sorted(zip(docs, scores), key=lambda t: t[1], reverse=True)
    docs_sorted, scores_sorted = zip(*ranked)
    return list(docs_sorted), list(scores_sorted)


def retrieve_from_file_embedding(
    query: HumanMessage | str, pdf_path: Path, top_k: int = config.TOP_K
) -> List[Document]:
    docs = pdf_to_docs(pdf_path)

    if not docs:
        return "Failed to extract text from PDF!"
    query_text = query.content if hasattr(query, "content") else query

    texts = [d.page_content for d in docs]
    doc_vecs = model.encode(texts, normalize_embeddings=True)
    q_vecs = model.encode([query_text], normalize_embeddings=True)[0]

    cos_sim = util.cos_sim(q_vecs, doc_vecs)[0].float().cpu().numpy()

    best_idx = cos_sim.argsort()[-top_k:][::-1]
    best_docs = [docs[i] for i in best_idx]
    best_scores = cos_sim[best_idx]

    with open(config.SCORE_PATH, "w") as f:
        json.dump(best_scores.tolist(), f)

    return best_docs


def get_index_dimension(vectordb):
    """Extract the dimension of vectors in the FAISS index"""
    try:
        # Get dimension from the FAISS index
        if hasattr(vectordb, "index"):
            return vectordb.index.d
        return None
    except Exception as e:
        print(f"Error getting index dimension: {e}")
        return None


def vectordb_retrieve(query: HumanMessage | str) -> List[Document]:
    """Vector database retrieval with dimension check"""
    query_text = query.content if hasattr(query, "content") else query

    try:
        # Load the vector database
        vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        # Get query embedding
        query_emb = model.encode(
            query_text,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        # Check dimensions before searching
        index_dim = get_index_dimension(vectordb)
        query_dim = len(query_emb)

        if index_dim and index_dim != query_dim:
            print(
                f"⚠️ Dimension mismatch: FAISS index expects {index_dim} dimensions, "
                f"but query embedding has {query_dim} dimensions."
            )
            print(
                f"This likely means the embedding model used to create the index is "
                f"different from the current model ({config.EMBED_MODEL_NAME})."
            )

            # Fall back to default retrieval
            print("Attempting to use standard retrieval instead of vector search...")
            return vectordb.similarity_search(query_text, k=config.TOP_K)

        # Proceed with vector search if dimensions match
        sem = vectordb.similarity_search_by_vector(query_emb, k=config.TOP_K)

        # Save similarity scores
        doc_vecs = model.encode(
            [d.page_content for d in sem],
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        cos_sim = util.cos_sim(query_emb, doc_vecs)[0].float().cpu().numpy()

        # Make sure the output directory exists
        os.makedirs(os.path.dirname(config.SCORE_PATH), exist_ok=True)

        with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump(cos_sim.tolist(), f, ensure_ascii=False)

        # Reranking if enabled
        if config.RERANK:
            reranked_docs, scores = _rerank(query_text, sem)
            return reranked_docs

        return sem

    except Exception as e:
        print(f"Error during vector retrieval: {e}")
        # Create a fallback document to prevent pipeline failure
        fallback_doc = Document(
            page_content=(
                "I couldn't find relevant information in the knowledge base. "
                "Please try rephrasing your question."
            )
        )
        return [fallback_doc]


def vectordb_hybrid_retrieve(
    query: HumanMessage | str, weights: List[float] = [0.5, 0.5]
) -> List[Document]:
    """FAISS + BM25 하이브리드 검색"""
    query_text = query.content if hasattr(query, "content") else query

    try:
        # Load vector database
        vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        # Create directory for scores
        Path("score").mkdir(exist_ok=True)

        # Set up retrievers and use hybrid search
        faiss_retriever = vectordb.as_retriever(search_kwargs={"k": config.TOP_K})
        all_docs = list(vectordb.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = config.TOP_K

        # Use ensemble retriever which is more robust to dimension issues
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=weights,
        )

        # Get documents
        sem = ensemble_retriever.get_relevant_documents(query_text)

        # Rerank results
        reranked_docs, scores = _rerank(query_text, sem)

        # Make sure the output directory exists
        os.makedirs(os.path.dirname(config.SCORE_PATH), exist_ok=True)

        with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump([float(s) for s in scores], f, ensure_ascii=False)

        return reranked_docs

    except Exception as e:
        print(f"Error during hybrid retrieval: {e}")
        # Create a fallback document to prevent pipeline failure
        fallback_doc = Document(
            page_content=(
                "I couldn't find relevant information in the knowledge base. "
                "Please try rephrasing your question."
            )
        )
        return [fallback_doc]


def summary_retrieve(query: HumanMessage | str) -> Tuple[List[Document], str]:
    """FAISS ­+ LLM 설명 + 임베딩 검색"""
    query_text = query.content if hasattr(query, "content") else query

    vectordb = FAISS.load_local(
        config.CONTENT_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    messages = utils.build_payload_for_summary_generation(query_text)
    response = utils.call_openai_api(messages)
    query_explanation = response["choices"][0]["message"]["content"]

    query_emb = model.encode(
        query_explanation,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    sem = vectordb.similarity_search_by_vector(query_emb, k=config.TOP_K)

    doc_vecs = model.encode(
        [d.page_content for d in sem],
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    cos_sim = util.cos_sim(query_emb, doc_vecs)[0].float().cpu().numpy()

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(config.SCORE_PATH), exist_ok=True)

    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump(cos_sim.tolist(), f, ensure_ascii=False)

    # 조건부로 reranking 적용
    if config.RERANK:
        reranked_docs, scores = _rerank(query_text, sem)
        return reranked_docs, query_explanation

    return sem, query_explanation


def summary_hybrid_retrieve(
    query: HumanMessage | str, weights: List[float] = [0.5, 0.5]
) -> Tuple[List[Document], str]:
    """FAISS + BM25 하이브리드 검색 + LLM 설명"""
    query_text = query.content if hasattr(query, "content") else query

    vectordb = FAISS.load_local(
        config.CONTENT_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # LLM으로 질문 설명 생성
    messages = utils.build_payload_for_summary_generation(query_text)
    response = utils.call_openai_api(messages)
    query_explanation = response["choices"][0]["message"]["content"]

    # 하이브리드 검색 설정
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": config.TOP_K})

    all_docs = list(vectordb.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = config.TOP_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=weights,
    )

    sem = ensemble_retriever.get_relevant_documents(query_explanation)

    reranked_docs, scores = _rerank(query_explanation, sem)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(config.SCORE_PATH), exist_ok=True)

    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump([float(s) for s in scores], f, ensure_ascii=False)
    return reranked_docs, query_explanation


def hyde_retrieve(query: str):
    query_text = query.content if hasattr(query, "content") else query

    vectordb = FAISS.load_local(
        config.CONTENT_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    hydes: List[np.ndarray] = []
    hypo_docs: List[str] = []

    for _ in range(5):
        messages = utils.build_payload_for_hyde(query_text)
        response = utils.call_openai_api(messages)
        hypo_doc = response["choices"][0]["message"]["content"]
        hypo_docs.append(hypo_doc)

        embedding = model.encode(
            hypo_doc,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu()
        hydes.append(embedding)

    # mean_hyde = np.mean(np.stack(hydes, axis=0), axis=0)
    # norm = np.linalg.norm(mean_hyde)
    # if norm > 0:
    #     mean_hyde /= norm
    mean_hyde = torch.stack(hydes).mean(dim=0)
    mean_hyde = F.normalize(mean_hyde, p=2, dim=0)

    mean_hyde_np = mean_hyde.float().numpy()

    sem = vectordb.similarity_search_by_vector(mean_hyde_np, k=config.TOP_K)

    sem_vecs = model.encode(
        [d.page_content for d in sem],
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).cpu()

    dtype = torch.float32
    mean_hyde = mean_hyde.to(dtype)
    sem_vecs = sem_vecs.to(dtype)
    # sem_hyde_cos_sim = util.cos_sim(mean_hyde, sem_vecs)[0].cpu().numpy()
    sem_hyde_cos_sim = util.cos_sim(mean_hyde.unsqueeze(0), sem_vecs)[0].numpy()

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(config.SCORE_PATH), exist_ok=True)

    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump(sem_hyde_cos_sim.tolist(), f, ensure_ascii=False)

    if config.RERANK:
        query_text = query
        reranked_docs, scores = _rerank(query_text, sem)
        return reranked_docs, hypo_docs

    return sem, hypo_docs


def hyde_hybrid_retrieve(
    query: HumanMessage | str, weights: List[float] = [0.5, 0.5]
) -> Tuple[List[Document], str]:
    """HyDE + 하이브리드 검색"""
    query_text = query.content if hasattr(query, "content") else query

    vectordb = FAISS.load_local(
        config.CONTENT_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    hydes: List[np.ndarray] = []
    hypo_docs: List[str] = []

    # HyDE 문서 생성
    for _ in range(5):
        messages = utils.build_payload_for_hyde(query_text)
        response = utils.call_openai_api(messages)
        hypo_doc = response["choices"][0]["message"]["content"]
        hypo_docs.append(hypo_doc)

        embedding = model.encode(
            hypo_doc,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        hydes.append(embedding)

    # 가설 문서 임베딩의 평균 계산
    mean_hyde = np.mean(np.stack(hydes, axis=0), axis=0)
    norm = np.linalg.norm(mean_hyde)
    if norm > 0:
        mean_hyde /= norm

    # 하이브리드 검색 설정
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": config.TOP_K})

    all_docs = list(vectordb.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = config.TOP_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=weights,
    )

    # 가설 문서로 검색
    combined_hypo_doc = " ".join(hypo_docs)
    sem = ensemble_retriever.get_relevant_documents(combined_hypo_doc)

    # reranking
    reranked_docs, scores = _rerank(query_text, sem)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(config.SCORE_PATH), exist_ok=True)

    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump([float(s) for s in scores], f, ensure_ascii=False)

    return reranked_docs, hypo_docs
