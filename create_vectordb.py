import json
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer

# 1. Markdown 파일 로드
file_path = "./data/ai.md"
loader = TextLoader(file_path, encoding="utf-8")  # 인코딩 문제 없도록 명시
documents = loader.load()

# docs에서 raw text 추출
raw_text = documents[0].page_content

# 2. Markdown 헤더 기준 Split
headers_to_split_on = [("#", "Header1"), ("##", "Header2")]
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)
split_docs = splitter.split_text(raw_text)


for doc in split_docs:
    print(f"{doc.page_content}")
    print(f"{doc.metadata}", end="\n=====================\n")

# 3. HuggingFace 임베딩 모델 정의
embedding_model = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3")

# 4. FAISS vector DB 생성
vectordb = FAISS.from_documents(split_docs, embedding_model)

# 5. 저장 (선택)
vectordb.save_local("./vectordb/faiss")

print("Successfully created and saved the FAISS vector database.")


# 5. Tokenizer 불러오기 (token count용)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 6. 토큰 수 계산 및 json 저장용 리스트 생성
token_info = []
token_counts = []

for idx, doc in enumerate(split_docs):
    tokens = tokenizer.encode(doc.page_content, add_special_tokens=False)
    token_count = len(tokens)
    token_counts.append(token_count)

    chunk_data = {
        "chunk_index": idx + 1,
        "token_count": token_count,
        "text": doc.page_content,
        "metadata": doc.metadata,
    }
    token_info.append(chunk_data)

# 7. 평균 토큰 수 출력
avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
print(f"\n📊 Average tokens per chunk: {avg_tokens:.2f}")

# 8. JSON 파일로 저장
json_save_path = "./vectordb/token_info.json"
with open(json_save_path, "w", encoding="utf-8") as f:
    json.dump(token_info, f, indent=2, ensure_ascii=False)

print(f"📝 Token info saved to {json_save_path}")
