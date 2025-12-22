import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# OpenAI APIキー取得
load_dotenv(dotenv_path=".env")


# jsonファイルとChromaDBのパス
JSON_PATH = "応答集.json"
CHROMA_PATH = "chroma_db"

def build_chroma():
    # Jsonファイルの読み込み
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Documentリストを作成
    docs = []
    for i, qa in enumerate(qa_data):
        q, a = qa.get("question", ""), qa.get("answer", "")
        if q and a:
            docs.append(Document(page_content=f"Q: {q}\nA: {a}"))

    # 埋め込み
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ChromaDB作成
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"ChromaDBを作成しました ({len(docs)}件) → {CHROMA_PATH}")

if __name__ == "__main__":
    build_chroma()
