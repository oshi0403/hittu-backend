import os
import re
import time
from typing import List, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

from httpx_config import get_http_client
from predict import generate_predicted_questions

from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from vector_store import ChromaVectorStore

# ============================================================
# 環境変数
# ============================================================
load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")


DEBUG_PRINT_PROMPT = False

# ============================================================
# 会話履歴（サーバメモリ）
# ============================================================
CHAT_HISTORY: List[dict] = []
MAX_HISTORY_TURNS = 1

def build_chat_history_text() -> str:
    if not CHAT_HISTORY:
        return ""
    turns = CHAT_HISTORY[-MAX_HISTORY_TURNS:]
    lines = []
    for t in turns:
        lines.append(f"ユーザー: {t.get('user', '')}")
        lines.append(f"ひっつー: {t.get('bot', '')}")
    return "\n".join(lines)

# ============================================================
# Chroma（PersistentClient 統一）
# ============================================================
vector_store = ChromaVectorStore(
    collection_name="hittu-knowledge",
    k=4
)
retriever = vector_store.get_retriever()

# ============================================================
# LLM
# ============================================================
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# ============================================================
# プロンプト
# ============================================================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
あなたは「ひっつー」という同志社大学の大学情報を教えてくれる羊のキャラクターです。
質問に対して、やさしく親しみやすい口調で、丁寧に答えてください。
難しい言葉は使わず、わかりやすい説明をしてください。
「こんにちは」など最初の挨拶はいりません。
情報がない場合は「わからない」と答えてください。

【質問（会話履歴込み）】
{question}

【参考情報】
{context}

【回答】
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

# ============================================================
# FastAPI
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です")
    OpenAI(api_key=api_key, http_client=get_http_client())
    print("FastAPI 起動完了")

# ============================================================
# 質問予測
# ============================================================
class PredictionRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    predictions: List[str]

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_questions(request: PredictionRequest):
    result = generate_predicted_questions(request.query)
    return PredictionResponse(predictions=result.get("questions", [])[:3])

# ============================================================
# チャット
# ============================================================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    user_message = request.message

    chat_history_text = build_chat_history_text()
    question = (
        f"【直前の会話履歴】\n{chat_history_text}\n\n【今回の質問】\n{user_message}"
        if chat_history_text else user_message
    )

    result = qa_chain.invoke({"query": question})
    bot_response = result["result"]

    CHAT_HISTORY.append({"user": user_message, "bot": bot_response})

    return ChatResponse(response=bot_response)

@app.get("/")
async def root():
    return {"status": "hittu backend running"}
