import os
import sys
import re
import httpx
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from httpx_config import get_http_client, get_async_http_client
from pydantic import BaseModel
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# .envファイルから環境変数を読み込む
load_dotenv()

# ChromaDB のパス
CHROMA_PATH = "chroma_db"

# EmbeddingとDB読み込み
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# LLMモデル設定
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# =============================
# 会話履歴（簡易版：サーバ起動中だけ保持）
# ※本番は session_id + DB/Redis など推奨
# =============================
CHAT_HISTORY = []  # [{"user": "...", "bot": "..."}]
MAX_HISTORY_TURNS = 3  # 直近何往復分を使うか


def build_chat_history_text() -> str:
    """
    CHAT_HISTORY からプロンプト用の会話履歴テキストを生成
    """
    lines = []
    for turn in CHAT_HISTORY[-MAX_HISTORY_TURNS:]:
        lines.append(f"ユーザー: {turn.get('user', '')}")
        lines.append(f"ひっつー: {turn.get('bot', '')}")
    return "\n".join(lines)


# プロンプト設定（ひっつー用）
prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
# あなたは「ひっつー」という同志社大学の大学情報を教えてくれる羊のキャラクターです。
# 質問に対して、やさしく親しみやすい口調で、丁寧に答えてください。
# 難しい言葉は使わず、わかりやすい説明をしてください。
# 情報がない場合は「わからない」と答えてください。
# 質問が曖昧な場合は、具体的な情報を求めてください。
# 会話履歴と質問に関連性があれば，会話履歴からわかる文脈も考慮して回答してください．

【会話履歴】
{chat_history}

【質問】
{question}

【ひっつーが参考にする情報】
{context}

【ひっつーの回答】
""",
)

# RetrievalQAチェーン
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

# Dify API関連の設定
DIFY_API_KEY = os.getenv("DIFY_API_KEY")
DIFY_WORKFLOW_URL = os.getenv("DIFY_WORKFLOW_URL")

# OpenAIクライアントを保持するグローバル変数（※今は応答では使っていないが初期化処理はそのまま）
openai_client: OpenAI = None

# FastAPI アプリ本体
app = FastAPI()

# CORS設定（必要に応じて調整）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # フロントのoriginに合わせて必要なら絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    アプリ起動時にOpenAIクライアントを初期化するイベントハンドラ
    （現在は /api/chat では使わないが、将来の拡張に備えて残している）
    """
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 'OPENAI_API_KEY' が設定されていません。")
    try:
        http_client = get_http_client()
        openai_client = OpenAI(api_key=api_key, http_client=http_client)
        print("FastAPI: OpenAIクライアントの準備が完了しました (プロキシ経由)。")
    except Exception as e:
        print(f"FastAPI: クライアントの初期化中にエラーが発生しました: {e}")
        raise RuntimeError(f"OpenAIクライアントの初期化に失敗しました: {e}")


# =============================
#  質問集のデータ（デバッグ用として）
# =============================
FAQ_LIST = [
    "チャットボットの利用方法は？",
    "アカウント登録は必要ですか？",
    "料金体系について教えてください。",
    "セキュリティ対策は？",
    "対応しているブラウザは？",
    "AIの応答が不自然です。",
    "メッセージ履歴を削除するにはどうすればいいですか？",
    "質問の予測機能はどのように動作しますか？",
    "問い合わせの対応時間は？",
    "技術的なサポートは受けられますか？",
]


# 質問予測用のリクエストモデル
class PredictionRequest(BaseModel):
    query: str


# 質問予測用のレスポンスモデル
class PredictionResponse(BaseModel):
    predictions: List[str]


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_questions(request: PredictionRequest):
    """
    ユーザーの入力に基づいて、Difyワークフローを呼び出し、
    関連性の高い質問を予測して返すエンドポイント。
    """
    if not DIFY_API_KEY or not DIFY_WORKFLOW_URL:
        raise HTTPException(
            status_code=500,
            detail="DifyのAPIキーまたはワークフローURLが環境変数に設定されていません。",
        )

    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": {"input": request.query},
        "response_mode": "blocking",
        "user": "unique_user_id",
    }

    print(f"Difyに送信するペイロード: {payload}")

    try:
        # ===== ここで計測開始（バックエンド内での「AI呼び出し処理時間」） =====
        start_time = time.perf_counter()

        async with get_async_http_client() as client:
            response = await client.post(
                DIFY_WORKFLOW_URL, headers=headers, json=payload, timeout=60.0
            )
            response.raise_for_status()

        dify_response = response.json()

        # ===== 計測終了 =====
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000  # ミリ秒に変換

        print(f"Difyワークフロー呼び出し＋応答処理時間: {elapsed_ms:.2f} ms")
        print(f"Difyからの完全な応答: {dify_response}")

        predicted_text = (
            dify_response.get("data", {}).get("outputs", {}).get("predicted_text", "")
        )

        if not predicted_text:
            print("Difyからの応答に予測結果が含まれていませんでした。")
            return PredictionResponse(predictions=[])

        # Dify から返ってきたテキストを前後トリム
        predicted_text = predicted_text.strip()

        predictions: List[str] = []

        # 行ごとに分割して箇条書き・番号付きリストを吸い出す
        for line in predicted_text.splitlines():
            # 行末の余分な空白（Markdown の "  " など）を削除
            line = line.rstrip()

            # 完全な空行はスキップ
            if not line.strip():
                continue

            # 先頭の箇条書き記号や番号を削除
            # 例: "- 質問…", "・質問…", "* 質問…", "1. 質問…" など
            cleaned = line.strip()
            cleaned = re.sub(r"^[-・*]\s*", "", cleaned)  # 箇条書き記号
            cleaned = re.sub(r"^\d+\.\s*", "", cleaned)  # "1. " など番号付き
            cleaned = cleaned.strip().strip("「」").strip()

            if cleaned:
                predictions.append(cleaned)

        # 念のため3件に制限
        predictions = predictions[:3]

        print(f"抽出された予測リスト: {predictions}")
        return PredictionResponse(predictions=predictions)

    except Exception as e:
        print(f"Difyワークフローの呼び出し中にエラーが発生しました: {e}")
        raise HTTPException(
            status_code=500, detail="質問の予測中にサーバーエラーが発生しました。"
        )


# =============================
#  チャットボット応答用モデル
# =============================
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    ユーザーのメッセージに対してチャットボットが応答を返すエンドポイント
    （ChromaDB + ひっつーで回答）
    """
    user_message = request.message

    try:
        # 会話履歴をプロンプトに挿入
        chat_history_text = build_chat_history_text()
        
            # ===== Retriever から context を取得（QAと同じもの）=====
        docs = retriever.get_relevant_documents(user_message)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        # ===== 実際にLLMに投げられるプロンプトを生成 =====
        final_prompt = prompt.format(
            context=context_text,
            question=user_message,
            chat_history=chat_history_text,
        )

        # ===== プロンプトをコンソールに表示 =====
        print("\n================= LLM に送信するプロンプト =================")
        print(final_prompt)
        print("===========================================================\n")

        # RetrievalQAチェーンを使って回答を生成
        start_time = time.monotonic()

        # query は RetrievalQA が受け取る入力キー
        # chat_history は PromptTemplate の追加変数として渡す
        result = qa_chain.invoke({"query": user_message, "chat_history": chat_history_text})
        bot_response_text = result["result"]

        end_time = time.monotonic()
        elapsed_ms = (end_time - start_time) * 1000

        print(f"ひっつー応答生成時間（RetrievalQAチェーン全体）: {elapsed_ms:.2f} ms")
        print(f"ユーザー入力: {user_message}")
        print(f"ひっつーの回答: {bot_response_text}")

        # ===== 今回のQ&Aを履歴に保存 =====
        CHAT_HISTORY.append({"user": user_message, "bot": bot_response_text})

        return ChatResponse(response=bot_response_text)

    except Exception as e:
        print(f"チャットAPI呼び出し中にエラーが発生しました: {e}")
        raise HTTPException(
            status_code=500, detail="メッセージの送信中にエラーが発生しました。"
        )


@app.get("/")
async def read_root():
    return {"message": "FastAPI backend with ChromaDB (ひっつー) is running"}
