import os
import re
import time
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# =============================
#  環境変数・ChromaDB 初期化
# =============================

# .env から OPENAI_API_KEY などを読み込む
load_dotenv()

# ChromaDB のパス（chat.py と同じ想定）
CHROMA_PATH = "chroma_db"

# Embedding と DB の読み込み
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# LLM モデル設定（chat.py と同じ）
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# =============================
#  質問予測用プロンプト
#  ※ 現在 Dify で使っているプロンプトをベースにしている
# =============================

predict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
##指示
あなたは同志社大学のお問い合わせチャットボットのための質問候補を考えるAIである。
以下のユーザ入力をもとにした知識検索結果（想定問答集の抜粋）を読み、
ユーザが実際にチャットボットに投げそうな質問文の候補を3つ日本語で出力せよ。

##ユーザ入力
{question}

##知識検索結果（想定問答集の該当部分）
{context}

##出力ルール
- 箇条書きで3つの質問文だけを出力する。
- 3つの質問は、ユーザ入力および検索結果から自然に派生した具体的な質問にする。
- 同志社大学に関する質問に限定する。
"""
)

# RetrievalQA チェーン（質問予測用）
predict_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": predict_prompt},
)

# =============================
#  テキストから質問文だけを抜き出す関数
# =============================

def extract_questions_from_text(text: str, max_count: int = 3) -> List[str]:
    """
    LLM から返ってきたテキストから「質問文」だけを取り出す。
    箇条書き・番号付きリストなどを軽くパースして整形する。
    """
    questions: List[str] = []

    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue

        cleaned = line.strip()

        # 先頭の記号や番号（- ・ * 1. など）を削除
        cleaned = re.sub(r"^[-・*]\s*", "", cleaned)   # 箇条書き
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)    # 1. など番号付き

        # 全角カギカッコを外すなど軽く整形
        cleaned = cleaned.strip().strip("「」").strip()

        if cleaned:
            questions.append(cleaned)

        if len(questions) >= max_count:
            break

    return questions

# =============================
#  質問予測のメイン処理
# =============================

def generate_predicted_questions(user_input: str):
    """
    1 回のユーザー入力に対して：
      - ChromaDB + LLM で質問候補を 3 つ生成
      - 所要時間（ms）を計測
      - 生テキストと抽出後の質問文をコンソールに表示
    """
    print("=" * 60)
    print(f"ユーザー入力: {user_input}")

    # 時間計測開始
    start_time = time.perf_counter()

    # RetrievalQA チェーンを実行
    # RetrievalQA は "query" を受け取り、内部で "question" にマッピングする
    result = predict_chain.invoke({"query": user_input})
    raw_text = result["result"]

    # 時間計測終了
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    # 質問文だけ抽出
    questions = extract_questions_from_text(raw_text, max_count=3)

    print(f"\n[処理時間] {elapsed_ms:.2f} ms")
    print("\n[LLM 生テキスト]")
    print(raw_text)
    print("\n[抽出された質問候補]")
    if not questions:
        print("（質問を抽出できませんでした）")
    else:
        for i, q in enumerate(questions, start=1):
            print(f"{i}. {q}")
    print("=" * 60)
    print()
    
    return {
        "questions": questions,         # 抽出された質問候補リスト
        "raw_text": raw_text,           # LLM の生出力（必要なら）
    }

# =============================
#  エントリポイント（コンソール用）
# =============================

if __name__ == "__main__":
    print("=== 質問予測テスト（ChromaDB + LLM） ===")
    print("空行のみで Enter を押すと終了します。\n")

    while True:
        try:
            user_input = input("ユーザー入力 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            print("終了します。")
            break

        generate_predicted_questions(user_input)
