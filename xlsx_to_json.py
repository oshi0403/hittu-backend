import argparse
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser(description="ExcelをJSONに変換")
    parser.add_argument("--input", default=".xlsxのパス")
    parser.add_argument("--output", default=".jsonのパス")
    args = parser.parse_args()

    # エクセルファイル読み込み
    df = pd.read_excel(args.input)

    # 列抽出（質問文と回答文の列を抽出）
    qa_list = []
    for _, row in df.iterrows():
        q = str(row.get("質問文", "")).strip()
        a = str(row.get("回答文", "")).strip()
        if q and a:
            qa_list.append({"question": q, "answer": a})

    # jsonファイルを保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print(f" {len(qa_list)}件を変換完了 → {args.output}")

if __name__ == "__main__":
    main()
