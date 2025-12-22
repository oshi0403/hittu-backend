# test_openai_cliの始め方

##　仮想環境の構築

`python -m venv {仮想環境名}`

## 仮想環境の立ち上げ

`{仮想環境名}\Scripts\activete`

## 必要なものをインストール
**研究室ではプロキシ対策しておかないと実行できない**

`pip install -r requirements.txt`
`pip install --proxy http://{ユーザー名}:{パスワード}@{プロキシサーバー}:{ポート番号} -r requirements.txt`

## 新しいやつ個別で追加したいとき

`pip install --proxy http://{ユーザー名}:{パスワード}@{プロキシサーバー}:{ポート番号} {インストールしたいもの}`

## FAST APIを動かすには
`uvicorn test_openai_cli:app --reload --port 8000`
`uvicorn chat:app --reload --port 8000`

### .envの作成

以下の二つの環境変数を設定してください。

```
    OPENAI_API_KEY = ""
    HTTPS_PROXY = ""
```

pip install --proxy http://cguh0053:Doushisha43@proxy.doshisha.ac.jp:8080 -r requirements.txt