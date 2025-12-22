import httpx
import inspect

try:
    print(f"httpxのバージョン: {httpx.__version__}")
    print(f"読み込み元ファイル: {httpx.__file__}")
    print("-" * 20)
    print("httpx.Clientが受け付ける引数:")
    print(inspect.signature(httpx.Client))
except Exception as e:
    print(f"調査中にエラーが発生しました: {e}")