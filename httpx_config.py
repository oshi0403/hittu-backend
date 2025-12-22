#!/usr/bin/env python3
"""
HTTPXクライアント設定モジュール

プロキシ設定を環境変数から読み込み、
設定済みのhttpx.Clientインスタンスを生成します。
"""

import os
import httpx

def get_http_client() -> httpx.Client:
    """
    環境変数に基づいてプロキシが設定されたhttpx.Clientを返します。

    環境変数 'HTTPS_PROXY' が設定されている場合、その値がプロキシとして使用されます。
    設定されていない場合は、プロキシなしの通常のクライアントが返されます。

    Returns:
        httpx.Client: 設定済みのHTTPXクライアントインスタンス
    """
    # 標準的な環境変数である 'HTTPS_PROXY' からプロキシURLを取得
    proxy_url = os.getenv('HTTPS_PROXY')

    if proxy_url:
        # プロキシが設定されている場合
        proxies = {
            "http://": proxy_url,
            "https://": proxy_url,
        }
        print(f"プロキシサーバー ({proxy_url}) を経由して接続します。")
        # タイムアウトを20秒に設定したプロキシクライアントを返す
        return httpx.Client(proxy=proxy_url, timeout=20.0)
    else:
        # プロキシが設定されていない場合は、通常のクライアントを返す
        return httpx.Client(timeout=20.0)
def get_async_http_client() -> httpx.AsyncClient:
    """
    非同期用のHTTPクライアントを返します。
    async def関数内でのAPI呼び出しに使用します。
    """
    proxy_url = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    
    if proxy_url:
        # 非同期クライアントも同様に proxy="http://..." で設定
        return httpx.AsyncClient(proxy=proxy_url)
    else:
        return httpx.AsyncClient()