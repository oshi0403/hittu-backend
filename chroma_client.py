import os
import chromadb

def get_chroma_client():
    """
    ChromaDB の PersistentClient を返す
    - ローカル: ./chroma_db
    - クラウド: /data/chroma（Azure Files）
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    return chromadb.PersistentClient(path=persist_dir)
