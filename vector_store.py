from typing import Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from chroma_client import get_chroma_client

class ChromaVectorStore:
    def __init__(self, collection_name: str, k: int = 4):
        self.client = get_chroma_client()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.vectordb = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

        self.retriever = self.vectordb.as_retriever(
            search_kwargs={"k": k}
        )

    def get_retriever(self):
        return self.retriever
