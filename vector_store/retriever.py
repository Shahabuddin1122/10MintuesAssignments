from typing import List
from vector_store.embedder import get_embedder
from langchain.vectorstores import FAISS


def query_vectorstore(query: str, path: str, top_k: int = 3) -> List[str]:
    embedder = get_embedder()
    db = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
    results = db.similarity_search("query: " + query, k=top_k)
    return [doc.page_content for doc in results]
