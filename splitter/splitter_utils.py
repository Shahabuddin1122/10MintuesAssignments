from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from vector_store.embedder import get_embedder


def paragraph_split(docs: List[Document], chunk_size=512) -> List[Document]:
    chunks = []
    for doc in docs:
        paras = doc.page_content.split("\n\n")
        buf = ""
        for para in paras:
            if len(buf) + len(para) <= chunk_size:
                buf += para + "\n\n"
            else:
                chunks.append(Document(page_content=buf.strip(), metadata=doc.metadata))
                buf = para + "\n\n"
        if buf:
            chunks.append(Document(page_content=buf.strip(), metadata=doc.metadata))
    return chunks


def semantic_split(docs: List[Document], threshold=0.85) -> List[Document]:
    embedder = get_embedder()
    chunks = []
    for doc in docs:
        paras = doc.page_content.split("\n\n")
        if len(paras) < 2:
            chunks.append(doc)
            continue
        buf = []
        for i in range(len(paras) - 1):
            buf.append(paras[i])
            v1 = embedder.embed_query("passage: " + paras[i])
            v2 = embedder.embed_query("passage: " + paras[i + 1])
            sim = cosine_similarity([v1], [v2])[0][0]
            if sim < threshold:
                chunks.append(Document(page_content="\n\n".join(buf).strip(), metadata=doc.metadata))
                buf = []
        buf.append(paras[-1])
        if buf:
            chunks.append(Document(page_content="\n\n".join(buf).strip(), metadata=doc.metadata))
    return chunks
