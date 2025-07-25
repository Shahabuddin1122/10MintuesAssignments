from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


def recursive_split(docs: List[Document], chunk_size=1024, overlap=100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "।"]
    )
    return splitter.split_documents(docs)
