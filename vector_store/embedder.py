from langchain.embeddings import HuggingFaceEmbeddings
from config.constants import EMBED_MODEL


def get_embedder(model_name: str = EMBED_MODEL):
    print("Embedding Model Downloading...")
    return HuggingFaceEmbeddings(model_name=model_name)
