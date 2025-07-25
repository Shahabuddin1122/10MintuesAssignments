import io
import sys

from vector_store.store import load_documents, save_vectorstore
from splitter.text_splitter import recursive_split
from splitter.splitter_utils import semantic_split, paragraph_split


def build_dual_stores(bn_files, en_files, strategy="semantic"):

    splitter = {
        "semantic": semantic_split,
        "paragraph": paragraph_split,
        "recursive": recursive_split
    }.get(strategy)

    if not splitter:
        raise ValueError(f"Invalid split strategy: {strategy}")

    if bn_files:
        print("Processing Bangla")
        bn_chunks = splitter(load_documents(bn_files))
        save_vectorstore(bn_chunks, f"vectorstore_bn_{strategy}")
    else:
        print("No Bangla files provided, skipping Bangla processing")

    if en_files:
        print("Processing English")
        en_chunks = splitter(load_documents(en_files))
        save_vectorstore(en_chunks, f"vectorstore_en_{strategy}")
    else:
        print("No English files provided, skipping English processing")


if __name__ == "__main__":
    bn_files = [
        "data/raw/bangla/HSC26-Bangla1st-Paper.pdf"
    ]
    en_files = []
    build_dual_stores(bn_files, en_files, strategy="semantic")