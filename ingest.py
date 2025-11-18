import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import (
    DATA_PATH,
    DB_FAISS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    TEXT_SPLITTER_CONFIG
)


def create_vector_db():
    data_path = "data/"
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist.")
        print("Please ensure you have a 'data/' folder with .txt files.")
        sys.exit(1)

    txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"Error: No .txt files found in '{data_path}'")
        sys.exit(1)

    print(f"Found {len(txt_files)} text files: {txt_files}")

    try:
        print("\n[1/4] Loading documents...")
        loader = DirectoryLoader(data_path, glob='*.txt', loader_cls=TextLoader)
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} documents")

        print("\n[2/4] Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
            chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")

        print("\n[3/4] Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE}
        )
        print(f"✓ Embedding model loaded: {EMBEDDING_MODEL}")

        print("\n[4/4] Creating FAISS vector store...")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_FAISS_PATH)
        print(f"✓ Vector store saved to {DB_FAISS_PATH}")

        print("\n" + "=" * 60)
        print("SUCCESS! Vector database created successfully.")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("Failed to create vector database.")
        sys.exit(1)


if __name__ == "__main__":
    create_vector_db()
