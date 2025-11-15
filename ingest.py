import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def create_vector_db():
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob='*.txt', loader_cls=TextLoader)
    documents = loader.load()
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Successfully created vector store at {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()