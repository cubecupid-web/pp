import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from config import (
    DB_FAISS_PATH,
    MODEL_NAME,
    RAG_CONFIG,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    RAG_PROMPT_TEMPLATE
)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE}
    )


@st.cache_resource
def get_vector_db():
    embeddings = get_embeddings()
    try:
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.error("Did you run 'ingest.py' and push the 'vectorstores' folder to GitHub?")
        st.stop()


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)


@st.cache_resource
def get_retriever():
    db = get_vector_db()
    return db.as_retriever(
        search_type=RAG_CONFIG["search_type"],
        search_kwargs=RAG_CONFIG["search_kwargs"]
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def build_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain_with_sources = RunnableParallel(
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "language": itemgetter("language"),
            "chat_history": itemgetter("chat_history"),
            "document_context": itemgetter("document_context")
        }
    ) | {
        "answer": (
            {
                "context": (lambda x: format_docs(x["context"])),
                "question": itemgetter("question"),
                "language": itemgetter("language"),
                "chat_history": itemgetter("chat_history"),
                "document_context": itemgetter("document_context")
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        ),
        "sources": itemgetter("context")
    }

    return rag_chain_with_sources
