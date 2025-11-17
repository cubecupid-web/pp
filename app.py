import streamlit as st
import google.generativeai as genai
import os
import uuid
import json
import io
import logging
import time
from datetime import datetime
from functools import lru_cache
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
import supabase

st.set_page_config(
    page_title="Nyay-Saathi",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
:root {
    --primary-color: #00FFD1;
    --background-color: #08070C;
    --secondary-background-color: #1B1C2A;
    --text-color: #FAFAFA;
}
body { font-family: 'sans serif'; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stButton > button {
    border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color);
    padding: 12px 24px; border-radius: 8px; font-weight: bold; transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background: var(--primary-color); color: var(--background-color); box-shadow: 0 0 15px var(--primary-color);
}
.stTextArea textarea {
    background-color: var(--secondary-background-color); color: var(--text-color);
    border: 1px solid var(--primary-color); border-radius: 8px;
}
div[data-testid="chat-message-container"] {
    background-color: var(--secondary-background-color);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--text-color); padding: 10px; transition: all 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--secondary-background-color); }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--secondary-background-color); color: var(--primary-color); border-bottom: 3px solid var(--primary-color);
}
div[data-testid="stVerticalBlock"] {
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Error configuring: {e}. Please check your API key in Streamlit Secrets.")
    st.stop()

try:
    supabase_url = st.secrets.get("VITE_SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    supabase_key = st.secrets.get("VITE_SUPABASE_SUPABASE_ANON_KEY") or os.getenv("VITE_SUPABASE_SUPABASE_ANON_KEY")
    supabase_client = supabase.create_client(supabase_url, supabase_key)
except Exception as e:
    logger.warning(f"Supabase connection failed: {e}. Running in offline mode.")
    supabase_client = None

DB_FAISS_PATH = "vectorstores/db_faiss"
MODEL_NAME = "gemini-2.5-flash"

rag_prompt_template = """
You are 'Nyay-Saathi,' a kind legal friend.
A common Indian citizen is asking for help.
You have two sources of information. Prioritize the MOST relevant one.
1. CONTEXT_FROM_GUIDES: (General guides from a database)
{context}

2. DOCUMENT_CONTEXT: (Specific text from a document the user uploaded)
{document_context}

Answer the user's 'new question' based on the most relevant context.
If the 'new question' is a follow-up, use the 'chat history' to understand it.
Do not use any legal jargon.
Give a simple, step-by-step action plan in the following language: {language}.
If no context is relevant, just say "I'm sorry, I don't have enough information on that. Please contact NALSA."

CHAT HISTORY:
{chat_history}

NEW QUESTION:
{question}

Your Simple, Step-by-Step Action Plan (in {language}):
"""

@st.cache_resource
def get_models_and_db():
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.6
            }
        )

        return retriever, llm
    except Exception as e:
        st.error(f"Error loading models or vector store: {e}")
        st.error("Did you run 'ingest.py' and push the 'vectorstores' folder to GitHub?")
        st.stop()

retriever, llm = get_models_and_db()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def sanitize_input(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > 5000:
        text = text[:5000]
    return text

@lru_cache(maxsize=128)
def get_cached_response(question_hash, language):
    if supabase_client:
        try:
            response = supabase_client.table("messages").select("content, sources_from_guides").eq("id", question_hash).maybeSingle().execute()
            if response.data:
                return response.data
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
    return None

def get_or_create_user():
    if "user_id" not in st.session_state:
        session_id = str(uuid.uuid4())
        st.session_state.user_id = session_id

        if supabase_client:
            try:
                response = supabase_client.table("users").insert({
                    "session_id": session_id,
                    "language": "Simple English"
                }).execute()
                st.session_state.db_user_id = response.data[0]["id"] if response.data else None
            except Exception as e:
                logger.warning(f"Failed to create user record: {e}")
                st.session_state.db_user_id = None
    return st.session_state.user_id

def get_or_create_conversation():
    if "conversation_id" not in st.session_state:
        conv_id = str(uuid.uuid4())
        st.session_state.conversation_id = conv_id

        if supabase_client and st.session_state.get("db_user_id"):
            try:
                response = supabase_client.table("conversations").insert({
                    "user_id": st.session_state.db_user_id,
                    "title": "New Conversation"
                }).execute()
                st.session_state.db_conversation_id = response.data[0]["id"] if response.data else None
            except Exception as e:
                logger.warning(f"Failed to create conversation: {e}")
                st.session_state.db_conversation_id = None
    return st.session_state.conversation_id

def save_message_to_db(role, content, sources, used_document):
    if supabase_client and st.session_state.get("db_conversation_id"):
        try:
            sources_list = [doc.metadata.get('source', 'Unknown') for doc in sources] if sources else []
            supabase_client.table("messages").insert({
                "conversation_id": st.session_state.db_conversation_id,
                "role": role,
                "content": content,
                "sources_from_guides": sources_list,
                "source_from_document": used_document
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save message: {e}")

def save_document_to_db(file_name, file_type, raw_text, explanation, language):
    if supabase_client and st.session_state.get("db_user_id"):
        try:
            doc_response = supabase_client.table("documents").insert({
                "user_id": st.session_state.db_user_id,
                "conversation_id": st.session_state.get("db_conversation_id"),
                "file_name": file_name,
                "file_type": file_type,
                "raw_text": raw_text,
                "explanation": explanation,
                "language": language
            }).execute()
            return doc_response.data[0]["id"] if doc_response.data else None
        except Exception as e:
            logger.warning(f"Failed to save document: {e}")
    return None

def save_feedback_to_db(message_index, rating):
    if supabase_client and st.session_state.get("db_user_id") and st.session_state.get("db_conversation_id"):
        try:
            messages = supabase_client.table("messages").select("id").eq("conversation_id", st.session_state.db_conversation_id).execute()
            if messages.data and len(messages.data) > message_index:
                message_id = messages.data[message_index]["id"]
                supabase_client.table("feedback").insert({
                    "message_id": message_id,
                    "user_id": st.session_state.db_user_id,
                    "rating": rating
                }).execute()
        except Exception as e:
            logger.warning(f"Failed to save feedback: {e}")

def log_analytics(event_type, response_time_ms=None, api_used=None):
    if supabase_client and st.session_state.get("db_user_id"):
        try:
            supabase_client.table("analytics").insert({
                "user_id": st.session_state.db_user_id,
                "event_type": event_type,
                "response_time_ms": response_time_ms,
                "api_used": api_used
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to log analytics: {e}")

rag_prompt = PromptTemplate.from_template(rag_prompt_template)

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

if "app_started" not in st.session_state:
    st.session_state.app_started = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_context" not in st.session_state:
    st.session_state.document_context = "No document uploaded."
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "uploaded_file_type" not in st.session_state:
    st.session_state.uploaded_file_type = None
if "samjhao_explanation" not in st.session_state:
    st.session_state.samjhao_explanation = None
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

def clear_session():
    st.session_state.messages = []
    st.session_state.document_context = "No document uploaded."
    st.session_state.uploaded_file_bytes = None
    st.session_state.uploaded_file_type = None
    st.session_state.samjhao_explanation = None
    st.session_state.uploaded_documents = []
    st.session_state.file_uploader_key += 1

get_or_create_user()
get_or_create_conversation()

if not st.session_state.app_started:
    st.title("Welcome to 🤝 Nyay-Saathi")
    st.subheader("Your AI legal friend, built for India.")
    st.markdown("This tool helps you understand complex legal documents and get clear, simple action plans.")
    st.markdown("---")

    if st.button("Click here to start", type="primary"):
        st.session_state.app_started = True
        log_analytics("app_started")
        st.rerun()

else:
    st.title("🤝 Nyay-Saathi (Justice Companion)")
    st.markdown("Your legal friend, in your pocket. Built for India.")

    col1, col2 = st.columns([3, 1])
    with col1:
        language = st.selectbox(
            "Choose your language:",
            ("Simple English", "Hindi (in Roman script)", "Kannada", "Tamil", "Telugu", "Marathi")
        )
    with col2:
        st.write("")
        st.write("")
        st.button("Start New Session ♻️", on_click=clear_session, type="primary")

    st.divider()

    tab1, tab2 = st.tabs(["**Samjhao** (Explain this Document)", "**Kya Karoon?** (Ask a Question)"])

    with tab1:
        st.header("Upload Legal Documents to Explain")
        st.write("Take a photo (or upload a PDF) of your legal notice or agreement.")

        uploaded_file = st.file_uploader(
            "Choose a file...",
            type=["jpg", "jpeg", "png", "pdf"],
            key=st.session_state.file_uploader_key
        )

        if uploaded_file is not None:
            new_file_bytes = uploaded_file.getvalue()
            if new_file_bytes != st.session_state.uploaded_file_bytes:
                st.session_state.uploaded_file_bytes = new_file_bytes
                st.session_state.uploaded_file_type = uploaded_file.type
                st.session_state.samjhao_explanation = None
                st.session_state.document_context = "No document uploaded."

        if st.session_state.uploaded_file_bytes is not None:
            file_bytes = st.session_state.uploaded_file_bytes
            file_type = st.session_state.uploaded_file_type

            if "image" in file_type:
                image = Image.open(io.BytesIO(file_bytes))
                st.image(image, caption="Your Uploaded Document", use_column_width=True)
            elif "pdf" in file_type:
                st.info("PDF file uploaded. Click 'Samjhao!' to explain.")

            if st.button("Samjhao!", type="primary", key="samjhao_button"):

                spinner_text = "Your friend is reading and explaining..."
                if "image" in file_type:
                    spinner_text = "Reading your image... (this can take 15-30s)"

                with st.spinner(spinner_text):
                    try:
                        start_time = time.time()
                        model = genai.GenerativeModel(MODEL_NAME)

                        prompt_text_multi = f"""
                        You are an AI assistant. The user has uploaded a document (MIME type: {file_type}).
                        Perform two tasks:
                        1. Extract all raw text from the document.
                        2. Explain the document in simple, everyday {language}.

                        Respond with ONLY a JSON object in this format:
                        {{
                          "raw_text": "The raw extracted text...",
                          "explanation": "Your simple {language} explanation..."
                        }}
                        """

                        data_part = {'mime_type': file_type, 'data': file_bytes}

                        max_retries = 3
                        response = None
                        for attempt in range(max_retries):
                            try:
                                response = model.generate_content([prompt_text_multi, data_part])
                                break
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    time.sleep(2 ** attempt)
                                else:
                                    raise e

                        clean_response_text = response.text.strip().replace("```json", "").replace("```", "")
                        response_json = json.loads(clean_response_text)

                        st.session_state.samjhao_explanation = response_json.get("explanation")
                        st.session_state.document_context = response_json.get("raw_text")

                        response_time = int((time.time() - start_time) * 1000)
                        log_analytics("document_explained", response_time_ms=response_time, api_used="gemini")

                        doc_id = save_document_to_db(
                            uploaded_file.name,
                            file_type,
                            response_json.get("raw_text", ""),
                            response_json.get("explanation", ""),
                            language
                        )

                        st.session_state.uploaded_documents.append({
                            "id": doc_id,
                            "name": uploaded_file.name,
                            "explanation": response_json.get("explanation")
                        })

                    except json.JSONDecodeError:
                        st.error("Failed to parse AI response. Please try again.")
                        logger.error(f"JSON decode error: {response.text if response else 'No response'}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        logger.error(f"Document processing error: {e}")
                        st.warning("Please try again or try a different document.")

        if st.session_state.samjhao_explanation:
            st.subheader(f"Here's what it means in {language}:")
            st.markdown(st.session_state.samjhao_explanation)

        if st.session_state.document_context != "No document uploaded." and st.session_state.samjhao_explanation:
            st.success("Context Saved! You can now ask questions about this document in the 'Kya Karoon?' tab.")

        if st.session_state.uploaded_documents:
            st.subheader("Uploaded Documents")
            for doc in st.session_state.uploaded_documents:
                st.write(f"📄 {doc['name']}")

    with tab2:
        st.header("Ask for a simple action plan")

        col1, col2 = st.columns([3,1])
        with col1:
             st.write("Scared? Confused? Ask a question and get a simple 3-step plan **based on real guides.**")
        with col2:
            if st.button("Clear Chat ♻️"):
                st.session_state.messages = []
                log_analytics("chat_cleared")
                st.rerun()

        if st.session_state.document_context != "No document uploaded.":
            with st.container():
                st.info(f"**Context Loaded:** I have your uploaded document in memory. Feel free to ask questions about it!")

        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                guides_sources = message.get("sources_from_guides")
                doc_context_used = message.get("source_from_document")

                if (guides_sources and len(guides_sources) > 0) or doc_context_used:
                    st.subheader("Sources I used:")

                    if doc_context_used:
                        st.warning(f"**From Your Uploaded Document:**\n\n...{st.session_state.document_context[:500]}...")

                    if guides_sources:
                        for doc in guides_sources:
                            st.info(f"**From {doc.metadata.get('source', 'Unknown Guide')}:**\n\n...{doc.page_content}...")

                if message["role"] == "assistant":
                    feedback_key = f"feedback_{i}"
                    c1, c2, _ = st.columns([1, 1, 5])
                    with c1:
                        if st.button("👍", key=f"{feedback_key}_up"):
                            save_feedback_to_db(i, "positive")
                            st.toast("Thanks for your feedback!")
                            log_analytics("feedback_positive")
                    with c2:
                        if st.button("👎", key=f"{feedback_key}_down"):
                            save_feedback_to_db(i, "negative")
                            st.toast("Thanks for your feedback!")
                            log_analytics("feedback_negative")

        if prompt := st.chat_input(f"Ask your follow-up question in {language}..."):

            prompt = sanitize_input(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_message_to_db("user", prompt, [], False)

            with st.spinner("Your friend is checking the guides..."):
                try:
                    start_time = time.time()

                    chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]])
                    current_doc_context = st.session_state.document_context

                    invoke_payload = {
                        "question": prompt,
                        "language": language,
                        "chat_history": chat_history_str,
                        "document_context": current_doc_context
                    }

                    response_dict = rag_chain_with_sources.invoke(invoke_payload)
                    response = response_dict["answer"]
                    docs = response_dict["sources"]

                    used_document = current_doc_context != "No document uploaded." and (not docs or len(docs) == 0)

                    response_time = int((time.time() - start_time) * 1000)
                    log_analytics("question_answered", response_time_ms=response_time, api_used="gemini")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources_from_guides": docs,
                        "source_from_document": used_document
                    })

                    save_message_to_db("assistant", response, docs, used_document)

                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"RAG processing error: {e}")
                    st.info("Please try rephrasing your question or try again in a moment.")

st.divider()
st.error("""
**Disclaimer:** I am an AI, not a lawyer. This is not legal advice.
Please consult a real lawyer or contact your local **NALSA (National Legal Services Authority)** for free legal aid.
""")
