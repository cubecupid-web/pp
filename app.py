import streamlit as st
import google.generativeai as genai
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from PIL import Image 
from langchain_core.messages import HumanMessage 
import base64
import json
import io

# --- CONFIGURATION & PAGE SETUP ---
# This MUST be the very first Streamlit command
st.set_page_config(
    page_title="Nyay-Saathi", 
    page_icon="🤝", 
    layout="wide", 
    initial_sidebar_state="collapsed" # Collapse sidebar for a cleaner look
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* ... (Your custom CSS is here, hidden for brevity) ... */
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
/* Style the chat messages */
div[data-testid="chat-message-container"] {
    background-color: var(--secondary-background-color);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}
/* This is the tab styling from before */
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--text-color); padding: 10px; transition: all 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--secondary-background-color); }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--secondary-background-color); color: var(--primary-color); border-bottom: 3px solid var(--primary-color);
}
/* Center the landing page elements */
div[data-testid="stVerticalBlock"] {
    align-items: center;
}
</style>
""", unsafe_allow_html=True)


# --- API KEY CONFIGURATION ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Error configuring: {e}. Please check your API key in Streamlit Secrets.")
    st.stop()

DB_FAISS_PATH = "vectorstores/db_faiss"
MODEL_NAME = "gemini-2.5-flash"


# --- RAG PROMPT TEMPLATE ---
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

# --- LOAD THE MODEL & VECTOR STORE ---
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
                "score_threshold": 0.3 # Your tuned threshold
            }
        )
        
        return retriever, llm
    except Exception as e:
        st.error(f"Error loading models or vector store: {e}")
        st.error("Did you run 'ingest.py' and push the 'vectorstores' folder to GitHub?")
        st.stop()

retriever, llm = get_models_and_db()

# --- NEW HELPER FUNCTION ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- THE RAG CHAIN ---
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

# --- SESSION STATE INITIALIZATION ---
# This runs once per session
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


# --- "START NEW SESSION" BUTTON ---
def clear_session():
    st.session_state.messages = []
    st.session_state.document_context = "No document uploaded."
    st.session_state.uploaded_file_bytes = None
    st.session_state.uploaded_file_type = None
    st.session_state.samjhao_explanation = None
    # This increments the key, forcing the file uploader to reset
    st.session_state.file_uploader_key += 1 


# --- THE APP UI ---

# --- LANDING PAGE ---
if not st.session_state.app_started:
    st.title("Welcome to 🤝 Nyay-Saathi")
    st.subheader("Your AI legal friend, built for India.")
    # You can add a logo here if you have one in your repo
    # st.image("logo.png", width=200) 
    st.markdown("This tool helps you understand complex legal documents and get clear, simple action plans.")
    st.markdown("---")
    
    if st.button("Click here to start", type="primary"):
        st.session_state.app_started = True
        st.rerun()

# --- MAIN APP ---
else:
    st.title("🤝 Nyay-Saathi (Justice Companion)")
    st.markdown("Your legal friend, in your pocket. Built for India.")

    # Place language selector and clear button side-by-side
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

    # --- THE TAB-BASED LAYOUT ---
    tab1, tab2 = st.tabs(["**Samjhao** (Explain this Document)", "**Kya Karoon?** (Ask a Question)"])

    # --- TAB 1: SAMJHAO (EXPLAIN) ---
    with tab1:
        st.header("Upload a Legal Document to Explain")
        st.write("Take a photo (or upload a PDF) of your legal notice or agreement.")
        
        uploaded_file = st.file_uploader(
            "Choose a file...", 
            type=["jpg", "jpeg", "png", "pdf"], 
            key=st.session_state.file_uploader_key # Use the stateful key
        )
        
        # Check if a *new* file has been uploaded
        if uploaded_file is not None:
            new_file_bytes = uploaded_file.getvalue()
            if new_file_bytes != st.session_state.uploaded_file_bytes:
                st.session_state.uploaded_file_bytes = new_file_bytes
                st.session_state.uploaded_file_type = uploaded_file.type
                st.session_state.samjhao_explanation = None # Clear old explanation
                st.session_state.document_context = "No document uploaded." # Clear old context
        
        # Display logic: ALWAYS read from session state
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
                        response = model.generate_content([prompt_text_multi, data_part])
                        clean_response_text = response.text.strip().replace("```json", "").replace("```", "")
                        response_json = json.loads(clean_response_text)
                        
                        st.session_state.samjhao_explanation = response_json.get("explanation")
                        st.session_state.document_context = response_json.get("raw_text")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.warning("The AI response might be in an invalid format. Please try again.")

        # Always display the explanation if it exists in the state
        if st.session_state.samjhao_explanation:
            st.subheader(f"Here's what it means in {language}:")
            st.markdown(st.session_state.samjhao_explanation)
        
        if st.session_state.document_context != "No document uploaded." and st.session_state.samjhao_explanation:
            st.success("Context Saved! You can now ask questions about this document in the 'Kya Karoon?' tab.")


    # --- TAB 2: KYA KAROON? (WHAT TO DO?) ---
    with tab2:
        st.header("Ask for a simple action plan")
        
        # --- NEW: In-tab clear chat button ---
        col1, col2 = st.columns([3,1])
        with col1:
             st.write("Scared? Confused? Ask a question and get a simple 3-step plan **based on real guides.**")
        with col2:
            if st.button("Clear Chat ♻️"):
                st.session_state.messages = []
                st.rerun()

        # Display a message if context is loaded
        if st.session_state.document_context != "No document uploaded.":
            with st.container():
                st.info(f"**Context Loaded:** I have your uploaded document in memory. Feel free to ask questions about it!")

        # Display all past messages
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
                
                # --- NEW: Feedback Buttons ---
                if message["role"] == "assistant":
                    feedback_key = f"feedback_{i}"
                    c1, c2, _ = st.columns([1, 1, 5])
                    with c1:
                        if st.button("👍", key=f"{feedback_key}_up"):
                            st.toast("Thanks for your feedback!")
                    with c2:
                        if st.button("👎", key=f"{feedback_key}_down"):
                            st.toast("Thanks for your feedback!")

        # Define the chat input box
        if prompt := st.chat_input(f"Ask your follow-up question in {language}..."):
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Your friend is checking the guides..."):
                try:
                    chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])
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
                    
                    used_document = False
                    
                    # --- NEW: "Source of Truth" Audit ---
                    if not docs and current_doc_context != "No document uploaded.":
                        # If no guides were found, we *must* audit the response.
                        with st.spinner("Auditing response source..."):
                            audit_model = genai.GenerativeModel(MODEL_NAME)
                            audit_prompt = f"""
                            You are an auditor.
                            Question: "{prompt}"
                            Answer: "{response}"
                            Context: "{current_doc_context}"
                            
                            Did the "Answer" come *primarily* from the "Context"?
                            Respond with ONLY the word 'YES' or 'NO'.
                            """
                            audit_response = audit_model.generate_content(audit_prompt)
                            
                            if "YES" in audit_response.text.upper():
                                used_document = True
                    # --- END AUDIT ---

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources_from_guides": docs,
                        "source_from_document": used_document
                    })
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"An error occurred during RAG processing: {e}")

# --- DISCLAIMER (At the bottom, full width) ---
st.divider()
st.error("""
**Disclaimer:** I am an AI, not a lawyer. This is not legal advice.
Please consult a real lawyer or contact your local **NALSA (National Legal Services Authority)** for free legal aid.
""")