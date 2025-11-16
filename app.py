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

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Nyay-Saathi", page_icon="🤝", layout="wide")

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
div[data-testid="column"] {
    background: transparent;
    border: none;
    border-radius: 10px;
    padding: 10px;
}
/* Style the chat messages */
div[data-testid="chat-message-container"] {
    background-color: var(--secondary-background-color);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
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
MODEL_NAME = "gemini-2.5-flash-lite"


# --- RAG PROMPT TEMPLATE (for Column 2) ---
rag_prompt_template = """
You are 'Nyay-Saathi,' a kind legal friend.
A common Indian citizen is asking for help.
Answer their 'new question' based ONLY on the context provided.
If the new question is a follow-up, use the 'chat history' to understand it.
Do not use any legal jargon.
Give a simple, step-by-step action plan in the following language: {language}.
If the context is not enough, just say "I'm sorry, I don't have enough information on that. Please contact NALSA."

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

NEW QUESTION:
{question}

Your Simple, Step-by-Step Action Plan (in {language}):
"""

# --- LOAD THE MODEL & VECTOR STORE ---
@st.cache_resource
def get_models_and_db(): # Renamed to clear cache
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)
        
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5
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
        "chat_history": itemgetter("chat_history")
    }
) | {
    "answer": (
        {
            "context": (lambda x: format_docs(x["context"])),
            "question": itemgetter("question"),
            "language": itemgetter("language"),
            "chat_history": itemgetter("chat_history")
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    ),
    "sources": itemgetter("context")
}

# --- THE APP UI ---
st.title("🤝 Nyay-Saathi (Justice Companion)")
st.markdown("Your legal friend, in your pocket. Built for India.")

# --- LANGUAGE SELECTOR ---
language = st.selectbox(
    "Choose your language:",
    ("Simple English", "Hindi (in Roman script)", "Kannada", "Tamil", "Telugu", "Marathi")
)

st.divider()

# --- THE LAYOUT ---
col1, col2 = st.columns(2)

# --- COLUMN 1: SAMJHAO (EXPLAIN) ---
# This part is unchanged
with col1:
    st.header("Samjhao (Explain this Document)")
    st.write("Take a photo (or upload a PDF) of your legal notice or agreement.")
    
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.type
        
        if "image" in file_type:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Uploaded Document", use_column_width=True)
        elif "pdf" in file_type:
            st.info("PDF file uploaded. Click 'Samjhao!' to explain.")
        
        if st.button("Samjhao!", type="primary", key="samjhao_button"):
            with st.spinner("Your friend is reading the document..."):
                try:
                    model = genai.GenerativeModel(MODEL_NAME)
                    prompt_text = f"""
                    You are 'Nyay-Saathi,' a kind legal friend.
                    The user has uploaded a document (MIME type: {file_type}).
                    First, extract all the text you can see from this document.
                    Then, explain that extracted text in simple, everyday {language}.
                    Do not use any legal jargon.
                    Identify the 3 most important parts for the user (like dates, names, or actions they must take).
                    The user is scared and confused. Be kind and reassuring.

                    Your Simple {language} Explanation:
                    """
                    
                    data_part = {'mime_type': file_type, 'data': file_bytes}
                    response = model.generate_content([prompt_text, data_part])
                    
                    if response.text:
                        st.subheader(f"Here's what it means in {language}:")
                        st.markdown(response.text)
                    else:
                        st.error("The AI could not read the document. Please try a clearer file.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- COLUMN 2: KYA KAROON? (WHAT TO DO?) ---
# --- THIS ENTIRE BLOCK IS UPDATED ---
with col2:
    st.header("Kya Karoon? (Ask a Question)")
    st.write("Scared? Confused? Ask a question and get a simple 3-step plan **based on real guides.**")

    # 1. Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. Display all past messages (THIS IS THE FIX)
    # This loop now runs *before* the chat input, so all messages appear above it.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources if they exist in the message
            if "sources" in message:
                st.subheader("Sources I used:")
                for doc in message["sources"]:
                    st.info(f"**From {doc.metadata.get('source', 'Unknown Guide')}:**\n\n...{doc.page_content}...")

    # 3. Define the chat input box (it will now appear at the bottom)
    if prompt := st.chat_input(f"Ask your follow-up question in {language}..."):
        
        # 4. Add user's new message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 5. Get the AI's response
        with st.spinner("Your friend is checking the guides..."):
            try:
                # 6. Create the chat history string
                chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])
                
                # 7. Create the payload
                invoke_payload = {
                    "question": prompt,
                    "language": language,
                    "chat_history": chat_history_str
                }
                
                # 8. Call the RAG chain
                response_dict = rag_chain_with_sources.invoke(invoke_payload) 
                response = response_dict["answer"]
                docs = response_dict["sources"]
                
                # 9. Add the AI's response to history (with sources)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": docs
                })
                
                # 10. Force the app to re-run from the top
                # This will make the loop in step 2 display the new messages
                st.rerun()
            
            except Exception as e:
                st.error(f"An error occurred during RAG processing: {e}")

# --- DISCLAIMER (At the bottom, full width) ---
st.divider()
st.error("""
**Disclaimer:** I am an AI, not a lawyer. This is not legal advice.
Please consult a real lawyer or contact your local **NALSA (National Legal Services Authority)** for free legal aid.
""")