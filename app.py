import streamlit as st
import google.generativeai as genai
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURATION & PAGE SETUP ---
# This MUST be the very first Streamlit command
st.set_page_config(page_title="Nyay-Saathi", page_icon="🤝", layout="wide")

# --- CUSTOM CSS FOR THE "PROPER WEBSITE" LAYOUT ---
st.markdown("""
<style>
/* --- THEME --- */
/* These are your colors from config.toml */
:root {
    --primary-color: #00FFD1;
    --background-color: #08070C;
    --secondary-background-color: #1B1C2A;
    --text-color: #FAFAFA;
}

/* --- MAIN FONT --- */
body {
    font-family: 'sans serif';
}

/* --- HIDE STREAMLIT BRANDING --- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* --- BUTTONS --- */
.stButton > button {
    border: 2px solid var(--primary-color);
    background: transparent;
    color: var(--primary-color);
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
}

.stButton > button:hover {
    background: var(--primary-color);
    color: var(--background-color);
    box-shadow: 0 0 15px var(--primary-color);
}

/* --- TABS --- */
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-color);
    padding: 10px;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--secondary-background-color);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--secondary-background-color);
    color: var(--primary-color);
    border-bottom: 3px solid var(--primary-color);
}

/* --- TEXT AREA --- */
.stTextArea textarea {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    border: 1px solid var(--primary-color);
    border-radius: 8px;
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

# --- PROMPTS ---
def get_samjhao_prompt(legal_text):
    return f"""
    The following is a confusing Indian legal document or text. 
    Explain it in simple, everyday Hindi (using Roman script). 
    Do not use any legal jargon.
    Identify the 3 most important parts for the user.
    The user is a common person who is scared and confused. Be kind and reassuring.

    Legal Text: "{legal_text}"
    Simple Hindi Explanation:
    """

rag_prompt_template = """
You are 'Nyay-Saathi,' a kind legal friend.
A common Indian citizen is asking for help.
Base your answer ONLY on the context provided below.
Do not use any legal jargon.
Give a simple, 3-step action plan.
If the context is not enough, just say "I'm sorry, I don't have enough information on that. Please contact NALSA."

CONTEXT:
{context}

QUESTION:
{question}

Your Simple, 3-Step Action Plan:
"""

# --- LOAD THE MODEL & VECTOR STORE ---
@st.cache_resource
def load_models_and_db():
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)
        return db.as_retriever(), llm
    except Exception as e:
        st.error(f"Error loading models or vector store: {e}")
        st.error("Did you run 'ingest.py' and push the 'vectorstores' folder to GitHub?")
        st.stop()

retriever, llm = load_models_and_db()

# --- THE RAG CHAIN ---
rag_prompt = PromptTemplate.from_template(rag_prompt_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- HELPER FUNCTION (For the simple "Samjhao" tab) ---
def get_llm_response(prompt):
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# --- THE APP UI ---
st.title("🤝 Nyay-Saathi (Justice Companion)")
st.markdown("Your legal friend, in your pocket. Built for India.")

tab1, tab2 = st.tabs(["**Samjhao** (Explain this to me)", "**Kya Karoon?** (What do I do?)"])

# --- TAB 1: SAMJHAO (EXPLAIN) ---
with tab1:
    st.header("Translate 'Legalese' into simple language")
    st.write("Confused by a legal notice, rent agreement, or court paper? Paste it here.")
    legal_text = st.text_area("Paste the confusing legal text here:", height=200)
    
    if st.button("Samjhao!", type="primary", key="samjhao_button"):
        if not legal_text:
            st.warning("Please paste some text to explain.")
        else:
            with st.spinner("Your friend is thinking..."):
                prompt = get_samjhao_prompt(legal_text)
                response = get_llm_response(prompt)
                if response:
                    st.subheader("Here's what it means in simple terms:")
                    st.markdown(response)

# --- TAB 2: KYA KAROON? (WHAT TO DO?) ---
with tab2:
    st.header("Ask for a simple action plan")
    st.write("Scared? Confused? Ask a question and get a simple 3-step plan **based on real guides.**")
    user_question = st.text_input("Ask your question (e.g., 'My landlord is threatening to evict me')")
    
    if st.button("Get Plan", type="primary", key="kya_karoon_button"):
        if not user_question:
            st.warning("Please ask a question.")
        else:
            with st.spinner("Your friend is checking the guides..."):
                try:
                    docs = retriever.get_relevant_documents(user_question)
                    response = rag_chain.invoke(user_question)
                    
                    if response:
                        st.subheader("Here is a simple 3-step plan:")
                        st.markdown(response)
                        st.divider()
                        st.subheader("Sources I used to answer you:")
                        if docs:
                            for doc in docs:
                                st.info(f"**From {doc.metadata.get('source', 'Unknown Guide')}:**\n\n...{doc.page_content}...")
                        else:
                            st.write("No specific sources were needed for this query.")
                except Exception as e:
                    st.error(f"An error occurred during RAG processing: {e}")

# --- DISCLAIMER ---
st.divider()
st.error("""
**Disclaimer:** I am an AI, not a lawyer. This is not legal advice.
Please consult a real lawyer or contact your local **NALSA (National Legal Services Authority)** for free legal aid.
""")