import streamlit as st
import google.generativeai as genai
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
/* We can add a subtle border to our columns */
div[data-testid="column"] {
    background: var(--secondary-background-color);
    border: 1px solid #1B1C2A;
    border-radius: 10px;
    padding: 20px;
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
MODEL_NAME = "gemini-1.5-flash-latest"


# --- RAG PROMPT TEMPLATE (for Column 2) ---
rag_prompt_template = """
You are 'Nyay-Saathi,' a kind legal friend.
A common Indian citizen is asking for help.
Base your answer ONLY on the context provided below.
Do not use any legal jargon.
Give a simple, 3-step action plan in the following language: {language}.
If the context is not enough, just say "I'm sorry, I don't have enough information on that. Please contact NALSA."

CONTEXT:
{context}

QUESTION:
{question}

Your Simple, 3-Step Action Plan (in {language}):
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

# --- THE RAG CHAIN (for Column 2) ---
rag_prompt = PromptTemplate.from_template(rag_prompt_template)
rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language")
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- THE APP UI ---
st.title("🤝 Nyay-Saathi (Justice Companion)")
st.markdown("Your legal friend, in your pocket. Built for India.")

# --- LANGUAGE SELECTOR ---
language = st.selectbox(
    "Choose your language:",
    ("Simple English", "Hindi (in Roman script)", "Kannada", "Tamil", "Telugu", "Marathi")
)

st.divider()

# --- THE NEW LAYOUT ---
# Create two columns of equal width
col1, col2 = st.columns(2)

# --- COLUMN 1: SAMJHAO (EXPLAIN) ---
with col1:
    st.header("Samjhao (Explain this Document)")
    st.write("Take a photo (or upload a PDF) of your legal notice or agreement.")
    
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.type
        
        # Display logic
        if "image" in file_type:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Uploaded Document", use_column_width=True)
        elif "pdf" in file_type:
            st.info("PDF file uploaded. Click 'Samjhao!' to explain.")
        
        if st.button("Samjhao!", type="primary", key="samjhao_button"):
            with st.spinner("Your friend is reading the document..."):
                try:
                    # Use native genai for multimodal
                    model = genai.GenerativeModel(MODEL_NAME)
                    
                    prompt_text = f"Explain this document in simple, everyday {language}. Identify the 3 most important parts. Be reassuring."
                    
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
with col2:
    st.header("Kya Karoon? (Ask a Question)")
    st.write("Scared? Confused? Ask a question and get a simple 3-step plan **based on real guides.**")
    user_question = st.text_input("Ask your question (e.g., 'My landlord is threatening to evict me')")
    
    if st.button("Get Plan", type="primary", key="kya_karoon_button"):
        if not user_question:
            st.warning("Please ask a question.")
        else:
            with st.spinner("Your friend is checking the guides..."):
                try:
                    docs = retriever.get_relevant_documents(user_question)
                    invoke_payload = {"question": user_question, "language": language}
                    response = rag_chain.invoke(invoke_payload)
                    
                    if response:
                        st.subheader(f"Here is a simple 3-step plan (in {language}):")
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

# --- DISCLAIMER (At the bottom, full width) ---
st.divider()
st.error("""
**Disclaimer:** I am an AI, not a lawyer. This is not legal advice.
Please consult a real lawyer or contact your local **NALSA (National Legal Services Authority)** for free legal aid.
""")