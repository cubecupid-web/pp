import gradio as gr
import google.generativeai as genai
import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeModel

# --- CONFIGURATION ---
# IMPORTANT: You still need to add GOOGLE_API_KEY to your Secrets
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except Exception as e:
    print(f"API Key not found. Set it in Secrets. Error: {e}")

DB_FAISS_PATH = "vectorstores/db_faiss"
MODEL_NAME = "gemini-2.5-flash-lite"

# --- PROMPTS ---
rag_prompt_template = """
You are 'Nyay-Saathi,' a kind legal friend.
Base your answer ONLY on the context provided.
Give a simple, 3-step action plan.
If the context is not enough, just say "I'm sorry, I don't have enough information on that."

CONTEXT:
{context}

QUESTION:
{question}

Your Simple, 3-Step Action Plan:
"""

# --- LOAD MODELS (This is the RAG "Brain") ---
# This part is the same as our Streamlit app
try:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatGoogleGenerativeModel(model=MODEL_NAME, temperature=0.7)
    retriever = db.as_retriever()
except Exception as e:
    print(f"Error loading models or DB: {e}. Did you run ingest.py and push 'vectorstores'?")

# --- THE RAG CHAIN ---
rag_prompt = PromptTemplate.from_template(rag_prompt_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- THE "WHAT TO DO?" FUNCTION (This is what Gradio calls) ---
def get_rag_answer(user_question):
    # Get the answer from the RAG chain
    response_text = rag_chain.invoke(user_question)
    
    # Get the sources
    docs = retriever.get_relevant_documents(user_question)
    sources_text = "\n\n--- SOURCES I USED ---\n"
    if docs:
        for doc in docs:
            sources_text += f"**From {doc.metadata.get('source', 'Unknown')}:**\n...{doc.page_content}...\n\n"
    
    return response_text + sources_text

# --- THE "SAMJHAO" (EXPLAIN) FUNCTION ---
def get_explain_answer(legal_text):
    prompt = f"""
    Explain the following legal text in simple, everyday Hindi (using Roman script). 
    Identify the 3 most important parts. Be kind and reassuring.

    Legal Text: "{legal_text}"
    """
    response = llm.invoke(prompt)
    return response

# --- BUILD THE GRADIO UI ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="Nyay-Saathi") as demo:
    gr.Markdown("# 🤝 Nyay-Saathi (Justice Companion)")
    gr.Markdown("Your legal friend, in your pocket. Built for India.")
    
    with gr.Tabs():
        # --- TAB 2: KYA KAROON? (RAG-Powered) ---
        with gr.TabItem("Kya Karoon? (What do I do?)"):
            gr.Markdown("Ask a question and get a simple 3-step plan **based on real guides.**")
            question_box = gr.Textbox(label="Your Question", placeholder="e.g., 'My landlord is threatening to evict me'")
            answer_box = gr.Markdown(label="Your Plan")
            q_button = gr.Button("Get Plan", variant="primary")
            q_button.click(get_rag_answer, inputs=question_box, outputs=answer_box)

        # --- TAB 1: SAMJHAO (EXPLAIN) ---
        with gr.TabItem("Samjhao (Explain this to me)"):
            gr.Markdown("Confused by a legal notice, rent agreement, or court paper? Paste it here.")
            text_box = gr.Textbox(label="Paste the confusing legal text here:", lines=5)
            explanation_box = gr.Markdown(label="Simple Explanation")
            s_button = gr.Button("Samjhao!", variant="primary")
            s_button.click(get_explain_answer, inputs=text_box, outputs=explanation_box)

    gr.Error("Disclaimer: I am an AI, not a lawyer. This is not legal advice. Please consult a real lawyer or contact NALSA.")

# --- LAUNCH THE APP ---
demo.launch()