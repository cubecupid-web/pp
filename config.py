DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"
MODEL_NAME = "gemini-2.5-flash"

RAG_CONFIG = {
    "search_type": "similarity_score_threshold",
    "search_kwargs": {
        "k": 3,
        "score_threshold": 0.3
    }
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50
}

SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png", "pdf"]
MAX_IMAGE_SIZE_MB = 10
IMAGE_COMPRESSION_QUALITY = 85

LANGUAGES = [
    "Simple English",
    "Hindi (in Roman script)",
    "Kannada",
    "Tamil",
    "Telugu",
    "Marathi"
]

RAG_PROMPT_TEMPLATE = """
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
