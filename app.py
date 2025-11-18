import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

from config import (
    MODEL_NAME,
    SUPPORTED_FILE_TYPES,
    LANGUAGES
)
from rag_pipeline import build_rag_chain
from document_processor import (
    extract_and_explain_document,
    audit_response_source,
    compress_image
)
from ui_components import (
    render_language_selector_and_buttons,
    render_document_context_info,
    render_chat_messages,
    render_disclaimer
)


st.set_page_config(
    page_title="Nyay-Saathi",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <link rel="stylesheet" href="file:.streamlit/styles.css">
    """,
    unsafe_allow_html=True
)

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Error configuring: {e}. Please check your API key in Streamlit Secrets.")
    st.stop()


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
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "Simple English"


def clear_session():
    st.session_state.messages = []
    st.session_state.document_context = "No document uploaded."
    st.session_state.uploaded_file_bytes = None
    st.session_state.uploaded_file_type = None
    st.session_state.samjhao_explanation = None
    st.session_state.file_uploader_key += 1


if not st.session_state.app_started:
    st.title("Welcome to 🤝 Nyay-Saathi")
    st.subheader("Your AI legal friend, built for India.")
    st.markdown("This tool helps you understand complex legal documents and get clear, simple action plans.")
    st.markdown("---")

    if st.button("Click here to start", type="primary"):
        st.session_state.app_started = True
        st.rerun()

else:
    st.title("🤝 Nyay-Saathi (Justice Companion)")
    st.markdown("Your legal friend, in your pocket. Built for India.")

    language = render_language_selector_and_buttons()
    st.session_state.selected_language = language
    st.divider()

    tab1, tab2 = st.tabs(["**Samjhao** (Explain this Document)", "**Kya Karoon?** (Ask a Question)"])

    with tab1:
        st.header("Upload a Legal Document to Explain")
        st.write("Take a photo (or upload a PDF) of your legal notice or agreement.")

        uploaded_file = st.file_uploader(
            "Choose a file...",
            type=SUPPORTED_FILE_TYPES,
            key=st.session_state.file_uploader_key
        )

        if uploaded_file is not None:
            new_file_bytes = uploaded_file.getvalue()
            if new_file_bytes != st.session_state.uploaded_file_bytes:
                if "image" in uploaded_file.type:
                    compressed_bytes = compress_image(new_file_bytes)
                    st.session_state.uploaded_file_bytes = compressed_bytes
                else:
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
                    explanation, raw_text = extract_and_explain_document(
                        file_bytes,
                        file_type,
                        language
                    )

                    if explanation and raw_text:
                        st.session_state.samjhao_explanation = explanation
                        st.session_state.document_context = raw_text

        if st.session_state.samjhao_explanation:
            st.subheader(f"Here's what it means in {language}:")
            st.markdown(st.session_state.samjhao_explanation)

        if (st.session_state.document_context != "No document uploaded." and
                st.session_state.samjhao_explanation):
            st.success("Context Saved! You can now ask questions about this document in the 'Kya Karoon?' tab.")

    with tab2:
        st.header("Ask for a simple action plan")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Scared? Confused? Ask a question and get a simple 3-step plan **based on real guides.**")
        with col2:
            if st.button("Clear Chat ♻️", key="clear_chat_btn"):
                st.session_state.messages = []
                st.rerun()

        render_document_context_info(st.session_state.document_context)

        render_chat_messages(
            st.session_state.messages,
            st.session_state.document_context,
            show_sources=True,
            show_feedback=True
        )

        if prompt := st.chat_input(f"Ask your follow-up question in {language}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Your friend is checking the guides..."):
                try:
                    rag_chain = build_rag_chain()

                    chat_history_str = "\n".join(
                        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]]
                    )
                    current_doc_context = st.session_state.document_context

                    invoke_payload = {
                        "question": prompt,
                        "language": language,
                        "chat_history": chat_history_str,
                        "document_context": current_doc_context
                    }

                    response_dict = rag_chain.invoke(invoke_payload)
                    response = response_dict["answer"]
                    docs = response_dict["sources"]

                    used_document = False

                    if not docs and current_doc_context != "No document uploaded.":
                        with st.spinner("Auditing response source..."):
                            used_document = audit_response_source(
                                prompt,
                                response,
                                current_doc_context
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources_from_guides": docs,
                        "source_from_document": used_document
                    })

                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during RAG processing: {e}")

    render_disclaimer()
