import streamlit as st
from typing import Optional, List, Dict


def render_sources(
    guides_sources: Optional[List],
    doc_context_used: bool,
    document_context: str,
    show_truncation_note: bool = True
):
    if (guides_sources and len(guides_sources) > 0) or doc_context_used:
        st.subheader("Sources I used:")

        if doc_context_used:
            context_preview = document_context[:500]
            preview_length = len(document_context)
            truncated_note = (
                f"\n\n*[Showing first 500 of {preview_length} characters]*"
                if show_truncation_note and preview_length > 500
                else ""
            )
            st.warning(f"**From Your Uploaded Document:**\n\n...{context_preview}...{truncated_note}")

        if guides_sources:
            for doc in guides_sources:
                source_name = doc.metadata.get("source", "Unknown Guide")
                content_preview = doc.page_content[:300]
                preview_length = len(doc.page_content)
                truncated_note = (
                    f"\n\n*[Showing first 300 of {preview_length} characters]*"
                    if show_truncation_note and preview_length > 300
                    else ""
                )
                st.info(f"**From {source_name}:**\n\n...{content_preview}...{truncated_note}")


def render_feedback_buttons(message_index: int):
    c1, c2, _ = st.columns([1, 1, 5])
    with c1:
        if st.button("👍", key=f"feedback_{message_index}_up"):
            st.toast("Thanks for your feedback!")
    with c2:
        if st.button("👎", key=f"feedback_{message_index}_down"):
            st.toast("Thanks for your feedback!")


def render_chat_messages(
    messages: List[Dict],
    document_context: str,
    show_sources: bool = True,
    show_feedback: bool = True
):
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if show_sources and message["role"] == "assistant":
                guides_sources = message.get("sources_from_guides")
                doc_context_used = message.get("source_from_document")
                render_sources(guides_sources, doc_context_used, document_context)

            if show_feedback and message["role"] == "assistant":
                render_feedback_buttons(i)


def render_language_selector_and_buttons(col_ratio=[3, 1]):
    col1, col2 = st.columns(col_ratio)
    with col1:
        language = st.selectbox(
            "Choose your language:",
            (
                "Simple English",
                "Hindi (in Roman script)",
                "Kannada",
                "Tamil",
                "Telugu",
                "Marathi"
            ),
            key="language_selector"
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Start New Session ♻️", key="new_session_btn", type="primary"):
            st.session_state.messages = []
            st.session_state.document_context = "No document uploaded."
            st.session_state.uploaded_file_bytes = None
            st.session_state.uploaded_file_type = None
            st.session_state.samjhao_explanation = None
            st.session_state.file_uploader_key += 1
            st.rerun()

    return language


def render_document_context_info(document_context: str):
    if document_context != "No document uploaded.":
        with st.container():
            st.info("**Context Loaded:** I have your uploaded document in memory. Feel free to ask questions about it!")


def render_disclaimer():
    st.divider()
    st.error(
        """
**Disclaimer:** I am an AI, not a lawyer. This is not legal advice.
Please consult a real lawyer or contact your local **NALSA (National Legal Services Authority)** for free legal aid.
"""
    )
