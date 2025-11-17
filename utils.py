import logging
import time
import supabase
import streamlit as st
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

def get_supabase_client(supabase_url: str, supabase_key: str) -> Optional[supabase.Client]:
    try:
        return supabase.create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.warning(f"Supabase connection failed: {e}")
        return None

def sanitize_input(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > 5000:
        text = text[:5000]
    return text

def get_or_create_user(supabase_client: Optional[supabase.Client], session_id: str) -> Optional[str]:
    if supabase_client:
        try:
            response = supabase_client.table("users").insert({
                "session_id": session_id,
                "language": "Simple English"
            }).execute()
            return response.data[0]["id"] if response.data else None
        except Exception as e:
            logger.warning(f"Failed to create user record: {e}")
    return None

def get_or_create_conversation(supabase_client: Optional[supabase.Client], db_user_id: str) -> Optional[str]:
    if supabase_client and db_user_id:
        try:
            response = supabase_client.table("conversations").insert({
                "user_id": db_user_id,
                "title": "New Conversation"
            }).execute()
            return response.data[0]["id"] if response.data else None
        except Exception as e:
            logger.warning(f"Failed to create conversation: {e}")
    return None

def save_message_to_db(
    supabase_client: Optional[supabase.Client],
    db_conversation_id: str,
    role: str,
    content: str,
    sources: List[Any],
    used_document: bool
) -> None:
    if supabase_client and db_conversation_id:
        try:
            sources_list = [doc.metadata.get('source', 'Unknown') for doc in sources] if sources else []
            supabase_client.table("messages").insert({
                "conversation_id": db_conversation_id,
                "role": role,
                "content": content,
                "sources_from_guides": sources_list,
                "source_from_document": used_document
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save message: {e}")

def save_document_to_db(
    supabase_client: Optional[supabase.Client],
    db_user_id: str,
    db_conversation_id: Optional[str],
    file_name: str,
    file_type: str,
    raw_text: str,
    explanation: str,
    language: str
) -> Optional[str]:
    if supabase_client and db_user_id:
        try:
            doc_response = supabase_client.table("documents").insert({
                "user_id": db_user_id,
                "conversation_id": db_conversation_id,
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

def save_feedback_to_db(
    supabase_client: Optional[supabase.Client],
    db_user_id: str,
    db_conversation_id: str,
    message_index: int,
    rating: str
) -> None:
    if supabase_client and db_user_id and db_conversation_id:
        try:
            messages = supabase_client.table("messages").select("id").eq("conversation_id", db_conversation_id).execute()
            if messages.data and len(messages.data) > message_index:
                message_id = messages.data[message_index]["id"]
                supabase_client.table("feedback").insert({
                    "message_id": message_id,
                    "user_id": db_user_id,
                    "rating": rating
                }).execute()
        except Exception as e:
            logger.warning(f"Failed to save feedback: {e}")

def log_analytics(
    supabase_client: Optional[supabase.Client],
    db_user_id: str,
    event_type: str,
    response_time_ms: Optional[int] = None,
    api_used: Optional[str] = None
) -> None:
    if supabase_client and db_user_id:
        try:
            supabase_client.table("analytics").insert({
                "user_id": db_user_id,
                "event_type": event_type,
                "response_time_ms": response_time_ms,
                "api_used": api_used
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to log analytics: {e}")

def retry_with_backoff(func: callable, max_retries: int = 3, base_delay: int = 1) -> Any:
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay ** attempt)
            else:
                raise e
