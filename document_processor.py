import streamlit as st
import google.generativeai as genai
import json
from PIL import Image
import io
from config import MODEL_NAME, MAX_IMAGE_SIZE_MB, IMAGE_COMPRESSION_QUALITY


def compress_image(image_bytes, max_size_mb=MAX_IMAGE_SIZE_MB, quality=IMAGE_COMPRESSION_QUALITY):
    try:
        image = Image.open(io.BytesIO(image_bytes))

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        compressed_bytes = output.getvalue()

        size_mb = len(compressed_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            quality = int(quality * 0.8)
            return compress_image(image_bytes, max_size_mb, quality)

        return compressed_bytes
    except Exception as e:
        st.error(f"Error compressing image: {e}")
        return image_bytes


def extract_and_explain_document(file_bytes, file_type, language):
    try:
        model = genai.GenerativeModel(MODEL_NAME)

        prompt_text = f"""
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

        data_part = {"mime_type": file_type, "data": file_bytes}
        response = model.generate_content([prompt_text, data_part])
        clean_response_text = response.text.strip().replace("```json", "").replace("```", "")

        try:
            response_json = json.loads(clean_response_text)
            return response_json.get("explanation"), response_json.get("raw_text")
        except json.JSONDecodeError:
            st.error("AI response was not in valid JSON format. Please try again.")
            return None, None

    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None, None


def audit_response_source(prompt, response, document_context, model_name=MODEL_NAME):
    try:
        audit_model = genai.GenerativeModel(model_name)
        audit_prompt = f"""
You are an auditor.
Question: "{prompt}"
Answer: "{response}"
Context: "{document_context}"

Did the "Answer" come *primarily* from the "Context"?
Respond with ONLY the word 'YES' or 'NO'.
"""
        audit_response = audit_model.generate_content(audit_prompt)
        return "YES" in audit_response.text.upper()
    except Exception as e:
        st.warning(f"Could not audit response source: {e}")
        return False
