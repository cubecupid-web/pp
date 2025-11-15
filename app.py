import streamlit as st
import google.generativeai as genai
import os

# --- CONFIGURATION ---
# IMPORTANT: Set your API key in Streamlit's "Secrets"
# Go to your Streamlit app's settings > Secrets and add:
# GOOGLE_API_KEY = "YOUR_KEY_HERE"

try:
    # This is the line for a deployed Streamlit Cloud app
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    # This is a fallback for local testing
    print(f"Streamlit secrets not found. Error: {e}")
    # You can uncomment the line below for a quick local test
    # but DO NOT commit your key to GitHub!
    # genai.configure(api_key="PASTE_YOUR_KEY_HERE_FOR_LOCAL_TEST_ONLY")


# --- MODEL & HELPER FUNCTION (The "Brain") ---
# We are using Gemini 1.5 Flash as you requested. It's fast and powerful.
MODEL_NAME = "gemini-2.5-flash-lite"

def get_llm_response(prompt):
    """Sends a prompt to the Gemini model and gets a response."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # A more user-friendly error
        st.error(f"An error occurred with the AI model. Please check your API key or try again. Details: {e}")
        return None

# --- THE PROMPTS (Your "Secret Sauce" - Unchanged) ---
def get_samjhao_prompt(legal_text):
    """Creates the prompt for the "Samjhao" (Explain) tab."""
    return f"""
    The following is a confusing Indian legal document or text. 
    Explain it in simple, everyday Hindi (using Roman script). 
    Do not use any legal jargon.
    Identify the 3 most important parts for the user.
    The user is a common person who is scared and confused. Be kind and reassuring.

    Legal Text: "{legal_text}"

    Simple Hindi Explanation:
    """

def get_kya_karoon_prompt(user_question):
    """Creates the prompt for the "Kya Karoon?" (What to do?) tab."""
    return f"""
    You are 'Nyay-Saathi,' a kind legal friend.
    A common Indian citizen is asking for help.
    Give a simple, 3-step action plan.
    **Do not use any legal jargon.**
    **Do not give legal advice.** Instead, guide them on *what* to do (e.g., "Find a lawyer," "File a complaint," "Gather these documents").
    Base your answer on public NALSA guidelines.
    End by strongly recommending they consult a real lawyer or NALSA.

    User's Question: "{user_question}"

    Your Simple, 3-Step Action Plan:
    """

# --- THE APP UI (The "Face" - Unchanged) ---

st.set_page_config(page_title="Nyay-Saathi", page_icon="🤝")
st.title("🤝 Nyay-Saathi (Justice Companion)")
st.markdown("Your legal friend, in your pocket. Built for India.")

# Create the two tabs
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
    st.write("Scared? Confused? Ask a question and get a simple 3-step plan.")

    user_question = st.text_input("Ask your question (e.g., 'My landlord is threatening to evict me')")
    
    if st.button("Get Plan", type="primary", key="kya_karoon_button"):
        if not user_question:
            st.warning("Please ask a question.")
        else:
            with st.spinner("Your friend is figuring out a plan..."):
                prompt = get_kya_karoon_prompt(user_question)
                response = get_llm_response(prompt)
                if response:
                    st.subheader("Here is a simple 3-step plan:")
                    st.markdown(response)

# --- DISCLAIMER (VERY IMPORTANT) ---
st.divider()
st.error("""
**Disclaimer:** I am an AI, not a lawyer. 
This is not legal advice. I am only here to help you understand things better. 
Please consult a real lawyer or contact your local **NALSA (National Legal Services Authority)** for free legal aid.
""")