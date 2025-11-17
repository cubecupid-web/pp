import streamlit as st
import pandas as pd
import os
import supabase
from datetime import datetime, timedelta

st.set_page_config(page_title="Nyay-Saathi Analytics", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root {
    --primary-color: #00FFD1;
    --background-color: #08070C;
    --secondary-background-color: #1B1C2A;
    --text-color: #FAFAFA;
}
body { font-family: 'sans serif'; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Nyay-Saathi Analytics Dashboard")

try:
    supabase_url = st.secrets.get("VITE_SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    supabase_key = st.secrets.get("VITE_SUPABASE_SUPABASE_ANON_KEY") or os.getenv("VITE_SUPABASE_SUPABASE_ANON_KEY")
    client = supabase.create_client(supabase_url, supabase_key)
except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    try:
        users_response = client.table("users").select("id").execute()
        user_count = len(users_response.data) if users_response.data else 0
        st.metric("Total Users", user_count)
    except Exception as e:
        st.metric("Total Users", "N/A")

with col2:
    try:
        messages_response = client.table("messages").select("id").execute()
        message_count = len(messages_response.data) if messages_response.data else 0
        st.metric("Total Messages", message_count)
    except Exception as e:
        st.metric("Total Messages", "N/A")

with col3:
    try:
        docs_response = client.table("documents").select("id").execute()
        doc_count = len(docs_response.data) if docs_response.data else 0
        st.metric("Documents Processed", doc_count)
    except Exception as e:
        st.metric("Documents Processed", "N/A")

st.divider()

tab1, tab2, tab3 = st.tabs(["Events", "Feedback", "Performance"])

with tab1:
    st.subheader("Recent Events")
    try:
        analytics_response = client.table("analytics").select("*").order("created_at", desc=True).limit(50).execute()
        if analytics_response.data:
            df = pd.DataFrame(analytics_response.data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No analytics data yet")
    except Exception as e:
        st.error(f"Failed to load events: {e}")

with tab2:
    st.subheader("Feedback Analysis")
    try:
        feedback_response = client.table("feedback").select("rating").execute()
        if feedback_response.data:
            ratings = [f["rating"] for f in feedback_response.data]
            positive = ratings.count("positive")
            negative = ratings.count("negative")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positive Feedback", positive)
            with col2:
                st.metric("Negative Feedback", negative)

            if positive + negative > 0:
                st.metric("Positive Rate", f"{(positive / (positive + negative) * 100):.1f}%")
        else:
            st.info("No feedback yet")
    except Exception as e:
        st.error(f"Failed to load feedback: {e}")

with tab3:
    st.subheader("Performance Metrics")
    try:
        perf_response = client.table("analytics").select("response_time_ms").execute()
        if perf_response.data:
            times = [a["response_time_ms"] for a in perf_response.data if a["response_time_ms"]]
            if times:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Response Time", f"{sum(times) / len(times):.0f}ms")
                with col2:
                    st.metric("Min Response Time", f"{min(times):.0f}ms")
                with col3:
                    st.metric("Max Response Time", f"{max(times):.0f}ms")
        else:
            st.info("No performance data yet")
    except Exception as e:
        st.error(f"Failed to load performance data: {e}")
