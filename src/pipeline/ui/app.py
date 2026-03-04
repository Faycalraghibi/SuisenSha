import os

import pandas as pd
import requests
import streamlit as st

# Basic config
st.set_page_config(
    page_title="SuisenSha — Movie Recommendations",
    page_icon="🎬",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def fetch_history(user_id: int):
    try:
        resp = requests.get(f"{API_URL}/users/{user_id}/history", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return None


def fetch_embedding_recs(user_id: int):
    try:
        resp = requests.get(f"{API_URL}/recommend/embedding/{user_id}", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return None


def fetch_sasrec_recs(user_id: int):
    try:
        resp = requests.get(f"{API_URL}/recommend/sasrec/{user_id}", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return None


def fetch_rag_explanation(user_id: int):
    try:
        resp = requests.get(f"{API_URL}/recommend/rag/{user_id}", timeout=120)
        return resp.json() if resp.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return None


# --- UI Layout ---

st.title("🎬 SuisenSha Recommendations")
st.markdown(
    "Compare **Embedding** vs **Transformer (SASRec)** models, "
    "and generate **AI Explanations (RAG)**."
)

# Sidebar
st.sidebar.header("User Selection")
# In a real app we'd fetch the list of users from the API,
# for now we'll hardcode some sample user IDs from MovieLens 100K evaluation set.
user_id = st.sidebar.number_input("User ID", min_value=1, max_value=943, value=1, step=1)

if st.sidebar.button("Fetch Details"):
    st.session_state["user_id"] = user_id
    st.session_state["rag_explanation"] = None  # Clear previous RAG

current_user = st.session_state.get("user_id", None)

if current_user:
    st.header(f"User {current_user} Profile")

    # 1. History
    with st.expander("👁️ View Watch History", expanded=True):
        history_data = fetch_history(current_user)
        if history_data:
            df_hist = pd.DataFrame(history_data["recent_history"])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.error("Failed to fetch user history.")

    st.divider()
    st.header("Top 10 Recommendations")

    # 2. Side-by-side models
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("FAISS Baseline (Item Embeddings)")
        emb_data = fetch_embedding_recs(current_user)
        if emb_data:
            df_emb = pd.DataFrame(emb_data["recommendations"])
            st.dataframe(df_emb, use_container_width=True, hide_index=True)
        else:
            st.error("Failed to fetch Embedding recommendations.")

    with col2:
        st.subheader("SASRec (Sequential Transformer)")
        sas_data = fetch_sasrec_recs(current_user)
        if sas_data:
            df_sas = pd.DataFrame(sas_data["recommendations"])
            st.dataframe(df_sas, use_container_width=True, hide_index=True)
        else:
            st.error("Failed to fetch SASRec recommendations.")

    st.divider()

    # 3. RAG Explanation
    st.header("🤖 AI Concierge (RAG)")
    st.markdown(
        "Use a generative language model to explain "
        "the recommendations based on the user's history."
    )

    if st.button("Generate Explanation", type="primary"):
        with st.spinner(
            "Generating natural language explanation... (this may take a moment on CPU)"
        ):
            rag_data = fetch_rag_explanation(current_user)
            if rag_data:
                st.session_state["rag_explanation"] = rag_data["rationale"]
            else:
                st.error("Failed to generate RAG explanation.")

    if st.session_state.get("rag_explanation"):
        st.info(st.session_state["rag_explanation"])

else:
    st.info("👈 Enter a User ID in the sidebar and click 'Fetch Details' to begin.")
