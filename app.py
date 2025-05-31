#!/usr/bin/env python3

import streamlit as st
from utils.vectorstore_utils import create_index_from_file
from rag.simple_rag import get_simple_rag_response
from rag.corrective_rag import get_corrective_rag_response
from rag.self_rag import get_self_rag_response
from rag.fusion_rag import get_fusion_rag_response
import tempfile
import os

# Set page config
st.set_page_config(page_title="RAG Explorer", layout="centered")

# App title
st.title("🔍 RAG Models Explorer")

# Sidebar - Document Upload Section
st.sidebar.header("📂 Upload Your Document")
uploaded_file = st.sidebar.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

# Process uploaded file
if uploaded_file:
    # Save to a temporary file
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        temp_file_path = tmpfile.name

    # Rebuild FAISS index
    with st.spinner("🔄 Processing and indexing document..."):
        create_index_from_file(temp_file_path)
    st.sidebar.success("✅ Document indexed successfully!")

# Model selection dropdown
model_choice = st.selectbox(
    "🧠 Choose a RAG model:",
    ("Simple RAG", "Corrective RAG", "Self-RAG", "Fusion RAG")
)

# User input query
user_query = st.text_input("❓ Enter your question:")

# Answer button
if st.button("🤖 Get Answer"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🧠 Thinking..."):
            if model_choice == "Simple RAG":
