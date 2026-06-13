import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag.rag_assistant import retrieve_context


st.title("Retail Product Knowledge RAG Assistant")

st.caption(
    "Ask questions about retail policies using ChromaDB vector search and document retrieval."
)

question = st.text_input("Ask a retail policy question")

if question:
    cleaned_question = question.strip()

    if len(cleaned_question) < 8 or cleaned_question.isdigit():
        st.warning("Please enter a proper retail policy question.")
        st.stop()

    results = retrieve_context(cleaned_question, n_results=1)

    distance = results["distances"][0][0]

    if distance > 0.9:
        st.warning("I could not find a relevant retail policy answer for this question.")
        st.stop()

    document = results["documents"][0][0]
    source = results["metadatas"][0][0]["filename"]

    st.subheader("Answer")
    st.write(document)

    st.subheader("Source")
    st.success(f"Source Document: {source}")

    with st.expander("Retrieval details"):
        st.write(f"Similarity distance: {distance:.4f}")