import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag.rag_assistant import retrieve_context


MAX_RELEVANCE_DISTANCE = 0.9
MIN_WORD_COUNT = 3


st.title("Retail Product Knowledge RAG Assistant")

st.caption(
    "Ask questions about retail policies using ChromaDB vector search and document retrieval."
)

question = st.text_input(
    "Ask a retail policy question",
    placeholder="Example: How long does express shipping take?",
    help="Ask about return policy, shipping policy, warranty, refunds, or damaged items.",
)

if question:
    cleaned_question = question.strip()

    has_alpha = any(char.isalpha() for char in cleaned_question)
    word_count = len(cleaned_question.split())

    if not has_alpha or word_count < MIN_WORD_COUNT:
        st.warning("Please enter a clear retail policy question with at least 3 words.")
        st.stop()

    try:
        with st.spinner("Searching policy documents..."):
            results = retrieve_context(cleaned_question, n_results=3)

    except Exception as error:
        st.error(f"Unable to search policy documents: {error}")
        st.stop()

    if (
        not results
        or not results.get("documents")
        or not results["documents"][0]
        or not results.get("distances")
        or not results["distances"][0]
    ):
        st.warning("No relevant policy documents were found.")
        st.stop()

    candidates = list(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    )

    relevant_candidates = [
        candidate for candidate in candidates if candidate[2] <= MAX_RELEVANCE_DISTANCE
    ]

    if not relevant_candidates:
        st.warning("I could not find a relevant retail policy answer for this question.")
        st.stop()

    document, metadata, distance = relevant_candidates[0]
    source = metadata["filename"]

    st.subheader("Answer")
    st.write(document)

    st.subheader("Source")
    st.success(f"Source Document: {source}")

    with st.expander("Retrieval details"):
        st.write(f"Similarity distance: {distance:.4f}")