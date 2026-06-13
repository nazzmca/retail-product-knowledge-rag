import streamlit as st
import sys
sys.path.insert(0, "../")

from src.rag.rag_assistant import retrieve_context, generate_answer

st.title("Retail Product Knowledge Assistant")

question = st.text_input(
    "Ask a retail policy question"
)

if question:
    if len(question.strip()) < 3:
        st.warning("Please enter a valid question.")
        st.stop()

    try:
        results = retrieve_context(question, n_results=3)
        
        if not results["documents"] or not results["documents"][0]:
            st.warning("No relevant documents found. Please try a different question.")
            st.stop()
        
        answer = generate_answer(question, results)
        
        st.subheader("Answer")
        st.write(answer)
        
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")