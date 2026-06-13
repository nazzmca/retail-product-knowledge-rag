import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "retail_knowledge_base"

@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return model, collection


model, collection = load_resources()

st.title("Retail Product Knowledge Assistant")

question = st.text_input(
    "Ask a retail policy question"
)

if question:

    query_embedding = model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    document = results["documents"][0][0]
    source = results["metadatas"][0][0]["filename"]

    st.subheader("Answer")

    st.write(document)

    st.subheader("Source")

    st.write(source)