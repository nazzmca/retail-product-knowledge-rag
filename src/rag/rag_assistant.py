import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "retail_knowledge_base"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve_context(question: str, n_results: int = 1):
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    return results


def generate_answer(question: str, results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)
    sources = [metadata["filename"] for metadata in metadatas]

    answer = f"""
Question:
{question}

Answer:
Based on the retrieved retail policy documents:

{context}

Sources:
{", ".join(sources)}
"""

    return answer


if __name__ == "__main__":
    question = "How long does express shipping take?"

    results = retrieve_context(question)
    answer = generate_answer(question, results)

    print(answer)