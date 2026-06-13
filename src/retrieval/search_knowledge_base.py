import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "retail_knowledge_base"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

query = "How long does shipping take?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nUser Question:")
print(query)

print("\nTop Retrieved Documents")
print("=" * 50)

for document, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"\nSource: {metadata['filename']}")
    print(f"Distance: {distance}")
    print(document)