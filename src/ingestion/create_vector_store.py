from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/documents"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "retail_knowledge_base"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

documents = []
ids = []
metadatas = []

for index, file in enumerate(Path(DOCS_PATH).glob("*.txt")):
    content = file.read_text()

    documents.append(content)
    ids.append(f"doc_{index}")
    metadatas.append({"filename": file.name})

embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print("\nVector store created successfully")
print("=" * 50)
print(f"Documents embedded: {len(documents)}")
print(f"Collection name: {COLLECTION_NAME}")
print(f"Chroma path: {CHROMA_PATH}")