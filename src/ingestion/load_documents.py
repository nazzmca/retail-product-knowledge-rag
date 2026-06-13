from pathlib import Path

DOCS_PATH = "data/documents"

documents = []

for file in Path(DOCS_PATH).glob("*.txt"):
    content = file.read_text()

    documents.append(
        {
            "filename": file.name,
            "content": content
        }
    )

print("\nDocuments Loaded")
print("=" * 50)

for doc in documents:
    print(f"\nFile: {doc['filename']}")
    print(doc["content"])

print(f"\nTotal Documents: {len(documents)}")