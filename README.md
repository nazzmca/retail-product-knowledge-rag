# Retail Product Knowledge RAG

Retail knowledge assistant built using embeddings, ChromaDB vector search, semantic retrieval, and Retrieval-Augmented Generation (RAG).

## Project Overview

This project demonstrates a simple end-to-end RAG pipeline using retail policy documents.

The application:

* Loads retail knowledge documents
* Creates vector embeddings
* Stores embeddings in ChromaDB
* Performs semantic similarity search
* Retrieves relevant context
* Generates source-based answers

---

## Architecture

```text
Retail Documents
        ↓
Document Loader
        ↓
Sentence Transformer Embeddings
        ↓
ChromaDB Vector Database
        ↓
Semantic Search
        ↓
Context Retrieval
        ↓
RAG Answer Generation
```

---

## Knowledge Base Documents

Current knowledge base contains:

* Return Policy
* Shipping Policy
* Warranty Policy

Location:

```text
data/documents
```

---

## Technology Stack

### AI / RAG

* Sentence Transformers
* ChromaDB
* Embeddings
* Semantic Search

### Development

* Python
* Git
* GitHub

---

## Project Structure

```text
retail-product-knowledge-rag
│
├── data
│   └── documents
│
├── src
│   ├── ingestion
│   ├── retrieval
│   └── rag
│
├── tests
│
├── README.md
└── requirements.txt
```

---

## Example Question

```text
How long does express shipping take?
```

## Example Answer

```text
Express shipping takes 1 to 2 business days.

Source:
shipping_policy.txt
```

---

## Current Features

### Completed

* Document ingestion
* Embedding generation
* ChromaDB vector store
* Semantic similarity search
* Source-based retrieval
* RAG answer generation

### Planned Enhancements

* Interactive question input
* PDF document ingestion
* LLM integration
* OpenAI support
* Web UI
* API endpoints
* Evaluation metrics

---

## Author

Nazmul Laskar

Senior Data Engineer | Platform Engineer | AI & Data Engineering

GitHub: https://github.com/nazzmca
