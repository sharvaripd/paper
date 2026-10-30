# 🚀 Paper Knowledge Assistant - Astronomy Research RAG

A **Retrieval-Augmented Generation (RAG)** application built to search, process, and query academic astronomy research papers. This tool utilizes language models and vector search to answer hyper-specific astrophysical questions accurately without hallucinations.

## 🛠️ Tech Stack & Architecture
* **Frontend UI:** Streamlit (Dynamic, easy-to-use web interface)
* **LLM Orchestration:** LangChain / LangChain Community
* **Document Processing:** PyPDFLoader & RecursiveCharacterTextTextSplitter
* **Embeddings & Vector Database:** HuggingFace Embeddings & FAISS Vector Store

## 📂 Features
* **PDF Ingestion:** Automatically reads and chunks dense multi-page academic papers.
* **Vector Semantic Search:** Embeds texts and cross-references queries to retrieve exact contextual snippets.
* **Astronomy Q&A:** Delivers accurate answers based strictly on the uploaded scientific papers.
