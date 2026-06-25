# Enterprise Document Intelligence Engine: Production-Grade RAG Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110.0-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector--v17-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Orchestrated-blue.svg)](https://www.docker.com/)

An enterprise-ready, fully containerized Retrieval-Augmented Generation (RAG) backend engine designed to ingest unstructured documents, perform deterministic semantic chunking, and execute context-aware synthesis using Google's Gemini 2.5 Flash and PostgreSQL (`pgvector`). 

This architecture is optimized for low-latency retrieval, zero-temperature factual generation, and mathematically rigorous vector space management.

---

## 🏗️ Architectural Topology & System Design

The system decouples the data-ingestion pipeline from the real-time inference loop, ensuring independent horizontal scalability for both database writes and LLM reads.

[ Client / UI ] 
          │
          ▼  (Port 8000)
┌──────────────────────────────────┐
│       FastAPI Ingestion API      │◀─── Async Event Loop
└──────────────────────────────────┘
   │                        │
   │ (Text Chunking)        │ (Query Embeddings)
   ▼                        ▼
┌──────────────┐        ┌────────────────┐
│  LangChain   │        │ Google Gemini  │ (gemini-embedding-001)
│ Engine Core  │        │ Embedding API  │ ───► Outputs 768-Dim Vectors
└──────────────┘        └────────────────┘
│                        │
└───────────┬────────────┘
▼ (Internal Docker Network)
┌──────────────────────────────────┐
│     PostgreSQL + pgvector        │◀─── Exposed via Port 5433
│   (HNSW Inverted Index Layer)    │
└──────────────────────────────────┘

### Core Engineering Features
* **Asynchronous Execution (`FastAPI`):** Non-blocking I/O operations utilize Python's native event loop to process file transformations without degrading server performance during concurrent client connections.
* **Deterministic Hashing & Idempotency:** Implements an MD5 hashing layer (`hashlib.md5`) matching structural payloads (Filename + Sequence Text). If duplicate documents or chunks are re-submitted, the system enforces a database-level upsert restriction, mitigating vector space pollution and redundant compute costs.
* **Network Isolation via Docker Compose:** The application stack is isolated across an internal bridge network. The API communicates with the database natively via internal port `5432`, while the external host port is mapped to `5433` to completely prevent conflicts with local database installations.

---

## 🧮 Mathematical Rigor & Vector Space Optimization

Rather than using basic keyword mapping, this engine relies on geometric distance metrics in high-dimensional space. Text chunks are mapped using `gemini-embedding-001` into a 768-dimensional vector space ($\mathbb{R}^{768}$).

### Semantic Match Calculation
The retrieval mechanism isolates highly relevant context by calculating the **Cosine Similarity** between the user query vector $\mathbf{Q}$ and the document chunk vectors $\mathbf{D}$:

$$\text{Cosine Similarity} = \frac{\mathbf{Q} \cdot \mathbf{D}}{\|\mathbf{Q}\| \|\mathbf{D}\|} = \frac{\sum_{i=1}^{768} Q_i D_i}{\sqrt{\sum_{i=1}^{768} Q_i^2} \sqrt{\sum_{i=1}^{768} D_i^2}}$$

By measuring the cosine of the angle between vectors rather than their absolute Euclidean magnitude, the engine evaluates semantic relevance independent of text length variations.

### Inverted Indexing with HNSW
To guarantee sub-linear $O(\log N)$ query scaling across millions of rows, the database utilizes a **Hierarchical Navigable Small World (HNSW)** index over the vector column, constructing a multi-layer graph that avoids the expensive computation of exhaustive flat searches ($O(N)$).

---

## 🛠️ Technology Stack & Dependencies

* **Runtime:** Python 3.12+
* **Framework:** FastAPI (Uvicorn ASGI Server)
* **AI Orchestration:** LangChain Core & LangChain Community
* **LLM Architecture:** Google GenAI (`gemini-2.5-flash`, `gemini-embedding-001`)
* **Vector Storage:** PostgreSQL 17 with `pgvector` extension
* **Infrastructure:** Docker & Docker Compose

---

## 📦 Local Deployment & Configuration

### 1. Repository Initialization
```bash
git clone [https://github.com/02atanu/rag-document-intelligence.git](https://github.com/02atanu/rag-document-intelligence.git)
cd rag-document-intelligence
