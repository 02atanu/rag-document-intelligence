# RAG Document Intelligence API

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110.0-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector--v17-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Orchestrated-blue.svg)](https://www.docker.com/)

A containerized Retrieval-Augmented Generation (RAG) backend. This system ingests PDF documents, chunks the text, stores the embeddings in PostgreSQL using `pgvector`, and uses Google's Gemini 2.5 Flash to answer questions based strictly on the uploaded context.

The architecture is built for speed and data hygiene, utilizing asynchronous endpoints and deterministic hashing to prevent database bloat.

---

## ⚡ Core Features

* **Async FastAPI Backend:** File parsing and vector math are handled without blocking the main event loop, allowing the API to handle concurrent requests smoothly.
* **Idempotent Data Ingestion:** Uses MD5 hashing on document names and text chunks. If you upload the same PDF twice, the database ignores the duplicate, saving compute costs and preventing vector space pollution.
* **HNSW Vector Indexing:** Uses a Hierarchical Navigable Small World (HNSW) index in Postgres. This keeps semantic search times incredibly fast ($O(\log N)$) even if the database scales to millions of document chunks.
* **Fully Dockerized:** The API and Vector Database run in isolated containers on an internal bridge network, meaning it works out-of-the-box on any machine without local dependency conflicts.

---

## 🏗️ System Architecture

```mermaid
graph TD
    %% Define Styles
    classDef client fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#333;
    classDef api fill:#005f73,stroke:#fff,stroke-width:2px,color:#fff;
    classDef core fill:#0a9396,stroke:#fff,stroke-width:2px,color:#fff;
    classDef db fill:#94d2bd,stroke:#333,stroke-width:2px,color:#333;

    %% Nodes Configuration 
    Client["Client / UI Workspace"] ::: client
    API["FastAPI Ingestion API <br>Async Event Loop"] ::: api
    LC["LangChain Engine Core <br>Text Chunking Pipeline"] ::: core
    Gemini["Google Gemini API <br>gemini-embedding-001"] ::: core
    DB[("PostgreSQL + pgvector <br>HNSW Inverted Index Layer")] ::: db

    %% Structural Relationships
    Client -->|"Ingress Port 8000"| API
    
    subgraph DockerNetwork ["Isolated Docker Bridge Network"]
        API -->|"1. Document Payload"| LC
        API -->|"2. Semantic Query"| Gemini
        LC -->|"3. Normalized Chunks"| DB
        Gemini -->|"4. 768-Dim Vector Arrays"| DB
    end

    DB -.->|"Egress Mapping Port 5433"| Client
```

---

## 🧮 How the Vector Search Works (Under the Hood)

Instead of relying on simple keyword matching (like a standard SQL `LIKE` query), this engine maps text chunks into a 768-dimensional mathematical space using `gemini-embedding-001`. 

When a user asks a question, the system finds the most relevant document chunks by calculating the **Cosine Similarity** between the question's vector (`Q`) and the document's vector (`D`):

```math
\text{Cosine Similarity} = \frac{\mathbf{Q} \cdot \mathbf{D}}{\|\mathbf{Q}\| \|\mathbf{D}\|} = \frac{\sum_{i=1}^{768} Q_i D_i}{\sqrt{\sum_{i=1}^{768} Q_i^2} \sqrt{\sum_{i=1}^{768} D_i^2}}
```

Because it measures the angle between vectors rather than the absolute distance, the search engine accurately matches meaning regardless of how long or short the text chunks are.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Framework:** FastAPI (Uvicorn ASGI)
* **LLM & Embeddings:** Google Gemini 2.5 Flash / `gemini-embedding-001`
* **RAG Orchestration:** LangChain
* **Database:** PostgreSQL 17 + `pgvector`
* **Infrastructure:** Docker & Docker Compose

---

## 🚀 Local Setup & Deployment

### 1. Clone the repository
```bash
git clone [https://github.com/02atanu/rag-document-intelligence.git](https://github.com/02atanu/rag-document-intelligence.git)
cd rag-document-intelligence
```

### 2. Add your API Key
Create a `.env` file in the root directory. Add your database credentials and Google AI Studio API key:
```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

GEMINI_API_KEY=your_actual_api_key_here
```
*(Note: `.env` is included in `.gitignore` to keep your keys secure).*

### 3. Boot up the Docker containers
```bash
docker compose up --build -d
```
The API will be live at `http://localhost:8000`. You can visit `http://localhost:8000/docs` to use the interactive Swagger UI to upload PDFs and query your documents.

### 4. Database Introspection (Optional)
If you want to view the raw vector embeddings inside your database, connect your preferred SQL client (like DBeaver or VS Code SQLTools) to `localhost:5433` using the credentials in your `.env` file.
