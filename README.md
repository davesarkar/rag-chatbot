# RAG Chatbot

A conversational AI assistant powered by a LangChain RAG (Retrieval-Augmented Generation) pipeline. Built by [Devashish Sarkar](https://linkedin.com/in/dave-sarkar).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Pipeline | LangChain, RAG |
| Vector Store | ChromaDB |
| LLM | Ollama (Llama 3.1) |
| Embeddings | Ollama (nomic-embed-text) |
| Backend | Python, FastAPI |
| Frontend | React, TypeScript, Vite |

## How It Works

```
User Question
     ↓
[Embed Question] → vector representation (nomic-embed-text via Ollama)
     ↓
[ChromaDB] → find top 4 similar document chunks (similarity search)
     ↓
[LangChain Chain] → combine chunks + history + question into prompt
     ↓
[Llama 3.1 via Ollama] → generate grounded response
     ↓
Answer + Sources returned to user
```

## Getting Started

### Prerequisites
- [Ollama](https://ollama.com/) installed and running locally
- Node.js & npm
- Python 3.10+

### 1. Pull required Ollama models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 2. Clone the repo

```bash
git clone https://github.com/davesarkar/rag-chatbot.git
cd rag-chatbot
```

### 3. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Open the app

```
http://localhost:5173
```

### 6. Add documents and ingest (optional)

Drop `.txt` files into `backend/documents/`, then run:

```bash
curl -X POST http://localhost:8000/ingest
```

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py                  # FastAPI routes
│   ├── rag_pipeline_ollama.py   # LangChain RAG logic (Ollama)
│   ├── rag_pipeline.py          # LangChain RAG logic (OpenAI variant)
│   ├── documents/               # Drop .txt files here for ingestion
│   ├── chroma_db/               # Persisted vector store
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   │       └── ChatWindow.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/chat` | Send message, get AI response |
| POST | `/ingest` | Ingest documents into vector store |

## Author

**Devashish Sarkar** — Full-Stack & AI Engineer  
[LinkedIn](https://linkedin.com/in/dave-sarkar) · [GitHub](https://github.com/davesarkar)
