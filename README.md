# RAG Chatbot

A conversational AI assistant powered by a LangChain RAG (Retrieval-Augmented Generation) pipeline. Built by [Devashish Sarkar](https://linkedin.com/in/dave-sarkar).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Pipeline | LangChain, RAG, LLM Agents |
| Vector Store | ChromaDB |
| LLM | OpenAI GPT-3.5-turbo |
| Backend | Python, FastAPI |
| Frontend | React.js, Vite |
| Containerization | Docker, Docker Compose |

## How It Works

```
User Question
     ↓
[Embed Question] → vector representation
     ↓
[ChromaDB] → find top 4 similar document chunks (similarity search)
     ↓
[LangChain Chain] → combine chunks + history + question into prompt
     ↓
[OpenAI LLM] → generate grounded response
     ↓
Answer + Sources returned to user
```

## Getting Started

### Prerequisites
- Docker & Docker Compose
- OpenAI API key

### Run Locally

1. Clone the repo:
```bash
git clone https://github.com/davesarkar/rag-chatbot.git
cd rag-chatbot
```

2. Set up environment:
```bash
cp backend/.env.example backend/.env
# Add your OPENAI_API_KEY to backend/.env
```

3. Add documents to ingest (optional):
```bash
# Drop .txt files into backend/documents/
```

4. Start everything with Docker:
```bash
docker-compose up --build
```

5. Open http://localhost:3000

6. Ingest documents (if you added any):
```bash
curl -X POST http://localhost:8000/ingest
```

### Run Without Docker

**Backend:**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # add your API key
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py           # FastAPI routes
│   ├── rag_pipeline.py   # LangChain RAG logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       └── ChatWindow.jsx
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
