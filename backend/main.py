from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from rag_pipeline_ollama import RAGPipeline

app = FastAPI(title="RAG Chatbot API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline once when server starts
rag = RAGPipeline()


# --- Request/Response Models ---
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []


# --- Routes ---
@app.get("/")
def health_check():
    return {"status": "RAG Chatbot API is running"}


import logging

logger = logging.getLogger(__name__)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = rag.query(
            question=request.message,
            chat_history=[(m.role, m.content) for m in request.history],
        )
        return ChatResponse(reply=result["answer"], sources=result.get("sources", []))
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def ingest_documents():
    try:
        count = rag.ingest()
        return {"status": f"Ingested {count} chunks successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
