import os
import glob
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

CHROMA_DB_DIR = "./chroma_db"
DOCS_DIR = "./documents"
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are a helpful AI assistant with expertise in AI,
software development, and technology.

If relevant context is provided below, prioritize it in your answer.
If no context is provided or it's not relevant, answer from your own knowledge
— never say you have no information.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""


def get_llm():
    """Return local Ollama or Groq based on environment"""
    if ENVIRONMENT == "local":
        from langchain_community.llms import Ollama

        return Ollama(model="llama3.1", temperature=0.7)
    else:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="llama-3.1-8b-instant", temperature=0.7, api_key=GROQ_API_KEY
        )


def get_embeddings():
    """Return local Ollama or HuggingFace embeddings based on environment"""
    if ENVIRONMENT == "local":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model="nomic-embed-text")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # lightweight, fast, good quality
        )


class RAGPipeline:
    def __init__(self):
        # 1. LLM — local or cloud based on environment
        self.llm = get_llm()

        # 2. Embeddings — local or HuggingFace based on environment
        self.embeddings = get_embeddings()

        # 3. Vector store
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings
        )

        # 4. Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # 5. Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

        # 6. Prompt
        self.prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=SYSTEM_PROMPT,
        )

        # 7. Chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt},
        )

    def ingest(self):
        """Load docs → split → embed → store in ChromaDB"""
        all_docs = []
        for filepath in glob.glob(f"{DOCS_DIR}/**/*.txt", recursive=True):
            loader = TextLoader(filepath, encoding="utf-8")
            all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()

        return len(chunks)

    def query(self, question: str, chat_history: list = []):
        """Embed question → retrieve chunks → generate answer"""
        result = self.chain.invoke({"question": question})

        sources = list(
            set(
                [
                    doc.metadata.get("source", "unknown")
                    for doc in result.get("source_documents", [])
                ]
            )
        )

        return {"answer": result["answer"], "sources": sources}
