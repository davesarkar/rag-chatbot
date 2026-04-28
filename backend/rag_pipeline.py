import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_DIR = "./chroma_db"
DOCS_DIR = "./documents"

SYSTEM_PROMPT = """You are a helpful AI assistant. Use the following retrieved 
context to answer the user's question accurately. If you don't know the answer 
from the context, say so honestly rather than making something up.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self):
        # 1. LLM — the brain
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY
        )

        # 2. Embeddings — converts text to vectors
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # 3. Vector store — stores and retrieves document chunks
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings
        )

        # 4. Retriever — finds relevant chunks from vector store
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # 5. Memory — keeps conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

        # 6. Prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=SYSTEM_PROMPT,
        )

        # 7. Chain — ties everything together
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt},
        )

    def ingest(self):
        """Load docs → split → embed → store in ChromaDB"""
        loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()

        return len(chunks)

    def query(self, question: str, chat_history: list = []):
        """Embed question → retrieve chunks → generate answer"""
        result = self.chain({"question": question})

        sources = list(
            set(
                [
                    doc.metadata.get("source", "unknown")
                    for doc in result.get("source_documents", [])
                ]
            )
        )

        return {"answer": result["answer"], "sources": sources}
