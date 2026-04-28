import ChatWindow from './components/ChatWindow';
import './App.css';

export default function App() {
    return (
        <div className="app">
            <header className="app-header">
                <div className="header-left">
                    <div className="logo">DS</div>
                    <div>
                        <h1>RAG Chatbot</h1>
                        <span className="subtitle">
                            LangChain · ChromaDB · Llama 3.1 · FastAPI
                        </span>
                    </div>
                </div>
                <a
                    href="https://github.com/davesarkar/rag-chatbot"
                    target="_blank"
                    rel="noreferrer"
                    className="github-link"
                >
                    GitHub ↗
                </a>
            </header>
            <ChatWindow />
        </div>
    );
}
