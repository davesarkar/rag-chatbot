import { useState, useRef, useEffect } from 'react';
import Message from './Message';
import type { Message as MessageType } from '../types/chat';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SUGGESTIONS = [
    'What is a RAG pipeline?',
    'How does ChromaDB store embeddings?',
    'What are LangChain agents?',
    'Explain chunking in RAG',
];

export default function ChatWindow() {
    const [messages, setMessages] = useState<MessageType[]>([
        {
            role: 'assistant',
            content:
                "Hi! I'm a RAG-powered assistant built with LangChain and ChromaDB running on a local Llama 3.1 model. Ask me anything!",
        },
    ]);
    const [input, setInput] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    // Auto scroll to latest message
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    const sendMessage = async (text?: string) => {
        const question = text || input.trim();
        if (!question || loading) return;

        const userMessage: MessageType = { role: 'user', content: question };
        const updatedMessages = [...messages, userMessage];

        setMessages(updatedMessages);
        setInput('');
        setLoading(true);

        try {
            const response = await fetch(`${API_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: question,
                    history: updatedMessages.slice(1, -1),
                }),
            });

            const data = await response.json();

            const assistantMessage: MessageType = {
                role: 'assistant',
                content: data.reply,
                sources: data.sources || [],
            };

            setMessages((prev) => [...prev, assistantMessage]);
        } catch {
            const errorMessage: MessageType = {
                role: 'assistant',
                content: 'Could not connect to backend. Is the server running?',
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="chat-window">
            {/* Messages area */}
            <div className="messages">
                {messages.map((msg, i) => (
                    <Message key={i} message={msg} />
                ))}

                {/* Typing indicator */}
                {loading && (
                    <div className="message assistant">
                        <div className="avatar">AI</div>
                        <div className="bubble typing">
                            <span />
                            <span />
                            <span />
                        </div>
                    </div>
                )}

                {/* Suggestions — only shown at start */}
                {messages.length === 1 && !loading && (
                    <div className="suggestions">
                        {SUGGESTIONS.map((s) => (
                            <button
                                key={s}
                                className="suggestion"
                                onClick={() => sendMessage(s)}
                            >
                                {s}
                            </button>
                        ))}
                    </div>
                )}

                <div ref={bottomRef} />
            </div>

            {/* Input area */}
            <div className="input-row">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Ask something about RAG, LangChain or AI..."
                    disabled={loading}
                />
                <button
                    onClick={() => sendMessage()}
                    disabled={loading || !input.trim()}
                >
                    {loading ? '...' : 'Send'}
                </button>
            </div>
        </div>
    );
}
