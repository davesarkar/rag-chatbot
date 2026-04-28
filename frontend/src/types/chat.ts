export interface Message {
    role: 'user' | 'assistant';
    content: string;
    sources?: string[];
}

export interface ChatRequest {
    message: string;
    history: Message[];
}

export interface ChatResponse {
    reply: string;
    sources: string[];
}
