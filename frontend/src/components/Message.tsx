import type { Message as MessageType } from '../types/chat';

interface Props {
    message: MessageType;
}

export default function Message({ message }: Props) {
    const isUser = message.role === 'user';

    return (
        <div className={`message ${isUser ? 'user' : 'assistant'}`}>
            <div className="avatar">{isUser ? 'You' : 'AI'}</div>
            <div className="bubble-wrap">
                <div className="bubble">{message.content}</div>
                {message.sources && message.sources.length > 0 && (
                    <div className="sources">
                        Sources: {message.sources.join(', ')}
                    </div>
                )}
            </div>
        </div>
    );
}
