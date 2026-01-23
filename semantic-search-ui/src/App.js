import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const LANGGRAPH_URL = 'http://localhost:2024';
const UPLOAD_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Create a new thread
  const createThread = async () => {
    try {
      const response = await fetch(`${LANGGRAPH_URL}/threads`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error('Failed to create thread');
      }

      const data = await response.json();
      console.log('[Thread] Created new thread:', data.thread_id);
      return data.thread_id;
    } catch (error) {
      console.error('[Thread] Error creating thread:', error);
      return null;
    }
  };

  // Start a new conversation
  const startNewChat = async () => {
    const hadPreviousChat = messages.length > 0 || threadId;
    setMessages([]);
    setThreadId(null);
    console.log('[Chat] Started new conversation');

    if (hadPreviousChat) {
      setMessages([{ role: 'system', content: 'Started a new conversation. Previous chat history cleared.' }]);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Create thread if we don't have one
      let currentThreadId = threadId;
      if (!currentThreadId) {
        currentThreadId = await createThread();
        if (currentThreadId) {
          setThreadId(currentThreadId);
        }
      }

      // Use thread endpoint if we have a thread, otherwise fall back to stateless
      const endpoint = currentThreadId
        ? `${LANGGRAPH_URL}/threads/${currentThreadId}/runs/stream`
        : `${LANGGRAPH_URL}/runs/stream`;

      console.log('[Chat] Sending message to:', endpoint);
      console.log('[Chat] Thread ID:', currentThreadId || 'stateless');

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assistant_id: 'agent',
          input: {
            messages: [{ role: 'user', content: input }]
          },
          stream_mode: ['messages']
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to reach LangGraph server');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              // Handle messages stream mode: data is [message, metadata] tuple
              if (Array.isArray(data) && data.length >= 1) {
                const msg = data[0];
                if (msg && (msg.type === 'ai' || msg.role === 'assistant')) {
                  const content = msg.content;
                  if (typeof content === 'string') {
                    assistantContent = content;
                  } else if (Array.isArray(content)) {
                    assistantContent = content.map(c => typeof c === 'string' ? c : (c.text || '')).join('');
                  } else if (content && content.text) {
                    assistantContent = content.text;
                  }
                }
              }
              // Handle values stream mode: data has messages array
              else if (data.messages && data.messages.length > 0) {
                const lastMsg = data.messages[data.messages.length - 1];
                if (lastMsg.type === 'ai' || lastMsg.role === 'assistant') {
                  const content = lastMsg.content;
                  if (typeof content === 'string') {
                    assistantContent = content;
                  } else if (Array.isArray(content)) {
                    assistantContent = content.map(c => typeof c === 'string' ? c : (c.text || '')).join('');
                  } else if (content && content.text) {
                    assistantContent = content.text;
                  }
                }
              }
            } catch (parseErr) {
              // Skip non-JSON lines
            }
          }
        }
      }

      if (assistantContent) {
        setMessages((prev) => [...prev, { role: 'assistant', content: assistantContent }]);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', content: 'No response received.' }]);
      }
    } catch (error) {
      console.error('[Chat] Error:', error);
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Error: Could not reach the server.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    console.log('[Upload] Starting upload for:', file.name, 'Size:', file.size);

    const formData = new FormData();
    formData.append('file', file);

    const uploadUrl = `${UPLOAD_URL}/upload`;
    console.log('[Upload] POST to:', uploadUrl);

    setMessages((prev) => [...prev, { role: 'system', content: `Uploading and vectorizing "${file.name}"...` }]);

    try {
      const response = await fetch(uploadUrl, {
        method: 'POST',
        body: formData,
      });

      console.log('[Upload] Response status:', response.status, response.statusText);

      if (response.ok) {
        const data = await response.json();
        console.log('[Upload] Success response:', data);
        setMessages((prev) => [...prev, { role: 'system', content: `"${file.name}" uploaded and vectorized: ${data.pages} pages, ${data.chunks} chunks stored.` }]);
      } else {
        const errorText = await response.text();
        console.log('[Upload] Error response body:', errorText);
        let error = {};
        try { error = JSON.parse(errorText); } catch(e) {}
        setMessages((prev) => [...prev, { role: 'system', content: `Failed to upload "${file.name}": ${error.detail || response.statusText || 'Unknown error'}` }]);
      }
    } catch (error) {
      console.error('[Upload] Fetch error:', error);
      setMessages((prev) => [...prev, { role: 'system', content: 'Error: Could not connect to server.' }]);
    }
    fileInputRef.current.value = '';
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Semantic Search Agent</h1>
        <div className="header-buttons">
          <button className="new-chat-btn" onClick={startNewChat}>New Chat</button>
          <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept=".pdf" style={{ display: 'none' }} />
          <button className="upload-btn" onClick={() => fileInputRef.current.click()}>Upload Book</button>
        </div>
      </header>

      <div className="chat-container">
        {threadId && <div className="thread-indicator">Thread: {threadId.slice(0, 8)}...</div>}
        <div className="messages">
          {messages.length === 0 && <div className="empty-state">Ask me anything about your books!</div>}
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.role === 'assistant' ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
            </div>
          ))}
          {isLoading && <div className="message assistant"><div className="message-content">Thinking...</div></div>}
          <div ref={messagesEndRef} />
        </div>

        <form className="input-form" onSubmit={sendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;
