import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './App.css';
import SourcesPanel from './components/SourcesPanel';

const BACKEND_URL = 'http://localhost:2024'; // Unified backend (LangGraph + custom routes)
const LANGGRAPH_URL = BACKEND_URL;
const UPLOAD_URL = BACKEND_URL;

// Helper function to parse SearchResults from Python repr format
const parseSearchResults = (content) => {
  // Format: "Returning structured response: search_result_message=\"...\" times_semantic_search_tool_called=X sources=[...]"

  if (!content.includes('search_result_message=')) {
    return null;
  }

  try {
    // Extract search_result_message - handle both single and double quotes with multiline support
    let message = '';
    let quoteChar = "'";
    let startIdx = content.indexOf("search_result_message='");

    if (startIdx === -1) {
      startIdx = content.indexOf('search_result_message="');
      quoteChar = '"';
    }

    if (startIdx !== -1) {
      startIdx += ('search_result_message=' + quoteChar).length;
      let endIdx = startIdx;
      let escaped = false;

      // Find the matching closing quote, handling escapes
      while (endIdx < content.length) {
        const char = content[endIdx];
        if (escaped) {
          escaped = false;
        } else if (char === '\\') {
          escaped = true;
        } else if (char === quoteChar) {
          // Found unescaped closing quote
          break;
        }
        endIdx++;
      }

      // Handle Python string escape sequences
      // We need to handle \\\\ first to preserve LaTeX commands like \\times
      let rawMessage = content.substring(startIdx, endIdx);

      // Step 1: Replace escaped backslashes (\\) with a placeholder to protect LaTeX
      rawMessage = rawMessage.replace(/\\\\/g, '\x00BACKSLASH\x00');

      // Step 2: Now safely replace Python escape sequences
      rawMessage = rawMessage
        .replace(/\\n/g, '\n')
        .replace(/\\t/g, '\t')
        .replace(/\\r/g, '\r')
        .replace(/\\'/g, "'")
        .replace(/\\"/g, '"');

      // Step 3: Restore the backslashes for LaTeX
      message = rawMessage.replace(/\x00BACKSLASH\x00/g, '\\');
    }

    // Extract times_semantic_search_tool_called
    const timesMatch = content.match(/times_semantic_search_tool_called=(\d+)/);
    const times = timesMatch ? parseInt(timesMatch[1], 10) : 0;

    // Extract sources - handle both list formats with proper quote-aware parsing
    const sourcesMatch = content.match(/sources=\[([^\]]*)\]/);
    let sources = [];
    if (sourcesMatch && sourcesMatch[1].trim()) {
      // Parse the Python list format: ['item1', 'item2'] or ["item1", "item2"]
      const sourcesStr = sourcesMatch[1];

      // Split on commas only when outside of quotes
      const items = [];
      let currentItem = '';
      let inQuote = false;
      let quoteChar = null;

      for (let i = 0; i < sourcesStr.length; i++) {
        const char = sourcesStr[i];

        if ((char === '"' || char === "'") && (i === 0 || sourcesStr[i-1] !== '\\')) {
          if (!inQuote) {
            inQuote = true;
            quoteChar = char;
          } else if (char === quoteChar) {
            inQuote = false;
            quoteChar = null;
          }
        } else if (char === ',' && !inQuote) {
          if (currentItem.trim()) {
            items.push(currentItem.trim());
          }
          currentItem = '';
          continue;
        }

        currentItem += char;
      }

      if (currentItem.trim()) {
        items.push(currentItem.trim());
      }

      sources = items
        .map(s => s.replace(/^['"]|['"]$/g, '').trim())
        .filter(s => s.length > 0);

      // Combine filename + page pairs into single entries
      const combinedSources = [];
      for (let i = 0; i < sources.length; i++) {
        const current = sources[i];
        const next = sources[i + 1];

        // If current is a filename and next is "page X", combine them
        if (current.endsWith('.pdf') && next && next.match(/^page\s+\d+$/i)) {
          combinedSources.push(`${current}, ${next}`);
          i++; // Skip the next item since we combined it
        } else {
          combinedSources.push(current);
        }
      }

      sources = combinedSources;
    }

    console.log('[parseSearchResults] Parsed message length:', message.length, 'times:', times, 'sources:', sources);

    return {
      search_result_message: message,
      times_semantic_search_tool_called: times,
      sources: sources
    };
  } catch (e) {
    console.error('[parseSearchResults] Parse error:', e);
    return null;
  }
};

// Helper function to extract and parse content from agent response
const extractContent = (content) => {
  console.log('[extractContent] Input type:', typeof content, 'Value:', content);

  // If already a SearchResults object
  if (content && typeof content === 'object' && !Array.isArray(content) && content.search_result_message !== undefined) {
    console.log('[extractContent] Already a SearchResults object');
    return content;
  }

  let rawContent = '';

  if (typeof content === 'string') {
    rawContent = content;
  } else if (Array.isArray(content)) {
    // Handle array of content blocks
    rawContent = content.map(c => {
      if (typeof c === 'string') return c;
      if (c.text) return c.text;
      if (typeof c === 'object') return JSON.stringify(c);
      return '';
    }).join('');
  } else if (content && content.text) {
    rawContent = content.text;
  } else if (content && typeof content === 'object') {
    // Might be the SearchResults object already
    rawContent = JSON.stringify(content);
  }

  console.log('[extractContent] Raw content:', rawContent.substring(0, 200));

  // Try to parse as SearchResults from Python repr format
  if (rawContent.includes('search_result_message=')) {
    const parsed = parseSearchResults(rawContent);
    if (parsed) {
      console.log('[extractContent] Successfully parsed SearchResults from Python repr');
      return parsed;
    }
  }

  // Try to parse as SearchResults JSON
  if (rawContent.trim()) {
    try {
      const parsed = JSON.parse(rawContent);
      if (parsed.search_result_message !== undefined) {
        console.log('[extractContent] Successfully parsed SearchResults from JSON');
        return parsed;
      }
    } catch (e) {
      console.log('[extractContent] Not valid JSON:', e.message);
    }
  }

  return rawContent;
};

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState(null);
  const [sourcesRefreshTrigger, setSourcesRefreshTrigger] = useState(0);
  const [selectedSources, setSelectedSources] = useState([]);
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

  // Toggle source selection
  const toggleSourceSelection = (source) => {
    setSelectedSources((prev) => {
      if (prev.includes(source)) {
        return prev.filter((s) => s !== source);
      } else {
        return [...prev, source];
      }
    });
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Append source filter if sources are selected
    let messageContent = input;
    if (selectedSources.length > 0) {
      const sourcesList = selectedSources.map(s => `"${s}"`).join(', ');
      messageContent = `${input}\n\n[Filter by sources: ${sourcesList}]`;
    }

    const userMessage = { role: 'user', content: input }; // Show original input to user
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
            messages: [{ role: 'user', content: messageContent }] // Use modified content with source filter
          },
          stream_mode: ['messages']
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to reach LangGraph server');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        console.log('[extractContent] Input type:', typeof chunk, 'Value:', chunk);

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              // Handle messages stream mode: data is [message, metadata] tuple
              if (Array.isArray(data) && data.length >= 1) {
                const msg = data[0];
                // Handle AI messages, assistant messages, and tool messages (SearchResults)
                if (msg && (msg.type === 'ai' || msg.role === 'assistant' || msg.type === 'tool')) {
                  const extracted = extractContent(msg.content);
                  if (extracted) {
                    assistantContent = extracted;
                    console.log('[Chat] Extracted content from', msg.type || msg.role, ':', typeof extracted === 'object' ? 'SearchResults object' : 'string');
                  }
                }
              }
              // Handle values stream mode: data has messages array
              else if (data.messages && data.messages.length > 0) {
                const lastMsg = data.messages[data.messages.length - 1];
                // Handle AI messages, assistant messages, and tool messages (SearchResults)
                if (lastMsg.type === 'ai' || lastMsg.role === 'assistant' || lastMsg.type === 'tool') {
                  const extracted = extractContent(lastMsg.content);
                  if (extracted) {
                    assistantContent = extracted;
                    console.log('[Chat] Extracted content from', lastMsg.type || lastMsg.role, ':', typeof extracted === 'object' ? 'SearchResults object' : 'string');
                  }
                }
              }
            } catch (parseErr) {
              // Skip non-JSON lines
              console.log('[Chat] Parse error:', parseErr.message);
            }
          }
        }
      }

      console.log('[Chat] Final assistantContent:', assistantContent);

      if (assistantContent && (typeof assistantContent === 'object' || assistantContent.trim() !== '')) {
        setMessages((prev) => [...prev, { role: 'assistant', content: assistantContent }]);
      } else {
        console.warn('[Chat] No valid content received');
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
        setSourcesRefreshTrigger(prev => prev + 1);
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

      <div className="main-layout">
        <div className="chat-container">
        {threadId && <div className="thread-indicator">Thread: {threadId.slice(0, 8)}...</div>}
        <div className="messages">
          {messages.length === 0 && <div className="empty-state">Ask me anything about your books!</div>}
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.role === 'assistant' ? (
                  typeof msg.content === 'object' && msg.content.search_result_message ? (
                    // Structured SearchResults format
                    <div className="search-results">
                      <div className="search-message">
                        <ReactMarkdown
                          remarkPlugins={[remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {msg.content.search_result_message}
                        </ReactMarkdown>
                      </div>
                      {msg.content.sources && msg.content.sources.length > 0 && (
                        <div className="sources-section">
                          <strong>Sources:</strong>
                          <div className="sources-list">
                            {msg.content.sources.map((source, i) => (
                              <div key={i} className="source-item">
                                <span className="source-index">[{i + 1}]</span> {source}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="search-stats">
                        <small>Semantic searches performed: {msg.content.times_semantic_search_tool_called}</small>
                      </div>
                    </div>
                  ) : (
                    // Regular string content
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}
                    </ReactMarkdown>
                  )
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
          {selectedSources.length > 0 && (
            <div className="selected-sources-indicator">
              <span className="indicator-label">Filtering by:</span>
              <div className="selected-sources-tags">
                {selectedSources.map((source, idx) => (
                  <span key={idx} className="source-tag">
                    {source}
                    <button
                      type="button"
                      className="remove-tag-btn"
                      onClick={() => toggleSourceSelection(source)}
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}
          <div className="input-row">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !input.trim()}>Send</button>
          </div>
        </form>
        </div>

        <SourcesPanel
          refreshTrigger={sourcesRefreshTrigger}
          selectedSources={selectedSources}
          onToggleSource={toggleSourceSelection}
        />
      </div>
    </div>
  );
}

export default App;
