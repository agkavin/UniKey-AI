import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Send, Key, StopCircle, Trash2, Copy, Check, ChevronDown, ChevronUp, Zap } from 'lucide-react'

// ─── Types ───────────────────────────────────────────────────────────────────

interface Message {
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
}

interface ConnectionConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl: string
}

// ─── Constants ────────────────────────────────────────────────────────────────

const PROVIDERS = [
  { value: 'groq', label: 'Groq', defaultModel: 'llama-3.3-70b-versatile', needsBaseUrl: false },
  { value: 'openai', label: 'OpenAI', defaultModel: 'gpt-4o-mini', needsBaseUrl: false },
  { value: 'anthropic', label: 'Anthropic', defaultModel: 'claude-3-5-haiku-20241022', needsBaseUrl: false },
  { value: 'gemini', label: 'Google Gemini', defaultModel: 'gemini-1.5-flash', needsBaseUrl: false },
  { value: 'mistral', label: 'Mistral', defaultModel: 'mistral-large-latest', needsBaseUrl: false },
  { value: 'cohere', label: 'Cohere', defaultModel: 'command-r-plus', needsBaseUrl: false },
  { value: 'ollama', label: 'Ollama (Local)', defaultModel: 'llama3.2:3b', needsBaseUrl: true },
  { value: 'openai-compatible', label: 'OpenAI-Compatible', defaultModel: 'local-model', needsBaseUrl: true },
]

const BACKEND_URL = '/api'  // proxied to :8000 via vite.config.ts

// ─── Helper Components ────────────────────────────────────────────────────────

function TypingDots() {
  return (
    <span className="flex items-center gap-1 py-1">
      <span className="dot-1 inline-block w-2 h-2 rounded-full bg-violet-400" />
      <span className="dot-2 inline-block w-2 h-2 rounded-full bg-violet-400" />
      <span className="dot-3 inline-block w-2 h-2 rounded-full bg-violet-400" />
    </span>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000) }}
      className={`shrink-0 p-1.5 rounded-lg border transition-all flex items-center justify-center ${copied
        ? 'bg-green-500/20 border-green-500/50 text-green-400'
        : 'bg-zinc-800 border-zinc-700 hover:border-zinc-500 text-zinc-400'}`}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [config, setConfig] = useState<ConnectionConfig>({
    provider: 'groq',
    apiKey: '',
    model: 'llama-3.3-70b-versatile',
    baseUrl: 'http://localhost:11434',
  })
  const [connected, setConnected] = useState(false)
  const [configOpen, setConfigOpen] = useState(true)

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const abortRef = useRef<AbortController | null>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [input])

  // When provider changes, update model to its default
  function handleProviderChange(value: string) {
    const p = PROVIDERS.find(p => p.value === value)!
    setConfig(c => ({ ...c, provider: value, model: p.defaultModel }))
  }

  function connect() {
    if (!config.apiKey.trim() || !config.model.trim()) {
      setError('API key and model are required.')
      return
    }
    setError('')
    setConnected(true)
    setConfigOpen(false)
    setMessages([])
  }

  function disconnect() {
    if (loading) abortRef.current?.abort()
    setConnected(false)
    setConfigOpen(true)
    setMessages([])
    setError('')
  }

  async function sendMessage() {
    const userInput = input.trim()
    if (!userInput || !connected || loading) return

    setInput('')
    setError('')

    const userMsg: Message = { role: 'user', content: userInput }
    const assistantMsg: Message = { role: 'assistant', content: '', streaming: true }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setLoading(true)

    abortRef.current = new AbortController()

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'x-ai-provider': config.provider,
        'x-ai-key': config.apiKey,
        'x-ai-model': config.model,
      }

      const selectedProvider = PROVIDERS.find(p => p.value === config.provider)
      if (selectedProvider?.needsBaseUrl && config.baseUrl) {
        headers['x-ai-base-url'] = config.baseUrl
      }

      const res = await fetch(`${BACKEND_URL}/chat/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ prompt: userInput }),
        signal: abortRef.current.signal,
      })

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(errBody.detail || `HTTP ${res.status}`)
      }

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        accumulated += chunk
        const snap = accumulated
        setMessages(prev => {
          const next = [...prev]
          next[next.length - 1] = { role: 'assistant', content: snap, streaming: true }
          return next
        })
      }

      // Mark streaming complete
      setMessages(prev => {
        const next = [...prev]
        next[next.length - 1] = { role: 'assistant', content: accumulated, streaming: false }
        return next
      })

    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      if (msg.toLowerCase().includes('abort')) {
        // User stopped — freeze current content
        setMessages(prev => {
          const next = [...prev]
          const last = next[next.length - 1]
          next[next.length - 1] = { ...last, streaming: false }
          return next
        })
      } else {
        setMessages(prev => {
          const next = [...prev]
          next[next.length - 1] = { role: 'assistant', content: `⚠️ ${msg}` }
          return next
        })
        setError(msg)
      }
    }

    setLoading(false)
    abortRef.current = null
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const selectedProvider = PROVIDERS.find(p => p.value === config.provider)!

  // ─── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="h-screen bg-zinc-950 text-white flex flex-col items-center overflow-hidden" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>

      {/* ── Header ── */}
      <header className="w-full max-w-3xl px-4 pt-10 pb-4 flex flex-col items-center gap-1 shrink-0">
        <div className="flex items-center gap-3">
          <div className="bg-violet-500/10 border border-violet-500/30 rounded-xl p-2.5">
            <Key className="text-violet-400" size={26} />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight leading-none">
              UNI<span className="text-violet-400">KEY</span>-AI
            </h1>
            <p className="text-zinc-500 text-xs mt-0.5">
              Bring Your Own Key · Any Provider · Zero Developer Cost
            </p>
          </div>
        </div>
      </header>

      {/* ── Connection Config Panel ── */}
      <div className="w-full max-w-3xl px-4 mb-2 shrink-0">
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden">

          {/* Panel header / toggle */}
          <button
            onClick={() => setConfigOpen(o => !o)}
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-zinc-800/50 transition-colors"
          >
            <div className="flex items-center gap-2.5">
              <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-zinc-600'}`} />
              <span className="text-sm font-semibold text-zinc-200">
                {connected
                  ? `${selectedProvider.label} · ${config.model}`
                  : 'Connect your API key'}
              </span>
              {connected && (
                <span className="text-[10px] font-bold text-green-500/80 bg-green-500/10 border border-green-500/20 rounded-full px-2 py-0.5 uppercase tracking-widest">
                  Connected
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {connected && (
                <button
                  onClick={e => { e.stopPropagation(); disconnect() }}
                  className="text-[11px] text-zinc-500 hover:text-red-400 transition-colors px-2 py-1 rounded-lg hover:bg-red-500/10"
                >
                  Disconnect
                </button>
              )}
              {configOpen ? <ChevronUp size={16} className="text-zinc-500" /> : <ChevronDown size={16} className="text-zinc-500" />}
            </div>
          </button>

          {/* Config fields */}
          {configOpen && (
            <div className="slide-down px-4 pb-4 flex flex-col gap-3 border-t border-zinc-800 pt-4">

              {/* Provider + Model row */}
              <div className="flex gap-3">
                <div className="flex-1 flex flex-col gap-1.5">
                  <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Provider</label>
                  <select
                    value={config.provider}
                    onChange={e => handleProviderChange(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-violet-500/60 transition-colors cursor-pointer"
                  >
                    {PROVIDERS.map(p => (
                      <option key={p.value} value={p.value} className="bg-zinc-900">{p.label}</option>
                    ))}
                  </select>
                </div>
                <div className="flex-1 flex flex-col gap-1.5">
                  <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Model</label>
                  <input
                    type="text"
                    value={config.model}
                    onChange={e => setConfig(c => ({ ...c, model: e.target.value }))}
                    placeholder={selectedProvider.defaultModel}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-xl px-3 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500/60 transition-colors"
                  />
                </div>
              </div>

              {/* API Key */}
              <div className="flex flex-col gap-1.5">
                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">API Key</label>
                <input
                  type="password"
                  value={config.apiKey}
                  onChange={e => setConfig(c => ({ ...c, apiKey: e.target.value }))}
                  placeholder={selectedProvider.needsBaseUrl ? 'dummy (not required for local)' : 'sk-... / gsk_...'}
                  className="w-full bg-zinc-950 border border-zinc-700 rounded-xl px-3 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500/60 transition-colors font-mono"
                />
              </div>

              {/* Base URL — only shown for local providers */}
              {selectedProvider.needsBaseUrl && (
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Base URL</label>
                  <input
                    type="text"
                    value={config.baseUrl}
                    onChange={e => setConfig(c => ({ ...c, baseUrl: e.target.value }))}
                    placeholder="http://localhost:11434"
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-xl px-3 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500/60 transition-colors font-mono"
                  />
                </div>
              )}

              {/* Connect button */}
              <button
                onClick={connect}
                disabled={!config.apiKey.trim() || !config.model.trim()}
                className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-800 disabled:text-zinc-600 text-white font-semibold text-sm transition-all"
              >
                <Zap size={15} />
                Connect
              </button>

              {/* Info line */}
              <p className="text-center text-[10px] text-zinc-600">
                Your key is sent directly with each request and is never stored on the server.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── Error Banner ── */}
      {error && (
        <div className="w-full max-w-3xl px-4 mb-2 shrink-0">
          <div className="bg-red-950/50 border border-red-800/60 rounded-xl px-4 py-2.5 text-red-300 text-xs font-mono">
            {error}
          </div>
        </div>
      )}

      {/* ── Chat Area ── */}
      <div className="w-full max-w-3xl px-4 flex-1 min-h-0 mb-4">
        <div
          className="chat-scroll h-full bg-zinc-900/50 border border-zinc-800 rounded-2xl p-4 overflow-y-auto shadow-2xl"
        >
          {messages.length === 0 ? (
            /* ── Empty State ── */
            <div className="h-full flex flex-col items-center justify-center gap-4 text-zinc-700 select-none">
              <Key size={52} strokeWidth={1} className="opacity-20" />
              <div className="text-center">
                {connected ? (
                  <>
                    <p className="font-medium text-sm text-zinc-500">Your key, your model, your chat</p>
                    <p className="text-xs mt-1 text-zinc-600">
                      Connected to <span className="text-violet-400">{selectedProvider.label}</span> · {config.model}
                    </p>
                  </>
                ) : (
                  <>
                    <p className="font-medium text-sm text-zinc-500">Connect your API key to start</p>
                    <p className="text-xs mt-1">Pick a provider, paste your key, and chat. Zero cost to the developer.</p>
                  </>
                )}
              </div>

              {/* Supported providers pill list */}
              {!connected && (
                <div className="flex flex-wrap gap-2 justify-center max-w-sm mt-1">
                  {PROVIDERS.map(p => (
                    <span key={p.value} className="text-[10px] font-bold text-violet-400/70 bg-violet-500/10 border border-violet-500/20 rounded-full px-2.5 py-1">
                      {p.label}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ) : (
            /* ── Messages ── */
            <div className="flex flex-col gap-5">
              {messages.map((msg, i) => (
                <div key={i} className={`msg-in flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>

                  {/* Assistant avatar */}
                  {msg.role === 'assistant' && (
                    <div className="shrink-0 w-7 h-7 bg-violet-500/15 border border-violet-500/30 rounded-lg flex items-center justify-center mt-0.5">
                      <Key size={13} className="text-violet-400" />
                    </div>
                  )}

                  {/* Bubble */}
                  <div className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${msg.role === 'user'
                    ? 'bg-violet-600/20 border border-violet-500/30 text-violet-50 rounded-br-sm'
                    : 'bg-zinc-800/70 border border-zinc-700/50 text-zinc-100 rounded-bl-sm'
                    }`}>
                    {msg.role === 'assistant' ? (
                      msg.content === '' && msg.streaming ? (
                        <TypingDots />
                      ) : (
                        <div className={`prose prose-sm prose-invert prose-dark max-w-none ${msg.streaming ? 'cursor-blink' : ''}`}>
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                        </div>
                      )
                    ) : (
                      <span className="whitespace-pre-wrap">{msg.content}</span>
                    )}
                  </div>

                  {/* User avatar + copy */}
                  {msg.role === 'user' && (
                    <div className="flex flex-col items-center gap-1">
                      <div className="shrink-0 w-7 h-7 bg-zinc-700/60 border border-zinc-600/40 rounded-lg flex items-center justify-center text-xs font-bold text-zinc-400">
                        U
                      </div>
                      <CopyButton text={msg.content} />
                    </div>
                  )}
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* ── Input Bar ── */}
      <div className="w-full max-w-3xl px-4 pb-8 shrink-0">
        <div className={`flex gap-2 items-end bg-zinc-900 border rounded-2xl px-3 py-2 transition-all duration-200 ${connected
          ? 'border-zinc-700 focus-within:border-violet-500/60 focus-within:shadow-[0_0_0_3px_rgba(139,92,246,0.08)]'
          : 'border-zinc-800 opacity-50 pointer-events-none'}`
        }>
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              title="Clear chat"
              className="shrink-0 p-2 hover:bg-zinc-800 rounded-xl transition-colors text-zinc-600 hover:text-red-400"
            >
              <Trash2 size={16} />
            </button>
          )}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!connected}
            rows={1}
            className="flex-1 bg-transparent resize-none focus:outline-none text-sm leading-relaxed py-1.5 placeholder-zinc-600 disabled:opacity-40"
            placeholder={connected ? `Message ${config.model} …  (Shift+Enter for newline)` : 'Connect your API key first…'}
            style={{ maxHeight: '160px' }}
          />
          {loading ? (
            <button
              onClick={() => abortRef.current?.abort()}
              title="Stop generation"
              className="shrink-0 p-2 rounded-xl bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 transition-all text-red-400"
            >
              <StopCircle size={18} />
            </button>
          ) : (
            <button
              onClick={sendMessage}
              disabled={!input.trim() || !connected}
              title="Send (Enter)"
              className="shrink-0 p-2 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-800 disabled:text-zinc-600 transition-all text-white"
            >
              <Send size={18} />
            </button>
          )}
        </div>
        <p className="text-center text-[10px] text-zinc-700 mt-2">
          Powered by{' '}
          <span className="text-violet-600 font-semibold">UniKey-AI</span>
          {' '}· Your key is used directly — the developer pays nothing
        </p>
      </div>

    </div>
  )
}
