import { useState, useCallback, useRef, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import {
  Key, Network, Upload, ChevronDown, ChevronUp,
  FileText, Sparkles, AlertCircle,
} from 'lucide-react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface GraphNode {
  id: string
  label: string
  type: string
  x?: number
  y?: number
  fx?: number
  fy?: number
}

interface GraphEdge {
  source: string | GraphNode
  target: string | GraphNode
  label: string
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphEdge[]
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

// Node type → color mapping
const NODE_COLORS: Record<string, string> = {
  Person: '#a78bfa', // violet
  Organization: '#60a5fa', // blue
  Location: '#34d399', // green
  Concept: '#fbbf24', // amber
  Event: '#f87171', // red
  Other: '#94a3b8', // slate
}

const LEGEND_TYPES = Object.keys(NODE_COLORS)

const SAMPLE_TEXT = `Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California in 1976.
Tim Cook became CEO of Apple after Steve Jobs passed away in 2011.
Apple is headquartered in the Apple Park campus, also located in Cupertino.
The company is known for products like the iPhone, Mac, and iPad.
Steve Jobs previously co-founded Pixar Animation Studios after leaving Apple in 1985.`

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

  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)

  const graphRef = useRef<HTMLDivElement>(null)
  const [graphDimensions, setGraphDimensions] = useState({ width: 800, height: 500 })

  // Measure graph container on resize
  useEffect(() => {
    const el = graphRef.current
    if (!el) return
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        setGraphDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        })
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

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
  }

  function disconnect() {
    setConnected(false)
    setConfigOpen(true)
    setGraphData(null)
    setError('')
  }

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => setText((ev.target?.result as string) ?? '')
    reader.readAsText(file)
  }

  async function extract() {
    if (!text.trim() || !connected) return
    setLoading(true)
    setError('')
    setGraphData(null)

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

      const res = await fetch('/api/extract', {
        method: 'POST',
        headers,
        body: JSON.stringify({ text }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(body.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      // react-force-graph uses 'links' not 'edges'
      setGraphData({
        nodes: data.graph.nodes,
        links: data.graph.edges,
      })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    }

    setLoading(false)
  }

  // Node drag pin/unpin
  const handleNodeDrag = useCallback((node: GraphNode) => {
    node.fx = node.x
    node.fy = node.y
  }, [])

  const handleNodeClick = useCallback((node: GraphNode) => {
    if (node.fx !== undefined) {
      node.fx = undefined
      node.fy = undefined
    } else {
      node.fx = node.x
      node.fy = node.y
    }
  }, [])

  const selectedProvider = PROVIDERS.find(p => p.value === config.provider)!

  // ─── Render ──────────────────────────────────────────────────────────────────
  return (
    <div
      className="h-screen bg-zinc-950 text-white flex flex-col items-center overflow-hidden"
      style={{ fontFamily: 'Inter, system-ui, sans-serif' }}
    >

      {/* ── Header ── */}
      <header className="w-full max-w-5xl px-4 pt-8 pb-3 flex flex-col items-center gap-1 shrink-0">
        <div className="flex items-center gap-3">
          <div className="bg-violet-500/10 border border-violet-500/30 rounded-xl p-2.5">
            <Network className="text-violet-400" size={26} />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight leading-none">
              UNI<span className="text-violet-400">KEY</span>-AI
              <span className="text-zinc-500 font-light mx-2">·</span>
              <span className="text-violet-300 font-semibold">Knowledge Graph</span>
            </h1>
            <p className="text-zinc-500 text-xs mt-0.5">
              Bring Your Own Key · Extract entities & relationships from any document
            </p>
          </div>
        </div>
      </header>

      {/* ── Main Layout ── */}
      <div className="w-full max-w-5xl px-4 flex-1 min-h-0 flex gap-3 pb-6">

        {/* ── Left Sidebar ── */}
        <div className="w-80 shrink-0 flex flex-col gap-2 overflow-y-auto">

          {/* Config Panel */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden shrink-0">
            <div
              role="button"
              tabIndex={0}
              onClick={() => setConfigOpen(o => !o)}
              onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') setConfigOpen(o => !o) }}
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-zinc-800/50 transition-colors cursor-pointer select-none"
            >
              <div className="flex items-center gap-2.5">
                <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-zinc-600'}`} />
                <span className="text-sm font-semibold text-zinc-200">
                  {connected ? `${selectedProvider.label} · ${config.model}` : 'Connect API key'}
                </span>
                {connected && (
                  <span className="text-[10px] font-bold text-green-500/80 bg-green-500/10 border border-green-500/20 rounded-full px-2 py-0.5 uppercase tracking-widest">
                    Live
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
                {configOpen ? <ChevronUp size={15} className="text-zinc-500" /> : <ChevronDown size={15} className="text-zinc-500" />}
              </div>
            </div>

            {configOpen && (
              <div className="slide-down px-4 pb-4 flex flex-col gap-3 border-t border-zinc-800 pt-4">
                {/* Provider + Model */}
                <div className="flex flex-col gap-1.5">
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
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Model</label>
                  <input
                    type="text"
                    value={config.model}
                    onChange={e => setConfig(c => ({ ...c, model: e.target.value }))}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-xl px-3 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500/60 transition-colors"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">API Key</label>
                  <input
                    type="password"
                    value={config.apiKey}
                    onChange={e => setConfig(c => ({ ...c, apiKey: e.target.value }))}
                    placeholder={selectedProvider.needsBaseUrl ? 'dummy' : 'sk-… / gsk_…'}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-xl px-3 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500/60 transition-colors font-mono"
                  />
                </div>
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
                <button
                  onClick={connect}
                  disabled={!config.apiKey.trim() || !config.model.trim()}
                  className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-800 disabled:text-zinc-600 text-white font-semibold text-sm transition-all"
                >
                  <Key size={14} />
                  Connect
                </button>
              </div>
            )}
          </div>

          {/* Document Input */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 flex flex-col gap-3 flex-1 min-h-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText size={14} className="text-violet-400" />
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Document</span>
              </div>
              <label className="cursor-pointer flex items-center gap-1.5 text-[11px] text-zinc-500 hover:text-violet-400 transition-colors bg-zinc-800 hover:bg-zinc-700 rounded-lg px-2.5 py-1.5">
                <Upload size={11} />
                Upload file
                <input type="file" accept=".txt,.md,.csv" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>

            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder={`Paste text here… or try the sample:\n\n${SAMPLE_TEXT.slice(0, 80)}…`}
              className="flex-1 w-full bg-zinc-950 border border-zinc-700 rounded-xl px-3 py-3 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-violet-500/60 transition-colors resize-none leading-relaxed min-h-[160px]"
            />

            <div className="flex gap-2">
              <button
                onClick={() => setText(SAMPLE_TEXT)}
                className="flex-1 py-2 rounded-xl text-xs text-zinc-500 hover:text-zinc-300 bg-zinc-800/50 hover:bg-zinc-800 border border-zinc-700/50 transition-all"
              >
                Load sample
              </button>
              <button
                onClick={extract}
                disabled={!connected || !text.trim() || loading}
                className="flex-1 flex items-center justify-center gap-2 py-2 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-800 disabled:text-zinc-600 text-white font-semibold text-sm transition-all"
              >
                {loading
                  ? <><span className="animate-spin text-xs">◐</span> Extracting…</>
                  : <><Sparkles size={14} /> Extract</>
                }
              </button>
            </div>

            {/* Stats */}
            {graphData && (
              <div className="fade-in flex gap-2">
                <div className="flex-1 bg-zinc-800/60 rounded-xl px-3 py-2 text-center">
                  <p className="text-xl font-black text-violet-400">{graphData.nodes.length}</p>
                  <p className="text-[10px] text-zinc-500 uppercase tracking-widest">Entities</p>
                </div>
                <div className="flex-1 bg-zinc-800/60 rounded-xl px-3 py-2 text-center">
                  <p className="text-xl font-black text-violet-400">{graphData.links.length}</p>
                  <p className="text-[10px] text-zinc-500 uppercase tracking-widest">Relations</p>
                </div>
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl px-4 py-3 shrink-0">
            <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-2">Legend</p>
            <div className="flex flex-col gap-1.5">
              {LEGEND_TYPES.map(type => (
                <div key={type} className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: NODE_COLORS[type] }} />
                  <span className="text-xs text-zinc-400">{type}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Graph Canvas ── */}
        <div className="flex-1 min-w-0 flex flex-col gap-2">

          {/* Error Banner */}
          {error && (
            <div className="shrink-0 flex items-start gap-2.5 bg-red-950/50 border border-red-800/60 rounded-xl px-4 py-3 text-red-300 text-xs font-mono">
              <AlertCircle size={14} className="shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Graph area */}
          <div
            ref={graphRef}
            className="flex-1 min-h-0 bg-zinc-900/50 border border-zinc-800 rounded-2xl overflow-hidden relative graph-canvas"
          >

            {graphData ? (
              <>
                {/* Hover tooltip */}
                {hoveredNode && (
                  <div className="absolute top-3 left-3 z-10 bg-zinc-900 border border-zinc-700 rounded-xl px-3 py-2 text-xs pointer-events-none fade-in">
                    <p className="font-bold text-white">{hoveredNode.label}</p>
                    <p className="flex items-center gap-1.5 mt-0.5">
                      <span className="w-2 h-2 rounded-full inline-block" style={{ background: NODE_COLORS[hoveredNode.type] ?? '#94a3b8' }} />
                      <span className="text-zinc-400">{hoveredNode.type}</span>
                    </p>
                  </div>
                )}
                <p className="absolute bottom-3 right-3 text-[10px] text-zinc-700 z-10">
                  Drag nodes · Click to pin/unpin
                </p>

                <ForceGraph2D
                  graphData={graphData}
                  width={graphDimensions.width}
                  height={graphDimensions.height}
                  backgroundColor="#09090b"
                  nodeLabel=""
                  nodeCanvasObject={(node: GraphNode, ctx, globalScale) => {
                    const radius = 6
                    const color = NODE_COLORS[node.type] ?? '#94a3b8'
                    // Circle
                    ctx.beginPath()
                    ctx.arc(node.x ?? 0, node.y ?? 0, radius, 0, 2 * Math.PI)
                    ctx.fillStyle = color + '33'
                    ctx.fill()
                    ctx.strokeStyle = color
                    ctx.lineWidth = 1.5
                    ctx.stroke()

                    // Label
                    const label = node.label
                    const fontSize = Math.max(10 / globalScale, 3)
                    ctx.font = `600 ${fontSize}px Inter, sans-serif`
                    ctx.textAlign = 'center'
                    ctx.textBaseline = 'middle'
                    ctx.fillStyle = '#f4f4f5'
                    ctx.fillText(label, node.x ?? 0, (node.y ?? 0) + radius + fontSize * 0.8)
                  }}
                  nodePointerAreaPaint={(node: GraphNode, color, ctx) => {
                    ctx.fillStyle = color
                    ctx.beginPath()
                    ctx.arc(node.x ?? 0, node.y ?? 0, 10, 0, 2 * Math.PI)
                    ctx.fill()
                  }}
                  linkColor={() => '#52525b'}
                  linkWidth={1.2}
                  linkDirectionalArrowLength={4}
                  linkDirectionalArrowRelPos={1}
                  linkLabel={(link: GraphEdge) => link.label}
                  linkCanvasObjectMode={() => 'after'}
                  linkCanvasObject={(link: GraphEdge, ctx, globalScale) => {
                    const source = link.source as GraphNode
                    const target = link.target as GraphNode
                    if (!source.x || !source.y || !target.x || !target.y) return

                    const midX = (source.x + target.x) / 2
                    const midY = (source.y + target.y) / 2
                    const fontSize = Math.max(8 / globalScale, 2.5)

                    ctx.font = `${fontSize}px Inter, sans-serif`
                    ctx.textAlign = 'center'
                    ctx.textBaseline = 'middle'
                    ctx.fillStyle = '#a1a1aa'
                    ctx.fillText(link.label, midX, midY)
                  }}
                  onNodeHover={(node) => setHoveredNode(node as GraphNode | null)}
                  onNodeDrag={handleNodeDrag}
                  onNodeClick={handleNodeClick}
                  cooldownTicks={120}
                  d3AlphaDecay={0.02}
                  d3VelocityDecay={0.3}
                />
              </>
            ) : (
              /* ── Empty State ── */
              <div className="h-full flex flex-col items-center justify-center gap-4 text-zinc-700 select-none p-8">
                <Network size={64} strokeWidth={0.8} className="opacity-20" />
                <div className="text-center max-w-xs">
                  <p className="font-semibold text-zinc-500">Your knowledge graph will appear here</p>
                  <p className="text-xs mt-2 text-zinc-600 leading-relaxed">
                    {connected
                      ? 'Paste or upload a document on the left, then click Extract to build the graph.'
                      : 'Connect your API key first, then paste a document and click Extract.'}
                  </p>
                </div>

                {/* 3-step hint */}
                <div className="flex items-center gap-3 text-[11px] text-zinc-700 mt-2">
                  <div className="flex items-center gap-1.5">
                    <span className="w-5 h-5 rounded-full border border-zinc-700 flex items-center justify-center text-[10px] font-bold">1</span>
                    Connect key
                  </div>
                  <span>→</span>
                  <div className="flex items-center gap-1.5">
                    <span className="w-5 h-5 rounded-full border border-zinc-700 flex items-center justify-center text-[10px] font-bold">2</span>
                    Paste text
                  </div>
                  <span>→</span>
                  <div className="flex items-center gap-1.5">
                    <span className="w-5 h-5 rounded-full border border-zinc-700 flex items-center justify-center text-[10px] font-bold">3</span>
                    Extract graph
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <p className="text-center text-[10px] text-zinc-700 shrink-0">
            Powered by <span className="text-violet-600 font-semibold">UniKey-AI</span>
            {' '}· Your key is used directly — the developer pays nothing
          </p>
        </div>
      </div>
    </div>
  )
}
