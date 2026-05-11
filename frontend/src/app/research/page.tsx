'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Search, Zap, ChevronRight, RotateCcw, Download, Clock, Database, CheckCircle2, AlertCircle, Loader2, ArrowRight, Layers, Shield, FileText } from 'lucide-react'
import { useResearchStore } from '@/store/research'
import type { AgentUIState, ResearchReport } from '@/types/research'

// ── Constants ─────────────────────────────────────────────────────────────────
const EXAMPLE_QUERIES = [
  'Analyze the current state of quantum computing: breakthroughs, limitations, and commercial timeline',
  'Compare RAG vs fine-tuning for enterprise LLM deployment — costs, accuracy, latency tradeoffs',
  'What are the key risks and opportunities in AI regulation across US, EU, and China?',
  'Trace the evolution of transformer architecture from 2017 to 2025 — major milestones',
]

const AGENT_CONFIG: Record<string, { icon: React.FC<{className?: string}>, color: string, bgColor: string }> = {
  orchestrator: { icon: Brain, color: 'text-purple-600', bgColor: 'bg-purple-50 border-purple-200' },
  planner: { icon: Layers, color: 'text-teal-600', bgColor: 'bg-teal-50 border-teal-200' },
  researcher_a: { icon: Search, color: 'text-blue-600', bgColor: 'bg-blue-50 border-blue-200' },
  researcher_b: { icon: Database, color: 'text-blue-600', bgColor: 'bg-blue-50 border-blue-200' },
  critic: { icon: Shield, color: 'text-amber-600', bgColor: 'bg-amber-50 border-amber-200' },
  synthesizer: { icon: FileText, color: 'text-orange-600', bgColor: 'bg-orange-50 border-orange-200' },
}

const AGENT_ORDER = ['orchestrator', 'planner', 'researcher_a', 'researcher_b', 'critic', 'synthesizer']

// ── Utility ───────────────────────────────────────────────────────────────────
function cn(...classes: (string | undefined | false)[]) {
  return classes.filter(Boolean).join(' ')
}

function ConfidenceBar({ value, className = '' }: { value: number; className?: string }) {
  const pct = Math.round(value * 100)
  const color = pct >= 80 ? 'bg-teal-500' : pct >= 60 ? 'bg-amber-400' : 'bg-red-400'
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <motion.div
          className={cn('h-full rounded-full', color)}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>
      <span className="text-xs text-gray-500 tabular-nums min-w-[2.5rem] text-right">{pct}%</span>
    </div>
  )
}

// ── Agent card ────────────────────────────────────────────────────────────────
function AgentCard({ agent, isLast }: { agent: AgentUIState; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const config = AGENT_CONFIG[agent.role] ?? {
    icon: Brain, color: 'text-gray-600', bgColor: 'bg-gray-50 border-gray-200'
  }
  const Icon = config.icon

  const statusIcon = {
    pending: <div className="w-2 h-2 rounded-full bg-gray-300" />,
    running: <Loader2 className="w-3.5 h-3.5 text-purple-500 animate-spin" />,
    done: <CheckCircle2 className="w-3.5 h-3.5 text-teal-500" />,
    error: <AlertCircle className="w-3.5 h-3.5 text-red-500" />,
    skipped: <div className="w-2 h-2 rounded-full bg-gray-200" />,
  }[agent.status]

  return (
    <div className="relative">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className={cn(
          'border rounded-xl overflow-hidden transition-all duration-200',
          agent.status === 'running' && 'border-purple-300 shadow-sm shadow-purple-100',
          agent.status === 'done' && 'border-teal-200',
          agent.status === 'error' && 'border-red-200',
          agent.status === 'pending' && 'border-gray-100',
        )}
      >
        <button
          onClick={() => agent.status !== 'pending' && setExpanded(!expanded)}
          className={cn(
            'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors',
            agent.status !== 'pending' && 'hover:bg-gray-50/60',
            agent.status === 'running' && 'bg-purple-50/40',
            agent.status === 'done' && 'bg-teal-50/30',
          )}
        >
          <div className={cn('flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center border', config.bgColor)}>
            <Icon className={cn('w-4 h-4', config.color)} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-900">{agent.displayName}</span>
              {agent.status === 'running' && (
                <span className="text-xs px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded-full">Running</span>
              )}
              {agent.status === 'done' && agent.durationMs && (
                <span className="text-xs text-gray-400 flex items-center gap-1">
                  <Clock className="w-3 h-3" />{(agent.durationMs / 1000).toFixed(1)}s
                </span>
              )}
            </div>
            {agent.message && (
              <p className="text-xs text-gray-500 mt-0.5 truncate">{agent.message}</p>
            )}
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {statusIcon}
            {agent.status !== 'pending' && (
              <ChevronRight className={cn('w-3.5 h-3.5 text-gray-400 transition-transform', expanded && 'rotate-90')} />
            )}
          </div>
        </button>

        <AnimatePresence>
          {expanded && agent.findings.length > 0 && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="border-t border-gray-100 px-4 py-3 bg-gray-50/50"
            >
              <ul className="space-y-1.5">
                {agent.findings.slice(0, 6).map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-gray-600">
                    <ArrowRight className="w-3 h-3 text-purple-400 flex-shrink-0 mt-0.5" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
              {agent.sourcesCount > 0 && (
                <p className="mt-2 text-xs text-gray-400">{agent.sourcesCount} sources retrieved</p>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {!isLast && (
        <div className="flex justify-start ml-8 my-1">
          <div className={cn(
            'w-px h-3 transition-colors',
            agent.status === 'done' ? 'bg-teal-300' : 'bg-gray-200'
          )} />
        </div>
      )}
    </div>
  )
}

// ── Report view ───────────────────────────────────────────────────────────────
function ReportView({ report }: { report: ResearchReport }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-6 space-y-4"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">{report.title}</h2>
          <p className="text-sm text-gray-500 mt-1">
            {report.total_sources} sources · {report.word_count.toLocaleString()} words · {report.refinement_iterations} refinement{report.refinement_iterations !== 1 ? 's' : ''}
          </p>
        </div>
        <button
          onClick={() => {
            const blob = new Blob([
              `# ${report.title}\n\n${report.executive_summary}\n\n` +
              report.sections.map(s => `## ${s.heading}\n\n${s.content}`).join('\n\n')
            ], { type: 'text/markdown' })
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url; a.download = 'research-report.md'; a.click()
            URL.revokeObjectURL(url)
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors flex-shrink-0"
        >
          <Download className="w-3.5 h-3.5" /> Export MD
        </button>
      </div>

      {/* Confidence + overall score */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-gray-50 rounded-xl p-3 text-center border border-gray-100">
          <div className="text-2xl font-semibold text-gray-900">{Math.round(report.overall_confidence * 100)}%</div>
          <div className="text-xs text-gray-500 mt-0.5">Overall confidence</div>
        </div>
        <div className="bg-gray-50 rounded-xl p-3 text-center border border-gray-100">
          <div className="text-2xl font-semibold text-gray-900">{report.total_sources}</div>
          <div className="text-xs text-gray-500 mt-0.5">Sources cited</div>
        </div>
        <div className="bg-gray-50 rounded-xl p-3 text-center border border-gray-100">
          <div className="text-2xl font-semibold text-gray-900">{report.sections.length}</div>
          <div className="text-xs text-gray-500 mt-0.5">Sections</div>
        </div>
      </div>

      {/* Executive summary */}
      <div className="border-l-2 border-purple-200 pl-4 py-1">
        <p className="text-sm text-gray-700 leading-relaxed">{report.executive_summary}</p>
      </div>

      {/* Sections */}
      <div className="space-y-4">
        {report.sections.map((section, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="border border-gray-100 rounded-xl p-4"
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <h3 className="text-sm font-semibold text-gray-900">{section.heading}</h3>
              <div className="flex-shrink-0 w-24">
                <ConfidenceBar value={section.confidence} />
              </div>
            </div>
            <p className="text-sm text-gray-700 leading-relaxed">{section.content}</p>
            {section.gaps_noted.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-100">
                <p className="text-xs text-gray-400 font-medium mb-1">Areas for further research</p>
                {section.gaps_noted.map((gap, j) => (
                  <p key={j} className="text-xs text-gray-400">· {gap}</p>
                ))}
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Sources */}
      {report.all_sources.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-2 uppercase tracking-wide">Sources</p>
          <div className="flex flex-wrap gap-2">
            {report.all_sources.slice(0, 20).map((source, i) => (
              <a
                key={i}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs px-2.5 py-1 bg-blue-50 text-blue-600 border border-blue-100 rounded-full hover:bg-blue-100 transition-colors truncate max-w-xs"
                title={source.title}
              >
                {source.title.slice(0, 40)}{source.title.length > 40 ? '…' : ''}
              </a>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function ResearchPage() {
  const {
    query, status, agents, report, totalSources, refinementCount,
    durationSeconds, tokensUsed, error,
    startResearch, resetPipeline, setQuery,
  } = useResearchStore()

  const [depth, setDepth] = useState<'quick' | 'standard' | 'deep'>('standard')
  const [localQuery, setLocalQuery] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const isRunning = !['idle', 'complete', 'failed'].includes(status)

  useEffect(() => { setLocalQuery(query) }, [query])

  const handleSubmit = async () => {
    if (!localQuery.trim() || isRunning) return
    await startResearch({
      query: localQuery.trim(),
      depth,
      max_sources: { quick: 10, standard: 20, deep: 40 }[depth],
      include_arxiv: true,
      language: 'en',
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit()
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-3xl mx-auto px-4 py-10">

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2.5 mb-2">
            <div className="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center">
              <Brain className="w-4 h-4 text-white" />
            </div>
            <h1 className="text-xl font-semibold text-gray-900">Multi-Agent Research</h1>
          </div>
          <p className="text-sm text-gray-500">
            Orchestrator · Planner · Dual Researchers · Critic · Synthesizer
          </p>
        </div>

        {/* Query input */}
        <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm mb-4">
          <textarea
            ref={textareaRef}
            value={localQuery}
            onChange={e => setLocalQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter your research question…"
            rows={3}
            disabled={isRunning}
            className="w-full text-sm text-gray-900 placeholder-gray-400 resize-none border-0 outline-none bg-transparent leading-relaxed disabled:opacity-50"
          />

          {/* Example chips */}
          {!localQuery && status === 'idle' && (
            <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-gray-100">
              {EXAMPLE_QUERIES.map((q, i) => (
                <button
                  key={i}
                  onClick={() => setLocalQuery(q)}
                  className="text-xs px-2.5 py-1 bg-gray-100 text-gray-500 rounded-full hover:bg-purple-50 hover:text-purple-700 transition-colors"
                >
                  {q.slice(0, 48)}…
                </button>
              ))}
            </div>
          )}

          <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-100">
            <div className="flex items-center gap-1">
              {(['quick', 'standard', 'deep'] as const).map(d => (
                <button
                  key={d}
                  onClick={() => setDepth(d)}
                  className={cn(
                    'px-2.5 py-1 text-xs rounded-lg font-medium capitalize transition-colors',
                    depth === d
                      ? 'bg-purple-600 text-white'
                      : 'text-gray-500 hover:bg-gray-100'
                  )}
                >
                  {d}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              {(status === 'complete' || status === 'failed') && (
                <button
                  onClick={resetPipeline}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <RotateCcw className="w-3 h-3" /> New
                </button>
              )}
              <button
                onClick={handleSubmit}
                disabled={!localQuery.trim() || isRunning}
                className={cn(
                  'flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium rounded-lg transition-all',
                  !localQuery.trim() || isRunning
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-purple-600 text-white hover:bg-purple-700 active:scale-95'
                )}
              >
                {isRunning ? (
                  <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Running</>
                ) : (
                  <><Zap className="w-3.5 h-3.5" /> Research</>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Status bar */}
        {status !== 'idle' && (
          <div className="flex items-center gap-4 px-1 mb-6 text-xs text-gray-500">
            {totalSources > 0 && (
              <span className="flex items-center gap-1">
                <Database className="w-3 h-3" /> {totalSources} sources
              </span>
            )}
            {refinementCount > 0 && (
              <span className="flex items-center gap-1 text-amber-600">
                <RotateCcw className="w-3 h-3" /> {refinementCount} refinement{refinementCount > 1 ? 's' : ''}
              </span>
            )}
            {durationSeconds && (
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" /> {durationSeconds.toFixed(1)}s
              </span>
            )}
            {tokensUsed && (
              <span>{tokensUsed.toLocaleString()} tokens</span>
            )}
            {status === 'complete' && (
              <span className="flex items-center gap-1 text-teal-600 ml-auto">
                <CheckCircle2 className="w-3 h-3" /> Complete
              </span>
            )}
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700 mb-4">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Agent pipeline */}
        {status !== 'idle' && (
          <div className="mb-6">
            <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">Pipeline</p>
            <div>
              {AGENT_ORDER.map((role, i) => (
                <AgentCard
                  key={role}
                  agent={agents[role]}
                  isLast={i === AGENT_ORDER.length - 1}
                />
              ))}
            </div>
          </div>
        )}

        {/* Report */}
        {report && <ReportView report={report} />}
      </div>
    </div>
  )
}
