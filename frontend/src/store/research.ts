/**
 * Zustand store — single source of truth for the research pipeline UI.
 * Manages SSE connection lifecycle, agent state, and report data.
 * Persists completed sessions to localStorage for history.
 */
import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type {
  AgentUIState,
  PipelineUIState,
  ResearchRequest,
  ResearchStatus,
  SSEEvent,
  AgentProgressData,
  SourcesFoundData,
  RefinementLoopData,
  PipelineCompleteData,
} from '@/types/research'

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1'

const AGENT_DISPLAY_NAMES: Record<string, string> = {
  orchestrator: 'Orchestrator',
  planner: 'Planner',
  researcher_a: 'Researcher A',
  researcher_b: 'Researcher B',
  critic: 'Critic',
  synthesizer: 'Synthesizer',
}

const AGENT_ORDER = [
  'orchestrator', 'planner', 'researcher_a', 'researcher_b', 'critic', 'synthesizer'
]

function makeDefaultAgent(role: string): AgentUIState {
  return {
    role,
    displayName: AGENT_DISPLAY_NAMES[role] ?? role,
    status: 'pending',
    message: '',
    durationMs: null,
    tokensUsed: null,
    findings: [],
    sourcesCount: 0,
  }
}

const DEFAULT_PIPELINE: PipelineUIState = {
  sessionId: null,
  query: '',
  status: 'idle',
  agents: Object.fromEntries(AGENT_ORDER.map(r => [r, makeDefaultAgent(r)])),
  report: null,
  totalSources: 0,
  refinementCount: 0,
  durationSeconds: null,
  tokensUsed: null,
  error: null,
  events: [],
}

interface ResearchStore extends PipelineUIState {
  // Actions
  startResearch: (request: ResearchRequest) => Promise<void>
  cancelResearch: () => void
  resetPipeline: () => void
  setQuery: (query: string) => void
  // Internal
  _sse: EventSource | null
  _updateAgent: (role: string, updates: Partial<AgentUIState>) => void
  _handleEvent: (event: SSEEvent) => void
}

export const useResearchStore = create<ResearchStore>()(
  devtools(
    (set, get) => ({
      ...DEFAULT_PIPELINE,
      _sse: null,

      setQuery: (query) => set({ query }),

      resetPipeline: () => {
        get()._sse?.close()
        set({ ...DEFAULT_PIPELINE, _sse: null })
      },

      cancelResearch: () => {
        get()._sse?.close()
        set({ status: 'failed', error: 'Cancelled by user', _sse: null })
      },

      _updateAgent: (role, updates) => {
        set(state => ({
          agents: {
            ...state.agents,
            [role]: { ...state.agents[role], ...updates },
          }
        }))
      },

      _handleEvent: (event: SSEEvent) => {
        const { _updateAgent } = get()
        const role = event.data?.agent ?? ''

        // Append to event log (keep last 200)
        set(state => ({
          events: [...state.events.slice(-199), event],
        }))

        switch (event.event) {
          case 'pipeline_start':
            set({ status: 'queued' })
            break

          case 'agent_start': {
            const agentRole = (event.data as { agent_role?: string }).agent_role ?? role
            set({ status: _mapStatus(agentRole) })
            _updateAgent(agentRole, {
              status: 'running',
              message: (event.data as { message?: string }).message ?? '',
            })
            break
          }

          case 'agent_progress': {
            const d = event.data as AgentProgressData
            const agentRole = d.agent_role ?? role
            if (!agentRole) break
            _updateAgent(agentRole, {
              status: 'running',
              message: d.message ?? '',
            })
            if (d.subtasks) {
              _updateAgent(agentRole, { findings: d.subtasks })
            }
            if (typeof d.refinement_needed === 'boolean') {
              set(state => ({
                refinementCount: d.refinement_needed
                  ? state.refinementCount
                  : state.refinementCount,
              }))
            }
            break
          }

          case 'agent_complete': {
            const d = event.data as { agent_role?: string; duration_ms?: number; tokens_used?: number }
            const agentRole = d.agent_role ?? role
            _updateAgent(agentRole, {
              status: 'done',
              durationMs: d.duration_ms ?? null,
              tokensUsed: d.tokens_used ?? null,
            })
            break
          }

          case 'agent_error': {
            const agentRole = (event.data as { agent_role?: string }).agent_role ?? role
            _updateAgent(agentRole, {
              status: 'error',
              message: (event.data as { error?: string }).error ?? 'Unknown error',
            })
            break
          }

          case 'sources_found': {
            const d = event.data as SourcesFoundData
            set({ totalSources: d.total_sources })
            // Attribute to the active researcher
            ;['researcher_a', 'researcher_b'].forEach(r => {
              if (get().agents[r]?.status === 'running') {
                _updateAgent(r, { sourcesCount: d.total_sources })
              }
            })
            break
          }

          case 'refinement_loop': {
            const d = event.data as RefinementLoopData
            set({ refinementCount: d.iteration, status: 'refining' })
            _updateAgent('critic', {
              message: `Refinement loop ${d.iteration}/${d.max_iterations}`,
              findings: d.targets,
            })
            break
          }

          case 'report_section': {
            const d = event.data as { heading?: string; section_index?: number; total_sections?: number }
            _updateAgent('synthesizer', {
              status: 'running',
              message: `Writing section ${(d.section_index ?? 0) + 1}/${d.total_sections ?? '?'}: ${d.heading ?? ''}`,
            })
            break
          }

          case 'pipeline_complete': {
            const d = event.data as PipelineCompleteData
            _updateAgent('synthesizer', { status: 'done' })
            set({
              status: 'complete',
              durationSeconds: d.duration_seconds,
              tokensUsed: d.tokens_used,
              report: d.report ?? null,
              _sse: null,
            })
            get()._sse?.close()
            break
          }

          case 'pipeline_error': {
            const d = event.data as { error?: string }
            set({
              status: 'failed',
              error: d.error ?? 'Pipeline failed',
              _sse: null,
            })
            get()._sse?.close()
            break
          }
        }
      },

      startResearch: async (request) => {
        const { resetPipeline, _handleEvent } = get()
        resetPipeline()

        set({ query: request.query, status: 'queued' })

        // 1. POST to start the pipeline
        const res = await fetch(`${API_BASE}/research`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        })

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: 'Request failed' }))
          set({ status: 'failed', error: err.detail ?? 'Failed to start research' })
          return
        }

        const { session_id } = await res.json()
        set({ sessionId: session_id })

        // 2. Open SSE stream
        const sse = new EventSource(`${API_BASE}/research/${session_id}/stream`)
        set({ _sse: sse })

        // Handle all named events
        const eventTypes: SSEEvent['event'][] = [
          'pipeline_start', 'agent_start', 'agent_progress', 'agent_complete',
          'agent_error', 'sources_found', 'refinement_loop', 'report_section',
          'pipeline_complete', 'pipeline_error', 'heartbeat',
        ]

        eventTypes.forEach(eventType => {
          sse.addEventListener(eventType, (e: MessageEvent) => {
            try {
              const parsed = JSON.parse(e.data) as SSEEvent
              _handleEvent({ ...parsed, event: eventType })
            } catch (err) {
              console.error('SSE parse error', err)
            }
          })
        })

        sse.addEventListener('done', () => {
          sse.close()
          set({ _sse: null })
        })

        sse.onerror = () => {
          // EventSource auto-reconnects — only set error if already in final state
          const { status } = get()
          if (status !== 'complete' && status !== 'failed') {
            console.warn('SSE connection interrupted, reconnecting...')
          }
        }
      },
    }),
    { name: 'research-store' }
  )
)

function _mapStatus(agentRole: string): ResearchStatus {
  const map: Record<string, ResearchStatus> = {
    orchestrator: 'queued',
    planner: 'planning',
    researcher_a: 'researching',
    researcher_b: 'researching',
    critic: 'critiquing',
    synthesizer: 'synthesizing',
  }
  return map[agentRole] ?? 'researching'
}
