// ── Enums ─────────────────────────────────────────────────────────────────────

export type AgentStatus = 'pending' | 'running' | 'done' | 'error' | 'skipped'
export type ResearchStatus =
  | 'queued'
  | 'planning'
  | 'researching'
  | 'critiquing'
  | 'refining'
  | 'synthesizing'
  | 'complete'
  | 'failed'
export type SourceType = 'web' | 'arxiv' | 'semantic_scholar' | 'pdf' | 'internal'

export type SSEEventType =
  | 'pipeline_start'
  | 'agent_start'
  | 'agent_progress'
  | 'agent_complete'
  | 'agent_error'
  | 'sources_found'
  | 'refinement_loop'
  | 'report_section'
  | 'pipeline_complete'
  | 'pipeline_error'
  | 'heartbeat'

// ── Domain models ─────────────────────────────────────────────────────────────

export interface Source {
  id: string
  url: string
  title: string
  snippet: string
  source_type: SourceType
  published_date: string | null
  credibility_score: number
  retrieved_at: string
  authors: string[]
  citation_count: number
}

export interface ReportSection {
  heading: string
  content: string
  confidence: number
  citations: Citation[]
  word_count: number
  gaps_noted: string[]
}

export interface Citation {
  source_id: string
  claim: string
  quote: string
  page_number: number | null
}

export interface ResearchReport {
  id: string
  title: string
  executive_summary: string
  sections: ReportSection[]
  all_sources: Source[]
  overall_confidence: number
  total_sources: number
  word_count: number
  generated_at: string
  refinement_iterations: number
  model_versions: Record<string, string>
}

export interface AgentTrace {
  agent_id: string
  status: AgentStatus
  started_at: string | null
  completed_at: string | null
  duration_ms: number | null
  token_usage: Record<string, number>
  tool_calls: Record<string, unknown>[]
  error: string | null
}

export interface SubTask {
  id: string
  heading: string
  objective: string
  assigned_to: string
  allowed_sources: SourceType[]
  scope_rules: string[]
  priority: number
  status: AgentStatus
  findings: string[]
  sources: Source[]
}

export interface PipelineState {
  session_id: string
  user_id: string | null
  query: string
  status: ResearchStatus
  sub_tasks: SubTask[]
  agent_traces: Record<string, AgentTrace>
  refinement_count: number
  report: ResearchReport | null
  created_at: string
  updated_at: string
  error_message: string | null
  metadata: Record<string, unknown>
}

// ── SSE event envelope ────────────────────────────────────────────────────────

export interface SSEEvent<T = Record<string, unknown>> {
  event: SSEEventType
  session_id: string
  timestamp: string
  data: T & { agent?: string }
}

// ── Specific event payloads ───────────────────────────────────────────────────

export interface AgentStartData {
  agent_role: string
  message: string
}

export interface AgentProgressData {
  message: string
  agent_role?: string
  analysis?: Record<string, unknown>
  subtasks?: string[]
  gaps?: string[]
  refinement_needed?: boolean
  sources?: number
  confidence?: number
}

export interface SourcesFoundData {
  total_sources: number
  by_type: Partial<Record<SourceType, number>>
}

export interface RefinementLoopData {
  iteration: number
  max_iterations: number
  targets: string[]
}

export interface PipelineCompleteData {
  session_id: string
  duration_seconds: number
  tokens_used: number
  report_sections: number
  total_sources: number
  overall_confidence: number
  report?: ResearchReport
}

// ── API types ─────────────────────────────────────────────────────────────────

export interface ResearchRequest {
  query: string
  depth: 'quick' | 'standard' | 'deep'
  max_sources: number
  include_arxiv: boolean
  language: string
}

export interface ResearchResponse {
  session_id: string
  status: ResearchStatus
  message: string
}

// ── UI state ──────────────────────────────────────────────────────────────────

export interface AgentUIState {
  role: string
  displayName: string
  status: AgentStatus
  message: string
  durationMs: number | null
  tokensUsed: number | null
  findings: string[]
  sourcesCount: number
}

export interface PipelineUIState {
  sessionId: string | null
  query: string
  status: ResearchStatus | 'idle'
  agents: Record<string, AgentUIState>
  report: ResearchReport | null
  totalSources: number
  refinementCount: number
  durationSeconds: number | null
  tokensUsed: number | null
  error: string | null
  events: SSEEvent[]
}
