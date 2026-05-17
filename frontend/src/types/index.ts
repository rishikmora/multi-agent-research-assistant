export type AgentId =
  | "planner"
  | "researcher_a"
  | "researcher_b"
  | "critic"
  | "skeptic"
  | "synthesizer";

export type AgentStatus = "idle" | "running" | "done" | "error";

export interface AgentState {
  id: AgentId;
  name: string;
  role: string;
  status: AgentStatus;
  output: string;
  confidence: number | null;
}

export interface DebateMessage {
  round: number;
  agent: AgentId;
  stance: string;
  text: string;
  confidence: number;
  consensus: number;
}

export interface BeliefSnapshot {
  prior: number;
  after_ra?: number;
  after_rb?: number;
  after_critic?: number;
  after_skeptic?: number;
  after_debate?: number;
  final: number;
}

export interface Metrics {
  hallucination_rate: number;
  citation_grounding: number;
  agent_agreement: number;
  quality_score: number;
  grade: string;
}

export interface PipelineState {
  status: "idle" | "running" | "complete" | "error";
  query: string;
  agents: Record<AgentId, AgentState>;
  debate: DebateMessage[];
  debateConsensus: number;
  synthesis: string;
  metrics: Metrics | null;
  beliefs: { primary: BeliefSnapshot; counter: BeliefSnapshot } | null;
  elapsed: number | null;
}
