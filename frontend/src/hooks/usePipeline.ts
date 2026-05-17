import { useState, useCallback, useRef } from "react";
import type {
  AgentId,
  AgentState,
  DebateMessage,
  PipelineState,
} from "../types";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const AGENT_META: Record<AgentId, { name: string; role: string }> = {
  planner:      { name: "Planner",      role: "Decomposition"   },
  researcher_a: { name: "Researcher A", role: "Primary evidence" },
  researcher_b: { name: "Researcher B", role: "Domain specialist"},
  critic:       { name: "Critic",       role: "Gap analysis"     },
  skeptic:      { name: "Skeptic",      role: "Counter-evidence" },
  synthesizer:  { name: "Synthesizer",  role: "Final synthesis"  },
};

const AGENT_IDS: AgentId[] = [
  "planner", "researcher_a", "researcher_b",
  "critic", "skeptic", "synthesizer",
];

function makeInitialAgents(): Record<AgentId, AgentState> {
  return Object.fromEntries(
    AGENT_IDS.map((id) => [
      id,
      { id, ...AGENT_META[id], status: "idle", output: "", confidence: null },
    ])
  ) as Record<AgentId, AgentState>;
}

const INITIAL: PipelineState = {
  status: "idle",
  query: "",
  agents: makeInitialAgents(),
  debate: [],
  debateConsensus: 0,
  synthesis: "",
  metrics: null,
  beliefs: null,
  elapsed: null,
};

export function usePipeline() {
  const [state, setState] = useState<PipelineState>(INITIAL);
  const esRef = useRef<EventSource | null>(null);

  const setAgent = useCallback(
    (id: AgentId, patch: Partial<AgentState>) =>
      setState((s) => ({
        ...s,
        agents: { ...s.agents, [id]: { ...s.agents[id], ...patch } },
      })),
    []
  );

  const run = useCallback(
    async (query: string) => {
      esRef.current?.close();
      setState({ ...INITIAL, status: "running", query });

      // POST to get the stream
      const res = await fetch(`${API}/research/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok || !res.body) {
        setState((s) => ({ ...s, status: "error" }));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      const processEvent = (raw: string) => {
        const lines = raw.split("\n");
        let eventName = "";
        let dataStr = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) eventName = line.slice(7).trim();
          if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
        }
        if (!eventName || !dataStr) return;
        let data: Record<string, unknown>;
        try { data = JSON.parse(dataStr); } catch { return; }

        switch (eventName) {
          case "agent_start":
            setAgent(data.agent as AgentId, { status: "running", output: data.message as string });
            break;
          case "agent_done":
            setAgent(data.agent as AgentId, {
              status: "done",
              output: data.output as string,
              confidence: data.confidence as number,
            });
            break;
          case "agent_error":
            setAgent(data.agent as AgentId, {
              status: "error",
              output: `Error: ${data.error}`,
            });
            break;
          case "debate_round_done":
            setState((s) => ({
              ...s,
              debateConsensus: data.consensus as number,
              debate: [
                ...s.debate,
                {
                  round: data.round as number,
                  agent: data.agent as AgentId,
                  stance: data.stance as string,
                  text: data.text as string,
                  confidence: data.confidence as number,
                  consensus: data.consensus as number,
                } as DebateMessage,
              ],
            }));
            break;
          case "pipeline_complete":
            setState((s) => ({
              ...s,
              status: "complete",
              synthesis: data.synthesis as string,
              metrics: data.metrics as PipelineState["metrics"],
              beliefs: data.belief_snapshots as PipelineState["beliefs"],
              elapsed: data.elapsed_seconds as number,
            }));
            break;
        }
      };

      const pump = async () => {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const parts = buf.split("\n\n");
          buf = parts.pop() ?? "";
          parts.forEach(processEvent);
        }
      };

      pump().catch(() =>
        setState((s) => ({ ...s, status: s.status === "complete" ? "complete" : "error" }))
      );
    },
    [setAgent]
  );

  const reset = useCallback(() => setState(INITIAL), []);

  return { state, run, reset };
}
