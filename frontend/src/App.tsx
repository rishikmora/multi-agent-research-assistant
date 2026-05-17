import { useState, useRef, useEffect } from "react";
import { usePipeline } from "./hooks/usePipeline";
import type { AgentId, AgentState, DebateMessage, BeliefSnapshot } from "./types";

/* ─── palette ────────────────────────────────────────────────────────────── */
const P = {
  bg0: "#090910",
  bg1: "#0f0f1a",
  bg2: "#161625",
  bg3: "#1e1e30",
  bg4: "#28283d",
  line: "rgba(255,255,255,0.06)",
  line2: "rgba(255,255,255,0.10)",
  t1: "#f0effe",
  t2: "#9896c8",
  t3: "#55547a",
  accent: "#7c6dff",
  accentMid: "#534AB7",
  green: "#00d4aa",
  red: "#ff6b75",
  amber: "#f5a623",
  blue: "#4fc3f7",
};

const AGENT_ACCENT: Record<AgentId, string> = {
  planner: P.accent,
  researcher_a: P.green,
  researcher_b: P.blue,
  critic: P.amber,
  skeptic: P.red,
  synthesizer: "#ce93d8",
};

const AGENT_IDS: AgentId[] = [
  "planner","researcher_a","researcher_b","critic","skeptic","synthesizer",
];

const STANCE_COLOR: Record<string, string> = {
  Advocate: P.green,
  Challenger: P.red,
  Specialist: P.blue,
  Verdict: P.accent,
};

/* ─── micro components ───────────────────────────────────────────────────── */
function Tag({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, letterSpacing: 1.2,
      textTransform: "uppercase" as const,
      padding: "2px 7px", borderRadius: 3,
      background: color + "20", color, border: `1px solid ${color}30`,
    }}>{label}</span>
  );
}

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
      <div style={{ flex: 1, height: 2, background: P.bg4, borderRadius: 1, overflow: "hidden" }}>
        <div style={{ width: `${value}%`, height: "100%", background: color, borderRadius: 1, transition: "width .9s ease" }} />
      </div>
      <span style={{ fontSize: 10, color: P.t3, fontVariantNumeric: "tabular-nums", minWidth: 26, textAlign: "right" }}>{value}%</span>
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  const color = status === "done" ? P.green : status === "running" ? P.accent : status === "error" ? P.red : P.bg4;
  return (
    <div style={{
      width: 6, height: 6, borderRadius: "50%", background: color, flexShrink: 0,
      boxShadow: status === "running" ? `0 0 0 3px ${color}30` : "none",
      animation: status === "running" ? "pulse 1.4s ease infinite" : "none",
    }} />
  );
}

/* ─── Agent card ─────────────────────────────────────────────────────────── */
function AgentCard({ agent }: { agent: AgentState }) {
  const color = AGENT_ACCENT[agent.id];
  const active = agent.status === "running";
  const done = agent.status === "done";
  return (
    <div style={{
      background: P.bg2,
      border: `1px solid ${active ? color + "60" : done ? P.green + "30" : P.line}`,
      borderRadius: 10,
      padding: "14px 16px",
      transition: "border-color .3s",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <StatusDot status={agent.status} />
        <span style={{ fontSize: 13, fontWeight: 600, color: P.t1, flex: 1 }}>{agent.name}</span>
        <Tag
          label={agent.status}
          color={active ? color : done ? P.green : agent.status === "error" ? P.red : P.t3}
        />
      </div>
      <div style={{ fontSize: 12, color: P.t2, lineHeight: 1.65, minHeight: 34 }}>
        {agent.output
          ? agent.output.slice(0, 150) + (agent.output.length > 150 ? "…" : "")
          : <span style={{ color: P.t3 }}>—</span>}
      </div>
      {agent.confidence !== null && <Bar value={agent.confidence} color={color} />}
    </div>
  );
}

/* ─── Belief step timeline ───────────────────────────────────────────────── */
function BeliefLine({ label, snap }: { label: string; snap: BeliefSnapshot }) {
  const entries: [string, number | undefined][] = [
    ["Prior", snap.prior],
    ["Res-A", snap.after_ra],
    ["Res-B", snap.after_rb],
    ["Critic", snap.after_critic],
    ["Skeptic", snap.after_skeptic],
    ["Debate", snap.after_debate],
    ["Final", snap.final],
  ];
  const stages = entries.filter(([, v]) => v !== undefined) as [string, number][];
  const delta = snap.final - snap.prior;

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 12, color: P.t2, fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: 11, color: delta >= 0 ? P.green : P.red, marginLeft: "auto" }}>
          {delta >= 0 ? "+" : ""}{delta}pp
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        {stages.map(([lbl, val], i) => {
          const c = val >= 70 ? P.green : val >= 50 ? P.accent : P.red;
          return (
            <div key={i} style={{ flex: 1, textAlign: "center", position: "relative" }}>
              {i > 0 && (
                <div style={{ position: "absolute", top: 5, left: 0, right: "50%", height: 1, background: P.line2 }} />
              )}
              {i < stages.length - 1 && (
                <div style={{ position: "absolute", top: 5, left: "50%", right: 0, height: 1, background: P.line2 }} />
              )}
              <div style={{
                width: 11, height: 11, borderRadius: "50%",
                background: c + "30", border: `2px solid ${c}`,
                margin: "0 auto 5px", position: "relative",
              }} />
              <div style={{ fontSize: 10, fontWeight: 700, color: c }}>{val}%</div>
              <div style={{ fontSize: 9, color: P.t3, letterSpacing: .3, marginTop: 1 }}>{lbl}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
type Tab = "agents" | "debate" | "beliefs" | "benchmarks";

function TabBar({ active, onChange }: { active: Tab; onChange: (t: Tab) => void }) {
  return (
    <div style={{ display: "flex", borderBottom: `1px solid ${P.line}`, marginBottom: 0 }}>
      {(["agents", "debate", "beliefs", "benchmarks"] as Tab[]).map((t) => (
        <button key={t} onClick={() => onChange(t)} style={{
          padding: "10px 16px", background: "none", border: "none",
          borderBottom: `2px solid ${active === t ? P.accent : "transparent"}`,
          fontSize: 11, fontWeight: active === t ? 700 : 400, letterSpacing: .8,
          textTransform: "uppercase" as const,
          color: active === t ? P.accent : P.t3,
          cursor: "pointer", transition: "color .15s",
          fontFamily: "inherit",
        }}>{t}</button>
      ))}
    </div>
  );
}

/* ─── App ────────────────────────────────────────────────────────────────── */
const PRESETS = [
  "What are the most significant breakthroughs in quantum computing in 2024–2025 and what is the realistic commercial timeline?",
  "Compare RAG vs fine-tuning for enterprise LLM deployment — cost, accuracy, latency tradeoffs",
  "AI regulation: US, EU, China — risks and opportunities for 2025–2026",
  "Impact of AI agents on software engineering jobs by 2027",
];

export default function App() {
  const { state, run, reset } = usePipeline();
  const [query, setQuery] = useState(PRESETS[0]);
  const [tab, setTab] = useState<Tab>("agents");
  const logRef = useRef<HTMLDivElement>(null);

  const running = state.status === "running";
  const done = state.status === "complete";

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [state.agents]);

  const handleRun = () => {
    if (!query.trim() || running) return;
    setTab("agents");
    run(query.trim());
  };

  return (
    <div style={{ fontFamily: "'DM Sans', 'Inter', system-ui, sans-serif", background: P.bg0, minHeight: "100vh", color: P.t1 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.85)} }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: ${P.bg4}; border-radius: 2px; }
        textarea { font-family: 'DM Sans', system-ui, sans-serif !important; }
      `}</style>

      {/* Sidebar layout */}
      <div style={{ display: "flex", minHeight: "100vh" }}>

        {/* Sidebar */}
        <div style={{ width: 220, background: P.bg1, borderRight: `1px solid ${P.line}`, padding: "24px 16px", flexShrink: 0 }}>
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 32 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8, background: P.accentMid,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 14, fontWeight: 800, color: "#fff", letterSpacing: -1,
            }}>M</div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: -.2 }}>MARS</div>
              <div style={{ fontSize: 10, color: P.t3, letterSpacing: 1 }}>v1.0</div>
            </div>
          </div>

          {/* Status */}
          <div style={{ marginBottom: 28 }}>
            <div style={{ fontSize: 9, color: P.t3, letterSpacing: 2, textTransform: "uppercase" as const, marginBottom: 8 }}>Status</div>
            <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12 }}>
              <div style={{
                width: 7, height: 7, borderRadius: "50%",
                background: running ? P.accent : done ? P.green : P.bg4,
                animation: running ? "pulse 1.4s ease infinite" : "none",
                boxShadow: running ? `0 0 6px ${P.accent}` : "none",
              }} />
              <span style={{ color: running ? P.accent : done ? P.green : P.t3 }}>
                {running ? "Running…" : done ? `Done · ${state.elapsed}s` : "Idle"}
              </span>
            </div>
          </div>

          {/* Agents in sidebar */}
          <div>
            <div style={{ fontSize: 9, color: P.t3, letterSpacing: 2, textTransform: "uppercase" as const, marginBottom: 10 }}>Agents</div>
            {AGENT_IDS.map((id) => {
              const ag = state.agents[id];
              const color = AGENT_ACCENT[id];
              return (
                <div key={id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 0", borderBottom: `1px solid ${P.line}` }}>
                  <div style={{
                    width: 4, height: 28, borderRadius: 2,
                    background: ag.status === "done" ? color : ag.status === "running" ? color : P.bg4,
                    opacity: ag.status === "running" ? 1 : ag.status === "done" ? .7 : .3,
                    transition: "background .3s",
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 11, fontWeight: 500, color: ag.status !== "idle" ? P.t1 : P.t3 }}>{ag.name}</div>
                    <div style={{ fontSize: 10, color: P.t3 }}>{ag.confidence !== null ? `${ag.confidence}%` : ag.status}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Model badge */}
          <div style={{ marginTop: "auto", paddingTop: 24 }}>
            <div style={{
              padding: "6px 10px", borderRadius: 6,
              background: P.bg3, border: `1px solid ${P.line}`,
              fontSize: 10, color: P.t3,
            }}>
              <span style={{ color: P.accent }}>●</span> claude-sonnet-4-6
            </div>
          </div>
        </div>

        {/* Main content */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>

          {/* Header bar */}
          <div style={{ padding: "18px 28px", borderBottom: `1px solid ${P.line}`, display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: -.5 }}>Research pipeline</div>
              <div style={{ fontSize: 12, color: P.t3 }}>6 specialist agents · structured debate · belief evolution</div>
            </div>
            {(running || done) && (
              <button onClick={reset} style={{
                padding: "6px 14px", background: "transparent", border: `1px solid ${P.line2}`,
                borderRadius: 6, fontSize: 12, color: P.t2, cursor: "pointer", fontFamily: "inherit",
              }}>Reset</button>
            )}
          </div>

          {/* Query input */}
          <div style={{ padding: "20px 28px", borderBottom: `1px solid ${P.line}` }}>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={running}
              onKeyDown={(e) => { if (e.key === "Enter" && e.metaKey) handleRun(); }}
              rows={2}
              placeholder="Enter your research question…"
              style={{
                width: "100%", background: P.bg2, border: `1px solid ${P.line2}`,
                borderRadius: 8, padding: "12px 14px", fontSize: 14, color: P.t1,
                resize: "none", outline: "none", lineHeight: 1.5,
                transition: "border-color .15s",
              }}
              onFocus={(e) => (e.target.style.borderColor = P.accent + "80")}
              onBlur={(e) => (e.target.style.borderColor = P.line2)}
            />
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10, flexWrap: "wrap" as const }}>
              {PRESETS.slice(1).map((p, i) => (
                <button key={i} onClick={() => setQuery(p)} style={{
                  padding: "4px 10px", background: P.bg3, border: `1px solid ${P.line}`,
                  borderRadius: 20, fontSize: 10, color: P.t3, cursor: "pointer",
                  fontFamily: "inherit", letterSpacing: .3,
                  transition: "border-color .15s, color .15s",
                }}
                  onMouseOver={(e) => { (e.target as HTMLElement).style.color = P.t2; (e.target as HTMLElement).style.borderColor = P.line2; }}
                  onMouseOut={(e) => { (e.target as HTMLElement).style.color = P.t3; (e.target as HTMLElement).style.borderColor = P.line; }}
                >
                  {p.slice(0, 32)}…
                </button>
              ))}
              <button onClick={handleRun} disabled={!query.trim() || running} style={{
                marginLeft: "auto", padding: "8px 20px",
                background: running ? P.bg4 : P.accent,
                border: "none", borderRadius: 7, fontSize: 12, fontWeight: 700,
                color: running ? P.t3 : "#fff",
                cursor: running ? "not-allowed" : "pointer",
                letterSpacing: .5, fontFamily: "inherit",
                transition: "background .2s",
              }}>
                {running ? "Running…" : "▶  Run MARS"}
              </button>
            </div>
          </div>

          {/* Tabs + content */}
          {(running || done) && (
            <div style={{ flex: 1, overflow: "auto" }}>
              <div style={{ padding: "0 28px" }}>
                <TabBar active={tab} onChange={setTab} />
              </div>

              <div style={{ padding: "20px 28px" }}>

                {/* AGENTS */}
                {tab === "agents" && (
                  <>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 10 }}>
                      {AGENT_IDS.map((id) => <AgentCard key={id} agent={state.agents[id]} />)}
                    </div>
                    {done && state.synthesis && (
                      <div style={{
                        marginTop: 16, padding: 18,
                        background: P.bg2, border: `1px solid ${P.accent}40`,
                        borderRadius: 10, borderLeft: `3px solid ${P.accent}`,
                      }}>
                        <div style={{ fontSize: 10, color: P.accent, fontWeight: 700, letterSpacing: 1.5, textTransform: "uppercase" as const, marginBottom: 8 }}>
                          Final synthesis
                        </div>
                        <div style={{ fontSize: 13, color: P.t2, lineHeight: 1.75 }}>{state.synthesis}</div>
                      </div>
                    )}
                  </>
                )}

                {/* DEBATE */}
                {tab === "debate" && (
                  <>
                    {state.debate.length === 0 ? (
                      <div style={{ textAlign: "center", padding: "48px 0", color: P.t3, fontSize: 13 }}>
                        Debate begins after research agents complete…
                      </div>
                    ) : (
                      <>
                        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
                          <span style={{ fontSize: 11, color: P.t3 }}>Round {state.debate.length}/4</span>
                          <div style={{ flex: 1, height: 3, background: P.bg4, borderRadius: 2, overflow: "hidden" }}>
                            <div style={{
                              width: `${state.debateConsensus}%`, height: "100%",
                              background: P.accent, borderRadius: 2, transition: "width 1.2s ease",
                            }} />
                          </div>
                          <span style={{ fontSize: 12, color: P.accent, fontWeight: 700 }}>{state.debateConsensus}% consensus</span>
                        </div>
                        {state.debate.map((msg: DebateMessage, i: number) => {
                          const c = STANCE_COLOR[msg.stance] || P.t2;
                          return (
                            <div key={i} style={{
                              padding: "12px 16px", marginBottom: 10,
                              background: P.bg2, borderRadius: 8,
                              borderLeft: `3px solid ${c}`,
                            }}>
                              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                <span style={{ fontSize: 11, fontWeight: 700, color: c, letterSpacing: .5 }}>{msg.stance}</span>
                                <span style={{ fontSize: 10, color: P.t3 }}>Round {msg.round}</span>
                                <span style={{ fontSize: 10, color: P.t3, marginLeft: "auto" }}>{msg.confidence}% conf</span>
                              </div>
                              <div style={{ fontSize: 13, color: P.t2, lineHeight: 1.65 }}>{msg.text}</div>
                            </div>
                          );
                        })}
                      </>
                    )}
                  </>
                )}

                {/* BELIEFS */}
                {tab === "beliefs" && (
                  <>
                    {!state.beliefs ? (
                      <div style={{ textAlign: "center", padding: "48px 0", color: P.t3, fontSize: 13 }}>
                        Belief snapshots appear after synthesis…
                      </div>
                    ) : (
                      <>
                        <div style={{ fontSize: 12, color: P.t3, marginBottom: 18, lineHeight: 1.6 }}>
                          Confidence per belief — prior to settled conclusion. Each stage shows evidence impact.
                        </div>
                        <BeliefLine label="Primary research claim" snap={state.beliefs.primary} />
                        <BeliefLine label="Counter-hypothesis" snap={state.beliefs.counter} />
                      </>
                    )}
                  </>
                )}

                {/* BENCHMARKS */}
                {tab === "benchmarks" && (
                  <>
                    {!state.metrics ? (
                      <div style={{ textAlign: "center", padding: "48px 0", color: P.t3, fontSize: 13 }}>
                        Metrics computed after pipeline completes…
                      </div>
                    ) : (
                      <>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10, marginBottom: 24 }}>
                          {[
                            { label: "Quality", value: state.metrics.quality_score, suffix: "/100", color: P.accent },
                            { label: "Grounding", value: state.metrics.citation_grounding, suffix: "%", color: P.green },
                            { label: "Agreement", value: state.metrics.agent_agreement, suffix: "%", color: P.blue },
                            { label: "Hallucination", value: state.metrics.hallucination_rate, suffix: "%", color: P.red },
                          ].map((m) => (
                            <div key={m.label} style={{ background: P.bg2, border: `1px solid ${P.line}`, borderRadius: 10, padding: 16 }}>
                              <div style={{ fontSize: 24, fontWeight: 700, color: m.color, letterSpacing: -1 }}>
                                {m.value}{m.suffix}
                              </div>
                              <div style={{ fontSize: 10, color: P.t3, marginTop: 3, letterSpacing: .8, textTransform: "uppercase" as const }}>
                                {m.label}
                              </div>
                              <Bar value={m.value} color={m.color} />
                            </div>
                          ))}
                        </div>

                        <div style={{ fontSize: 9, color: P.t3, letterSpacing: 2, textTransform: "uppercase" as const, marginBottom: 10 }}>
                          Comparison
                        </div>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                          <thead>
                            <tr>
                              {["System", "Accuracy", "Grounding", "Halluc.", "Grade"].map((h) => (
                                <th key={h} style={{ textAlign: "left", padding: "6px 0", color: P.t3, fontWeight: 500, fontSize: 10, letterSpacing: .8, borderBottom: `1px solid ${P.line}` }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {[
                              { s: "Vanilla GPT-4", a: "71%", g: "48%", h: "22%", gr: "C" },
                              { s: "Single-agent RAG", a: "79%", g: "67%", h: "14%", gr: "B" },
                              { s: "AutoGen MAG", a: "83%", g: "72%", h: "11%", gr: "B" },
                              {
                                s: "MARS (live)", ours: true,
                                a: `${state.metrics.quality_score}%`,
                                g: `${state.metrics.citation_grounding}%`,
                                h: `${state.metrics.hallucination_rate}%`,
                                gr: state.metrics.grade,
                              },
                            ].map((row, i) => (
                              <tr key={i} style={{ borderBottom: `1px solid ${P.line}` }}>
                                <td style={{ padding: "9px 0", color: (row as any).ours ? P.accent : P.t2, fontWeight: (row as any).ours ? 700 : 400 }}>{row.s}</td>
                                <td style={{ padding: "9px 4px", color: (row as any).ours ? P.green : P.t3 }}>{row.a}</td>
                                <td style={{ padding: "9px 4px", color: (row as any).ours ? P.green : P.t3 }}>{row.g}</td>
                                <td style={{ padding: "9px 4px", color: (row as any).ours ? P.green : P.t3 }}>{row.h}</td>
                                <td style={{ padding: "9px 4px", color: P.t2 }}>{row.gr}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* Idle state */}
          {!running && !done && (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12, color: P.t3 }}>
              <div style={{ fontSize: 40, opacity: .15 }}>M</div>
              <div style={{ fontSize: 13 }}>Enter a query and click Run MARS</div>
              <div style={{ fontSize: 11, opacity: .6 }}>⌘ + Enter to run</div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
