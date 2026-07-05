import { useState, useEffect, useCallback } from "react";

/* ─── Types matching the backend memory API ─────────────────────────────── */

interface EpisodicStats {
  total_episodes: number;
  avg_confidence: number | null;
  avg_duration_seconds: number | null;
  total_tokens_consumed: number;
  avg_contradictions_per_session: number | null;
}

interface SemanticStats {
  total_entries: number;
  avg_confidence: number | null;
  avg_corroboration: number | null;
  contested_entries: number;
}

interface PendingConflict {
  id: string;
  existing_claim: string;
  conflicting_claim: string;
  created_at: string;
}

interface MemoryHealth {
  episodic: EpisodicStats;
  semantic: SemanticStats;
  pending_conflicts: PendingConflict[];
  pending_conflict_count: number;
}

interface RetrievedContextPreview {
  formatted_context: string;
  episodic_hits: number;
  semantic_hits: number;
  has_content: boolean;
  retrieval_latency_ms: number | null;
}

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

/* ─── palette (matches modern dark theme) ───────────────────────────────── */
const P = {
  bg2: "#161625", bg3: "#1e1e30", bg4: "#28283d",
  line: "rgba(255,255,255,0.06)", line2: "rgba(255,255,255,0.10)",
  t1: "#f0effe", t2: "#9896c8", t3: "#55547a",
  accent: "#7c6dff", green: "#00d4aa", red: "#ff6b75", amber: "#f5a623",
};

function StatCard({ label, value, sub, color = P.t1 }: {
  label: string; value: string | number; sub?: string; color?: string;
}) {
  return (
    <div style={{ background: P.bg2, border: `1px solid ${P.line}`, borderRadius: 10, padding: 14 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color, letterSpacing: -0.5 }}>{value}</div>
      <div style={{ fontSize: 10, color: P.t3, marginTop: 3, letterSpacing: 0.8, textTransform: "uppercase" }}>
        {label}
      </div>
      {sub && <div style={{ fontSize: 11, color: P.t2, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function TierHeader({ tier, name, description }: { tier: number; name: string; description: string }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 12, marginTop: 24 }}>
      <span style={{
        fontSize: 10, fontWeight: 700, color: P.accent, background: P.accent + "20",
        padding: "2px 8px", borderRadius: 20, letterSpacing: 0.5,
      }}>TIER {tier}</span>
      <span style={{ fontSize: 14, fontWeight: 600, color: P.t1 }}>{name}</span>
      <span style={{ fontSize: 11, color: P.t3, marginLeft: "auto" }}>{description}</span>
    </div>
  );
}

/* ─── Memory Panel ───────────────────────────────────────────────────────── */

export function MemoryPanel() {
  const [health, setHealth] = useState<MemoryHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [testQuery, setTestQuery] = useState("quantum computing breakthroughs");
  const [retrievalPreview, setRetrievalPreview] = useState<RetrievedContextPreview | null>(null);
  const [retrievingPreview, setRetrievingPreview] = useState(false);

  const fetchHealth = useCallback(async () => {
    try {
      setError(null);
      const res = await fetch(`${API}/memory/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: MemoryHealth = await res.json();
      setHealth(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load memory health");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);   // Poll every 15s
    return () => clearInterval(interval);
  }, [fetchHealth]);

  const runRetrievalPreview = async () => {
    if (!testQuery.trim()) return;
    setRetrievingPreview(true);
    try {
      const res = await fetch(
        `${API}/memory/preview?query=${encodeURIComponent(testQuery)}`
      );
      const data: RetrievedContextPreview = await res.json();
      setRetrievalPreview(data);
    } catch {
      setRetrievalPreview(null);
    } finally {
      setRetrievingPreview(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: "center", color: P.t3, fontSize: 13 }}>
        Loading memory system status…
      </div>
    );
  }

  if (error || !health) {
    return (
      <div style={{
        padding: 16, background: P.red + "10", border: `1px solid ${P.red}30`,
        borderRadius: 10, color: P.red, fontSize: 13,
      }}>
        Memory system unavailable: {error}
      </div>
    );
  }

  const { episodic, semantic, pending_conflicts, pending_conflict_count } = health;

  return (
    <div style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      {/* ── Tier 2: Episodic ─────────────────────────────────────────────── */}
      <TierHeader tier={2} name="Episodic memory" description="Past research sessions · PostgreSQL" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10 }}>
        <StatCard label="Total episodes" value={episodic.total_episodes} />
        <StatCard
          label="Avg confidence"
          value={episodic.avg_confidence !== null ? `${Math.round(episodic.avg_confidence * 100)}%` : "—"}
          color={episodic.avg_confidence && episodic.avg_confidence > 0.7 ? P.green : P.amber}
        />
        <StatCard
          label="Avg duration"
          value={episodic.avg_duration_seconds !== null ? `${Math.round(episodic.avg_duration_seconds)}s` : "—"}
        />
        <StatCard
          label="Total tokens"
          value={episodic.total_tokens_consumed.toLocaleString()}
        />
        <StatCard
          label="Avg contradictions"
          value={episodic.avg_contradictions_per_session?.toFixed(1) ?? "—"}
          color={P.amber}
        />
      </div>

      {/* ── Tier 3: Semantic ─────────────────────────────────────────────── */}
      <TierHeader tier={3} name="Semantic memory" description="Distilled cross-session facts · pgvector HNSW" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10 }}>
        <StatCard label="Total facts" value={semantic.total_entries} color={P.accent} />
        <StatCard
          label="Avg confidence"
          value={semantic.avg_confidence !== null ? `${Math.round(semantic.avg_confidence * 100)}%` : "—"}
        />
        <StatCard
          label="Avg corroboration"
          value={semantic.avg_corroboration?.toFixed(1) ?? "—"}
          sub="Episodes supporting each fact"
        />
        <StatCard
          label="Contested facts"
          value={semantic.contested_entries}
          color={semantic.contested_entries > 0 ? P.red : P.green}
        />
      </div>

      {/* ── Pending conflicts ────────────────────────────────────────────── */}
      {pending_conflict_count > 0 && (
        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: P.red, marginBottom: 8 }}>
            ⚠ {pending_conflict_count} pending conflict{pending_conflict_count !== 1 ? "s" : ""} — needs Critic review
          </div>
          {pending_conflicts.map((c) => (
            <div key={c.id} style={{
              background: P.red + "08", border: `1px solid ${P.red}25`, borderRadius: 8,
              padding: 12, marginBottom: 8, fontSize: 12,
            }}>
              <div style={{ color: P.t2, marginBottom: 4 }}>
                <span style={{ color: P.t3 }}>Existing: </span>{c.existing_claim}
              </div>
              <div style={{ color: P.t2 }}>
                <span style={{ color: P.t3 }}>New: </span>{c.conflicting_claim}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Live retrieval preview ───────────────────────────────────────── */}
      <div style={{ marginTop: 28, paddingTop: 20, borderTop: `1px solid ${P.line}` }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: P.t1, marginBottom: 10 }}>
          Test memory retrieval
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={testQuery}
            onChange={(e) => setTestQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runRetrievalPreview()}
            placeholder="Try a query to see what memory would inject…"
            style={{
              flex: 1, background: P.bg3, border: `1px solid ${P.line2}`, borderRadius: 6,
              padding: "8px 12px", fontSize: 12, color: P.t1, outline: "none",
            }}
          />
          <button
            onClick={runRetrievalPreview}
            disabled={retrievingPreview}
            style={{
              padding: "8px 16px", background: P.accent, border: "none", borderRadius: 6,
              fontSize: 11, fontWeight: 600, color: "#fff", cursor: "pointer",
            }}
          >
            {retrievingPreview ? "…" : "Preview"}
          </button>
        </div>

        {retrievalPreview && (
          <div style={{ marginTop: 12 }}>
            <div style={{ display: "flex", gap: 12, fontSize: 11, color: P.t3, marginBottom: 8 }}>
              <span>{retrievalPreview.episodic_hits} episodic hit{retrievalPreview.episodic_hits !== 1 ? "s" : ""}</span>
              <span>{retrievalPreview.semantic_hits} semantic hit{retrievalPreview.semantic_hits !== 1 ? "s" : ""}</span>
              {retrievalPreview.retrieval_latency_ms !== null && (
                <span>{retrievalPreview.retrieval_latency_ms.toFixed(1)}ms</span>
              )}
            </div>
            {retrievalPreview.has_content ? (
              <pre style={{
                background: P.bg3, border: `1px solid ${P.line}`, borderRadius: 8,
                padding: 12, fontSize: 11, color: P.t2, whiteSpace: "pre-wrap",
                fontFamily: "var(--font-mono, monospace)", margin: 0, lineHeight: 1.6,
              }}>
                {retrievalPreview.formatted_context}
              </pre>
            ) : (
              <div style={{ fontSize: 12, color: P.t3, fontStyle: "italic" }}>
                No prior memory would be injected for this query — first-time topic.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
