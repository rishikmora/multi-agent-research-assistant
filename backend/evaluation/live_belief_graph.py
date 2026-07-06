"""
Live Belief Graph Engine — Week 4, System 3 (THE WOW FEATURE)

WHAT MAKES THIS THE SIGNATURE FEATURE
Every other system in MARS produces a result you see AFTER it finishes.
The live belief graph is different: it emits a structured event for every
single mutation to the evidence graph AS IT HAPPENS DURING PIPELINE
EXECUTION — a claim node appearing the instant an agent asserts it, an
edge appearing the instant the Critic detects a contradiction, a node's
confidence visibly decaying the instant the Skeptic weakens it, a node's
confidence recovering the instant a debate round resolves the conflict.
No other system in the academic or commercial research-assistant
landscape renders belief-graph formation live during execution — reports
render the graph AFTER the fact, as a static artifact. This module is
what makes that live rendering possible: a single source of truth for
graph state that emits typed events on every mutation, consumable by an
SSE stream for a frontend to animate in real time.

ARCHITECTURE
LiveBeliefGraph wraps an in-memory node/edge store (no NetworkX
dependency needed at this scale — a session's belief graph has tens, not
millions, of nodes) and an event emitter. Every mutation method
(add_claim_node, add_relationship, update_node_confidence,
resolve_contradiction) both mutates state AND returns a GraphEvent
describing exactly what changed, in a form directly serializable to an
SSE `data:` payload. The pipeline's agent classes call these methods
inline as they work; the SSE handler subscribes to the emitted events
and forwards them to the frontend without needing to know anything about
graph internals.

THIS IS DELIBERATELY NOT WEEK 1'S EVIDENCE GRAPH REASONING MODULE
Week 1 already built a NetworkX-backed evidence graph with contradiction
clustering and multi-hop reasoning, for POST-hoc analysis of a completed
session. This module has a narrower, complementary purpose: minimal,
fast, event-emitting state tracking DURING execution, optimized for low-
latency mutation + broadcast, not for graph algorithms. At pipeline
completion, `to_static_graph()` exports this live state in a shape that
can be handed to Week 1's `EvidenceGraph` for the deeper post-hoc
analysis (contradiction clusters, causal chains) if desired — the two
modules compose rather than duplicate each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

CONFIDENCE_DECAY_ON_CONTRADICTION = 0.15   # How much a contradicted node's confidence drops
CONFIDENCE_RECOVERY_ON_RESOLUTION = 0.10   # How much confidence recovers when resolved


class NodeType(str, Enum):
    CLAIM = "claim"
    SOURCE = "source"
    SYNTHESIS = "synthesis"


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"


class GraphEventType(str, Enum):
    NODE_ADDED = "node_added"
    NODE_CONFIDENCE_CHANGED = "node_confidence_changed"
    EDGE_ADDED = "edge_added"
    CONTRADICTION_DETECTED = "contradiction_detected"
    CONTRADICTION_RESOLVED = "contradiction_resolved"
    NODE_REMOVED = "node_removed"


@dataclass
class BeliefNode:
    id: str
    node_type: NodeType
    label: str
    confidence: float
    agent_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "type": self.node_type.value, "label": self.label,
            "confidence": round(self.confidence, 4), "agent_id": self.agent_id,
            "metadata": self.metadata,
        }


@dataclass
class BeliefEdge:
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 0.5
    explanation: str = ""
    resolved: bool = False   # For contradiction edges — has this been addressed?

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "source": self.source_id, "target": self.target_id,
            "type": self.edge_type.value, "weight": round(self.weight, 4),
            "explanation": self.explanation, "resolved": self.resolved,
        }


@dataclass
class GraphEvent:
    """A single, typed, serializable mutation event — this IS the live stream."""
    event_type: GraphEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    node: dict[str, Any] | None = None
    edge: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_sse_payload(self) -> dict[str, Any]:
        """Directly consumable by an SSE `data:` field — no further
        transformation needed by the caller."""
        payload: dict[str, Any] = {
            "event": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.node is not None:
            payload["node"] = self.node
        if self.edge is not None:
            payload["edge"] = self.edge
        if self.extra:
            payload.update(self.extra)
        return payload


class LiveBeliefGraphError(Exception):
    pass


class LiveBeliefGraph:
    """
    The WOW feature's backend. One instance per pipeline session.

    Usage (called inline by agents during pipeline execution):
        graph = LiveBeliefGraph(session_id, on_event=sse_queue.put_nowait)

        node = graph.add_claim_node("IBM Condor has 1121 qubits", agent_id="researcher_a", confidence=0.85)
        graph.add_relationship(node.id, other_node.id, EdgeType.SUPPORTS)

        # When Critic finds a contradiction:
        graph.flag_contradiction(node_a.id, node_b.id, explanation="Conflicting qubit counts")

        # When debate resolves it:
        graph.resolve_contradiction(edge_id)
    """

    def __init__(
        self,
        session_id: str,
        on_event: Callable[[GraphEvent], None] | None = None,
    ):
        self.session_id = session_id
        self._on_event = on_event
        self.nodes: dict[str, BeliefNode] = {}
        self.edges: dict[str, BeliefEdge] = {}

    def _emit(self, event: GraphEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as exc:
                # Live-stream emission is best-effort — a broken SSE
                # connection or full queue must never break graph state
                # mutation, which is the actual source of truth.
                log.warning("live_belief_graph.emit_failed",
                          session_id=self.session_id, error=str(exc))

    def add_claim_node(
        self,
        label: str,
        agent_id: str,
        confidence: float,
        node_type: NodeType = NodeType.CLAIM,
        metadata: dict[str, Any] | None = None,
    ) -> BeliefNode:
        node = BeliefNode(
            id=str(uuid4()), node_type=node_type, label=label,
            confidence=confidence, agent_id=agent_id, metadata=metadata or {},
        )
        self.nodes[node.id] = node
        self._emit(GraphEvent(event_type=GraphEventType.NODE_ADDED, node=node.to_dict()))
        log.info("live_belief_graph.node_added",
                session_id=self.session_id, node_id=node.id, agent=agent_id)
        return node

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 0.5,
        explanation: str = "",
    ) -> BeliefEdge:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise LiveBeliefGraphError(
                f"Cannot add edge: source={source_id} or target={target_id} not in graph"
            )

        edge = BeliefEdge(
            id=str(uuid4()), source_id=source_id, target_id=target_id,
            edge_type=edge_type, weight=weight, explanation=explanation,
        )
        self.edges[edge.id] = edge

        event_type = (
            GraphEventType.CONTRADICTION_DETECTED
            if edge_type == EdgeType.CONTRADICTS
            else GraphEventType.EDGE_ADDED
        )
        self._emit(GraphEvent(event_type=event_type, edge=edge.to_dict()))

        if edge_type == EdgeType.CONTRADICTS:
            self._apply_contradiction_decay(source_id, target_id)

        return edge

    def flag_contradiction(
        self, node_a_id: str, node_b_id: str, explanation: str = ""
    ) -> BeliefEdge:
        """Convenience wrapper — the common case of the Critic detecting
        a contradiction between two existing claim nodes."""
        return self.add_relationship(
            node_a_id, node_b_id, EdgeType.CONTRADICTS,
            weight=0.7, explanation=explanation,
        )

    def _apply_contradiction_decay(self, node_a_id: str, node_b_id: str) -> None:
        """When a contradiction is flagged, BOTH involved nodes visibly
        lose confidence — this is what makes the graph feel alive: you
        watch nodes shrink/dim the instant a conflict appears, not after
        the fact in a static report."""
        for node_id in (node_a_id, node_b_id):
            node = self.nodes.get(node_id)
            if node is None:
                continue
            old_confidence = node.confidence
            node.confidence = max(0.0, node.confidence - CONFIDENCE_DECAY_ON_CONTRADICTION)
            self._emit(GraphEvent(
                event_type=GraphEventType.NODE_CONFIDENCE_CHANGED,
                node=node.to_dict(),
                extra={"previous_confidence": round(old_confidence, 4), "reason": "contradiction_detected"},
            ))

    def resolve_contradiction(self, edge_id: str, resolution_note: str = "") -> BeliefEdge:
        """When a debate round or Critic re-check resolves a flagged
        contradiction: the edge is marked resolved, and both endpoint
        nodes recover SOME confidence (not full — a resolved contradiction
        still means the claim survived a real challenge, which should be
        reflected as somewhat lower-trust than a claim that was never
        challenged at all)."""
        edge = self.edges.get(edge_id)
        if edge is None:
            raise LiveBeliefGraphError(f"Edge not found: {edge_id}")
        if edge.edge_type != EdgeType.CONTRADICTS:
            raise LiveBeliefGraphError(f"Edge {edge_id} is not a contradiction edge")

        edge.resolved = True
        edge.explanation = resolution_note or edge.explanation

        for node_id in (edge.source_id, edge.target_id):
            node = self.nodes.get(node_id)
            if node is None:
                continue
            old_confidence = node.confidence
            node.confidence = min(1.0, node.confidence + CONFIDENCE_RECOVERY_ON_RESOLUTION)
            self._emit(GraphEvent(
                event_type=GraphEventType.NODE_CONFIDENCE_CHANGED,
                node=node.to_dict(),
                extra={"previous_confidence": round(old_confidence, 4), "reason": "contradiction_resolved"},
            ))

        self._emit(GraphEvent(event_type=GraphEventType.CONTRADICTION_RESOLVED, edge=edge.to_dict()))
        log.info("live_belief_graph.contradiction_resolved",
                session_id=self.session_id, edge_id=edge_id)
        return edge

    def update_node_confidence(
        self, node_id: str, new_confidence: float, reason: str = ""
    ) -> BeliefNode:
        node = self.nodes.get(node_id)
        if node is None:
            raise LiveBeliefGraphError(f"Node not found: {node_id}")

        old_confidence = node.confidence
        node.confidence = max(0.0, min(1.0, new_confidence))
        self._emit(GraphEvent(
            event_type=GraphEventType.NODE_CONFIDENCE_CHANGED,
            node=node.to_dict(),
            extra={"previous_confidence": round(old_confidence, 4), "reason": reason},
        ))
        return node

    def get_unresolved_contradictions(self) -> list[BeliefEdge]:
        return [
            e for e in self.edges.values()
            if e.edge_type == EdgeType.CONTRADICTS and not e.resolved
        ]

    def stats(self) -> dict[str, Any]:
        contradictions = [e for e in self.edges.values() if e.edge_type == EdgeType.CONTRADICTS]
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "contradictions_total": len(contradictions),
            "contradictions_unresolved": len(self.get_unresolved_contradictions()),
            "avg_confidence": round(
                sum(n.confidence for n in self.nodes.values()) / len(self.nodes), 4
            ) if self.nodes else None,
        }

    def to_static_graph(self) -> dict[str, Any]:
        """
        Export the final graph state for handoff to Week 1's post-hoc
        EvidenceGraph analysis, or for archival in episodic memory. This
        is the bridge between "live during execution" and "durable
        record after completion."
        """
        return {
            "session_id": self.session_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "stats": self.stats(),
        }
