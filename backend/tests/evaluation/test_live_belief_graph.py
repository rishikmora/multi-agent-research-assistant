"""
Tests for LiveBeliefGraph (Week 4, System 3 — THE WOW FEATURE).
Run: pytest tests/evaluation/test_live_belief_graph.py -v
"""
import pytest

from app.evaluation.live_belief_graph import (
    LiveBeliefGraph,
    LiveBeliefGraphError,
    NodeType,
    EdgeType,
    GraphEventType,
    CONFIDENCE_DECAY_ON_CONTRADICTION,
    CONFIDENCE_RECOVERY_ON_RESOLUTION,
)


class TestNodeCreation:
    def test_add_claim_node_creates_and_stores(self):
        graph = LiveBeliefGraph("session-1")
        node = graph.add_claim_node("IBM Condor has 1121 qubits", agent_id="researcher_a", confidence=0.85)

        assert node.id in graph.nodes
        assert graph.nodes[node.id].label == "IBM Condor has 1121 qubits"
        assert graph.nodes[node.id].confidence == 0.85

    def test_add_claim_node_emits_event(self):
        events = []
        graph = LiveBeliefGraph("session-1", on_event=events.append)
        graph.add_claim_node("claim text", agent_id="researcher_a", confidence=0.8)

        assert len(events) == 1
        assert events[0].event_type == GraphEventType.NODE_ADDED

    def test_event_payload_is_sse_ready(self):
        events = []
        graph = LiveBeliefGraph("session-1", on_event=events.append)
        graph.add_claim_node("claim text", agent_id="researcher_a", confidence=0.8)

        payload = events[0].to_sse_payload()
        assert payload["event"] == "node_added"
        assert "node" in payload
        assert payload["node"]["confidence"] == 0.8

    def test_event_emission_failure_does_not_break_graph_mutation(self):
        """The core resilience property: a broken SSE connection must
        never prevent the graph's actual state from updating correctly."""
        def broken_emit(event):
            raise ConnectionError("SSE connection closed")

        graph = LiveBeliefGraph("session-1", on_event=broken_emit)
        node = graph.add_claim_node("claim text", agent_id="researcher_a", confidence=0.8)

        # Node was still added despite the emit failure
        assert node.id in graph.nodes


class TestEdgeCreation:
    def test_add_relationship_between_existing_nodes(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("claim A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("claim B", agent_id="b", confidence=0.7)

        edge = graph.add_relationship(n1.id, n2.id, EdgeType.SUPPORTS)
        assert edge.id in graph.edges
        assert edge.source_id == n1.id
        assert edge.target_id == n2.id

    def test_edge_to_missing_node_raises(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("claim A", agent_id="a", confidence=0.8)

        with pytest.raises(LiveBeliefGraphError):
            graph.add_relationship(n1.id, "nonexistent_node", EdgeType.SUPPORTS)

    def test_supports_edge_emits_edge_added_event(self):
        events = []
        graph = LiveBeliefGraph("session-1", on_event=events.append)
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)
        events.clear()

        graph.add_relationship(n1.id, n2.id, EdgeType.SUPPORTS)
        assert events[0].event_type == GraphEventType.EDGE_ADDED

    def test_contradicts_edge_emits_contradiction_detected_event(self):
        events = []
        graph = LiveBeliefGraph("session-1", on_event=events.append)
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)
        events.clear()

        graph.add_relationship(n1.id, n2.id, EdgeType.CONTRADICTS)
        event_types = [e.event_type for e in events]
        assert GraphEventType.CONTRADICTION_DETECTED in event_types


class TestContradictionDecay:
    def test_contradiction_reduces_both_nodes_confidence(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.80)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.75)

        graph.add_relationship(n1.id, n2.id, EdgeType.CONTRADICTS)

        assert graph.nodes[n1.id].confidence == pytest.approx(0.80 - CONFIDENCE_DECAY_ON_CONTRADICTION, abs=0.001)
        assert graph.nodes[n2.id].confidence == pytest.approx(0.75 - CONFIDENCE_DECAY_ON_CONTRADICTION, abs=0.001)

    def test_confidence_decay_emits_visible_change_events(self):
        events = []
        graph = LiveBeliefGraph("session-1", on_event=events.append)
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.80)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.75)
        events.clear()

        graph.add_relationship(n1.id, n2.id, EdgeType.CONTRADICTS)

        confidence_events = [e for e in events if e.event_type == GraphEventType.NODE_CONFIDENCE_CHANGED]
        assert len(confidence_events) == 2   # Both endpoints decay

    def test_decay_never_goes_below_zero(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.05)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.03)

        graph.add_relationship(n1.id, n2.id, EdgeType.CONTRADICTS)

        assert graph.nodes[n1.id].confidence >= 0.0
        assert graph.nodes[n2.id].confidence >= 0.0

    def test_flag_contradiction_convenience_method(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)

        edge = graph.flag_contradiction(n1.id, n2.id, explanation="conflicting qubit counts")
        assert edge.edge_type == EdgeType.CONTRADICTS
        assert edge.explanation == "conflicting qubit counts"


class TestContradictionResolution:
    def test_resolve_marks_edge_resolved(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)
        edge = graph.flag_contradiction(n1.id, n2.id)

        resolved_edge = graph.resolve_contradiction(edge.id, "Debate settled the conflict")
        assert resolved_edge.resolved is True

    def test_resolve_recovers_some_confidence(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.80)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.75)
        edge = graph.flag_contradiction(n1.id, n2.id)

        decayed_conf_1 = graph.nodes[n1.id].confidence
        graph.resolve_contradiction(edge.id)

        assert graph.nodes[n1.id].confidence == pytest.approx(
            decayed_conf_1 + CONFIDENCE_RECOVERY_ON_RESOLUTION, abs=0.001
        )

    def test_recovery_does_not_fully_restore_original_confidence(self):
        """A resolved contradiction should leave the claim at LOWER trust
        than if it had never been challenged — recovery is partial, not full."""
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.80)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.75)
        edge = graph.flag_contradiction(n1.id, n2.id)
        graph.resolve_contradiction(edge.id)

        assert graph.nodes[n1.id].confidence < 0.80   # Never fully recovers

    def test_resolve_missing_edge_raises(self):
        graph = LiveBeliefGraph("session-1")
        with pytest.raises(LiveBeliefGraphError):
            graph.resolve_contradiction("nonexistent_edge")

    def test_resolve_non_contradiction_edge_raises(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)
        edge = graph.add_relationship(n1.id, n2.id, EdgeType.SUPPORTS)

        with pytest.raises(LiveBeliefGraphError):
            graph.resolve_contradiction(edge.id)

    def test_resolve_emits_contradiction_resolved_event(self):
        events = []
        graph = LiveBeliefGraph("session-1", on_event=events.append)
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)
        edge = graph.flag_contradiction(n1.id, n2.id)
        events.clear()

        graph.resolve_contradiction(edge.id)
        event_types = [e.event_type for e in events]
        assert GraphEventType.CONTRADICTION_RESOLVED in event_types


class TestUnresolvedContradictions:
    def test_get_unresolved_returns_only_unresolved_contradiction_edges(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)
        n3 = graph.add_claim_node("C", agent_id="c", confidence=0.6)

        edge1 = graph.flag_contradiction(n1.id, n2.id)
        graph.flag_contradiction(n2.id, n3.id)
        graph.resolve_contradiction(edge1.id)

        unresolved = graph.get_unresolved_contradictions()
        assert len(unresolved) == 1

    def test_supports_edges_never_appear_in_unresolved_contradictions(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)
        graph.add_relationship(n1.id, n2.id, EdgeType.SUPPORTS)

        assert graph.get_unresolved_contradictions() == []


class TestDirectConfidenceUpdate:
    def test_update_node_confidence(self):
        graph = LiveBeliefGraph("session-1")
        node = graph.add_claim_node("A", agent_id="a", confidence=0.5)

        updated = graph.update_node_confidence(node.id, 0.9, reason="debate_verdict")
        assert updated.confidence == 0.9

    def test_update_clamps_to_valid_range(self):
        graph = LiveBeliefGraph("session-1")
        node = graph.add_claim_node("A", agent_id="a", confidence=0.5)

        graph.update_node_confidence(node.id, 1.5)
        assert graph.nodes[node.id].confidence == 1.0

        graph.update_node_confidence(node.id, -0.5)
        assert graph.nodes[node.id].confidence == 0.0

    def test_update_missing_node_raises(self):
        graph = LiveBeliefGraph("session-1")
        with pytest.raises(LiveBeliefGraphError):
            graph.update_node_confidence("nonexistent", 0.5)


class TestStats:
    def test_stats_reflect_graph_state(self):
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.6)
        graph.flag_contradiction(n1.id, n2.id)

        stats = graph.stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert stats["contradictions_total"] == 1
        assert stats["contradictions_unresolved"] == 1

    def test_stats_on_empty_graph(self):
        graph = LiveBeliefGraph("session-1")
        stats = graph.stats()
        assert stats["total_nodes"] == 0
        assert stats["avg_confidence"] is None


class TestStaticExport:
    def test_to_static_graph_json_serializable(self):
        import json
        graph = LiveBeliefGraph("session-1")
        n1 = graph.add_claim_node("A", agent_id="a", confidence=0.8)
        n2 = graph.add_claim_node("B", agent_id="b", confidence=0.7)
        graph.add_relationship(n1.id, n2.id, EdgeType.SUPPORTS)

        static = graph.to_static_graph()
        json.dumps(static)
        assert len(static["nodes"]) == 2
        assert len(static["edges"]) == 1
        assert static["session_id"] == "session-1"
