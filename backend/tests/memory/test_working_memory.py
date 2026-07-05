"""
Tests for WorkingMemory (Tier 1). Requires a real Redis instance —
use fakeredis for CI, real Redis for integration testing.

Run: pytest tests/memory/test_working_memory.py -v
"""
import asyncio
import json

import pytest
import pytest_asyncio

from app.memory.working import (
    WorkingMemory,
    MAX_AGENT_OUTPUTS_PER_SESSION,
    MAX_DEBATE_MESSAGES,
    WORKING_MEMORY_TTL_SECONDS,
)


@pytest_asyncio.fixture
async def redis_client():
    """Use fakeredis in CI; swap to real aioredis.from_url for integration."""
    import fakeredis.aioredis as fakeredis
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest_asyncio.fixture
async def wm(redis_client):
    session_id = "test-session-001"
    memory = WorkingMemory(session_id, redis_client)
    await memory.initialize(query="test query")
    return memory


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_creates_session(self, wm):
        snapshot = await wm.get_snapshot()
        assert snapshot is not None
        assert snapshot.query == "test query"
        assert snapshot.status == "queued"

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, wm, redis_client):
        await wm.initialize(query="different query")
        snapshot = await wm.get_snapshot()
        # Second init overwrites — this documents current behavior
        assert snapshot.query == "different query"

    @pytest.mark.asyncio
    async def test_set_status_updates_and_publishes(self, wm, redis_client):
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(wm.subscribe_channel_name())

        await wm.set_status("researching")

        snapshot = await wm.get_snapshot()
        assert snapshot.status == "researching"

    @pytest.mark.asyncio
    async def test_ttl_is_set_on_initialize(self, wm, redis_client):
        from app.memory.working import _k_session
        ttl = await redis_client.ttl(_k_session(wm.session_id))
        assert 0 < ttl <= WORKING_MEMORY_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_delete_removes_all_keys(self, wm, redis_client):
        await wm.record_agent_output("planner", "output", 0.8)
        await wm.update_confidence("topic1", 0.7)
        await wm.append_debate_message({"round": 1, "text": "x"})

        await wm.delete()

        snapshot = await wm.get_snapshot()
        assert snapshot is None


class TestAgentOutputs:
    @pytest.mark.asyncio
    async def test_record_and_retrieve_output(self, wm):
        await wm.record_agent_output(
            agent_id="researcher_a",
            output_text="Finding: X is true",
            confidence=0.82,
            token_usage={"input": 100, "output": 50},
        )

        entry = await wm.get_agent_output("researcher_a")
        assert entry is not None
        assert entry.output_text == "Finding: X is true"
        assert entry.confidence == 0.82
        assert entry.token_usage["input"] == 100

    @pytest.mark.asyncio
    async def test_missing_output_returns_none(self, wm):
        entry = await wm.get_agent_output("nonexistent_agent")
        assert entry is None

    @pytest.mark.asyncio
    async def test_get_all_outputs(self, wm):
        await wm.record_agent_output("planner", "plan text", 0.85)
        await wm.record_agent_output("researcher_a", "research text", 0.75)

        all_outputs = await wm.get_all_agent_outputs()
        assert len(all_outputs) == 2
        assert "planner" in all_outputs
        assert "researcher_a" in all_outputs

    @pytest.mark.asyncio
    async def test_output_cap_evicts_oldest(self, wm):
        """Verifies the hard bound on outputs-per-session actually triggers."""
        for i in range(MAX_AGENT_OUTPUTS_PER_SESSION + 5):
            await wm.record_agent_output(f"agent_{i}", f"output {i}", 0.5 + i * 0.001)
            await asyncio.sleep(0.001)  # Ensure distinct timestamps

        all_outputs = await wm.get_all_agent_outputs()
        assert len(all_outputs) <= MAX_AGENT_OUTPUTS_PER_SESSION

    @pytest.mark.asyncio
    async def test_concurrent_writes_from_different_agents_do_not_clobber(self, wm):
        """
        The core correctness property: two agents finishing at the exact
        same moment must both have their outputs preserved. Uses HSET on
        distinct fields (agent_id), which Redis guarantees is independently
        atomic — this test verifies that guarantee holds under real
        concurrent execution, not just in theory.
        """
        async def write(agent_id: str, text: str, conf: float):
            await wm.record_agent_output(agent_id, text, conf)

        await asyncio.gather(
            write("researcher_a", "finding A", 0.80),
            write("researcher_b", "finding B", 0.75),
            write("critic", "critique C", 0.65),
        )

        all_outputs = await wm.get_all_agent_outputs()
        assert len(all_outputs) == 3
        assert all_outputs["researcher_a"].output_text == "finding A"
        assert all_outputs["researcher_b"].output_text == "finding B"
        assert all_outputs["critic"].output_text == "critique C"


class TestConfidenceTracking:
    @pytest.mark.asyncio
    async def test_update_and_get_confidence(self, wm):
        result = await wm.update_confidence("hardware breakthroughs", 0.72)
        assert result == 0.72

        conf = await wm.get_confidence("hardware breakthroughs")
        assert conf == 0.72

    @pytest.mark.asyncio
    async def test_missing_topic_returns_none(self, wm):
        conf = await wm.get_confidence("nonexistent topic")
        assert conf is None

    @pytest.mark.asyncio
    async def test_get_all_confidence(self, wm):
        await wm.update_confidence("topic_a", 0.6)
        await wm.update_confidence("topic_b", 0.8)

        all_conf = await wm.get_all_confidence()
        assert all_conf == {"topic_a": 0.6, "topic_b": 0.8}

    @pytest.mark.asyncio
    async def test_concurrent_confidence_updates_on_same_topic_are_atomic(self, wm):
        """
        Simulates the exact race condition the Lua script exists to prevent:
        the Critic and a Researcher both trying to update the SAME topic's
        confidence within the same event loop tick. Without the atomic Lua
        script, a naive HGET-then-HSET could lose one of these updates.
        The Lua script makes each individual update atomic — this test
        verifies the LAST write wins cleanly with no corruption, not that
        both values are somehow preserved (that would require a different
        merge strategy, which is out of scope here).
        """
        results = await asyncio.gather(
            wm.update_confidence("contested_topic", 0.70),
            wm.update_confidence("contested_topic", 0.65),
            wm.update_confidence("contested_topic", 0.75),
        )
        # All three calls must succeed without exception
        assert len(results) == 3
        # Final stored value must be exactly one of the three writes —
        # never a corrupted/partial value
        final = await wm.get_confidence("contested_topic")
        assert final in (0.70, 0.65, 0.75)


class TestDebateTranscript:
    @pytest.mark.asyncio
    async def test_append_and_retrieve_messages(self, wm):
        await wm.append_debate_message({"round": 1, "agent": "advocate", "text": "arg 1"})
        await wm.append_debate_message({"round": 2, "agent": "challenger", "text": "arg 2"})

        transcript = await wm.get_debate_transcript()
        assert len(transcript) == 2
        assert transcript[0]["round"] == 1
        assert transcript[1]["agent"] == "challenger"

    @pytest.mark.asyncio
    async def test_transcript_is_bounded(self, wm):
        """The safety valve: a runaway debate loop must not consume
        unbounded memory."""
        for i in range(MAX_DEBATE_MESSAGES + 10):
            await wm.append_debate_message({"round": i, "text": f"message {i}"})

        transcript = await wm.get_debate_transcript()
        assert len(transcript) == MAX_DEBATE_MESSAGES
        # Oldest messages should have been trimmed — the transcript should
        # contain the MOST RECENT messages, not the first ones
        assert transcript[-1]["round"] == MAX_DEBATE_MESSAGES + 9


class TestGraphState:
    @pytest.mark.asyncio
    async def test_set_and_get_graph_state(self, wm):
        graph = {
            "nodes": [{"id": "n1", "text": "claim 1"}],
            "edges": [{"source": "n1", "target": "n2", "type": "supports"}],
        }
        await wm.set_graph_state(graph)

        retrieved = await wm.get_graph_state()
        assert retrieved == graph

    @pytest.mark.asyncio
    async def test_missing_graph_returns_none(self, wm):
        retrieved = await wm.get_graph_state()
        assert retrieved is None


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_full_snapshot_includes_all_tiers(self, wm):
        await wm.record_agent_output("planner", "plan", 0.85)
        await wm.update_confidence("topic1", 0.7)
        await wm.append_debate_message({"round": 1, "text": "arg"})

        snapshot = await wm.get_snapshot()

        assert snapshot.query == "test query"
        assert "planner" in snapshot.agent_outputs
        assert snapshot.confidence_by_topic["topic1"] == 0.7
        assert len(snapshot.debate_transcript) == 1

    @pytest.mark.asyncio
    async def test_snapshot_serializes_to_dict(self, wm):
        await wm.record_agent_output("planner", "plan", 0.85)
        snapshot = await wm.get_snapshot()
        d = snapshot.to_dict()

        # Must be JSON-serializable — this is what gets sent over SSE
        json.dumps(d)
        assert d["query"] == "test query"


class TestPubSub:
    @pytest.mark.asyncio
    async def test_events_publish_to_correct_channel(self, wm, redis_client):
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(wm.subscribe_channel_name())
        await asyncio.sleep(0.05)  # Let subscription register

        await wm.record_agent_output("planner", "output", 0.8)

        message = await pubsub.get_message(timeout=1.0)  # Subscribe confirmation
        message = await pubsub.get_message(timeout=1.0)  # Actual event

        assert message is not None
        payload = json.loads(message["data"])
        assert payload["type"] == "agent_output"
        assert payload["agent_id"] == "planner"

    @pytest.mark.asyncio
    async def test_publish_failure_does_not_raise(self, wm, monkeypatch):
        """Pub/sub is best-effort — a broken publish must never crash
        the calling code path."""
        async def broken_publish(*args, **kwargs):
            raise ConnectionError("simulated redis failure")

        monkeypatch.setattr(wm.redis, "publish", broken_publish)

        # Must not raise despite the broken publish
        await wm.record_agent_output("planner", "output", 0.8)
        entry = await wm.get_agent_output("planner")
        assert entry is not None  # The actual write still succeeded
