"""
Integration tests for the full research pipeline.
Uses pytest-asyncio, real LLM calls are mocked.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.agents.agents import (
    OrchestratorAgent,
    PlannerAgent,
    CriticAgent,
    SynthesizerAgent,
)
from app.core.llm_client import TokenBudget
from app.schemas.research import (
    AgentStatus,
    PipelineState,
    ResearchStatus,
    SSEEvent,
    SubTask,
)


@pytest.fixture
def budget():
    return TokenBudget(max_tokens=100_000)


@pytest.fixture
def event_queue():
    return asyncio.Queue()


@pytest.fixture
def session_id():
    return str(uuid4())


@pytest.fixture
def basic_state():
    return PipelineState(
        query="What are the latest breakthroughs in quantum computing?",
        status=ResearchStatus.QUEUED,
    )


@pytest.fixture
def state_with_tasks(basic_state):
    """State pre-loaded with planner output."""
    basic_state.sub_tasks = [
        SubTask(
            heading="Hardware advances",
            objective="Research recent hardware improvements in quantum computing",
            assigned_to="researcher_a",
            scope_rules=["Focus on 2024-2025", "Exclude software layer"],
            priority=1,
        ),
        SubTask(
            heading="Commercial applications",
            objective="Find commercial quantum computing use cases",
            assigned_to="researcher_b",
            scope_rules=["Focus on real deployments", "Not theoretical"],
            priority=2,
        ),
    ]
    return basic_state


class TestOrchestratorAgent:
    @pytest.mark.asyncio
    async def test_orchestrator_classifies_query(self, budget, event_queue, session_id, basic_state):
        mock_response = json.dumps({
            "complexity": "moderate",
            "estimated_subtopics": 4,
            "recommended_depth": "standard",
            "key_dimensions": ["hardware", "error_correction", "commercial"],
            "domain": "technology",
            "temporal_sensitivity": "recent",
        })

        with patch.object(
            OrchestratorAgent, '_execute', new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = basic_state

            agent = OrchestratorAgent(budget, event_queue, session_id)
            result = await agent.run(basic_state)

        assert result is not None
        assert mock_exec.called

    @pytest.mark.asyncio
    async def test_orchestrator_handles_llm_failure(self, budget, event_queue, session_id, basic_state):
        """If LLM fails, orchestrator should raise and set error status."""
        with patch('app.core.agents.agents.LLMClient.complete', side_effect=Exception("LLM timeout")):
            agent = OrchestratorAgent(budget, event_queue, session_id)
            with pytest.raises(Exception, match="LLM timeout"):
                await agent.run(basic_state)

            trace = basic_state.agent_traces.get("orchestrator")
            assert trace is not None
            assert trace.status == AgentStatus.ERROR


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_planner_creates_subtasks(self, budget, event_queue, session_id, basic_state):
        mock_plan = json.dumps([
            {
                "heading": "Hardware advances",
                "objective": "Research quantum hardware",
                "assigned_to": "researcher_a",
                "allowed_sources": ["web", "arxiv"],
                "scope_rules": ["Focus on 2024-2025"],
                "priority": 1,
            },
            {
                "heading": "Error correction",
                "objective": "Research quantum error correction",
                "assigned_to": "researcher_b",
                "allowed_sources": ["arxiv"],
                "scope_rules": ["Theoretical aspects only"],
                "priority": 2,
            },
        ])

        with patch('app.core.agents.agents.LLMClient.complete', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_plan
            agent = PlannerAgent(budget, event_queue, session_id)
            result = await agent.run(basic_state)

        assert len(result.sub_tasks) == 2
        assert result.sub_tasks[0].heading == "Hardware advances"
        assert result.sub_tasks[0].assigned_to == "researcher_a"
        assert result.sub_tasks[1].assigned_to == "researcher_b"

    @pytest.mark.asyncio
    async def test_planner_alternates_assignments(self, budget, event_queue, session_id, basic_state):
        """Verifies tasks are distributed across both researchers."""
        mock_plan = json.dumps([
            {"heading": f"Task {i}", "objective": f"Objective {i}",
             "assigned_to": "researcher_a" if i % 2 == 0 else "researcher_b",
             "priority": i + 1}
            for i in range(4)
        ])

        with patch('app.core.agents.agents.LLMClient.complete', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_plan
            agent = PlannerAgent(budget, event_queue, session_id)
            result = await agent.run(basic_state)

        a_tasks = [t for t in result.sub_tasks if t.assigned_to == "researcher_a"]
        b_tasks = [t for t in result.sub_tasks if t.assigned_to == "researcher_b"]
        assert len(a_tasks) > 0
        assert len(b_tasks) > 0


class TestCriticAgent:
    @pytest.mark.asyncio
    async def test_critic_triggers_refinement_when_needed(
        self, budget, event_queue, session_id, state_with_tasks
    ):
        state_with_tasks.sub_tasks[0].status = AgentStatus.DONE
        state_with_tasks.sub_tasks[0].findings = ["Finding 1", "Finding 2"]
        state_with_tasks.sub_tasks[1].status = AgentStatus.DONE
        state_with_tasks.sub_tasks[1].findings = ["Finding 3"]

        mock_critique = json.dumps({
            "coverage_gaps": ["China quantum program not covered"],
            "contradictions": [],
            "confidence_scores": {"Hardware advances": 0.85, "Commercial applications": 0.72},
            "quality_issues": [],
            "refinement_needed": True,
            "refinement_targets": ["China quantum investment and national programs"],
        })

        with patch('app.core.agents.agents.LLMClient.complete', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_critique
            agent = CriticAgent(budget, event_queue, session_id)
            result = await agent.run(state_with_tasks)

        assert result.refinement_count == 1
        assert result.status == ResearchStatus.REFINING
        # Gap-fill task should be added
        gap_tasks = [t for t in result.sub_tasks if "Gap fill" in t.heading]
        assert len(gap_tasks) >= 1

    @pytest.mark.asyncio
    async def test_critic_respects_max_iterations(self, budget, event_queue, session_id, state_with_tasks):
        """Critic should NOT trigger refinement if already at max iterations."""
        state_with_tasks.refinement_count = 2  # Already at max
        state_with_tasks.sub_tasks[0].status = AgentStatus.DONE
        state_with_tasks.sub_tasks[0].findings = ["Finding 1"]

        mock_critique = json.dumps({
            "coverage_gaps": ["Still missing something"],
            "refinement_needed": True,
            "refinement_targets": ["topic X"],
            "confidence_scores": {},
            "contradictions": [],
            "quality_issues": [],
        })

        with patch('app.core.agents.agents.LLMClient.complete', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_critique
            agent = CriticAgent(budget, event_queue, session_id)
            result = await agent.run(state_with_tasks)

        # Should NOT trigger another refinement (count already at max=2)
        assert result.refinement_count == 2


class TestPipelineState:
    def test_pipeline_state_serialization(self, basic_state):
        """PipelineState must serialize/deserialize cleanly."""
        json_str = basic_state.model_dump_json()
        restored = PipelineState.model_validate_json(json_str)
        assert restored.query == basic_state.query
        assert restored.session_id == basic_state.session_id

    def test_token_budget_enforcement(self):
        budget = TokenBudget(max_tokens=100)

        async def test():
            await budget.consume(50)
            assert budget.used == 50
            assert budget.remaining == 50
            with pytest.raises(RuntimeError, match="Token budget exceeded"):
                await budget.consume(60)

        asyncio.run(test())
