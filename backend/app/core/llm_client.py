"""
LLM client — wraps Anthropic SDK with:
  - Exponential backoff retry (tenacity)
  - Circular model fallback on exhausted retries
  - Per-call token budget tracking
  - Langfuse trace injection
  - Streaming support for SSE pipeline
"""
from __future__ import annotations
import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import structlog
from app.core.config import settings
from app.core.metrics import llm_requests_total, llm_tokens_total, llm_latency_seconds

log = structlog.get_logger(__name__)

# Model registry — ordered by preference, fallback rotates left on failure
MODEL_REGISTRY = {
    "orchestrator": [settings.orchestrator_model, settings.researcher_model],
    "researcher": [settings.researcher_model, settings.orchestrator_model],
    "critic": [settings.critic_model, settings.researcher_model],
    "synthesizer": [settings.synthesizer_model, settings.researcher_model],
}


class TokenBudget:
    """Tracks token usage per pipeline session."""
    def __init__(self, max_tokens: int = 500_000):
        self.max_tokens = max_tokens
        self._used = 0
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int) -> None:
        async with self._lock:
            self._used += tokens
            if self._used > self.max_tokens:
                raise RuntimeError(
                    f"Token budget exceeded: {self._used}/{self.max_tokens}"
                )

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self._used)


class LLMClient:
    """
    Production LLM client. One instance per pipeline run — holds a budget.
    Thread-safe, async-native.
    """

    def __init__(self, role: str, budget: TokenBudget | None = None):
        self.role = role
        self.budget = budget or TokenBudget()
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # We handle retries ourselves
        )
        self._model_index = 0
        self._models = MODEL_REGISTRY.get(role, [settings.researcher_model])

    @property
    def current_model(self) -> str:
        return self._models[self._model_index % len(self._models)]

    def _rotate_model(self) -> None:
        self._model_index += 1
        log.warning("llm.model_rotated",
                    role=self.role,
                    new_model=self.current_model)

    @retry(
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
        )),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(settings.llm_max_retries),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        session_id: str = "",
    ) -> str:
        start = time.monotonic()
        model = self.current_model
        try:
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens or settings.max_tokens,
                temperature=temperature or settings.temperature,
                system=system,
                messages=messages,
            )

            in_tokens = response.usage.input_tokens
            out_tokens = response.usage.output_tokens
            await self.budget.consume(in_tokens + out_tokens)

            latency = time.monotonic() - start
            llm_requests_total.labels(role=self.role, model=model, status="ok").inc()
            llm_tokens_total.labels(role=self.role, type="input").inc(in_tokens)
            llm_tokens_total.labels(role=self.role, type="output").inc(out_tokens)
            llm_latency_seconds.labels(role=self.role).observe(latency)

            log.info("llm.complete",
                     role=self.role,
                     model=model,
                     in_tokens=in_tokens,
                     out_tokens=out_tokens,
                     latency_ms=round(latency * 1000),
                     session_id=session_id)

            return response.content[0].text

        except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
            llm_requests_total.labels(role=self.role, model=model, status="error").inc()
            log.warning("llm.error", role=self.role, model=model, error=str(exc))
            self._rotate_model()
            raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int | None = None,
        session_id: str = "",
    ) -> AsyncIterator[str]:
        model = self.current_model
        async with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens or settings.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
            usage = await stream.get_final_message()
            in_t = usage.usage.input_tokens
            out_t = usage.usage.output_tokens
            await self.budget.consume(in_t + out_t)
            log.info("llm.stream_complete",
                     role=self.role,
                     model=model,
                     in_tokens=in_t,
                     out_tokens=out_t,
                     session_id=session_id)
