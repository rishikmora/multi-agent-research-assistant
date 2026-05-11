"""
Prometheus metrics — collected async, scraped by Prometheus every 15s.
Exposed at /metrics endpoint.
"""
from prometheus_client import Counter, Histogram, Gauge, Summary

# LLM metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM API calls",
    ["role", "model", "status"],
)
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total tokens consumed",
    ["role", "type"],  # type: input | output
)
llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM call latency",
    ["role"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
)

# Pipeline metrics
pipeline_runs_total = Counter(
    "pipeline_runs_total",
    "Total research pipeline runs",
    ["status"],
)
pipeline_duration_seconds = Histogram(
    "pipeline_duration_seconds",
    "End-to-end pipeline duration",
    buckets=[5, 10, 30, 60, 120, 180, 300],
)
pipeline_active_gauge = Gauge(
    "pipeline_active_count",
    "Currently running pipelines",
)
refinement_iterations_total = Counter(
    "refinement_iterations_total",
    "Total critic refinement iterations triggered",
)

# Source metrics
sources_retrieved_total = Counter(
    "sources_retrieved_total",
    "Sources retrieved by type",
    ["source_type"],
)
confidence_score_summary = Summary(
    "confidence_score",
    "Distribution of report confidence scores",
)

# API metrics
api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "path", "status_code"],
)
api_latency_seconds = Histogram(
    "api_latency_seconds",
    "API endpoint latency",
    ["method", "path"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)
