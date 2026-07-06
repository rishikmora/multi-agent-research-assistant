"""
Regression Tracking Store — Week 4, System 2

WHY THIS EXISTS
Running FActScore once and printing a number is a demo. Running it after
every pipeline change, storing the result, and being able to say "this
prompt change moved hallucination rate from 14% to 8% across the same 10
benchmark queries" is engineering. Regression tracking is the difference
between "I think it got better" and a defensible, measured claim — and
it is table stakes in any production ML system, because model behavior,
prompt behavior, and retrieval behavior all drift over time in ways that
are invisible without a persistent baseline to compare against.

DESIGN
- Every evaluation run is stored as an immutable EvalRun record, tagged
  with a `run_label` (e.g. "baseline", "after-mar-integration",
  "gpt-5-migration") so runs can be grouped and compared meaningfully.
- `compare_runs()` computes per-metric deltas between any two labeled
  runs — this is what powers "before vs after" reporting.
- `detect_regression()` flags when a NEW run's metrics are meaningfully
  worse than the best prior run on the same benchmark query set, using a
  configurable tolerance band (small fluctuation is normal noise, not a
  regression — the threshold exists specifically to avoid false alarms
  on every run).
- Storage here is an in-memory store with a pluggable persistence hook
  (`persist_fn`) — production deployments wire this to PostgreSQL, but
  the tracking LOGIC (comparison, regression detection) is fully
  independent of where the rows physically live, which is what's under
  test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

# A metric moving by less than this fraction is treated as noise, not a
# real regression — prevents false-alarm flapping on every run.
DEFAULT_REGRESSION_TOLERANCE = 0.03   # 3 percentage points

# Metrics where LOWER is better — regression detection must invert direction
LOWER_IS_BETTER_METRICS = {"hallucination_rate", "contradiction_rate"}


@dataclass
class EvalRun:
    """One immutable evaluation snapshot."""
    id: str = field(default_factory=lambda: str(uuid4()))
    run_label: str = ""              # e.g. "baseline", "after-mar-integration"
    benchmark_query_set: str = ""    # Identifies which fixed query set was used
    session_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_label": self.run_label,
            "benchmark_query_set": self.benchmark_query_set,
            "session_id": self.session_id,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MetricDelta:
    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    delta_pct: float | None
    improved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_value": round(self.baseline_value, 4),
            "current_value": round(self.current_value, 4),
            "delta": round(self.delta, 4),
            "delta_pct": round(self.delta_pct, 2) if self.delta_pct is not None else None,
            "improved": self.improved,
        }


@dataclass
class RegressionReport:
    has_regression: bool
    regressed_metrics: list[MetricDelta] = field(default_factory=list)
    improved_metrics: list[MetricDelta] = field(default_factory=list)
    unchanged_metrics: list[MetricDelta] = field(default_factory=list)
    compared_against_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_regression": self.has_regression,
            "regressed_metrics": [m.to_dict() for m in self.regressed_metrics],
            "improved_metrics": [m.to_dict() for m in self.improved_metrics],
            "unchanged_metrics": [m.to_dict() for m in self.unchanged_metrics],
            "compared_against_run_id": self.compared_against_run_id,
        }


class RegressionTrackingError(Exception):
    pass


def _is_improvement(metric_name: str, delta: float) -> bool:
    """delta = current - baseline. Direction of 'improvement' depends on
    whether the metric is lower-is-better or higher-is-better."""
    if metric_name in LOWER_IS_BETTER_METRICS:
        return delta < 0
    return delta > 0


class RegressionTracker:
    """
    Stores and compares evaluation runs over time.

    Usage:
        tracker = RegressionTracker()
        tracker.record_run(EvalRun(run_label="baseline", metrics={"hallucination_rate": 0.14, ...}))
        tracker.record_run(EvalRun(run_label="after-mar", metrics={"hallucination_rate": 0.08, ...}))

        report = tracker.detect_regression(
            new_run_id=..., benchmark_query_set="standard_10"
        )
    """

    def __init__(
        self,
        tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
        persist_fn: Callable[[EvalRun], None] | None = None,
    ):
        self.tolerance = tolerance
        self._persist_fn = persist_fn
        self._runs: dict[str, EvalRun] = {}

    def record_run(self, run: EvalRun) -> None:
        self._runs[run.id] = run
        if self._persist_fn is not None:
            try:
                self._persist_fn(run)
            except Exception as exc:
                # Persistence failure must not lose the in-memory record —
                # the run is still trackable this session even if the
                # durable write failed.
                log.warning("regression_tracker.persist_failed",
                          run_id=run.id, error=str(exc))
        log.info("regression_tracker.run_recorded",
                run_id=run.id, label=run.run_label,
                query_set=run.benchmark_query_set)

    def get_run(self, run_id: str) -> EvalRun | None:
        return self._runs.get(run_id)

    def get_runs_for_query_set(self, benchmark_query_set: str) -> list[EvalRun]:
        return sorted(
            (r for r in self._runs.values() if r.benchmark_query_set == benchmark_query_set),
            key=lambda r: r.timestamp,
        )

    def get_best_prior_run(
        self, benchmark_query_set: str, metric_name: str, exclude_run_id: str | None = None
    ) -> EvalRun | None:
        """Finds the best-performing prior run on a given metric, for
        regression comparison. 'Best' respects metric direction."""
        candidates = [
            r for r in self.get_runs_for_query_set(benchmark_query_set)
            if r.id != exclude_run_id and metric_name in r.metrics
        ]
        if not candidates:
            return None

        lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
        return min(
            candidates, key=lambda r: r.metrics[metric_name]
        ) if lower_is_better else max(
            candidates, key=lambda r: r.metrics[metric_name]
        )

    def compare_runs(self, baseline_run_id: str, current_run_id: str) -> list[MetricDelta]:
        baseline = self._runs.get(baseline_run_id)
        current = self._runs.get(current_run_id)
        if baseline is None or current is None:
            raise RegressionTrackingError(
                f"Run not found: baseline={baseline_run_id}, current={current_run_id}"
            )

        deltas = []
        all_metric_names = set(baseline.metrics) | set(current.metrics)
        for name in sorted(all_metric_names):
            baseline_val = baseline.metrics.get(name)
            current_val = current.metrics.get(name)
            if baseline_val is None or current_val is None:
                continue

            delta = current_val - baseline_val
            delta_pct = (delta / baseline_val * 100) if baseline_val != 0 else None

            deltas.append(MetricDelta(
                metric_name=name,
                baseline_value=baseline_val,
                current_value=current_val,
                delta=delta,
                delta_pct=delta_pct,
                improved=_is_improvement(name, delta),
            ))
        return deltas

    def detect_regression(
        self, new_run_id: str, benchmark_query_set: str
    ) -> RegressionReport:
        """
        Compares a new run against the best prior run on the same
        benchmark query set, per metric. Flags a metric as regressed only
        if it moved AGAINST the improvement direction by more than
        `self.tolerance` — small fluctuation within tolerance is not
        flagged, preventing noisy false alarms on every single run.
        """
        new_run = self._runs.get(new_run_id)
        if new_run is None:
            raise RegressionTrackingError(f"Run not found: {new_run_id}")

        regressed: list[MetricDelta] = []
        improved: list[MetricDelta] = []
        unchanged: list[MetricDelta] = []
        best_comparison_run_id = ""

        for metric_name, current_value in new_run.metrics.items():
            best_prior = self.get_best_prior_run(
                benchmark_query_set, metric_name, exclude_run_id=new_run_id
            )
            if best_prior is None:
                continue   # No baseline to compare against yet

            best_comparison_run_id = best_prior.id
            baseline_value = best_prior.metrics[metric_name]
            delta = current_value - baseline_value
            delta_pct = (delta / baseline_value * 100) if baseline_value != 0 else None

            metric_delta = MetricDelta(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                delta=delta,
                delta_pct=delta_pct,
                improved=_is_improvement(metric_name, delta),
            )

            if abs(delta) < self.tolerance:
                unchanged.append(metric_delta)
            elif metric_delta.improved:
                improved.append(metric_delta)
            else:
                regressed.append(metric_delta)

        report = RegressionReport(
            has_regression=len(regressed) > 0,
            regressed_metrics=regressed,
            improved_metrics=improved,
            unchanged_metrics=unchanged,
            compared_against_run_id=best_comparison_run_id,
        )

        if report.has_regression:
            log.warning("regression_tracker.regression_detected",
                      run_id=new_run_id,
                      regressed=[m.metric_name for m in regressed])
        else:
            log.info("regression_tracker.no_regression",
                    run_id=new_run_id, improved=[m.metric_name for m in improved])

        return report

    def get_timeseries(
        self, benchmark_query_set: str, metric_name: str
    ) -> list[tuple[datetime, float]]:
        """Time-ordered (timestamp, value) pairs for charting a metric's
        history — this is what the dashboard's time-series chart consumes."""
        runs = self.get_runs_for_query_set(benchmark_query_set)
        return [
            (r.timestamp, r.metrics[metric_name])
            for r in runs if metric_name in r.metrics
        ]
