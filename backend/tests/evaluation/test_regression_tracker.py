"""
Tests for RegressionTracker (Week 4, System 2).
Run: pytest tests/evaluation/test_regression_tracker.py -v
"""
import pytest

from app.evaluation.regression_tracker import (
    RegressionTracker,
    EvalRun,
    RegressionTrackingError,
    DEFAULT_REGRESSION_TOLERANCE,
)


def make_run(label: str, query_set: str = "standard_10", **metrics) -> EvalRun:
    return EvalRun(run_label=label, benchmark_query_set=query_set, metrics=metrics)


class TestRecordAndRetrieve:
    def test_record_and_get_run(self):
        tracker = RegressionTracker()
        run = make_run("baseline", hallucination_rate=0.14)
        tracker.record_run(run)

        retrieved = tracker.get_run(run.id)
        assert retrieved is not None
        assert retrieved.run_label == "baseline"

    def test_missing_run_returns_none(self):
        tracker = RegressionTracker()
        assert tracker.get_run("nonexistent") is None

    def test_get_runs_for_query_set_filters_correctly(self):
        tracker = RegressionTracker()
        tracker.record_run(make_run("run1", query_set="set_a", hallucination_rate=0.1))
        tracker.record_run(make_run("run2", query_set="set_b", hallucination_rate=0.2))

        runs = tracker.get_runs_for_query_set("set_a")
        assert len(runs) == 1
        assert runs[0].run_label == "run1"

    def test_runs_sorted_by_timestamp(self):
        import time
        tracker = RegressionTracker()
        r1 = make_run("first", hallucination_rate=0.1)
        tracker.record_run(r1)
        time.sleep(0.01)
        r2 = make_run("second", hallucination_rate=0.1)
        tracker.record_run(r2)

        runs = tracker.get_runs_for_query_set("standard_10")
        assert runs[0].id == r1.id
        assert runs[1].id == r2.id

    def test_persist_fn_called_on_record(self):
        persisted = []
        tracker = RegressionTracker(persist_fn=lambda r: persisted.append(r.id))
        run = make_run("baseline", hallucination_rate=0.1)
        tracker.record_run(run)
        assert run.id in persisted

    def test_persist_failure_does_not_lose_in_memory_record(self):
        def broken_persist(run):
            raise RuntimeError("db connection failed")

        tracker = RegressionTracker(persist_fn=broken_persist)
        run = make_run("baseline", hallucination_rate=0.1)
        tracker.record_run(run)   # Must not raise

        assert tracker.get_run(run.id) is not None


class TestCompareRuns:
    def test_compare_computes_correct_deltas(self):
        tracker = RegressionTracker()
        baseline = make_run("baseline", hallucination_rate=0.14, grounding_rate=0.60)
        current = make_run("current", hallucination_rate=0.08, grounding_rate=0.75)
        tracker.record_run(baseline)
        tracker.record_run(current)

        deltas = tracker.compare_runs(baseline.id, current.id)
        hall_delta = next(d for d in deltas if d.metric_name == "hallucination_rate")
        ground_delta = next(d for d in deltas if d.metric_name == "grounding_rate")

        assert hall_delta.delta == pytest.approx(-0.06, abs=0.001)
        assert hall_delta.improved is True   # Lower hallucination = improvement
        assert ground_delta.delta == pytest.approx(0.15, abs=0.001)
        assert ground_delta.improved is True   # Higher grounding = improvement

    def test_compare_missing_run_raises(self):
        tracker = RegressionTracker()
        run = make_run("baseline", hallucination_rate=0.1)
        tracker.record_run(run)

        with pytest.raises(RegressionTrackingError):
            tracker.compare_runs(run.id, "nonexistent")

    def test_metric_only_in_one_run_excluded_from_comparison(self):
        tracker = RegressionTracker()
        baseline = make_run("baseline", hallucination_rate=0.1)
        current = make_run("current", hallucination_rate=0.1, new_metric=0.5)
        tracker.record_run(baseline)
        tracker.record_run(current)

        deltas = tracker.compare_runs(baseline.id, current.id)
        metric_names = {d.metric_name for d in deltas}
        assert "new_metric" not in metric_names   # Only in current, not baseline


class TestRegressionDetection:
    def test_no_prior_runs_no_regression_possible(self):
        tracker = RegressionTracker()
        run = make_run("first", hallucination_rate=0.1)
        tracker.record_run(run)

        report = tracker.detect_regression(run.id, "standard_10")
        assert report.has_regression is False

    def test_worse_metric_beyond_tolerance_flagged_as_regression(self):
        tracker = RegressionTracker(tolerance=0.03)
        baseline = make_run("baseline", hallucination_rate=0.08)
        tracker.record_run(baseline)

        worse = make_run("worse", hallucination_rate=0.20)   # +12pp, way beyond tolerance
        tracker.record_run(worse)

        report = tracker.detect_regression(worse.id, "standard_10")
        assert report.has_regression is True
        assert any(m.metric_name == "hallucination_rate" for m in report.regressed_metrics)

    def test_small_fluctuation_within_tolerance_not_flagged(self):
        tracker = RegressionTracker(tolerance=0.03)
        baseline = make_run("baseline", hallucination_rate=0.08)
        tracker.record_run(baseline)

        slight_change = make_run("slight", hallucination_rate=0.095)   # +1.5pp, within tolerance
        tracker.record_run(slight_change)

        report = tracker.detect_regression(slight_change.id, "standard_10")
        assert report.has_regression is False
        assert any(m.metric_name == "hallucination_rate" for m in report.unchanged_metrics)

    def test_improvement_not_flagged_as_regression(self):
        tracker = RegressionTracker(tolerance=0.03)
        baseline = make_run("baseline", hallucination_rate=0.20)
        tracker.record_run(baseline)

        better = make_run("better", hallucination_rate=0.08)
        tracker.record_run(better)

        report = tracker.detect_regression(better.id, "standard_10")
        assert report.has_regression is False
        assert any(m.metric_name == "hallucination_rate" for m in report.improved_metrics)

    def test_compares_against_best_prior_not_most_recent(self):
        """If run 2 was worse than run 1, and run 3 is between them,
        detection should compare against the BEST prior (run 1), not
        just the most recent (run 2) — otherwise a system could regress
        twice in a row without ever being flagged."""
        tracker = RegressionTracker(tolerance=0.03)
        best = make_run("best", hallucination_rate=0.05)
        tracker.record_run(best)
        mediocre = make_run("mediocre", hallucination_rate=0.15)
        tracker.record_run(mediocre)

        new_run = make_run("new", hallucination_rate=0.10)   # Better than mediocre, worse than best
        tracker.record_run(new_run)

        report = tracker.detect_regression(new_run.id, "standard_10")
        # Compared against best (0.05), 0.10 is a regression of +0.05
        assert report.has_regression is True
        assert report.compared_against_run_id == best.id

    def test_mixed_regression_and_improvement_across_metrics(self):
        tracker = RegressionTracker(tolerance=0.03)
        baseline = make_run("baseline", hallucination_rate=0.10, grounding_rate=0.60)
        tracker.record_run(baseline)

        mixed = make_run("mixed", hallucination_rate=0.20, grounding_rate=0.85)
        tracker.record_run(mixed)

        report = tracker.detect_regression(mixed.id, "standard_10")
        assert report.has_regression is True
        regressed_names = {m.metric_name for m in report.regressed_metrics}
        improved_names = {m.metric_name for m in report.improved_metrics}
        assert "hallucination_rate" in regressed_names
        assert "grounding_rate" in improved_names

    def test_detect_regression_missing_run_raises(self):
        tracker = RegressionTracker()
        with pytest.raises(RegressionTrackingError):
            tracker.detect_regression("nonexistent", "standard_10")


class TestTimeSeries:
    def test_timeseries_returns_chronological_values(self):
        import time
        tracker = RegressionTracker()
        tracker.record_run(make_run("r1", hallucination_rate=0.20))
        time.sleep(0.01)
        tracker.record_run(make_run("r2", hallucination_rate=0.15))
        time.sleep(0.01)
        tracker.record_run(make_run("r3", hallucination_rate=0.08))

        series = tracker.get_timeseries("standard_10", "hallucination_rate")
        values = [v for _, v in series]
        assert values == [0.20, 0.15, 0.08]

    def test_timeseries_excludes_runs_missing_the_metric(self):
        tracker = RegressionTracker()
        tracker.record_run(make_run("r1", hallucination_rate=0.1))
        tracker.record_run(make_run("r2", grounding_rate=0.5))   # No hallucination_rate

        series = tracker.get_timeseries("standard_10", "hallucination_rate")
        assert len(series) == 1


class TestSerialization:
    def test_eval_run_to_dict_json_serializable(self):
        import json
        run = make_run("baseline", hallucination_rate=0.1)
        json.dumps(run.to_dict())

    def test_regression_report_to_dict_json_serializable(self):
        import json
        tracker = RegressionTracker()
        baseline = make_run("baseline", hallucination_rate=0.1)
        tracker.record_run(baseline)
        current = make_run("current", hallucination_rate=0.2)
        tracker.record_run(current)

        report = tracker.detect_regression(current.id, "standard_10")
        json.dumps(report.to_dict())
