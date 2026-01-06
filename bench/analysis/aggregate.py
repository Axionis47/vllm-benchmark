"""Aggregate trace data into summary metrics."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from bench.metrics.latency import (
    LatencyMetrics,
    TraceRecord,
    compute_latency_metrics,
)
from bench.metrics.throughput import (
    ThroughputMetrics,
    ThroughputTraceRecord,
    compute_throughput_metrics,
)


@dataclass
class AggregatedMetrics:
    """Combined latency and throughput metrics."""

    config_id: str
    latency: LatencyMetrics
    throughput: ThroughputMetrics

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config_id": self.config_id,
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
        }


def load_traces_from_jsonl(path: Path) -> list[dict]:
    """Load trace records from JSONL file.

    Args:
        path: Path to traces.jsonl file

    Returns:
        List of trace record dictionaries
    """
    traces = []
    with open(path) as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def load_summary(path: Path) -> dict[str, object]:
    """Load summary from JSON file.

    Args:
        path: Path to summary.json file

    Returns:
        Summary dictionary
    """
    with open(path) as f:
        result: dict[str, object] = json.load(f)
        return result


def aggregate_traces(
    traces_path: Path,
    summary_path: Path | None = None,
    config_id: str = "unknown",
) -> AggregatedMetrics:
    """Aggregate traces into combined metrics.

    Args:
        traces_path: Path to traces.jsonl file
        summary_path: Optional path to summary.json for duration
        config_id: Configuration identifier

    Returns:
        Aggregated metrics
    """
    raw_traces = load_traces_from_jsonl(traces_path)

    # Convert to latency trace records
    latency_traces = [
        TraceRecord(
            ttft_ms=t["ttft_ms"],
            total_time_ms=t["total_time_ms"],
            completion_tokens=t["completion_tokens"],
        )
        for t in raw_traces
        if t.get("status") == "success"
    ]

    # Convert to throughput trace records
    throughput_traces = [
        ThroughputTraceRecord(
            prompt_tokens=t["prompt_tokens"],
            completion_tokens=t["completion_tokens"],
            status=t["status"],
        )
        for t in raw_traces
    ]

    # Get duration from summary or estimate
    duration_sec: float = 1.0
    if summary_path and summary_path.exists():
        summary = load_summary(summary_path)
        duration_val = summary.get("duration_sec")
        if isinstance(duration_val, (int, float)):
            duration_sec = float(duration_val)
    else:
        # Estimate from traces
        if raw_traces:
            total_time_ms = sum(t["total_time_ms"] for t in raw_traces)
            duration_sec = total_time_ms / 1000 / len(raw_traces)  # Rough estimate
        else:
            duration_sec = 1.0

    latency_metrics = compute_latency_metrics(latency_traces)
    throughput_metrics = compute_throughput_metrics(throughput_traces, duration_sec)

    return AggregatedMetrics(
        config_id=config_id,
        latency=latency_metrics,
        throughput=throughput_metrics,
    )


def save_metrics_csv(metrics: AggregatedMetrics, path: Path) -> None:
    """Save aggregated metrics to CSV.

    Args:
        metrics: Aggregated metrics to save
        path: Output CSV path
    """
    rows = [
        # TTFT metrics
        {
            "config_id": metrics.config_id,
            "metric": "ttft_ms",
            "mean": metrics.latency.ttft_mean,
            "p50": metrics.latency.ttft_p50,
            "p95": metrics.latency.ttft_p95,
            "p99": metrics.latency.ttft_p99,
            "min": metrics.latency.ttft_min,
            "max": metrics.latency.ttft_max,
            "stddev": metrics.latency.ttft_stddev,
        },
        # TPOT metrics
        {
            "config_id": metrics.config_id,
            "metric": "tpot_ms",
            "mean": metrics.latency.tpot_mean,
            "p50": metrics.latency.tpot_p50,
            "p95": metrics.latency.tpot_p95,
            "p99": metrics.latency.tpot_p99,
            "min": metrics.latency.tpot_min,
            "max": metrics.latency.tpot_max,
            "stddev": metrics.latency.tpot_stddev,
        },
        # E2E metrics
        {
            "config_id": metrics.config_id,
            "metric": "e2e_ms",
            "mean": metrics.latency.e2e_mean,
            "p50": metrics.latency.e2e_p50,
            "p95": metrics.latency.e2e_p95,
            "p99": metrics.latency.e2e_p99,
            "min": metrics.latency.e2e_min,
            "max": metrics.latency.e2e_max,
            "stddev": metrics.latency.e2e_stddev,
        },
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config_id", "metric", "mean", "p50", "p95", "p99", "min", "max", "stddev"],
        )
        writer.writeheader()
        writer.writerows(rows)
