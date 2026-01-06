"""Latency metrics computation."""

from dataclasses import dataclass
from typing import Sequence
import statistics


@dataclass
class LatencyMetrics:
    """Computed latency metrics."""

    # Time to First Token (ms)
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    ttft_min: float
    ttft_max: float
    ttft_stddev: float

    # Time Per Output Token (ms)
    tpot_mean: float
    tpot_p50: float
    tpot_p95: float
    tpot_p99: float
    tpot_min: float
    tpot_max: float
    tpot_stddev: float

    # End-to-end latency (ms)
    e2e_mean: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float
    e2e_min: float
    e2e_max: float
    e2e_stddev: float

    sample_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "ttft": {
                "mean": self.ttft_mean,
                "p50": self.ttft_p50,
                "p95": self.ttft_p95,
                "p99": self.ttft_p99,
                "min": self.ttft_min,
                "max": self.ttft_max,
                "stddev": self.ttft_stddev,
            },
            "tpot": {
                "mean": self.tpot_mean,
                "p50": self.tpot_p50,
                "p95": self.tpot_p95,
                "p99": self.tpot_p99,
                "min": self.tpot_min,
                "max": self.tpot_max,
                "stddev": self.tpot_stddev,
            },
            "e2e": {
                "mean": self.e2e_mean,
                "p50": self.e2e_p50,
                "p95": self.e2e_p95,
                "p99": self.e2e_p99,
                "min": self.e2e_min,
                "max": self.e2e_max,
                "stddev": self.e2e_stddev,
            },
            "sample_count": self.sample_count,
        }


def percentile(data: Sequence[float], p: float) -> float:
    """Compute percentile of a sorted sequence.

    Args:
        data: Sorted sequence of values
        p: Percentile (0-100)

    Returns:
        Value at the given percentile
    """
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


def compute_stats(values: Sequence[float]) -> dict:
    """Compute summary statistics for a sequence of values.

    Args:
        values: Sequence of numeric values

    Returns:
        Dictionary with mean, p50, p95, p99, min, max, stddev
    """
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stddev": 0.0,
        }

    sorted_values = sorted(values)
    return {
        "mean": statistics.mean(values),
        "p50": percentile(sorted_values, 50),
        "p95": percentile(sorted_values, 95),
        "p99": percentile(sorted_values, 99),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


@dataclass
class TraceRecord:
    """Single trace record from benchmark."""

    ttft_ms: float
    total_time_ms: float
    completion_tokens: int


def compute_latency_metrics(traces: Sequence[TraceRecord]) -> LatencyMetrics:
    """Compute latency metrics from trace records.

    Args:
        traces: Sequence of trace records

    Returns:
        Computed latency metrics
    """
    if not traces:
        raise ValueError("No traces provided")

    # Extract values
    ttft_values = [t.ttft_ms for t in traces]
    e2e_values = [t.total_time_ms for t in traces]

    # TPOT = (total_time - ttft) / completion_tokens
    tpot_values = []
    for t in traces:
        if t.completion_tokens > 0:
            decode_time = t.total_time_ms - t.ttft_ms
            tpot = decode_time / t.completion_tokens
            tpot_values.append(tpot)

    ttft_stats = compute_stats(ttft_values)
    tpot_stats = compute_stats(tpot_values)
    e2e_stats = compute_stats(e2e_values)

    return LatencyMetrics(
        ttft_mean=ttft_stats["mean"],
        ttft_p50=ttft_stats["p50"],
        ttft_p95=ttft_stats["p95"],
        ttft_p99=ttft_stats["p99"],
        ttft_min=ttft_stats["min"],
        ttft_max=ttft_stats["max"],
        ttft_stddev=ttft_stats["stddev"],
        tpot_mean=tpot_stats["mean"],
        tpot_p50=tpot_stats["p50"],
        tpot_p95=tpot_stats["p95"],
        tpot_p99=tpot_stats["p99"],
        tpot_min=tpot_stats["min"],
        tpot_max=tpot_stats["max"],
        tpot_stddev=tpot_stats["stddev"],
        e2e_mean=e2e_stats["mean"],
        e2e_p50=e2e_stats["p50"],
        e2e_p95=e2e_stats["p95"],
        e2e_p99=e2e_stats["p99"],
        e2e_min=e2e_stats["min"],
        e2e_max=e2e_stats["max"],
        e2e_stddev=e2e_stats["stddev"],
        sample_count=len(traces),
    )

