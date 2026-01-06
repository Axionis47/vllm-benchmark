"""Metrics computation for benchmark results."""

from bench.metrics.latency import compute_latency_metrics, LatencyMetrics
from bench.metrics.throughput import compute_throughput_metrics, ThroughputMetrics

__all__ = [
    "compute_latency_metrics",
    "LatencyMetrics",
    "compute_throughput_metrics",
    "ThroughputMetrics",
]

