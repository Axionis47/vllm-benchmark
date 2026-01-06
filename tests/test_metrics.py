"""Tests for metrics computation."""

import pytest

from bench.metrics.latency import (
    TraceRecord,
    compute_latency_metrics,
    compute_stats,
    percentile,
)
from bench.metrics.throughput import (
    ThroughputTraceRecord,
    compute_throughput_metrics,
)


class TestPercentile:
    """Tests for percentile computation."""

    def test_percentile_empty(self) -> None:
        """Test percentile of empty list."""
        assert percentile([], 50) == 0.0

    def test_percentile_single(self) -> None:
        """Test percentile of single element."""
        assert percentile([10.0], 50) == 10.0
        assert percentile([10.0], 99) == 10.0

    def test_percentile_median(self) -> None:
        """Test median calculation."""
        data = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile(data, 50) == 3.0

    def test_percentile_p95(self) -> None:
        """Test p95 calculation."""
        data = sorted(range(1, 101))  # 1 to 100
        p95 = percentile([float(x) for x in data], 95)
        assert 94.0 <= p95 <= 96.0


class TestComputeStats:
    """Tests for compute_stats function."""

    def test_empty_values(self) -> None:
        """Test with empty values."""
        stats = compute_stats([])
        assert stats["mean"] == 0.0
        assert stats["p50"] == 0.0

    def test_single_value(self) -> None:
        """Test with single value."""
        stats = compute_stats([42.0])
        assert stats["mean"] == 42.0
        assert stats["p50"] == 42.0
        assert stats["stddev"] == 0.0

    def test_multiple_values(self) -> None:
        """Test with multiple values."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_stats(values)
        assert stats["mean"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0


class TestLatencyMetrics:
    """Tests for latency metrics computation."""

    def test_compute_latency_metrics(self) -> None:
        """Test latency metrics from synthetic traces."""
        traces = [
            TraceRecord(ttft_ms=50.0, total_time_ms=500.0, completion_tokens=30),
            TraceRecord(ttft_ms=45.0, total_time_ms=450.0, completion_tokens=28),
            TraceRecord(ttft_ms=55.0, total_time_ms=550.0, completion_tokens=32),
            TraceRecord(ttft_ms=48.0, total_time_ms=480.0, completion_tokens=29),
            TraceRecord(ttft_ms=52.0, total_time_ms=520.0, completion_tokens=31),
        ]

        metrics = compute_latency_metrics(traces)

        assert metrics.sample_count == 5
        assert 45.0 <= metrics.ttft_mean <= 55.0
        assert metrics.ttft_min == 45.0
        assert metrics.ttft_max == 55.0
        assert metrics.tpot_mean > 0
        assert metrics.e2e_mean > 0

    def test_empty_traces_raises(self) -> None:
        """Test that empty traces raises ValueError."""
        with pytest.raises(ValueError, match="No traces"):
            compute_latency_metrics([])

    def test_tpot_calculation(self) -> None:
        """Test TPOT is correctly calculated."""
        # TPOT = (total_time - ttft) / completion_tokens
        traces = [
            TraceRecord(ttft_ms=100.0, total_time_ms=400.0, completion_tokens=30),
        ]
        metrics = compute_latency_metrics(traces)
        # Expected TPOT = (400 - 100) / 30 = 10.0 ms
        assert abs(metrics.tpot_mean - 10.0) < 0.01


class TestThroughputMetrics:
    """Tests for throughput metrics computation."""

    def test_compute_throughput_metrics(self) -> None:
        """Test throughput metrics from synthetic traces."""
        traces = [
            ThroughputTraceRecord(prompt_tokens=50, completion_tokens=30, status="success"),
            ThroughputTraceRecord(prompt_tokens=60, completion_tokens=40, status="success"),
            ThroughputTraceRecord(prompt_tokens=55, completion_tokens=35, status="success"),
        ]

        metrics = compute_throughput_metrics(traces, duration_sec=10.0)

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 3
        assert metrics.failed_requests == 0
        assert metrics.total_output_tokens == 105
        assert metrics.total_input_tokens == 165
        assert metrics.output_tokens_per_sec == 10.5
        assert metrics.requests_per_sec == 0.3

    def test_throughput_with_failures(self) -> None:
        """Test throughput metrics with failed requests."""
        traces = [
            ThroughputTraceRecord(prompt_tokens=50, completion_tokens=30, status="success"),
            ThroughputTraceRecord(prompt_tokens=0, completion_tokens=0, status="error"),
            ThroughputTraceRecord(prompt_tokens=55, completion_tokens=35, status="success"),
        ]

        metrics = compute_throughput_metrics(traces, duration_sec=5.0)

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1

    def test_empty_traces_raises(self) -> None:
        """Test that empty traces raises ValueError."""
        with pytest.raises(ValueError, match="No traces"):
            compute_throughput_metrics([], duration_sec=1.0)

    def test_zero_duration_raises(self) -> None:
        """Test that zero duration raises ValueError."""
        traces = [
            ThroughputTraceRecord(prompt_tokens=50, completion_tokens=30, status="success"),
        ]
        with pytest.raises(ValueError, match="Duration must be positive"):
            compute_throughput_metrics(traces, duration_sec=0.0)
