"""Throughput metrics computation."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ThroughputMetrics:
    """Computed throughput metrics."""

    # Tokens per second
    output_tokens_per_sec: float
    input_tokens_per_sec: float
    total_tokens_per_sec: float

    # Requests per second
    requests_per_sec: float
    successful_requests_per_sec: float

    # Totals
    total_output_tokens: int
    total_input_tokens: int
    total_requests: int
    successful_requests: int
    failed_requests: int

    # Duration
    duration_sec: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tokens_per_sec": {
                "output": self.output_tokens_per_sec,
                "input": self.input_tokens_per_sec,
                "total": self.total_tokens_per_sec,
            },
            "requests_per_sec": {
                "total": self.requests_per_sec,
                "successful": self.successful_requests_per_sec,
            },
            "totals": {
                "output_tokens": self.total_output_tokens,
                "input_tokens": self.total_input_tokens,
                "requests": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
            },
            "duration_sec": self.duration_sec,
        }


@dataclass
class ThroughputTraceRecord:
    """Trace record for throughput computation."""

    prompt_tokens: int
    completion_tokens: int
    status: str


def compute_throughput_metrics(
    traces: Sequence[ThroughputTraceRecord],
    duration_sec: float,
) -> ThroughputMetrics:
    """Compute throughput metrics from trace records.

    Args:
        traces: Sequence of trace records
        duration_sec: Total benchmark duration in seconds

    Returns:
        Computed throughput metrics
    """
    if not traces:
        raise ValueError("No traces provided")

    if duration_sec <= 0:
        raise ValueError("Duration must be positive")

    total_output_tokens = sum(t.completion_tokens for t in traces)
    total_input_tokens = sum(t.prompt_tokens for t in traces)
    total_requests = len(traces)
    successful_requests = sum(1 for t in traces if t.status == "success")
    failed_requests = total_requests - successful_requests

    return ThroughputMetrics(
        output_tokens_per_sec=total_output_tokens / duration_sec,
        input_tokens_per_sec=total_input_tokens / duration_sec,
        total_tokens_per_sec=(total_output_tokens + total_input_tokens) / duration_sec,
        requests_per_sec=total_requests / duration_sec,
        successful_requests_per_sec=successful_requests / duration_sec,
        total_output_tokens=total_output_tokens,
        total_input_tokens=total_input_tokens,
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        duration_sec=duration_sec,
    )

