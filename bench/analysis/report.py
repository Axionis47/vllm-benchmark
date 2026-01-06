"""Generate markdown reports from benchmark results."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bench.analysis.aggregate import AggregatedMetrics


def generate_report(
    metrics: AggregatedMetrics,
    output_path: Path,
    git_sha: Optional[str] = None,
    title: str = "Benchmark Report",
) -> None:
    """Generate a markdown report from aggregated metrics.

    Args:
        metrics: Aggregated benchmark metrics
        output_path: Path to write the report
        git_sha: Optional git commit SHA
        title: Report title
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    report = f"""# {title}

**Generated**: {timestamp}
**Config ID**: {metrics.config_id}
"""

    if git_sha:
        report += f"**Git SHA**: {git_sha}\n"

    report += f"""
## Summary

| Metric | Value |
|--------|-------|
| Total Requests | {metrics.throughput.total_requests} |
| Successful | {metrics.throughput.successful_requests} |
| Failed | {metrics.throughput.failed_requests} |
| Duration | {metrics.throughput.duration_sec:.2f}s |
| Throughput | {metrics.throughput.output_tokens_per_sec:.1f} tok/s |

## Latency Metrics

### Time to First Token (TTFT)

| Percentile | Value (ms) |
|------------|------------|
| Mean | {metrics.latency.ttft_mean:.2f} |
| p50 | {metrics.latency.ttft_p50:.2f} |
| p95 | {metrics.latency.ttft_p95:.2f} |
| p99 | {metrics.latency.ttft_p99:.2f} |
| Min | {metrics.latency.ttft_min:.2f} |
| Max | {metrics.latency.ttft_max:.2f} |

### Time Per Output Token (TPOT)

| Percentile | Value (ms) |
|------------|------------|
| Mean | {metrics.latency.tpot_mean:.2f} |
| p50 | {metrics.latency.tpot_p50:.2f} |
| p95 | {metrics.latency.tpot_p95:.2f} |
| p99 | {metrics.latency.tpot_p99:.2f} |
| Min | {metrics.latency.tpot_min:.2f} |
| Max | {metrics.latency.tpot_max:.2f} |

### End-to-End Latency

| Percentile | Value (ms) |
|------------|------------|
| Mean | {metrics.latency.e2e_mean:.2f} |
| p50 | {metrics.latency.e2e_p50:.2f} |
| p95 | {metrics.latency.e2e_p95:.2f} |
| p99 | {metrics.latency.e2e_p99:.2f} |
| Min | {metrics.latency.e2e_min:.2f} |
| Max | {metrics.latency.e2e_max:.2f} |

## Throughput Metrics

| Metric | Value |
|--------|-------|
| Output Tokens/sec | {metrics.throughput.output_tokens_per_sec:.2f} |
| Input Tokens/sec | {metrics.throughput.input_tokens_per_sec:.2f} |
| Requests/sec | {metrics.throughput.requests_per_sec:.2f} |

## Notes

This report was generated using {'mock' if 'mock' in metrics.config_id.lower() or 'smoke' in str(output_path).lower() else 'real'} benchmarking mode.
Sample count: {metrics.latency.sample_count} successful requests.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to {output_path}")


def generate_comparison_report(
    metrics_list: list[AggregatedMetrics],
    output_path: Path,
    title: str = "Benchmark Comparison",
) -> None:
    """Generate a comparison report for multiple configurations.

    Args:
        metrics_list: List of aggregated metrics to compare
        output_path: Path to write the report
        title: Report title
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    report = f"""# {title}

**Generated**: {timestamp}

## Configuration Comparison

### Latency (TTFT p50, ms)

| Config | Mean | p50 | p95 | p99 |
|--------|------|-----|-----|-----|
"""

    for m in metrics_list:
        report += f"| {m.config_id} | {m.latency.ttft_mean:.2f} | {m.latency.ttft_p50:.2f} | {m.latency.ttft_p95:.2f} | {m.latency.ttft_p99:.2f} |\n"

    report += """
### Throughput

| Config | Output tok/s | Requests/s |
|--------|--------------|------------|
"""

    for m in metrics_list:
        report += f"| {m.config_id} | {m.throughput.output_tokens_per_sec:.2f} | {m.throughput.requests_per_sec:.2f} |\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Comparison report written to {output_path}")

