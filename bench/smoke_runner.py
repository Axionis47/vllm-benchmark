#!/usr/bin/env python3
"""Smoke test runner for CI validation.

This module runs a quick benchmark using the mock client to validate
the entire pipeline works correctly without requiring GPU hardware.
"""

import asyncio
import subprocess
from pathlib import Path

from bench.analysis.aggregate import aggregate_traces, save_metrics_csv
from bench.analysis.report import generate_report
from bench.runner.loadgen import run_mock_benchmark


def get_git_sha() -> str | None:
    """Get current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


async def run_smoke_test() -> None:
    """Run the smoke test and generate outputs."""
    print("=" * 60)
    print("LLM Inference Benchmark - Smoke Test")
    print("=" * 60)

    # Output directories
    results_dir = Path("results/smoke_test")
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path("docs/REPORT_SMOKE.md")

    # Run mock benchmark
    print("\n[1/4] Running mock benchmark...")
    result = await run_mock_benchmark(
        num_requests=20,
        concurrency=2,
        output_dir=results_dir,
    )

    print(f"      Completed {result.successful_requests}/{result.total_requests} requests")
    print(f"      Duration: {result.duration_sec:.2f}s")

    # Aggregate metrics
    print("\n[2/4] Aggregating metrics...")
    metrics = aggregate_traces(
        traces_path=results_dir / "traces.jsonl",
        summary_path=results_dir / "summary.json",
        config_id="smoke_test_mock",
    )

    # Save metrics CSV
    print("\n[3/4] Saving metrics CSV...")
    save_metrics_csv(metrics, results_dir / "metrics.csv")

    # Generate report
    print("\n[4/4] Generating report...")
    git_sha = get_git_sha()
    generate_report(
        metrics=metrics,
        output_path=report_path,
        git_sha=git_sha,
        title="Smoke Test Report",
    )

    # Summary
    print("\n" + "=" * 60)
    print("Smoke Test Complete!")
    print("=" * 60)
    print("\nOutputs:")
    print(f"  - Traces:  {results_dir / 'traces.jsonl'}")
    print(f"  - Summary: {results_dir / 'summary.json'}")
    print(f"  - Metrics: {results_dir / 'metrics.csv'}")
    print(f"  - Report:  {report_path}")

    print("\nKey Metrics (mock):")
    print(f"  - TTFT p50: {metrics.latency.ttft_p50:.2f} ms")
    print(f"  - TPOT p50: {metrics.latency.tpot_p50:.2f} ms")
    print(f"  - Throughput: {metrics.throughput.output_tokens_per_sec:.1f} tok/s")


def main() -> None:
    """Entry point for smoke test."""
    asyncio.run(run_smoke_test())


if __name__ == "__main__":
    main()

