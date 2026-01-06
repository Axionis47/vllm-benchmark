#!/usr/bin/env python3
"""Aggregate matrix benchmark results into final report.

Reads each config's full summaries and produces:
- docs/REPORT_FINAL.md with delta-vs-C1 table
- CSV with all metrics
"""
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime, timezone


def load_summary(summary_path: Path) -> dict | None:
    """Load summary.json from a config output directory."""
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def load_manifest(manifest_path: Path) -> dict | None:
    """Load run_manifest.json from a config output directory."""
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def load_gpu_peak(gpu_path: Path) -> float | None:
    """Load peak GPU memory from gpu.csv."""
    if not gpu_path.exists():
        return None
    max_mem = 0.0
    with open(gpu_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mem = float(row.get("mem_used_mib", 0))
                max_mem = max(max_mem, mem)
            except (ValueError, TypeError):
                pass
    return max_mem if max_mem > 0 else None


def compute_delta(value: float, baseline: float) -> str:
    """Compute delta as percentage string."""
    if baseline == 0:
        return "N/A"
    delta = ((value - baseline) / baseline) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def aggregate_results(results_dir: Path) -> list[dict]:
    """Aggregate results from all config directories."""
    configs = []
    
    # Find all config subdirectories
    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir() or config_dir.name.startswith("."):
            continue
        if config_dir.name in ["smoke_prompts.jsonl"]:
            continue
            
        summary = load_summary(config_dir / "summary.json")
        manifest = load_manifest(config_dir / "run_manifest.json")
        gpu_peak = load_gpu_peak(config_dir / "gpu.csv")
        
        if not summary:
            print(f"Warning: No summary found for {config_dir.name}")
            continue
            
        ns = summary.get("non_streaming", {})
        s = summary.get("streaming", {})
        
        config_data = {
            "config": config_dir.name,
            "model_id": manifest.get("model_id", "unknown") if manifest else "unknown",
            
            # Non-streaming metrics
            "ns_req_per_sec": ns.get("throughput_req_per_sec", 0),
            "ns_tokens_per_sec": ns.get("tokens_per_sec", 0),
            "ns_e2e_p50": ns.get("latency_p50", 0),
            "ns_e2e_p95": ns.get("latency_p95", 0),
            "ns_e2e_p99": ns.get("latency_p99", 0),
            "ns_success_rate": ns.get("success_rate", 0),
            
            # Streaming metrics
            "s_req_per_sec": s.get("throughput_req_per_sec", 0),
            "s_ttft_p50": s.get("ttft_p50", 0),
            "s_ttft_p95": s.get("ttft_p95", 0),
            "s_e2e_p50": s.get("latency_p50", 0),
            "s_e2e_p95": s.get("latency_p95", 0),
            "s_success_rate": s.get("success_rate", 0),
            
            # GPU
            "gpu_peak_mib": gpu_peak,
            
            # Overall
            "failure_rate": summary.get("overall", {}).get("failure_rate", 0),
        }
        configs.append(config_data)
    
    return configs


def generate_report(configs: list[dict], output_path: Path, smoke_status: dict | None = None):
    """Generate markdown report with delta-vs-C1 table."""
    
    # Get baseline (C1)
    baseline = next((c for c in configs if c["config"] == "C1"), None)
    if not baseline:
        print("Warning: No C1 baseline found")
        baseline = configs[0] if configs else {}
    
    report = f"""# vLLM Benchmark Matrix Report

**Generated:** {datetime.now(timezone.utc).isoformat()}

## Summary

This report compares vLLM configurations C1-C8 against the baseline (C1).

### Configurations Tested

| Config | Description |
|--------|-------------|
| C1 | Baseline (default vLLM settings) |
| C2 | CUDA graphs disabled (--enforce-eager) |
| C3 | Chunked prefill enabled |
| C4 | Tokenizer pool (size=4) |
| C5 | FP8 quantized weights |
| C6 | FP8 KV cache quantization |
| C7 | Swap space (16 GiB) |
| C8 | Ngram speculative decoding |

"""

    if smoke_status:
        report += """## Smoke Gate Status

| Config | Status |
|--------|--------|
"""
        for config, status in smoke_status.items():
            icon = "✓" if status == "passed" else "✗"
            report += f"| {config} | {icon} {status} |\n"
        report += "\n"

    report += """## Non-Streaming Results

| Config | Req/s | Tokens/s | E2E p50 (s) | E2E p95 (s) | E2E p99 (s) | Success Rate | Δ Req/s |
|--------|-------|----------|-------------|-------------|-------------|--------------|---------|
"""
    
    for c in configs:
        delta = compute_delta(c["ns_req_per_sec"], baseline.get("ns_req_per_sec", 1))
        report += (
            f"| {c['config']} | {c['ns_req_per_sec']:.2f} | {c['ns_tokens_per_sec']:.1f} | "
            f"{c['ns_e2e_p50']:.3f} | {c['ns_e2e_p95']:.3f} | {c['ns_e2e_p99']:.3f} | "
            f"{c['ns_success_rate']:.2%} | {delta} |\n"
        )

    report += """
## Streaming Results

| Config | Req/s | TTFT p50 (s) | TTFT p95 (s) | E2E p50 (s) | E2E p95 (s) | Δ TTFT p50 |
|--------|-------|--------------|--------------|-------------|-------------|------------|
"""
    
    for c in configs:
        delta = compute_delta(c["s_ttft_p50"], baseline.get("s_ttft_p50", 1))
        report += (
            f"| {c['config']} | {c['s_req_per_sec']:.2f} | {c['s_ttft_p50']:.3f} | "
            f"{c['s_ttft_p95']:.3f} | {c['s_e2e_p50']:.3f} | {c['s_e2e_p95']:.3f} | {delta} |\n"
        )

    report += """
## Resource Usage

| Config | Peak VRAM (MiB) | Failure Rate |
|--------|-----------------|--------------|
"""
    for c in configs:
        gpu = f"{c['gpu_peak_mib']:.0f}" if c['gpu_peak_mib'] else "N/A"
        report += f"| {c['config']} | {gpu} | {c['failure_rate']:.2%} |\n"

    report += f"""
## Methodology

- **Prompts:** 1800 total (200 per task × 3 buckets × 3 tasks)
- **Concurrency:** 4
- **Model:** mistralai/Mistral-7B-Instruct-v0.3 (or FP8 variant for C5)
- **Max model length:** 8192
- **GPU:** NVIDIA L4 (24GB)

## Notes

- All latency measurements are in seconds
- Delta values compare against C1 baseline
- Positive delta for req/s means faster; negative delta for latency means faster
"""
    
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")


def save_csv(configs: list[dict], output_path: Path):
    """Save aggregated metrics to CSV."""
    if not configs:
        return

    fieldnames = list(configs[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(configs)
    print(f"CSV saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate matrix results")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Path to full_matrix_* results directory")
    parser.add_argument("--output", type=Path, default=Path("docs/REPORT_FINAL.md"))
    parser.add_argument("--csv-output", type=Path, default=None,
                        help="Optional CSV output path")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        return 1

    print(f"Aggregating results from {args.results_dir}")

    # Load smoke status if available
    smoke_status = None
    smoke_summary = args.results_dir.parent / "smoke_matrix_*" / "matrix_summary.json"
    # Try to find the most recent smoke matrix
    smoke_dirs = list(args.results_dir.parent.glob("smoke_matrix_*"))
    if smoke_dirs:
        latest_smoke = max(smoke_dirs, key=lambda p: p.name)
        summary_path = latest_smoke / "matrix_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                smoke_data = json.load(f)
                smoke_status = {k: v["status"] for k, v in smoke_data.get("results", {}).items()}

    configs = aggregate_results(args.results_dir)

    if not configs:
        print("ERROR: No config results found")
        return 1

    print(f"Found {len(configs)} configs: {[c['config'] for c in configs]}")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Generate report
    generate_report(configs, args.output, smoke_status)

    # Save CSV if requested
    if args.csv_output:
        save_csv(configs, args.csv_output)
    else:
        # Default CSV next to the markdown
        csv_path = args.output.with_suffix(".csv")
        save_csv(configs, csv_path)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

