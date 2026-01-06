#!/usr/bin/env python3
"""Recompute summary from existing traces with corrected success classification.

Corrected rules:
- A request is SUCCESSFUL only if:
  a) success == True (original)
  b) tokens (completion_tokens) > 0
  c) prompt_tokens > 0
  d) latency > 0

- Failed requests are EXCLUDED from latency distributions
"""
import json
import sys
from pathlib import Path
from typing import Any


def load_traces(traces_path: Path) -> list[dict[str, Any]]:
    """Load JSONL traces file."""
    traces = []
    with open(traces_path) as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def validate_success(trace: dict) -> bool:
    """Check if trace is truly successful with corrected rules."""
    if not trace.get("success"):
        return False
    if trace.get("tokens", trace.get("completion_tokens", 0)) <= 0:
        return False
    if trace.get("prompt_tokens", 0) <= 0:
        return False
    if trace.get("latency", 0) <= 0:
        return False
    return True


def compute_corrected_summary(traces: list[dict]) -> dict[str, Any]:
    """Compute summary with corrected success classification."""
    # Reclassify traces
    truly_successful = [t for t in traces if validate_success(t)]
    truly_failed = [t for t in traces if not validate_success(t)]
    
    # Get latencies from truly successful only
    latencies = sorted([t["latency"] for t in truly_successful])
    
    summary = {
        "total_requests": len(traces),
        "successful": len(truly_successful),
        "failed": len(truly_failed),
        "failed_breakdown": {},
    }
    
    # Categorize failures
    for t in truly_failed:
        if not t.get("success"):
            reason = "original_failure"
        elif t.get("tokens", 0) == 0:
            reason = "zero_completion_tokens"
        elif t.get("prompt_tokens", 0) == 0:
            reason = "zero_prompt_tokens"
        elif t.get("latency", 0) <= 0:
            reason = "invalid_latency"
        else:
            reason = "unknown"
        summary["failed_breakdown"][reason] = summary["failed_breakdown"].get(reason, 0) + 1
    
    if latencies:
        summary["latency_min"] = latencies[0]
        summary["latency_median"] = latencies[len(latencies) // 2]
        summary["latency_p95"] = latencies[int(len(latencies) * 0.95)]
        summary["latency_max"] = latencies[-1]
        
        total_tokens = sum(t.get("tokens", 0) for t in truly_successful)
        total_prompt = sum(t.get("prompt_tokens", 0) for t in truly_successful)
        summary["total_completion_tokens"] = total_tokens
        summary["total_prompt_tokens"] = total_prompt
    else:
        summary["latency_min"] = 0
        summary["latency_median"] = 0
        summary["latency_p95"] = 0
        summary["latency_max"] = 0
    
    return summary


def load_original_summary(summary_path: Path) -> dict[str, Any]:
    """Load original summary for comparison."""
    with open(summary_path) as f:
        return json.load(f)


def print_comparison(original: dict, corrected: dict) -> str:
    """Print before/after comparison."""
    lines = []
    lines.append("=" * 80)
    lines.append("BEFORE vs AFTER COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Metric':<25} {'BEFORE':>15} {'AFTER':>15} {'DELTA':>15}")
    lines.append("-" * 80)
    
    metrics = [
        ("successful", "successful", "successful"),
        ("failed", "failed", "failed"),
        ("latency_min", "latency_min", "latency_min"),
        ("latency_median", "latency_median", "latency_median"),
        ("latency_p95", "latency_p95", "latency_p95"),
        ("latency_max", "latency_max", "latency_max"),
    ]
    
    for label, orig_key, corr_key in metrics:
        orig_val = original.get(orig_key, 0)
        corr_val = corrected.get(corr_key, 0)
        
        if isinstance(orig_val, float):
            delta = corr_val - orig_val
            lines.append(f"{label:<25} {orig_val:>15.4f} {corr_val:>15.4f} {delta:>+15.4f}")
        else:
            delta = corr_val - orig_val
            lines.append(f"{label:<25} {orig_val:>15} {corr_val:>15} {delta:>+15}")
    
    lines.append("")
    lines.append("Failed breakdown:")
    for reason, count in corrected.get("failed_breakdown", {}).items():
        lines.append(f"  {reason}: {count}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    traces_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/full_run_c1_traces.jsonl")
    original_path = traces_path.parent / "full_run_c1_summary.json"
    
    traces = load_traces(traces_path)
    corrected = compute_corrected_summary(traces)
    
    if original_path.exists():
        original = load_original_summary(original_path)
        comparison = print_comparison(original, corrected)
        print(comparison)
    
    # Save corrected summary
    output_path = traces_path.parent / "full_run_c1_summary_fixed.json"
    with open(output_path, "w") as f:
        json.dump(corrected, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    print(f"\nCorrected summary:")
    print(json.dumps(corrected, indent=2))

