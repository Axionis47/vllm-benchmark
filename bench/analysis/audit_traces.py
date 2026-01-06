#!/usr/bin/env python3
"""Audit trace files for measurement bugs and invariant violations."""

import json
import sys
from pathlib import Path
from typing import Any


def load_traces(traces_path: Path) -> list[dict[str, Any]]:
    """Load JSONL traces file."""
    traces = []
    with open(traces_path) as f:
        for i, line in enumerate(f):
            if line.strip():
                trace = json.loads(line)
                trace.setdefault("line_num", i + 1)
                traces.append(trace)
    return traces


def audit_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit traces for measurement bugs and invariant violations."""
    findings = {
        "total_traces": len(traces),
        "zero_token_traces": [],
        "zero_prompt_token_traces": [],
        "negative_latency_traces": [],
        "very_low_latency_traces": [],  # < 0.1s with tokens
        "timestamp_violations": [],
        "lowest_latency_traces": [],
        "summary": {},
    }

    # Sort by latency to find minimum
    sorted_by_latency = sorted(traces, key=lambda x: x.get("latency", float("inf")))

    for trace in traces:
        idx = trace.get("idx", trace.get("line_num", "?"))
        latency = trace.get("latency", 0)
        tokens = trace.get("tokens", trace.get("completion_tokens", 0))
        prompt_tokens = trace.get("prompt_tokens", 0)
        success = trace.get("success", False)

        # Check for zero completion tokens marked as success
        if success and tokens == 0:
            findings["zero_token_traces"].append({
                "idx": idx,
                "latency": latency,
                "tokens": tokens,
                "prompt_tokens": prompt_tokens,
                "task": trace.get("task"),
                "bucket": trace.get("bucket"),
            })

        # Check for zero prompt tokens (response parsing failure)
        if success and prompt_tokens == 0:
            findings["zero_prompt_token_traces"].append({"idx": idx, "latency": latency})

        # Check for negative latency
        if latency < 0:
            findings["negative_latency_traces"].append({"idx": idx, "latency": latency})

        # Check for suspiciously low latency with tokens
        if success and tokens > 0 and latency < 0.1:
            findings["very_low_latency_traces"].append({
                "idx": idx, "latency": latency, "tokens": tokens
            })

    # Top 15 lowest latency
    findings["lowest_latency_traces"] = [
        {
            "idx": t.get("idx", t.get("line_num")),
            "latency": t.get("latency"),
            "tokens": t.get("tokens", t.get("completion_tokens", 0)),
            "prompt_tokens": t.get("prompt_tokens", 0),
            "task": t.get("task"),
            "bucket": t.get("bucket"),
            "success": t.get("success"),
        }
        for t in sorted_by_latency[:15]
    ]

    # Summary
    findings["summary"] = {
        "total_traces": len(traces),
        "zero_token_count": len(findings["zero_token_traces"]),
        "zero_prompt_token_count": len(findings["zero_prompt_token_traces"]),
        "negative_latency_count": len(findings["negative_latency_traces"]),
        "very_low_latency_count": len(findings["very_low_latency_traces"]),
        "min_latency": min(t.get("latency", float("inf")) for t in traces),
        "max_latency": max(t.get("latency", 0) for t in traces),
    }

    return findings


def print_audit_report(findings: dict[str, Any]) -> str:
    """Generate human-readable audit report."""
    lines = []
    lines.append("=" * 80)
    lines.append("TRACE AUDIT REPORT")
    lines.append("=" * 80)

    s = findings["summary"]
    lines.append(f"\nTotal traces: {s['total_traces']}")
    lines.append(f"Zero token traces (success=true, tokens=0): {s['zero_token_count']}")
    lines.append(f"Zero prompt token traces: {s['zero_prompt_token_count']}")
    lines.append(f"Negative latency traces: {s['negative_latency_count']}")
    lines.append(f"Very low latency (<0.1s with tokens): {s['very_low_latency_count']}")
    lines.append(f"Min latency: {s['min_latency']:.6f}s")
    lines.append(f"Max latency: {s['max_latency']:.6f}s")

    lines.append("\n" + "-" * 80)
    lines.append("TOP 15 LOWEST LATENCY TRACES")
    lines.append("-" * 80)
    lines.append(f"{'idx':>6} {'latency':>12} {'tokens':>8} {'prompt_tok':>10} {'task':>15} {'bucket':>6}")
    for t in findings["lowest_latency_traces"]:
        lines.append(f"{t['idx']:>6} {t['latency']:>12.6f} {t['tokens']:>8} {t['prompt_tokens']:>10} {t['task'] or 'N/A':>15} {t['bucket'] or 'N/A':>6}")

    if findings["zero_token_traces"]:
        lines.append("\n" + "-" * 80)
        lines.append(f"ZERO TOKEN TRACES (first 10 of {len(findings['zero_token_traces'])})")
        lines.append("-" * 80)
        for t in findings["zero_token_traces"][:10]:
            lines.append(f"  idx={t['idx']}, latency={t['latency']:.6f}s, task={t['task']}, bucket={t['bucket']}")

    lines.append("\n" + "=" * 80)
    lines.append("ROOT CAUSE ANALYSIS")
    lines.append("=" * 80)
    if s["zero_token_count"] > 0:
        lines.append(f"\n⚠️  CRITICAL: {s['zero_token_count']} traces marked success=true with tokens=0")
        lines.append("   This is a SUCCESS CLASSIFICATION BUG.")
        lines.append("   Requests returning 0 tokens should be marked as FAILED.")
        lines.append(f"   The minimum latency of {s['min_latency']:.6f}s is from these invalid traces.")

    return "\n".join(lines)


if __name__ == "__main__":
    traces_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/full_run_c1_traces.jsonl")
    traces = load_traces(traces_path)
    findings = audit_traces(traces)
    report = print_audit_report(findings)
    print(report)
    # Also output JSON
    output_dir = traces_path.parent
    with open(output_dir / "audit_traces_before.json", "w") as f:
        json.dump(findings, f, indent=2)
    with open(output_dir / "audit_traces_before.txt", "w") as f:
        f.write(report)
    print(f"\nSaved: {output_dir}/audit_traces_before.json")
    print(f"Saved: {output_dir}/audit_traces_before.txt")

