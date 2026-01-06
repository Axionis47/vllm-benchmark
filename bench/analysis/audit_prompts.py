#!/usr/bin/env python3
"""Audit prompts.jsonl to verify S/M/L bucket constraints with CONTEXT-SAFE accounting.

CONTEXT-SAFE ACCOUNTING:
  - request_prompt_tokens computed via apply_chat_template (includes [INST] overhead)
  - Buckets based on request_prompt_tokens (not raw prompt tokens)
  - request_prompt_tokens + max_new_tokens <= MAX_MODEL_LEN (8192) always

Bucket constraints (on request_prompt_tokens):
  S: 1-512 tokens
  M: 513-2048 tokens
  L: 2049-8192 tokens

Uses the same tokenizer as the serving model (Mistral).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


BUCKET_CONSTRAINTS = {
    "S": (1, 512),
    "M": (513, 2048),
    "L": (2049, 8192),
}

MAX_MODEL_LEN = 8192


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    """Load JSONL prompts file."""
    prompts = []
    with open(prompts_path) as f:
        for i, line in enumerate(f):
            if line.strip():
                prompt = json.loads(line)
                prompt.setdefault("line_num", i + 1)
                prompts.append(prompt)
    return prompts


def tokenize_prompts_context_safe(prompts: list[dict], model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
    """Tokenize prompts using apply_chat_template (as vLLM will see them).

    This computes request_prompt_tokens = len(apply_chat_template(..., add_generation_prompt=True))
    """
    tokenizer = None

    if HAS_TRANSFORMERS:
        try:
            print(f"Loading tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load tokenizer ({e}), using fallback")

    if tokenizer is None:
        print("FATAL: Cannot audit context-safety without Mistral tokenizer!")
        print("Install: pip install transformers")
        sys.exit(1)

    print(f"Computing request_prompt_tokens via apply_chat_template...")
    for i, p in enumerate(prompts):
        text = p.get("prompt", "")
        # This is exactly how vLLM tokenizes the prompt
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        p["request_prompt_tokens"] = len(token_ids)

        if (i + 1) % 200 == 0:
            print(f"  Tokenized {i + 1}/{len(prompts)}")

    return prompts


def audit_bucket_constraints(prompts: list[dict]) -> dict[str, Any]:
    """Audit prompts against bucket constraints AND context-safety."""
    findings = {
        "total_prompts": len(prompts),
        "by_task_bucket": defaultdict(lambda: {"count": 0, "token_counts": [], "totals": []}),
        "bucket_violations": [],
        "overflow_violations": [],
        "stats_by_task_bucket": {},
    }

    for p in prompts:
        task = p.get("task", "unknown")
        bucket = p.get("bucket", "unknown")
        request_tokens = p.get("request_prompt_tokens", 0)
        max_new_tokens = p.get("max_new_tokens", 256)
        total = request_tokens + max_new_tokens
        key = f"{task}/{bucket}"

        findings["by_task_bucket"][key]["count"] += 1
        findings["by_task_bucket"][key]["token_counts"].append(request_tokens)
        findings["by_task_bucket"][key]["totals"].append(total)

        # Check bucket constraint
        if bucket in BUCKET_CONSTRAINTS:
            min_tokens, max_tokens = BUCKET_CONSTRAINTS[bucket]
            if request_tokens < min_tokens or request_tokens > max_tokens:
                findings["bucket_violations"].append({
                    "line_num": p.get("line_num"),
                    "task": task,
                    "bucket": bucket,
                    "request_prompt_tokens": request_tokens,
                    "expected_range": f"{min_tokens}-{max_tokens}",
                })

        # Check context-safety
        if total > MAX_MODEL_LEN:
            findings["overflow_violations"].append({
                "line_num": p.get("line_num"),
                "task": task,
                "bucket": bucket,
                "request_prompt_tokens": request_tokens,
                "max_new_tokens": max_new_tokens,
                "total": total,
            })

    # Compute stats per task/bucket
    for key, data in findings["by_task_bucket"].items():
        tokens = sorted(data["token_counts"])
        totals = sorted(data["totals"])
        if tokens:
            findings["stats_by_task_bucket"][key] = {
                "count": len(tokens),
                "min": tokens[0],
                "median": tokens[len(tokens) // 2],
                "max": tokens[-1],
                "max_total": totals[-1],
            }

    return findings


def print_audit_report(findings: dict) -> str:
    """Generate human-readable audit report with context-safety."""
    lines = []
    lines.append("=" * 80)
    lines.append("PROMPTS AUDIT REPORT (Context-Safe Accounting)")
    lines.append("=" * 80)
    lines.append(f"\nTotal prompts: {findings['total_prompts']}")

    lines.append("\n" + "-" * 80)
    lines.append("COUNTS BY TASK/BUCKET (request_prompt_tokens via apply_chat_template)")
    lines.append("-" * 80)
    lines.append(f"{'Task/Bucket':<20} {'Count':>6} {'Min':>7} {'Median':>8} {'Max':>7} {'MaxTotal':>10}")

    for key in sorted(findings["stats_by_task_bucket"].keys()):
        s = findings["stats_by_task_bucket"][key]
        lines.append(f"{key:<20} {s['count']:>6} {s['min']:>7} {s['median']:>8} {s['max']:>7} {s['max_total']:>10}")

    lines.append("\n" + "-" * 80)
    lines.append("CONTEXT-SAFETY CHECK: max(request_prompt_tokens + max_new_tokens)")
    lines.append(f"Must be <= {MAX_MODEL_LEN}")
    lines.append("-" * 80)

    for key in sorted(findings["stats_by_task_bucket"].keys()):
        s = findings["stats_by_task_bucket"][key]
        status = "✓ SAFE" if s['max_total'] <= MAX_MODEL_LEN else "✗ OVERFLOW"
        lines.append(f"{key:<20} max_total={s['max_total']:>5} {status}")

    lines.append("\n" + "-" * 80)
    lines.append("BUCKET CONSTRAINT VIOLATIONS")
    lines.append("-" * 80)

    if findings["bucket_violations"]:
        lines.append(f"⚠️  FOUND {len(findings['bucket_violations'])} BUCKET VIOLATIONS:")
        for v in findings["bucket_violations"][:10]:
            lines.append(f"  Line {v['line_num']}: {v['task']}/{v['bucket']} has {v['request_prompt_tokens']} tokens (expected {v['expected_range']})")
        if len(findings["bucket_violations"]) > 10:
            lines.append(f"  ... and {len(findings['bucket_violations']) - 10} more")
    else:
        lines.append("✓ No bucket violations. All prompts satisfy bucket constraints.")

    lines.append("\n" + "-" * 80)
    lines.append("CONTEXT OVERFLOW VIOLATIONS")
    lines.append("-" * 80)

    if findings["overflow_violations"]:
        lines.append(f"⚠️  FOUND {len(findings['overflow_violations'])} OVERFLOW VIOLATIONS:")
        for v in findings["overflow_violations"][:10]:
            lines.append(f"  Line {v['line_num']}: {v['task']}/{v['bucket']} total={v['total']} ({v['request_prompt_tokens']}+{v['max_new_tokens']}) > {MAX_MODEL_LEN}")
        if len(findings["overflow_violations"]) > 10:
            lines.append(f"  ... and {len(findings['overflow_violations']) - 10} more")
    else:
        lines.append(f"✓ No overflow violations. All prompts context-safe (<= {MAX_MODEL_LEN}).")

    return "\n".join(lines)


if __name__ == "__main__":
    prompts_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/datasets/processed/prompts.jsonl")

    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} not found")
        sys.exit(1)

    prompts = load_prompts(prompts_path)
    prompts = tokenize_prompts_context_safe(prompts)
    findings = audit_bucket_constraints(prompts)
    report = print_audit_report(findings)
    print(report)

    # Save output
    output_path = Path("results/audit_prompts.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nSaved: {output_path}")

    # Exit with error if any violations found
    has_violations = findings["bucket_violations"] or findings["overflow_violations"]
    if has_violations:
        print("\n⚠️  AUDIT FAILED: Violations found!")
        sys.exit(1)
    else:
        print("\n✓ AUDIT PASSED: All prompts context-safe and within bucket constraints.")

