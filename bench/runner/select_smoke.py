#!/usr/bin/env python3
"""Select smoke test prompts for quick validation.

Deterministically selects 40 prompts that stress-test context limits:
- 15 summarization/L (highest request_prompt_tokens)
- 10 qa/L
- 10 dialogue/L
- 5 summarization/S (lowest complexity baseline)

Seed=42 for reproducibility.
"""
import argparse
import json
import random
from pathlib import Path


def load_prompts(prompts_path: Path) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def select_smoke_prompts(prompts: list[dict], seed: int = 42) -> list[dict]:
    """Select 40 smoke test prompts deterministically.

    Selection strategy:
    - Focus on L bucket (highest context length stress)
    - Include some S bucket for baseline comparison
    - Sort by request_prompt_tokens descending within each group
    """
    random.seed(seed)

    # Group prompts by (task, bucket)
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in prompts:
        key = (p.get("task", ""), p.get("bucket", ""))
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    # Sort each group by request_prompt_tokens descending
    for key in groups:
        groups[key].sort(key=lambda x: x.get("request_prompt_tokens", 0), reverse=True)

    selected = []

    # Selection quotas
    quotas = [
        (("summarization", "L"), 15),
        (("qa", "L"), 10),
        (("dialogue", "L"), 10),
        (("summarization", "S"), 5),
    ]

    for (task, bucket), count in quotas:
        key = (task, bucket)
        if key in groups:
            # Take top N by request_prompt_tokens
            available = groups[key][:count]
            selected.extend(available)
            print(f"  Selected {len(available)}/{count} from {task}/{bucket}")
        else:
            print(f"  WARNING: No prompts found for {task}/{bucket}")

    # Shuffle to avoid ordering bias during benchmark
    random.shuffle(selected)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Select smoke test prompts")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("bench/datasets/processed/prompts.jsonl"),
        help="Path to full prompts file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/smoke_prompts.jsonl"),
        help="Output path for smoke prompts",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Loading prompts from {args.prompts}...")
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    print(f"\nSelecting smoke prompts (seed={args.seed})...")
    smoke_prompts = select_smoke_prompts(prompts, seed=args.seed)
    print(f"\nSelected {len(smoke_prompts)} smoke prompts")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save smoke prompts
    with open(args.output, "w") as f:
        for p in smoke_prompts:
            f.write(json.dumps(p) + "\n")

    print(f"Saved to {args.output}")

    # Print stats
    print("\nSmoke prompt statistics:")
    by_task_bucket: dict[tuple[str, str], list[int]] = {}
    for p in smoke_prompts:
        key = (p.get("task", ""), p.get("bucket", ""))
        if key not in by_task_bucket:
            by_task_bucket[key] = []
        by_task_bucket[key].append(p.get("request_prompt_tokens", 0))

    for (task, bucket), tokens in sorted(by_task_bucket.items()):
        print(f"  {task}/{bucket}: n={len(tokens)}, "
              f"min={min(tokens)}, max={max(tokens)}")


if __name__ == "__main__":
    main()

