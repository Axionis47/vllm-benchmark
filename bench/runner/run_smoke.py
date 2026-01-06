#!/usr/bin/env python3
"""Smoke test runner for a single vLLM configuration.

Runs smoke prompts twice:
1. Non-streaming mode
2. Streaming mode

Asserts 100% success and 0 HTTP 400 errors.
Concurrency: 2 (conservative for smoke test)

Saves:
- smoke_traces.jsonl
- smoke_summary.json
- run_manifest.json
"""
import asyncio
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import Any
from datetime import datetime

import aiohttp
import yaml

# Import manifest generator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bench.runner.run_manifest import generate_manifest


def load_prompts(prompts_path: Path) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def load_config(config_path: Path) -> dict[str, Any]:
    """Load config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


async def run_request_nonstreaming(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Run a single non-streaming request."""
    start = time.time()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        async with session.post(endpoint, json=payload) as resp:
            end = time.time()
            latency = end - start
            raw_body = await resp.text()

            if resp.status != 200:
                return {
                    "success": False,
                    "http_status": resp.status,
                    "latency": latency,
                    "tokens": 0,
                    "error_reason": f"http_{resp.status}",
                    "streaming": False,
                }

            result = json.loads(raw_body)
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            success = completion_tokens > 0 and latency > 0
            return {
                "success": success,
                "http_status": resp.status,
                "latency": latency,
                "tokens": completion_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "error_reason": None if success else "zero_tokens",
                "streaming": False,
            }
    except Exception as e:
        return {
            "success": False,
            "http_status": None,
            "latency": time.time() - start,
            "tokens": 0,
            "error_reason": f"exception: {e}",
            "streaming": False,
        }


async def run_request_streaming(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Run a single streaming request, measure TTFT."""
    start = time.time()
    ttft = None
    tokens = 0
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        async with session.post(endpoint, json=payload) as resp:
            if resp.status != 200:
                return {
                    "success": False,
                    "http_status": resp.status,
                    "latency": time.time() - start,
                    "tokens": 0,
                    "ttft": None,
                    "error_reason": f"http_{resp.status}",
                    "streaming": True,
                }

            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if ttft is None:
                            ttft = time.time() - start
                        # Count tokens from delta
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            tokens += 1
                    except json.JSONDecodeError:
                        pass

            end = time.time()
            latency = end - start
            success = tokens > 0 and latency > 0

            return {
                "success": success,
                "http_status": resp.status,
                "latency": latency,
                "tokens": tokens,
                "ttft": ttft,
                "error_reason": None if success else "zero_tokens",
                "streaming": True,
            }
    except Exception as e:
        return {
            "success": False,
            "http_status": None,
            "latency": time.time() - start,
            "tokens": 0,
            "ttft": None,
            "error_reason": f"exception: {e}",
            "streaming": True,
        }


async def run_smoke_batch(
    prompts: list[dict],
    endpoint: str,
    model: str,
    streaming: bool,
    concurrency: int = 2,
    timeout: int = 300,
) -> list[dict]:
    """Run a batch of smoke requests."""
    sem = asyncio.Semaphore(concurrency)
    run_fn = run_request_streaming if streaming else run_request_nonstreaming

    async def bounded_request(session: aiohttp.ClientSession, p: dict, idx: int):
        async with sem:
            max_tokens = p.get("max_new_tokens", 64)
            result = await run_fn(session, endpoint, model, p["prompt"], max_tokens)
            result["idx"] = idx
            result["task"] = p.get("task", "")
            result["bucket"] = p.get("bucket", "")
            return result

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
        tasks = [bounded_request(session, p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

    return list(results)


def compute_summary(results: list[dict], duration: float, streaming: bool) -> dict:
    """Compute summary statistics."""
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    # Check for HTTP 400 specifically
    http_400_count = sum(1 for r in results if r.get("http_status") == 400)

    summary = {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "http_400_count": http_400_count,
        "success_rate": len(successful) / len(results) if results else 0,
        "duration_sec": duration,
        "streaming": streaming,
    }

    if successful:
        latencies = sorted([r["latency"] for r in successful])
        summary["latency_min"] = latencies[0]
        summary["latency_median"] = latencies[len(latencies) // 2]
        summary["latency_max"] = latencies[-1]

        if streaming:
            ttfts = [r["ttft"] for r in successful if r.get("ttft")]
            if ttfts:
                ttfts.sort()
                summary["ttft_min"] = ttfts[0]
                summary["ttft_median"] = ttfts[len(ttfts) // 2]
                summary["ttft_max"] = ttfts[-1]

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Smoke test runner")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    parser.add_argument("--prompts", type=Path, default=Path("results/smoke_prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    # Load config and prompts
    config = load_config(args.config)
    model = config.get("served_model_name", "mistral7b")
    prompts = load_prompts(args.prompts)

    print(f"Config: {config['name']}")
    print(f"Prompts: {len(prompts)}")
    print(f"Endpoint: {args.endpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_traces = []
    summaries = {}

    # Run non-streaming
    print("\n=== Non-streaming smoke test ===")
    start = time.time()
    ns_results = await run_smoke_batch(
        prompts, args.endpoint, model, streaming=False,
        concurrency=args.concurrency, timeout=args.timeout
    )
    ns_duration = time.time() - start
    ns_summary = compute_summary(ns_results, ns_duration, streaming=False)
    summaries["non_streaming"] = ns_summary
    all_traces.extend(ns_results)

    print(f"  Success: {ns_summary['successful']}/{ns_summary['total']}")
    print(f"  HTTP 400: {ns_summary['http_400_count']}")

    # Run streaming
    print("\n=== Streaming smoke test ===")
    start = time.time()
    s_results = await run_smoke_batch(
        prompts, args.endpoint, model, streaming=True,
        concurrency=args.concurrency, timeout=args.timeout
    )
    s_duration = time.time() - start
    s_summary = compute_summary(s_results, s_duration, streaming=True)
    summaries["streaming"] = s_summary
    all_traces.extend(s_results)

    print(f"  Success: {s_summary['successful']}/{s_summary['total']}")
    print(f"  HTTP 400: {s_summary['http_400_count']}")

    # Check assertions
    total_success = ns_summary["successful"] + s_summary["successful"]
    total_requests = ns_summary["total"] + s_summary["total"]
    total_http_400 = ns_summary["http_400_count"] + s_summary["http_400_count"]

    passed = (total_success == total_requests) and (total_http_400 == 0)

    summaries["overall"] = {
        "config_name": config["name"],
        "total_requests": total_requests,
        "total_success": total_success,
        "total_http_400": total_http_400,
        "passed": passed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Save outputs
    with open(args.output_dir / "smoke_traces.jsonl", "w") as f:
        for t in all_traces:
            f.write(json.dumps(t) + "\n")

    with open(args.output_dir / "smoke_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    # Generate manifest
    manifest = generate_manifest(model_id=config["model_id"])
    manifest["config_name"] = config["name"]
    manifest["prompts_file"] = str(args.prompts)
    manifest["concurrency"] = args.concurrency

    with open(args.output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SMOKE TEST {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*60}")
    print(f"Config: {config['name']}")
    print(f"Total: {total_success}/{total_requests} success")
    print(f"HTTP 400: {total_http_400}")

    if not passed:
        # Show failed requests
        failed = [t for t in all_traces if not t.get("success")]
        print(f"\nFailed requests ({len(failed)}):")
        for f_req in failed[:5]:
            print(f"  idx={f_req['idx']}, error={f_req.get('error_reason')}, "
                  f"http={f_req.get('http_status')}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

