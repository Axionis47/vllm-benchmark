#!/usr/bin/env python3
"""Benchmark runner with proper success classification and invariants.

A request is SUCCESSFUL only if:
  a) HTTP status == 200
  b) parsed response contains at least 1 generated token (completion_tokens > 0)
  c) latency > 0 (end_ts >= start_ts)

Failed requests are EXCLUDED from latency distributions.
"""
import asyncio
import json
import time
import argparse
from pathlib import Path
from typing import Any
import aiohttp


def redact_prompt(prompt: str, max_chars: int = 500) -> str:
    """Redact prompt text after max_chars."""
    if len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars] + f"...[REDACTED {len(prompt) - max_chars} chars]"


async def run_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    prompt_tokens_pre_send: int = 0,
) -> dict[str, Any]:
    """Run a single request with full evidence capture."""
    start = time.time()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Build redacted request for evidence
    request_json = {
        "model": model,
        "messages": [{"role": "user", "content": redact_prompt(prompt)}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Base evidence fields
    evidence = {
        "request_json": json.dumps(request_json),
        "prompt_chars": len(prompt),
        "prompt_tokens_pre_send": prompt_tokens_pre_send,
        "streaming": False,
    }

    try:
        async with session.post(endpoint, json=payload) as resp:
            end = time.time()
            latency = end - start

            # Capture raw response
            raw_body = await resp.text()
            response_snippet = raw_body[:500] if len(raw_body) > 500 else raw_body

            evidence.update({
                "http_status": resp.status,
                "response_body_snippet": response_snippet,
            })

            # Invariant: HTTP status must be 200
            if resp.status != 200:
                return {
                    "success": False,
                    "latency": latency,
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens_parsed": 0,
                    "error_reason": f"http_{resp.status}",
                    "parse_ok": False,
                    "finish_reason": None,
                    **evidence,
                }

            # Parse JSON
            try:
                result = json.loads(raw_body)
                parse_ok = True
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "latency": latency,
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens_parsed": 0,
                    "error_reason": f"json_parse_error: {e}",
                    "parse_ok": False,
                    "finish_reason": None,
                    **evidence,
                }

            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)

            # Extract finish_reason
            choices = result.get("choices", [])
            finish_reason = choices[0].get("finish_reason") if choices else None

            evidence.update({
                "parse_ok": parse_ok,
                "finish_reason": finish_reason,
                "completion_tokens_parsed": completion_tokens,
            })

            # Invariant: Must have at least 1 generated token
            if completion_tokens == 0:
                return {
                    "success": False,
                    "latency": latency,
                    "tokens": 0,
                    "prompt_tokens": prompt_tokens,
                    "error_reason": "zero_completion_tokens",
                    **evidence,
                }

            # Invariant: prompt_tokens_pre_send must match (sanity check)
            # Invariant: latency must be positive
            if latency <= 0:
                return {
                    "success": False,
                    "latency": latency,
                    "tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "error_reason": "non_positive_latency",
                    **evidence,
                }

            # SUCCESS: all invariants passed
            return {
                "success": True,
                "latency": latency,
                "tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "error_reason": None,
                **evidence,
            }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "latency": time.time() - start,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens_parsed": 0,
            "error_reason": "timeout",
            "parse_ok": False,
            "finish_reason": None,
            "http_status": None,
            "response_body_snippet": None,
            **evidence,
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens_parsed": 0,
            "error_reason": f"exception: {str(e)}",
            "parse_ok": False,
            "finish_reason": None,
            "http_status": None,
            "response_body_snippet": None,
            **evidence,
        }


def compute_stats(results: list[dict], duration: float) -> dict[str, Any]:
    """Compute stats from results, EXCLUDING failed requests from latency."""
    # Filter to only truly successful requests
    successful = [r for r in results if r.get("success") and r.get("tokens", 0) > 0]
    failed = [r for r in results if not r.get("success") or r.get("tokens", 0) == 0]

    stats = {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "duration_sec": duration,
        "throughput_req_per_sec": len(successful) / duration if duration > 0 else 0,
    }

    if successful:
        latencies = sorted([r["latency"] for r in successful])
        stats["latency_min"] = latencies[0]
        stats["latency_median"] = latencies[len(latencies) // 2]
        stats["latency_p95"] = latencies[int(len(latencies) * 0.95)]
        stats["latency_max"] = latencies[-1]
        stats["total_prompt_tokens"] = sum(r.get("prompt_tokens", 0) for r in successful)
        stats["total_completion_tokens"] = sum(r.get("tokens", 0) for r in successful)
        stats["tokens_per_sec"] = stats["total_completion_tokens"] / duration
    else:
        stats["latency_min"] = 0
        stats["latency_median"] = 0
        stats["latency_p95"] = 0
        stats["latency_max"] = 0

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Benchmark runner with proper invariants")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL file")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="mistral7b")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max_tokens (default: use per-prompt max_new_tokens)")
    parser.add_argument("--output-dir", default="results/run")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    args = parser.parse_args()

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    prompts = prompts[: args.max_requests]
    print(f"Loaded {len(prompts)} prompts")

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    sem = asyncio.Semaphore(args.concurrency)
    start_time = time.time()

    async def bounded_request(session, prompt_data, idx):
        async with sem:
            prompt = prompt_data.get("prompt", "")
            # Use per-prompt max_new_tokens if available, else fall back to --max-tokens or 256
            max_tokens = args.max_tokens if args.max_tokens is not None else prompt_data.get("max_new_tokens", 256)
            result = await run_request(
                session, args.endpoint, args.model, prompt, max_tokens
            )
            result["idx"] = idx
            result["task"] = prompt_data.get("task", "")
            result["bucket"] = prompt_data.get("bucket", "")
            result["max_new_tokens_used"] = max_tokens
            if (idx + 1) % 50 == 0:
                print(f"  Completed {idx + 1}/{len(prompts)}")
            return result

    print(f"Running {len(prompts)} requests with concurrency={args.concurrency}")
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [bounded_request(session, p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time

    # Compute stats (excludes failed from latency)
    stats = compute_stats(results, duration)

    # Print results
    sep = "=" * 60
    print(f"\n{sep}")
    print("BENCHMARK RESULTS")
    print(sep)
    print(f"Total requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Duration: {stats['duration_sec']:.2f}s")
    print(f"Throughput: {stats['throughput_req_per_sec']:.2f} req/s")

    if stats["successful"] > 0:
        print(f"\nLatency (seconds) - successful only:")
        print(f"  Min: {stats['latency_min']:.3f}")
        print(f"  Median: {stats['latency_median']:.3f}")
        print(f"  P95: {stats['latency_p95']:.3f}")
        print(f"  Max: {stats['latency_max']:.3f}")
        print(f"\nTokens:")
        print(f"  Total prompt tokens: {stats.get('total_prompt_tokens', 0)}")
        print(f"  Total completion tokens: {stats.get('total_completion_tokens', 0)}")
        print(f"  Tokens/sec: {stats.get('tokens_per_sec', 0):.2f}")

    # Show failed request breakdown if any
    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\nFailed request breakdown:")
        error_counts = {}
        for r in failed:
            err = r.get("error", "unknown")
            error_counts[err] = error_counts.get(err, 0) + 1
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {err}: {count}")

    # Save traces
    with open(output_dir / "traces.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())

