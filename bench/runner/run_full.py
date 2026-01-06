#!/usr/bin/env python3
"""Full benchmark runner for a single vLLM configuration.

Runs 1800 prompts twice:
1. Non-streaming mode
2. Streaming mode

Concurrency: 4 (fixed)

Saves:
- traces_nonstreaming.jsonl
- traces_streaming.jsonl
- summary.json
- gpu.csv (sampled every 250ms)
- run_manifest.json
"""
import asyncio
import json
import time
import argparse
import subprocess
import hashlib
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import aiohttp
import yaml

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


def file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def start_gpu_sampler(output_path: Path, interval_ms: int = 250) -> subprocess.Popen:
    """Start nvidia-smi sampling in background."""
    # Write header first
    with open(output_path, "w") as f:
        f.write("timestamp,gpu_util,mem_used_mib,mem_total_mib,power_w\n")
    
    cmd = f"""
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> {output_path}
        sleep {interval_ms / 1000}
    done
    """
    return subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def run_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    streaming: bool,
) -> dict[str, Any]:
    """Run a single request with full evidence capture."""
    start = time.time()
    ttft = None
    tokens = 0
    prompt_tokens = 0

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": streaming,
    }

    try:
        async with session.post(endpoint, json=payload) as resp:
            if resp.status != 200:
                return {
                    "success": False,
                    "http_status": resp.status,
                    "latency": time.time() - start,
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "ttft": None,
                    "error_reason": f"http_{resp.status}",
                    "streaming": streaming,
                }

            if streaming:
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
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                tokens += 1
                        except json.JSONDecodeError:
                            pass
            else:
                raw_body = await resp.text()
                result = json.loads(raw_body)
                usage = result.get("usage", {})
                tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)

            end = time.time()
            latency = end - start
            success = tokens > 0 and latency > 0

            return {
                "success": success,
                "http_status": resp.status,
                "latency": latency,
                "tokens": tokens,
                "prompt_tokens": prompt_tokens,
                "ttft": ttft,
                "error_reason": None if success else "zero_tokens",
                "streaming": streaming,
            }
    except Exception as e:
        return {
            "success": False,
            "http_status": None,
            "latency": time.time() - start,
            "tokens": 0,
            "prompt_tokens": 0,
            "ttft": None,
            "error_reason": f"exception: {e}",
            "streaming": streaming,
        }


async def run_batch(
    prompts: list[dict],
    endpoint: str,
    model: str,
    streaming: bool,
    concurrency: int = 4,
    timeout: int = 600,
) -> tuple[list[dict], float]:
    """Run a batch of requests."""
    sem = asyncio.Semaphore(concurrency)
    start_time = time.time()

    async def bounded_request(session: aiohttp.ClientSession, p: dict, idx: int):
        async with sem:
            max_tokens = p.get("max_new_tokens", 64)
            result = await run_request(
                session, endpoint, model, p["prompt"], max_tokens, streaming
            )
            result["idx"] = idx
            result["task"] = p.get("task", "")
            result["bucket"] = p.get("bucket", "")
            result["request_prompt_tokens"] = p.get("request_prompt_tokens", 0)
            result["max_new_tokens"] = max_tokens
            if (idx + 1) % 100 == 0:
                print(f"    Progress: {idx + 1}/{len(prompts)}")
            return result

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
        tasks = [bounded_request(session, p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

    duration = time.time() - start_time
    return list(results), duration


def compute_stats(results: list[dict], duration: float) -> dict[str, Any]:
    """Compute statistics from results."""
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    stats = {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0,
        "duration_sec": duration,
        "throughput_req_per_sec": len(successful) / duration if duration > 0 else 0,
    }

    if successful:
        latencies = sorted([r["latency"] for r in successful])
        stats["latency_min"] = latencies[0]
        stats["latency_p50"] = latencies[len(latencies) // 2]
        stats["latency_p95"] = latencies[int(len(latencies) * 0.95)]
        stats["latency_p99"] = latencies[int(len(latencies) * 0.99)]
        stats["latency_max"] = latencies[-1]

        total_tokens = sum(r.get("tokens", 0) for r in successful)
        stats["total_tokens"] = total_tokens
        stats["tokens_per_sec"] = total_tokens / duration if duration > 0 else 0

        # TTFT stats for streaming
        ttfts = [r["ttft"] for r in successful if r.get("ttft") is not None]
        if ttfts:
            ttfts.sort()
            stats["ttft_p50"] = ttfts[len(ttfts) // 2]
            stats["ttft_p95"] = ttfts[int(len(ttfts) * 0.95)]

    # Error breakdown
    error_counts: dict[str, int] = {}
    for r in failed:
        err = r.get("error_reason", "unknown")
        error_counts[err] = error_counts.get(err, 0) + 1
    stats["error_breakdown"] = error_counts

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Full benchmark runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("bench/datasets/processed/prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-gpu-sample", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model = config.get("served_model_name", "mistral7b")
    prompts = load_prompts(args.prompts)

    print(f"Config: {config['name']}")
    print(f"Prompts: {len(prompts)}")
    print(f"Concurrency: {args.concurrency}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Start GPU sampler
    gpu_sampler = None
    if not args.no_gpu_sample:
        gpu_path = args.output_dir / "gpu.csv"
        print(f"Starting GPU sampler -> {gpu_path}")
        gpu_sampler = start_gpu_sampler(gpu_path)

    try:
        all_stats = {}

        # Non-streaming run
        print("\n=== Non-streaming run ===")
        ns_results, ns_duration = await run_batch(
            prompts, args.endpoint, model, streaming=False,
            concurrency=args.concurrency, timeout=args.timeout
        )
        ns_stats = compute_stats(ns_results, ns_duration)
        ns_stats["streaming"] = False
        all_stats["non_streaming"] = ns_stats

        with open(args.output_dir / "traces_nonstreaming.jsonl", "w") as f:
            for r in ns_results:
                f.write(json.dumps(r) + "\n")

        print(f"  Success: {ns_stats['successful']}/{ns_stats['total']}")
        print(f"  Duration: {ns_stats['duration_sec']:.1f}s")
        print(f"  Throughput: {ns_stats['throughput_req_per_sec']:.2f} req/s")

        # Streaming run
        print("\n=== Streaming run ===")
        s_results, s_duration = await run_batch(
            prompts, args.endpoint, model, streaming=True,
            concurrency=args.concurrency, timeout=args.timeout
        )
        s_stats = compute_stats(s_results, s_duration)
        s_stats["streaming"] = True
        all_stats["streaming"] = s_stats

        with open(args.output_dir / "traces_streaming.jsonl", "w") as f:
            for r in s_results:
                f.write(json.dumps(r) + "\n")

        print(f"  Success: {s_stats['successful']}/{s_stats['total']}")
        print(f"  Duration: {s_stats['duration_sec']:.1f}s")
        if "ttft_p50" in s_stats:
            print(f"  TTFT p50: {s_stats['ttft_p50']:.3f}s")

    finally:
        if gpu_sampler:
            gpu_sampler.terminate()
            gpu_sampler.wait()

    # Compute overall stats
    total_success = ns_stats["successful"] + s_stats["successful"]
    total_requests = ns_stats["total"] + s_stats["total"]

    all_stats["overall"] = {
        "config_name": config["name"],
        "total_requests": total_requests,
        "total_success": total_success,
        "failure_rate": 1 - (total_success / total_requests) if total_requests > 0 else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save summary
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # Generate manifest
    manifest = generate_manifest(model_id=config["model_id"])
    manifest["config_name"] = config["name"]
    manifest["config_description"] = config.get("description", "")
    manifest["prompts_file"] = str(args.prompts)
    manifest["prompts_sha256"] = file_sha256(args.prompts)
    manifest["prompts_count"] = len(prompts)
    manifest["concurrency"] = args.concurrency
    manifest["max_request_plus_new_tokens"] = max(
        p.get("request_prompt_tokens", 0) + p.get("max_new_tokens", 0)
        for p in prompts
    )

    with open(args.output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FULL BENCHMARK COMPLETE: {config['name']}")
    print(f"{'='*60}")
    print(f"Total: {total_success}/{total_requests} success")
    print(f"Failure rate: {all_stats['overall']['failure_rate']:.2%}")
    print(f"Output: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

