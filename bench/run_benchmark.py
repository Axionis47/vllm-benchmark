#!/usr/bin/env python3
"""Run benchmark against vLLM endpoint using prompts.jsonl."""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.runner.client import BenchmarkClient, CompletionRequest, CompletionResponse
from bench.runner.loadgen import LoadGenConfig, LoadGenerator, LoadGenResult


@dataclass
class PromptRecord:
    """A prompt record from prompts.jsonl."""
    task: str
    id: str
    bucket: str
    prompt: str
    reference: str
    max_new_tokens: int


def load_prompts(path: Path) -> list[PromptRecord]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(PromptRecord(
                task=data["task"],
                id=data["id"],
                bucket=data["bucket"],
                prompt=data["prompt"],
                reference=data["reference"],
                max_new_tokens=data["max_new_tokens"],
            ))
    return prompts


async def run_benchmark(
    prompts: list[PromptRecord],
    client: BenchmarkClient,
    concurrency: int,
    output_dir: Path,
    bucket_filter: str | None = None,
    task_filter: str | None = None,
    limit: int | None = None,
) -> LoadGenResult:
    """Run benchmark on filtered prompts."""
    # Filter prompts
    filtered = prompts
    if bucket_filter:
        filtered = [p for p in filtered if p.bucket == bucket_filter]
    if task_filter:
        filtered = [p for p in filtered if p.task == task_filter]
    if limit:
        filtered = filtered[:limit]
    
    print(f"Running benchmark on {len(filtered)} prompts")
    print(f"  Concurrency: {concurrency}")
    print(f"  Output: {output_dir}")
    
    config = LoadGenConfig(
        concurrency=concurrency,
        num_requests=len(filtered),
        max_new_tokens=filtered[0].max_new_tokens if filtered else 128,
        stream=True,
        warmup_requests=min(5, len(filtered)),
        warmup_delay_sec=2.0,
        output_dir=output_dir,
    )
    
    generator = LoadGenerator(client, config)
    prompt_texts = [p.prompt for p in filtered]
    
    return await generator.run(prompt_texts)


def start_gpu_sampler(output_path: Path, interval_ms: int = 200) -> subprocess.Popen:
    """Start nvidia-smi sampling in background."""
    cmd = f"""
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader >> {output_path}
        sleep {interval_ms / 1000}
    done
    """
    return subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmark")
    parser.add_argument("--prompts", type=Path, default=Path("bench/datasets/processed/prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/run"))
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="mistral7b")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--bucket", choices=["S", "M", "L"], help="Filter by bucket")
    parser.add_argument("--task", choices=["summarization", "qa", "dialogue"], help="Filter by task")
    parser.add_argument("--limit", type=int, help="Limit number of prompts")
    parser.add_argument("--no-gpu-sample", action="store_true", help="Disable GPU sampling")
    args = parser.parse_args()
    
    # Load prompts
    print(f"Loading prompts from {args.prompts}")
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Start GPU sampler
    gpu_sampler = None
    if not args.no_gpu_sample:
        gpu_log = args.output / "gpu_samples.csv"
        print(f"Starting GPU sampler -> {gpu_log}")
        gpu_sampler = start_gpu_sampler(gpu_log)
    
    # Create client
    client = BenchmarkClient(base_url=args.endpoint, model=args.model, timeout=300.0)
    
    # Check health
    if not await client.health_check():
        print("ERROR: Endpoint not healthy")
        sys.exit(1)
    print("Endpoint healthy")
    
    try:
        result = await run_benchmark(
            prompts=prompts,
            client=client,
            concurrency=args.concurrency,
            output_dir=args.output,
            bucket_filter=args.bucket,
            task_filter=args.task,
            limit=args.limit,
        )
        
        print(f"\n=== Results ===")
        print(f"Total requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Duration: {result.duration_sec:.2f}s")
        print(f"Throughput: {result.successful_requests / result.duration_sec:.2f} req/s")
        
    finally:
        await client.close()
        if gpu_sampler:
            gpu_sampler.terminate()
            gpu_sampler.wait()


if __name__ == "__main__":
    asyncio.run(main())

