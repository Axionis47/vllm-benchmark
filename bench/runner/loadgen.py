"""Async load generator for benchmark runs."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bench.runner.client import (
    BaseClient,
    CompletionRequest,
    CompletionResponse,
    MockClient,
)
from bench.prompts.templates import get_synthetic_prompts


@dataclass
class LoadGenConfig:
    """Configuration for load generation."""

    concurrency: int = 1
    num_requests: int = 100
    max_new_tokens: int = 128
    stream: bool = True
    warmup_requests: int = 5
    warmup_delay_sec: float = 2.0
    output_dir: Optional[Path] = None


@dataclass
class LoadGenResult:
    """Result of a load generation run."""

    config: LoadGenConfig
    traces: list[CompletionResponse]
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int

    @property
    def duration_sec(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "config": {
                "concurrency": self.config.concurrency,
                "num_requests": self.config.num_requests,
                "max_new_tokens": self.config.max_new_tokens,
                "stream": self.config.stream,
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
        }


class LoadGenerator:
    """Async load generator for benchmark runs."""

    def __init__(self, client: BaseClient, config: LoadGenConfig):
        self.client = client
        self.config = config
        self._traces: list[CompletionResponse] = []
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def _send_request(
        self, prompt: str, request_id: str
    ) -> CompletionResponse:
        """Send a single request with concurrency control."""
        assert self._semaphore is not None
        async with self._semaphore:
            request = CompletionRequest(
                prompt=prompt,
                max_tokens=self.config.max_new_tokens,
                stream=self.config.stream,
                request_id=request_id,
            )
            return await self.client.complete(request)

    async def _warmup(self, prompts: list[str]) -> None:
        """Run warmup requests (results discarded)."""
        if self.config.warmup_requests <= 0:
            return

        print(f"Running {self.config.warmup_requests} warmup requests...")
        warmup_tasks = [
            self._send_request(prompts[i % len(prompts)], f"warmup_{i}")
            for i in range(self.config.warmup_requests)
        ]
        await asyncio.gather(*warmup_tasks)
        print(f"Warmup complete, waiting {self.config.warmup_delay_sec}s...")
        await asyncio.sleep(self.config.warmup_delay_sec)

    async def run(self, prompts: Optional[list[str]] = None) -> LoadGenResult:
        """Run the load generation benchmark."""
        if prompts is None:
            prompts = get_synthetic_prompts()

        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        self._traces = []

        # Warmup phase
        await self._warmup(prompts)

        # Main benchmark phase
        print(f"Starting benchmark: {self.config.num_requests} requests, "
              f"concurrency={self.config.concurrency}")

        start_time = time.time()

        tasks = [
            self._send_request(
                prompts[i % len(prompts)],
                f"req_{i:06d}"
            )
            for i in range(self.config.num_requests)
        ]

        self._traces = await asyncio.gather(*tasks)
        end_time = time.time()

        successful = sum(1 for t in self._traces if t.status == "success")
        failed = sum(1 for t in self._traces if t.status != "success")

        result = LoadGenResult(
            config=self.config,
            traces=self._traces,
            start_time=start_time,
            end_time=end_time,
            total_requests=self.config.num_requests,
            successful_requests=successful,
            failed_requests=failed,
        )

        # Save traces if output directory specified
        if self.config.output_dir:
            self._save_traces(result)

        return result

    def _save_traces(self, result: LoadGenResult) -> None:
        """Save trace data to JSONL file."""
        output_dir = self.config.output_dir
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        traces_file = output_dir / "traces.jsonl"
        with open(traces_file, "w") as f:
            for trace in result.traces:
                record = {
                    "request_id": trace.request_id,
                    "prompt_tokens": trace.prompt_tokens,
                    "completion_tokens": trace.completion_tokens,
                    "ttft_ms": trace.ttft_ms,
                    "total_time_ms": trace.total_time_ms,
                    "status": trace.status,
                    "error": trace.error,
                }
                f.write(json.dumps(record) + "\n")

        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Saved traces to {traces_file}")
        print(f"Saved summary to {summary_file}")


async def run_mock_benchmark(
    num_requests: int = 20,
    concurrency: int = 2,
    output_dir: Optional[Path] = None,
) -> LoadGenResult:
    """Convenience function to run a mock benchmark."""
    client = MockClient(seed=42)
    config = LoadGenConfig(
        concurrency=concurrency,
        num_requests=num_requests,
        max_new_tokens=64,
        warmup_requests=2,
        warmup_delay_sec=0.5,
        output_dir=output_dir,
    )
    generator = LoadGenerator(client, config)
    return await generator.run()

