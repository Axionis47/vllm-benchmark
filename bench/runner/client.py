"""HTTP client for OpenAI-compatible inference endpoints."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
import asyncio
import random

import httpx


@dataclass
class CompletionRequest:
    """Request for text completion."""

    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    request_id: Optional[str] = None


@dataclass
class CompletionResponse:
    """Response from completion request with timing info."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    text: str
    ttft_ms: float  # Time to first token
    total_time_ms: float  # End-to-end latency
    token_times_ms: list[float] = field(default_factory=list)  # Per-token times
    status: str = "success"
    error: Optional[str] = None


class BaseClient(ABC):
    """Abstract base class for inference clients."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send completion request and return response with timing."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the endpoint is healthy."""
        pass


class BenchmarkClient(BaseClient):
    """HTTP client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
        model: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the endpoint is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send completion request with streaming support using chat completions API."""
        client = await self._get_client()
        request_id = request.request_id or f"req_{time.time_ns()}"

        # Use chat completions API for instruct models
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        token_times: list[float] = []
        completion_text = ""
        completion_tokens = 0
        prompt_tokens = 0

        try:
            if request.stream:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                import json
                                chunk = json.loads(data_str)
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                # Count tokens from delta content
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    token_times.append(time.perf_counter())
                                    completion_tokens += 1
                                    completion_text += content
                                # Get usage if available
                                usage = chunk.get("usage")
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", 0)
                                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
            else:
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                first_token_time = time.perf_counter()
                data = response.json()
                completion_text = data["choices"][0]["message"]["content"]
                prompt_tokens = data["usage"]["prompt_tokens"]
                completion_tokens = data["usage"]["completion_tokens"]

            end_time = time.perf_counter()
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            total_time = (end_time - start_time) * 1000

            return CompletionResponse(
                request_id=request_id,
                prompt_tokens=prompt_tokens or len(request.prompt.split()),
                completion_tokens=completion_tokens,
                text=completion_text,
                ttft_ms=ttft,
                total_time_ms=total_time,
                token_times_ms=[(t - start_time) * 1000 for t in token_times],
                status="success",
            )

        except Exception as e:
            end_time = time.perf_counter()
            return CompletionResponse(
                request_id=request_id,
                prompt_tokens=0,
                completion_tokens=0,
                text="",
                ttft_ms=0,
                total_time_ms=(end_time - start_time) * 1000,
                status="error",
                error=str(e),
            )


class MockClient(BaseClient):
    """Mock client for testing without GPU."""

    def __init__(
        self,
        base_ttft_ms: float = 50.0,
        base_tpot_ms: float = 15.0,
        jitter_pct: float = 0.1,
        seed: int = 42,
    ):
        self.base_ttft_ms = base_ttft_ms
        self.base_tpot_ms = base_tpot_ms
        self.jitter_pct = jitter_pct
        self._rng = random.Random(seed)

    def _jitter(self, value: float) -> float:
        """Add random jitter to a value."""
        jitter = self._rng.uniform(-self.jitter_pct, self.jitter_pct)
        return value * (1 + jitter)

    async def health_check(self) -> bool:
        return True

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate mock response with deterministic timing."""
        request_id = request.request_id or f"mock_{time.time_ns()}"

        # Simulate TTFT
        ttft = self._jitter(self.base_ttft_ms)
        await asyncio.sleep(ttft / 1000)

        # Simulate token generation
        num_tokens = min(request.max_tokens, self._rng.randint(20, 100))
        token_times: list[float] = []
        cumulative_time = ttft

        for _ in range(num_tokens):
            token_time = self._jitter(self.base_tpot_ms)
            cumulative_time += token_time
            token_times.append(cumulative_time)
            await asyncio.sleep(token_time / 1000)

        return CompletionResponse(
            request_id=request_id,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=num_tokens,
            text="[mock response]" * (num_tokens // 2),
            ttft_ms=ttft,
            total_time_ms=cumulative_time,
            token_times_ms=token_times,
            status="success",
        )

