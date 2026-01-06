"""Tests for smoke test functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from bench.runner.client import CompletionRequest, MockClient
from bench.runner.loadgen import LoadGenConfig, LoadGenerator, run_mock_benchmark


class TestMockClient:
    """Tests for MockClient."""

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test mock client health check."""
        client = MockClient()
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_complete_returns_response(self) -> None:
        """Test mock client returns a valid response."""
        client = MockClient(seed=42)
        request = CompletionRequest(
            prompt="Test prompt",
            max_tokens=50,
            request_id="test_001",
        )

        response = await client.complete(request)

        assert response.request_id == "test_001"
        assert response.status == "success"
        assert response.ttft_ms > 0
        assert response.total_time_ms > response.ttft_ms
        assert response.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_deterministic_with_seed(self) -> None:
        """Test mock client is deterministic with same seed."""
        client1 = MockClient(seed=123)
        client2 = MockClient(seed=123)

        request = CompletionRequest(prompt="Test", max_tokens=50)

        response1 = await client1.complete(request)
        response2 = await client2.complete(request)

        assert response1.completion_tokens == response2.completion_tokens


class TestLoadGenerator:
    """Tests for LoadGenerator."""

    @pytest.mark.asyncio
    async def test_run_with_mock_client(self) -> None:
        """Test load generator runs with mock client."""
        client = MockClient(seed=42)
        config = LoadGenConfig(
            concurrency=2,
            num_requests=10,
            max_new_tokens=32,
            warmup_requests=1,
            warmup_delay_sec=0.1,
        )

        generator = LoadGenerator(client, config)
        result = await generator.run()

        assert result.total_requests == 10
        assert result.successful_requests == 10
        assert result.failed_requests == 0
        assert len(result.traces) == 10
        assert result.duration_sec > 0

    @pytest.mark.asyncio
    async def test_saves_traces_to_file(self) -> None:
        """Test that traces are saved to output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"

            client = MockClient(seed=42)
            config = LoadGenConfig(
                concurrency=1,
                num_requests=5,
                max_new_tokens=16,
                warmup_requests=0,
                warmup_delay_sec=0,
                output_dir=output_dir,
            )

            generator = LoadGenerator(client, config)
            await generator.run()

            # Check files were created
            assert (output_dir / "traces.jsonl").exists()
            assert (output_dir / "summary.json").exists()

            # Validate traces file
            with open(output_dir / "traces.jsonl") as f:
                lines = f.readlines()
                assert len(lines) == 5

                first_trace = json.loads(lines[0])
                assert "request_id" in first_trace
                assert "ttft_ms" in first_trace
                assert "status" in first_trace


class TestRunMockBenchmark:
    """Tests for run_mock_benchmark convenience function."""

    @pytest.mark.asyncio
    async def test_run_mock_benchmark(self) -> None:
        """Test the convenience function runs successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "smoke"

            result = await run_mock_benchmark(
                num_requests=5,
                concurrency=1,
                output_dir=output_dir,
            )

            assert result.successful_requests == 5
            assert (output_dir / "traces.jsonl").exists()


class TestSmokeTestOutputs:
    """Tests for smoke test output file generation."""

    @pytest.mark.asyncio
    async def test_smoke_test_creates_expected_files(self) -> None:
        """Test that smoke test creates all expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run minimal mock benchmark
            await run_mock_benchmark(
                num_requests=10,
                concurrency=2,
                output_dir=output_dir,
            )

            # Verify traces file
            traces_file = output_dir / "traces.jsonl"
            assert traces_file.exists()
            with open(traces_file) as f:
                traces = [json.loads(line) for line in f]
                assert len(traces) == 10

            # Verify summary file
            summary_file = output_dir / "summary.json"
            assert summary_file.exists()
            with open(summary_file) as f:
                summary = json.load(f)
                assert summary["total_requests"] == 10
                assert summary["successful_requests"] == 10

    @pytest.mark.asyncio
    async def test_trace_record_schema(self) -> None:
        """Test trace records have expected schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            await run_mock_benchmark(
                num_requests=3,
                concurrency=1,
                output_dir=output_dir,
            )

            with open(output_dir / "traces.jsonl") as f:
                trace = json.loads(f.readline())

            required_fields = {
                "request_id",
                "prompt_tokens",
                "completion_tokens",
                "ttft_ms",
                "total_time_ms",
                "status",
            }

            assert required_fields.issubset(trace.keys())
