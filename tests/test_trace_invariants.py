"""Tests for trace invariants and success classification."""

import json
import pytest
from pathlib import Path


def validate_trace(trace: dict) -> tuple[bool, str | None]:
    """Validate a trace record meets all invariants.
    
    Invariants for a SUCCESSFUL trace:
    1. success == True
    2. tokens (completion_tokens) > 0
    3. prompt_tokens > 0
    4. latency > 0
    5. No error field or error is None
    
    Returns (is_valid, error_message)
    """
    if not trace.get("success"):
        # Failed traces are valid (correctly marked)
        return True, None
    
    # Check invariants for successful traces
    tokens = trace.get("tokens", trace.get("completion_tokens", 0))
    prompt_tokens = trace.get("prompt_tokens", 0)
    latency = trace.get("latency", 0)
    
    if tokens <= 0:
        return False, f"SUCCESS trace with tokens={tokens} (must be > 0)"
    
    if prompt_tokens <= 0:
        return False, f"SUCCESS trace with prompt_tokens={prompt_tokens} (must be > 0)"
    
    if latency <= 0:
        return False, f"SUCCESS trace with latency={latency} (must be > 0)"
    
    if trace.get("error"):
        return False, f"SUCCESS trace with error={trace.get('error')}"
    
    return True, None


class TestTraceInvariants:
    """Test trace validation logic."""
    
    def test_valid_successful_trace(self):
        """A valid successful trace passes all invariants."""
        trace = {
            "success": True,
            "latency": 5.5,
            "tokens": 128,
            "prompt_tokens": 512,
            "idx": 0,
            "task": "summarization",
            "bucket": "M",
        }
        is_valid, error = validate_trace(trace)
        assert is_valid, f"Valid trace should pass: {error}"
    
    def test_zero_tokens_is_invalid_success(self):
        """A trace with tokens=0 cannot be success=True."""
        trace = {
            "success": True,
            "latency": 0.014,
            "tokens": 0,
            "prompt_tokens": 0,
            "idx": 525,
        }
        is_valid, error = validate_trace(trace)
        assert not is_valid, "Zero tokens should invalidate success"
        assert "tokens=0" in error
    
    def test_zero_prompt_tokens_is_invalid_success(self):
        """A trace with prompt_tokens=0 cannot be success=True."""
        trace = {
            "success": True,
            "latency": 5.0,
            "tokens": 128,
            "prompt_tokens": 0,
            "idx": 100,
        }
        is_valid, error = validate_trace(trace)
        assert not is_valid, "Zero prompt_tokens should invalidate success"
        assert "prompt_tokens=0" in error
    
    def test_negative_latency_is_invalid(self):
        """A trace with negative latency cannot be success=True."""
        trace = {
            "success": True,
            "latency": -0.5,
            "tokens": 128,
            "prompt_tokens": 512,
            "idx": 0,
        }
        is_valid, error = validate_trace(trace)
        assert not is_valid, "Negative latency should invalidate success"
        assert "latency=" in error
    
    def test_failed_trace_is_valid(self):
        """A properly marked failed trace is valid."""
        trace = {
            "success": False,
            "latency": 0.5,
            "tokens": 0,
            "prompt_tokens": 0,
            "error": "Timeout",
            "idx": 0,
        }
        is_valid, error = validate_trace(trace)
        assert is_valid, "Failed trace should be valid"


class TestLatencyAggregation:
    """Test that latency aggregation excludes failed requests."""
    
    def test_exclude_failed_from_latency_stats(self):
        """Failed requests must be excluded from latency calculations."""
        traces = [
            {"success": True, "latency": 10.0, "tokens": 100, "prompt_tokens": 500},
            {"success": True, "latency": 15.0, "tokens": 100, "prompt_tokens": 500},
            {"success": True, "latency": 20.0, "tokens": 100, "prompt_tokens": 500},
            {"success": False, "latency": 0.01, "tokens": 0, "prompt_tokens": 0},  # Should be excluded
            {"success": True, "latency": 0.02, "tokens": 0, "prompt_tokens": 0},  # Invalid success - exclude
        ]
        
        # Only count traces that are BOTH success=True AND tokens>0
        valid_successful = [
            t for t in traces 
            if t.get("success") and t.get("tokens", 0) > 0
        ]
        
        assert len(valid_successful) == 3, "Only 3 truly successful traces"
        
        latencies = sorted([t["latency"] for t in valid_successful])
        assert latencies == [10.0, 15.0, 20.0]
        assert min(latencies) == 10.0, "Min latency should be 10.0, not 0.01 or 0.02"


class TestStreamingInvariants:
    """Test streaming-specific invariants."""
    
    def test_ttft_before_end(self):
        """Time to first token must be before end time."""
        # This would be tested with actual streaming traces
        # For now, test the invariant logic
        trace = {
            "success": True,
            "ttft_ms": 100,
            "total_time_ms": 5000,
            "tokens": 100,
            "prompt_tokens": 500,
        }
        ttft = trace.get("ttft_ms", 0)
        total = trace.get("total_time_ms", 0)
        assert ttft <= total, "TTFT must be <= total time"

