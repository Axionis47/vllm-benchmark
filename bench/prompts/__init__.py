"""Prompt templates and bucketing for benchmarks."""

from bench.prompts.templates import get_synthetic_prompts
from bench.prompts.bucketing import bucket_prompts, PromptBucket

__all__ = ["get_synthetic_prompts", "bucket_prompts", "PromptBucket"]

