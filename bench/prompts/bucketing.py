"""Prompt length bucketing utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class PromptBucket(Enum):
    """Prompt length bucket categories."""

    SHORT = "short"      # ~50 tokens
    MEDIUM = "medium"    # ~200 tokens
    LONG = "long"        # ~500 tokens


@dataclass
class BucketedPrompt:
    """A prompt with its bucket classification."""

    text: str
    bucket: PromptBucket
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is an approximation; actual tokenization depends on the model.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Simple heuristic: ~4 chars per token, plus adjustment for whitespace
    char_count = len(text)
    word_count = len(text.split())

    # Average of character-based and word-based estimates
    char_estimate = char_count / 4
    word_estimate = word_count * 1.3  # ~1.3 tokens per word on average

    return int((char_estimate + word_estimate) / 2)


def classify_bucket(token_count: int) -> PromptBucket:
    """Classify a token count into a bucket.

    Args:
        token_count: Number of tokens

    Returns:
        The appropriate bucket for this token count
    """
    if token_count < 100:
        return PromptBucket.SHORT
    elif token_count < 350:
        return PromptBucket.MEDIUM
    else:
        return PromptBucket.LONG


def bucket_prompts(prompts: Sequence[str]) -> list[BucketedPrompt]:
    """Bucket a list of prompts by estimated length.

    Args:
        prompts: List of prompt texts

    Returns:
        List of BucketedPrompt objects with classifications
    """
    bucketed: list[BucketedPrompt] = []

    for text in prompts:
        tokens = estimate_tokens(text)
        bucket = classify_bucket(tokens)
        bucketed.append(BucketedPrompt(
            text=text,
            bucket=bucket,
            estimated_tokens=tokens,
        ))

    return bucketed


def get_bucket_distribution(prompts: Sequence[BucketedPrompt]) -> dict[PromptBucket, int]:
    """Get the distribution of prompts across buckets.

    Args:
        prompts: List of bucketed prompts

    Returns:
        Dictionary mapping bucket to count
    """
    distribution: dict[PromptBucket, int] = {
        PromptBucket.SHORT: 0,
        PromptBucket.MEDIUM: 0,
        PromptBucket.LONG: 0,
    }

    for prompt in prompts:
        distribution[prompt.bucket] += 1

    return distribution

