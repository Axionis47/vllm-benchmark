"""Synthetic prompt templates for benchmarking."""

# Short prompts (~50 tokens)
SHORT_PROMPTS = [
    "Explain what machine learning is in simple terms.",
    "Write a haiku about programming.",
    "What are the three laws of thermodynamics?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "List five benefits of regular exercise.",
]

# Medium prompts (~200 tokens)
MEDIUM_PROMPTS = [
    """You are an expert software engineer. Please review the following code and
    suggest improvements for readability, performance, and best practices:

    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)

    Provide your suggestions in a numbered list format.""",

    """Write a detailed explanation of how neural networks work, covering:
    1. The basic structure of a neural network
    2. How forward propagation works
    3. The role of activation functions
    4. How backpropagation updates weights
    Please make this accessible to someone with basic programming knowledge.""",

    """Compare and contrast three popular programming languages: Python, Rust, and Go.
    For each language, discuss:
    - Primary use cases
    - Performance characteristics
    - Learning curve
    - Ecosystem and community support
    Conclude with recommendations for when to use each.""",
]

# Long prompts (~500 tokens)
LONG_PROMPTS = [
    """You are a senior technical architect at a Fortune 500 company. Your team is
    building a new microservices-based e-commerce platform that needs to handle
    Black Friday-level traffic (approximately 100,000 concurrent users with
    50,000 transactions per minute).

    Current stack considerations:
    - The team has experience with Python, Java, and Go
    - You have budget for cloud infrastructure (AWS or GCP)
    - The system needs to be highly available (99.99% uptime SLA)
    - Data consistency is critical for inventory management
    - The platform must support real-time inventory updates across multiple warehouses
    - Customer data must be GDPR and CCPA compliant

    Please provide a detailed technical architecture proposal including:
    1. Service decomposition strategy
    2. Database choices (SQL vs NoSQL, specific recommendations)
    3. Message queue and event streaming approach
    4. Caching strategy
    5. Load balancing and auto-scaling configuration
    6. Disaster recovery and backup approach
    7. Monitoring and observability stack
    8. Security considerations

    For each component, explain your reasoning and any trade-offs involved.""",
]


def get_synthetic_prompts(
    short_count: int = 5,
    medium_count: int = 3,
    long_count: int = 1,
) -> list[str]:
    """Get a list of synthetic prompts for benchmarking.

    Args:
        short_count: Number of short prompts to include
        medium_count: Number of medium prompts to include
        long_count: Number of long prompts to include

    Returns:
        List of prompts mixed from different length buckets
    """
    prompts: list[str] = []

    # Cycle through each category
    for i in range(short_count):
        prompts.append(SHORT_PROMPTS[i % len(SHORT_PROMPTS)])

    for i in range(medium_count):
        prompts.append(MEDIUM_PROMPTS[i % len(MEDIUM_PROMPTS)])

    for i in range(long_count):
        prompts.append(LONG_PROMPTS[i % len(LONG_PROMPTS)])

    return prompts


def get_prompt_by_bucket(bucket: str) -> str:
    """Get a single prompt from a specific bucket.

    Args:
        bucket: One of 'short', 'medium', 'long'

    Returns:
        A prompt from the specified bucket
    """
    buckets = {
        "short": SHORT_PROMPTS,
        "medium": MEDIUM_PROMPTS,
        "long": LONG_PROMPTS,
    }
    prompts = buckets.get(bucket.lower())
    if prompts is None:
        raise ValueError(f"Unknown bucket: {bucket}. Use 'short', 'medium', or 'long'")
    return prompts[0]

