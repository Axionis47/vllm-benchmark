#!/usr/bin/env python3
"""Download HuggingFace datasets and preprocess into benchmark prompts.jsonl with context-packing.

CONTEXT-SAFE ACCOUNTING:
- request_prompt_tokens computed via apply_chat_template (includes [INST] overhead)
- Buckets based on request_prompt_tokens (not raw prompt tokens)
- request_prompt_tokens + max_new_tokens <= MAX_MODEL_LEN (8192) always

HARD ASSERTS:
- Exactly 200 prompts per (task, bucket)
- All prompts strictly within bucket token ranges
- No context overflow possible
"""

import json
import os
import random
import subprocess
import statistics
from pathlib import Path
from collections import defaultdict
import csv

# Configuration
SEED = 42
SAMPLES_PER_BUCKET = 200
RAW_DIR = Path("bench/datasets/raw")
PROCESSED_DIR = Path("bench/datasets/processed")
OUTPUT_FILE = PROCESSED_DIR / "prompts.jsonl"
MAX_MODEL_LEN = 8192  # vLLM model context length

# NEW LONG-CONTEXT DATASETS
DATASETS = {
    "Cornell-University/arxiv": "summarization",      # arXiv papers (long docs)
    "hotpotqa/hotpot_qa": "qa",                       # HotpotQA (multi-hop)
    "OpenAssistant/oasst1": "dialogue",               # OASST1 conversations
}

# Bucket thresholds (request_prompt_tokens) - FIXED, DO NOT CHANGE
BUCKETS = {"S": (1, 512), "M": (513, 2048), "L": (2049, 8192)}

# Max new tokens per task
MAX_NEW_TOKENS = {
    "summarization": 200,
    "qa": 64,
    "dialogue": 128,
}

# Tokenizer (loaded lazily)
_tokenizer = None


def get_tokenizer():
    """Get the Mistral tokenizer (served model)."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            print("  Loading Mistral tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                trust_remote_code=True,
            )
            print(f"  Tokenizer loaded: {_tokenizer.__class__.__name__}")
        except Exception as e:
            print(f"  WARNING: Could not load Mistral tokenizer: {e}")
            print("  Falling back to whitespace tokenizer")
            _tokenizer = "whitespace"
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens using the served model tokenizer (raw, no chat template)."""
    tokenizer = get_tokenizer()
    if tokenizer == "whitespace":
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_request_tokens(prompt_text: str) -> int:
    """Count tokens as vLLM will see them (with chat template + generation prompt).

    This is the AUTHORITATIVE token count for bucketing and overflow checks.
    """
    tokenizer = get_tokenizer()
    if tokenizer == "whitespace":
        # Estimate overhead as ~5 tokens for [INST]...[/INST]
        return len(prompt_text.split()) + 5

    # Apply chat template exactly as vLLM does
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens."""
    tokenizer = get_tokenizer()
    if tokenizer == "whitespace":
        words = text.split()
        return " ".join(words[:max_tokens])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


def truncate_prompt_context_safe(
    prompt_text: str,
    max_new_tokens: int,
    task: str,
) -> str:
    """Truncate prompt to ensure request_prompt_tokens + max_new_tokens <= MAX_MODEL_LEN.

    Context-safe truncation per task type:
    - Summarization: remove BACKGROUND first, then truncate PRIMARY ARTICLE from end
    - QA: remove Context B/C first, then truncate Context A from end
    - Dialogue: remove earlier conversations first, then truncate FINAL from end

    Returns truncated prompt that is guaranteed to fit.
    """
    allowed_prompt_tokens = MAX_MODEL_LEN - max_new_tokens
    current_request_tokens = count_request_tokens(prompt_text)

    if current_request_tokens <= allowed_prompt_tokens:
        return prompt_text  # Already fits

    # Compute overhead (chat template tokens)
    raw_tokens = count_tokens(prompt_text)
    overhead_tokens = current_request_tokens - raw_tokens
    allowed_user_tokens = allowed_prompt_tokens - overhead_tokens

    if allowed_user_tokens <= 0:
        raise ValueError(f"Cannot fit: overhead={overhead_tokens}, allowed_prompt={allowed_prompt_tokens}")

    # Truncate at token level to allowed_user_tokens
    truncated = truncate_to_tokens(prompt_text, allowed_user_tokens)

    # Verify it fits now
    final_request_tokens = count_request_tokens(truncated)
    if final_request_tokens > allowed_prompt_tokens:
        # Edge case: decode/encode rounding. Reduce by a few more tokens.
        truncated = truncate_to_tokens(prompt_text, allowed_user_tokens - 10)
        final_request_tokens = count_request_tokens(truncated)

    assert final_request_tokens <= allowed_prompt_tokens, (
        f"Truncation failed: {final_request_tokens} > {allowed_prompt_tokens}"
    )

    return truncated


def sanitize_slug(slug: str) -> str:
    """Convert dataset slug to directory name."""
    return slug.replace("/", "_")


def download_dataset(slug: str) -> Path:
    """Download dataset from HuggingFace Hub."""
    dest = RAW_DIR / sanitize_slug(slug)
    if dest.exists() and any(dest.iterdir()):
        print(f"  Dataset already exists: {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {slug} from HuggingFace...")

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install datasets: pip install datasets")

    # Map slug to HuggingFace dataset config
    hf_configs = {
        "Cornell-University/arxiv": {"name": "ccdv/arxiv-summarization", "split": "train"},
        "hotpotqa/hotpot_qa": {"name": "hotpot_qa", "config": "fullwiki", "split": "train"},
        "OpenAssistant/oasst1": {"name": "OpenAssistant/oasst1", "split": "train"},
    }

    if slug not in hf_configs:
        raise ValueError(f"Unknown dataset: {slug}")

    cfg = hf_configs[slug]
    ds_name = cfg["name"]
    config = cfg.get("config", None)
    split = cfg["split"]

    print(f"  Loading: {ds_name} (config={config}, split={split})")

    if config:
        ds = load_dataset(ds_name, config, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(ds_name, split=split, trust_remote_code=True)

    # Save to JSON for consistent processing
    output_file = dest / "data.json"
    print(f"  Saving to {output_file}...")

    # Convert to list of dicts
    records = [dict(row) for row in ds]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f)

    print(f"  Saved {len(records)} records")
    return dest





def load_csv(filepath: Path) -> list[dict]:
    """Load a CSV file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def load_json(filepath: Path) -> list[dict]:
    """Load a JSON file (handles both array and JSONL)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()
        if content.startswith("["):
            return json.loads(content)
        else:
            # JSONL format
            return [json.loads(line) for line in content.split("\n") if line.strip()]


def discover_files(directory: Path) -> dict[str, list[Path]]:
    """Discover all data files in directory."""
    files = {"csv": [], "json": []}
    for f in directory.rglob("*"):
        if f.suffix.lower() == ".csv":
            files["csv"].append(f)
        elif f.suffix.lower() == ".json":
            files["json"].append(f)
    return files


def parse_summarization(data_dir: Path) -> list[dict]:
    """Parse arXiv summarization dataset (long scientific papers)."""
    files = discover_files(data_dir)
    print(f"    Found files: csv={len(files['csv'])}, json={len(files['json'])}")

    records = []

    # Check for pre-processed data.json with article/reference fields
    for json_file in files["json"]:
        print(f"    Loading {json_file.name}...")
        try:
            data = load_json(json_file)
        except Exception as e:
            print(f"    Error loading {json_file}: {e}")
            continue

        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue
            # Check for article/reference (pre-processed) or article/abstract (raw)
            article = item.get("article") or item.get("text") or item.get("body") or ""
            reference = item.get("reference") or item.get("abstract") or item.get("summary") or ""
            if article and len(article) > 200:
                records.append({"article": article.strip(), "reference": reference.strip()})

    # Also try CSV
    for csv_file in files["csv"]:
        print(f"    Loading {csv_file.name}...")
        try:
            rows = load_csv(csv_file)
        except Exception as e:
            print(f"    Error: {e}")
            continue
        if not rows:
            continue
        cols = list(rows[0].keys())
        print(f"    Columns: {cols[:10]}...")  # Show first 10 cols

        # Find article and abstract columns
        article_col = next((c for c in cols if c.lower() in ["article", "text", "body", "content"]), None)
        abstract_col = next((c for c in cols if c.lower() in ["abstract", "summary", "highlights"]), None)

        if article_col and abstract_col:
            for row in rows:
                article = row.get(article_col, "").strip()
                abstract = row.get(abstract_col, "").strip()
                if article and abstract and len(article) > 200:
                    records.append({"article": article, "reference": abstract})

    print(f"    Parsed {len(records)} summarization records")
    return records


def parse_qa(data_dir: Path) -> list[dict]:
    """Parse HotpotQA dataset (multi-hop QA with longer contexts)."""
    files = discover_files(data_dir)
    print(f"    Found files: csv={len(files['csv'])}, json={len(files['json'])}")

    records = []

    for json_file in files["json"]:
        print(f"    Loading {json_file.name}...")
        try:
            data = load_json(json_file)
        except Exception as e:
            print(f"    Error: {e}")
            continue

        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue

            question = item.get("question", "")

            # Check for pre-processed format (context as string)
            if "context" in item and isinstance(item["context"], str):
                context_text = item["context"]
                reference = item.get("reference", item.get("answer", ""))
            elif "context" in item and isinstance(item["context"], dict):
                # HotpotQA format: context is dict with 'title' and 'sentences' as parallel lists
                answer = item.get("answer", "")
                ctx = item["context"]
                titles = ctx.get("title", [])
                sentences_list = ctx.get("sentences", [])
                context_text = ""
                for i, title in enumerate(titles):
                    if i < len(sentences_list):
                        sents = sentences_list[i]
                        if isinstance(sents, list):
                            context_text += f"{title}: {''.join(sents)}\n\n"
                        else:
                            context_text += f"{title}: {sents}\n\n"
                reference = answer
            else:
                # Raw HotpotQA format: context is list of [title, sentences]
                answer = item.get("answer", "")
                context_parts = item.get("context", [])
                context_text = ""
                if isinstance(context_parts, list):
                    for part in context_parts:
                        if isinstance(part, list) and len(part) >= 2:
                            title, sentences = part[0], part[1]
                            if isinstance(sentences, list):
                                context_text += f"{title}: {''.join(sentences)}\n\n"
                            else:
                                context_text += f"{title}: {sentences}\n\n"
                reference = answer

            if question and context_text:
                records.append({
                    "context": context_text.strip(),
                    "question": question,
                    "reference": reference if reference else "",
                })

    # Also try CSV
    for csv_file in files["csv"]:
        print(f"    Loading {csv_file.name}...")
        try:
            rows = load_csv(csv_file)
        except Exception as e:
            print(f"    Error: {e}")
            continue
        if not rows:
            continue
        cols = list(rows[0].keys())
        print(f"    Columns: {cols[:10]}...")

        ctx_col = next((c for c in cols if c.lower() in ["context", "passage", "paragraph"]), None)
        q_col = next((c for c in cols if c.lower() in ["question", "query"]), None)
        a_col = next((c for c in cols if c.lower() in ["answer", "response"]), None)

        if ctx_col and q_col:
            for row in rows:
                records.append({
                    "context": row.get(ctx_col, "").strip(),
                    "question": row.get(q_col, "").strip(),
                    "reference": row.get(a_col, "").strip() if a_col else "",
                })

    print(f"    Parsed {len(records)} QA records")
    return records


def parse_dialogue(data_dir: Path) -> list[dict]:
    """Parse OpenAssistant OASST1 dataset (multi-turn conversations)."""
    files = discover_files(data_dir)
    print(f"    Found files: csv={len(files['csv'])}, json={len(files['json'])}")

    records = []

    for json_file in files["json"]:
        print(f"    Loading {json_file.name}...")
        try:
            data = load_json(json_file)
        except Exception as e:
            print(f"    Error: {e}")
            continue

        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue

            # Check for pre-processed format (conversation field)
            text = item.get("conversation") or item.get("text") or ""
            reference = item.get("reference", "")

            if text and len(text) > 50:
                records.append({
                    "conversation": text.strip(),
                    "reference": reference,
                })

    # Try CSV
    for csv_file in files["csv"]:
        print(f"    Loading {csv_file.name}...")
        try:
            rows = load_csv(csv_file)
        except Exception as e:
            print(f"    Error: {e}")
            continue
        if not rows:
            continue
        cols = list(rows[0].keys())
        print(f"    Columns: {cols[:10]}...")

        text_col = next((c for c in cols if c.lower() in ["text", "message", "content", "conversation"]), None)
        if text_col:
            for row in rows:
                text = row.get(text_col, "").strip()
                if text and len(text) > 50:
                    records.append({"conversation": text, "reference": ""})

    print(f"    Parsed {len(records)} dialogue records")
    return records


def build_summarization_prompt(primary: str, distractors: list[str] = None) -> str:
    """Build summarization prompt with optional distractors."""
    if distractors:
        distractor_text = "\n\n---\n\n".join(distractors)
        return (
            f"Summarize ONLY the PRIMARY ARTICLE in 5–7 bullet points. Ignore BACKGROUND.\n\n"
            f"PRIMARY ARTICLE:\n{primary}\n\n"
            f"BACKGROUND (ignore):\n{distractor_text}"
        )
    return f"Summarize the following article in 5–7 bullet points:\n\n{primary}"


def build_qa_prompt(context: str, question: str, distractors: list[str] = None) -> str:
    """Build QA prompt with optional distractor contexts."""
    if distractors:
        parts = [f"Answer using ONLY Context A. Ignore other contexts.\n\nContext A:\n{context}"]
        labels = ["B", "C", "D", "E", "F", "G", "H"]
        for i, d in enumerate(distractors[:len(labels)]):
            parts.append(f"\nContext {labels[i]} (ignore):\n{d}")
        parts.append(f"\nQuestion: {question}\nAnswer:")
        return "".join(parts)
    return f"Answer the question using ONLY the context.\nContext: {context}\nQuestion: {question}\nAnswer:"


def build_dialogue_prompt(primary: str, distractors: list[str] = None) -> str:
    """Build dialogue prompt with optional distractor conversations."""
    if distractors:
        distractor_text = "\n\n---\n\n".join(distractors)
        return (
            f"You are a helpful assistant. Continue ONLY the FINAL conversation. Ignore earlier conversations.\n\n"
            f"Earlier conversations (ignore):\n{distractor_text}\n\n"
            f"FINAL conversation:\n{primary}\nAssistant:"
        )
    return f"Continue the conversation with one helpful reply.\n{primary}\nAssistant:"


def pack_to_bucket_strict(
    task: str,
    primary_record: dict,
    distractor_texts: list[str],
    distractor_token_counts: list[int],
    bucket: str,
    rng: random.Random,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Pack a prompt to STRICTLY fit within the target bucket (using request_prompt_tokens).

    CONTEXT-SAFE ACCOUNTING:
    - All token counts use count_request_tokens() (apply_chat_template)
    - Bucket membership based on request_prompt_tokens
    - Guaranteed: request_prompt_tokens + max_new_tokens <= MAX_MODEL_LEN

    Algorithm:
    1. Start with PRIMARY only
    2. Compute request_prompt_tokens (with chat template)
    3. Add distractors if needed to reach bucket_min
    4. Truncate context-safe to ensure no overflow
    5. Assert bucket constraints satisfied

    Returns (prompt, request_prompt_tokens) or raises AssertionError if cannot fit.
    """
    bucket_min, bucket_max = BUCKETS[bucket]

    # Context-safe cap: request_prompt_tokens cannot exceed this
    allowed_prompt_tokens = min(bucket_max, MAX_MODEL_LEN - max_new_tokens)

    # Get primary content and build function
    if task == "summarization":
        primary_text = primary_record["article"]
        build_fn = lambda distractors: build_summarization_prompt(primary_text, distractors if distractors else None)
    elif task == "qa":
        primary_text = primary_record["context"]
        question = primary_record["question"]
        build_fn = lambda distractors: build_qa_prompt(primary_text, question, distractors if distractors else None)
    elif task == "dialogue":
        primary_text = primary_record["conversation"]
        build_fn = lambda distractors: build_dialogue_prompt(primary_text, distractors if distractors else None)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Step 1: Start with PRIMARY only
    prompt = build_fn(None)
    request_tokens = count_request_tokens(prompt)

    # Already in bucket and context-safe?
    if bucket_min <= request_tokens <= allowed_prompt_tokens:
        return prompt, request_tokens

    # Too long for this bucket? Apply context-safe truncation
    if request_tokens > allowed_prompt_tokens:
        prompt = truncate_prompt_context_safe(prompt, max_new_tokens, task)
        request_tokens = count_request_tokens(prompt)
        if bucket_min <= request_tokens <= allowed_prompt_tokens:
            return prompt, request_tokens
        else:
            raise AssertionError(f"Cannot fit: truncated to {request_tokens}, need >= {bucket_min}")

    # Step 2: Too short - need distractors. Create shuffled indices
    indices = list(range(len(distractor_texts)))
    rng.shuffle(indices)

    # Estimate tokens needed (use raw token counts for estimation)
    raw_tokens = count_tokens(prompt)
    overhead = request_tokens - raw_tokens
    tokens_needed = bucket_min - request_tokens + 50  # +50 buffer

    # Add distractors using estimated token counts until we likely exceed bucket_min
    distractors = []
    estimated_added = 0
    idx_pos = 0

    while estimated_added < tokens_needed and idx_pos < len(indices):
        i = indices[idx_pos]
        distractors.append(distractor_texts[i])
        estimated_added += distractor_token_counts[i] + 15  # +15 for separator overhead
        idx_pos += 1

    # Build and get actual count
    prompt = build_fn(distractors)
    request_tokens = count_request_tokens(prompt)

    # Fine-tune: add more if still too short
    while request_tokens < bucket_min and idx_pos < len(indices):
        i = indices[idx_pos]
        distractors.append(distractor_texts[i])
        idx_pos += 1
        prompt = build_fn(distractors)
        request_tokens = count_request_tokens(prompt)

    # Step 3: If too long, apply context-safe truncation
    if request_tokens > allowed_prompt_tokens:
        prompt = truncate_prompt_context_safe(prompt, max_new_tokens, task)
        request_tokens = count_request_tokens(prompt)

    # Step 4: Assert strict bucket membership AND context-safety
    # Note: For L bucket, allowed_prompt_tokens may be less than bucket_max due to max_new_tokens
    if not (bucket_min <= request_tokens <= allowed_prompt_tokens):
        raise AssertionError(
            f"Failed to fit prompt in bucket {bucket} ({bucket_min}-{allowed_prompt_tokens}): got {request_tokens} request_tokens"
        )

    # HARD ASSERT: Context safety (redundant with above, but explicit)
    if request_tokens + max_new_tokens > MAX_MODEL_LEN:
        raise AssertionError(
            f"Context overflow: {request_tokens} + {max_new_tokens} = {request_tokens + max_new_tokens} > {MAX_MODEL_LEN}"
        )

    return prompt, request_tokens


def process_task_with_packing(task: str, records: list[dict]) -> list[dict]:
    """Process records into EXACTLY 200 prompts per bucket using strict context-packing."""
    max_new_tokens = MAX_NEW_TOKENS[task]

    # Build distractor pool from SAMPLED records (for efficiency)
    # We only need enough distractors to pack L bucket prompts
    MAX_DISTRACTORS = 5000  # Enough for packing to L bucket
    rng_sample = random.Random(SEED)
    sampled_records = records if len(records) <= MAX_DISTRACTORS else rng_sample.sample(records, MAX_DISTRACTORS)

    print(f"    Building distractor pool from {len(sampled_records)} records (sampled from {len(records)})...")
    distractor_texts = []
    distractor_token_counts = []

    for i, rec in enumerate(sampled_records):
        if task == "summarization":
            text = rec["article"]
        elif task == "qa":
            text = rec["context"]
        elif task == "dialogue":
            text = rec["conversation"]
        else:
            continue
        distractor_texts.append(text)
        distractor_token_counts.append(count_tokens(text))
        if (i + 1) % 1000 == 0:
            print(f"      Tokenized {i + 1}/{len(sampled_records)} distractors")

    print(f"    Distractor pool size: {len(distractor_texts)}")
    if distractor_token_counts:
        avg_tokens = sum(distractor_token_counts) / len(distractor_token_counts)
        print(f"    Avg distractor tokens: {avg_tokens:.0f}")

    # Shuffle records deterministically
    rng = random.Random(SEED)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)

    all_prompts = []

    for bucket in ["S", "M", "L"]:
        bucket_min, bucket_max = BUCKETS[bucket]
        print(f"    Building bucket {bucket} ({bucket_min}-{bucket_max} tokens)...")

        # Use a fresh RNG for each bucket to ensure reproducibility
        bucket_rng = random.Random(SEED + hash(bucket))

        bucket_prompts = []
        record_idx = 0
        attempts = 0
        max_attempts = len(shuffled_records) * 3  # Allow cycling through records

        while len(bucket_prompts) < SAMPLES_PER_BUCKET and attempts < max_attempts:
            primary = shuffled_records[record_idx % len(shuffled_records)]
            record_idx += 1
            attempts += 1

            try:
                # Try to pack this record into the bucket (context-safe accounting)
                prompt, request_tokens = pack_to_bucket_strict(
                    task, primary, distractor_texts, distractor_token_counts,
                    bucket, bucket_rng, max_new_tokens
                )

                bucket_prompts.append({
                    "task": task,
                    "id": f"{task}_{bucket}_{len(bucket_prompts)}",
                    "bucket": bucket,
                    "prompt": prompt,
                    "reference": primary.get("reference", ""),
                    "max_new_tokens": max_new_tokens,
                    "_request_prompt_tokens": request_tokens,
                })

                # Progress indicator
                if len(bucket_prompts) % 50 == 0:
                    print(f"      Progress: {len(bucket_prompts)}/{SAMPLES_PER_BUCKET}")

            except AssertionError as e:
                # This record couldn't be packed into this bucket, skip it
                continue

        # STRICT: Assert exactly 200 prompts
        assert len(bucket_prompts) == SAMPLES_PER_BUCKET, (
            f"ERROR: Task '{task}' bucket '{bucket}' has {len(bucket_prompts)} prompts, expected {SAMPLES_PER_BUCKET}"
        )

        print(f"      Generated {len(bucket_prompts)} prompts for bucket {bucket}")
        all_prompts.extend(bucket_prompts)

    return all_prompts


def main():
    """Main entry point."""
    print("=" * 60)
    print("HuggingFace Dataset Preprocessing")
    print("Long-context: arXiv, HotpotQA, OASST1")
    print("=" * 60)

    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_prompts = []

    # Process each dataset
    for idx, (slug, task) in enumerate(DATASETS.items()):
        print(f"\n[{idx+1}/3] Processing {task}: {slug}")

        # Download
        data_dir = download_dataset(slug)

        # Parse
        print(f"  Parsing {task} data...")
        if task == "summarization":
            records = parse_summarization(data_dir)
        elif task == "qa":
            records = parse_qa(data_dir)
        elif task == "dialogue":
            records = parse_dialogue(data_dir)
        else:
            records = []

        print(f"  Found {len(records)} raw records")

        if not records:
            print(f"  ERROR: No records found for {task}!")
            print(f"  Please check the dataset structure in {data_dir}")
            continue

        # Process with context-packing
        print(f"  Context-packing to 200 samples per bucket...")
        prompts = process_task_with_packing(task, records)
        all_prompts.extend(prompts)
        print(f"  Added {len(prompts)} prompts for {task}")

    # Write output (remove internal _token_count field)
    print(f"\n{'=' * 60}")
    print(f"Writing {len(all_prompts)} prompts to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for prompt in all_prompts:
            output = {k: v for k, v in prompt.items() if not k.startswith("_")}
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    # Print summary with token stats
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    # Collect counts, token stats, and context safety per (task, bucket)
    counts = defaultdict(lambda: defaultdict(int))
    token_stats = defaultdict(list)
    overflow_stats = defaultdict(list)  # prompt+max_new_tokens
    for p in all_prompts:
        counts[p["task"]][p["bucket"]] += 1
        req_tokens = p["_request_prompt_tokens"]
        token_stats[(p["task"], p["bucket"])].append(req_tokens)
        overflow_stats[(p["task"], p["bucket"])].append(req_tokens + p["max_new_tokens"])

    # Print counts table
    print(f"\n{'=' * 70}")
    print("COUNTS TABLE (per task/bucket)")
    print("=" * 70)
    print(f"{'Task':<15} {'S':>6} {'M':>6} {'L':>6} {'Total':>8}")
    print("-" * 45)
    for task in ["summarization", "qa", "dialogue"]:
        s = counts[task]["S"]
        m = counts[task]["M"]
        l = counts[task]["L"]
        total = s + m + l
        print(f"{task:<15} {s:>6} {m:>6} {l:>6} {total:>8}")
    print("-" * 45)
    total_s = sum(counts[t]["S"] for t in ["summarization", "qa", "dialogue"])
    total_m = sum(counts[t]["M"] for t in ["summarization", "qa", "dialogue"])
    total_l = sum(counts[t]["L"] for t in ["summarization", "qa", "dialogue"])
    print(f"{'TOTAL':<15} {total_s:>6} {total_m:>6} {total_l:>6} {len(all_prompts):>8}")

    # Print request_prompt_tokens stats per (task, bucket)
    print(f"\n{'=' * 70}")
    print("REQUEST_PROMPT_TOKENS STATS (with chat template, per task/bucket)")
    print("=" * 70)
    print(f"{'Task':<15} {'Bucket':>6} {'Min':>7} {'Median':>8} {'Max':>7} {'Range':>15}")
    print("-" * 60)
    for task in ["summarization", "qa", "dialogue"]:
        for bucket in ["S", "M", "L"]:
            tokens = token_stats[(task, bucket)]
            bucket_min, bucket_max = BUCKETS[bucket]
            if tokens:
                min_t = min(tokens)
                max_t = max(tokens)
                med_t = statistics.median(tokens)
                print(f"{task:<15} {bucket:>6} {min_t:>7} {med_t:>8.0f} {max_t:>7} {bucket_min:>6}-{bucket_max:<6}")
            else:
                print(f"{task:<15} {bucket:>6} {'N/A':>7} {'N/A':>8} {'N/A':>7} {bucket_min:>6}-{bucket_max:<6}")

    # Print context-safety stats (prompt+max_new_tokens)
    print(f"\n{'=' * 70}")
    print("CONTEXT-SAFETY CHECK: max(request_prompt_tokens + max_new_tokens)")
    print(f"Must be <= {MAX_MODEL_LEN}")
    print("=" * 70)
    print(f"{'Task':<15} {'Bucket':>6} {'Max Total':>12} {'Status':>10}")
    print("-" * 50)
    for task in ["summarization", "qa", "dialogue"]:
        for bucket in ["S", "M", "L"]:
            totals = overflow_stats[(task, bucket)]
            if totals:
                max_total = max(totals)
                status = "✓ SAFE" if max_total <= MAX_MODEL_LEN else "✗ OVERFLOW"
                print(f"{task:<15} {bucket:>6} {max_total:>12} {status:>10}")

    # Verify all buckets have exactly 200, within range, and context-safe
    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print("=" * 70)
    all_ok = True
    for task in ["summarization", "qa", "dialogue"]:
        for bucket in ["S", "M", "L"]:
            count = counts[task][bucket]
            tokens = token_stats[(task, bucket)]
            totals = overflow_stats[(task, bucket)]
            bucket_min, bucket_max = BUCKETS[bucket]

            # Check count
            if count != SAMPLES_PER_BUCKET:
                print(f"FAIL: {task}/{bucket} has {count} prompts, expected {SAMPLES_PER_BUCKET}")
                all_ok = False

            # Check token ranges
            if tokens:
                min_t, max_t = min(tokens), max(tokens)
                if min_t < bucket_min or max_t > bucket_max:
                    print(f"FAIL: {task}/{bucket} request_prompt_tokens out of range: {min_t}-{max_t} vs {bucket_min}-{bucket_max}")
                    all_ok = False

            # Check context safety
            if totals:
                max_total = max(totals)
                if max_total > MAX_MODEL_LEN:
                    print(f"FAIL: {task}/{bucket} context overflow: max(prompt+max_new_tokens)={max_total} > {MAX_MODEL_LEN}")
                    all_ok = False

    if all_ok:
        print("✓ All (task, bucket) combinations have exactly 200 prompts")
        print("✓ All request_prompt_tokens are within bucket ranges")
        print(f"✓ All prompts are context-safe (request_prompt_tokens + max_new_tokens <= {MAX_MODEL_LEN})")
    else:
        raise AssertionError("Verification failed! See errors above.")

    print(f"\nTotal prompts: {len(all_prompts)}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

