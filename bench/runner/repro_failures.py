#!/usr/bin/env python3
"""Micro reproduction for summarization/L failures.

Sends 5 prompts from summarization/L to live vLLM endpoint with full evidence capture.
"""
import asyncio
import json
import sys
from pathlib import Path

import aiohttp

# Import tokenizer for pre-send token count
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    HAS_TOKENIZER = True
except Exception:
    tokenizer = None
    HAS_TOKENIZER = False


def count_tokens(text: str) -> int:
    """Count tokens using Mistral tokenizer."""
    if HAS_TOKENIZER and tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Fallback estimate
    return int(len(text.split()) * 1.3)


def redact_prompt(prompt: str, max_chars: int = 500) -> str:
    """Redact prompt text after max_chars."""
    if len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars] + f"...[REDACTED {len(prompt) - max_chars} chars]"


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    idx: int,
) -> tuple[dict, dict]:
    """Send single request with full evidence capture."""
    prompt_tokens_pre_send = count_tokens(prompt)
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    request_record = {
        "idx": idx,
        "prompt_chars": len(prompt),
        "prompt_tokens_pre_send": prompt_tokens_pre_send,
        "max_tokens": max_tokens,
        "request_json": json.dumps({
            **payload,
            "messages": [{"role": "user", "content": redact_prompt(prompt)}],
        }),
    }
    
    try:
        async with session.post(endpoint, json=payload) as resp:
            raw_body = await resp.text()
            
            response_record = {
                "idx": idx,
                "http_status": resp.status,
                "response_body_snippet": raw_body[:1000] if len(raw_body) > 1000 else raw_body,
                "response_body_full": raw_body,
            }
            
            if resp.status == 200:
                try:
                    result = json.loads(raw_body)
                    usage = result.get("usage", {})
                    choices = result.get("choices", [])
                    response_record.update({
                        "parse_ok": True,
                        "completion_tokens_parsed": usage.get("completion_tokens", 0),
                        "prompt_tokens_server": usage.get("prompt_tokens", 0),
                        "finish_reason": choices[0].get("finish_reason") if choices else None,
                        "content_snippet": (choices[0].get("message", {}).get("content", "")[:200] 
                                          if choices else None),
                    })
                except json.JSONDecodeError as e:
                    response_record.update({
                        "parse_ok": False,
                        "parse_error": str(e),
                    })
            else:
                response_record["parse_ok"] = False
                
            return request_record, response_record
            
    except Exception as e:
        return request_record, {
            "idx": idx,
            "http_status": None,
            "error": str(e),
            "parse_ok": False,
        }


async def main():
    endpoint = "http://localhost:8000/v1/chat/completions"
    model = "mistral7b"  # Served model name (not HF path)
    max_tokens = 256
    
    # Load prompts
    prompts_path = Path("bench/datasets/processed/prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for i, line in enumerate(f):
            p = json.loads(line)
            p["original_idx"] = i
            if p.get("task") == "summarization" and p.get("bucket") == "L":
                prompts.append(p)
    
    # Take first 5
    prompts = prompts[:5]
    print(f"Loaded {len(prompts)} summarization/L prompts")
    
    requests = []
    responses = []
    
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for p in prompts:
            print(f"Sending idx={p['original_idx']}...")
            req, resp = await send_request(
                session, endpoint, model, p["prompt"], max_tokens, p["original_idx"]
            )
            requests.append(req)
            responses.append(resp)
            print(f"  http_status={resp.get('http_status')}, "
                  f"completion_tokens={resp.get('completion_tokens_parsed', 0)}, "
                  f"finish_reason={resp.get('finish_reason')}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "repro_summarization_L_request.jsonl", "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
    
    with open(results_dir / "repro_summarization_L_response.jsonl", "w") as f:
        for r in responses:
            f.write(json.dumps(r) + "\n")
    
    print(f"\nSaved to results/repro_summarization_L_*.jsonl")
    
    # Print summary
    print("\n" + "=" * 80)
    print("REPRO SUMMARY")
    print("=" * 80)
    for req, resp in zip(requests, responses):
        print(f"\nidx={req['idx']}")
        print(f"  prompt_tokens_pre_send: {req['prompt_tokens_pre_send']}")
        print(f"  http_status: {resp.get('http_status')}")
        print(f"  completion_tokens: {resp.get('completion_tokens_parsed', 0)}")
        print(f"  prompt_tokens_server: {resp.get('prompt_tokens_server', 0)}")
        print(f"  finish_reason: {resp.get('finish_reason')}")
        if resp.get("completion_tokens_parsed", 0) == 0:
            print(f"  ⚠️ FAILURE - response snippet: {resp.get('response_body_snippet', '')[:300]}")


if __name__ == "__main__":
    asyncio.run(main())

