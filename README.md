# L4 Inference Benchmark Lab (vLLM) - C1-C8 Optimisation Study

A reproducible benchmarking suite to measure **LLM inference performance trade-offs** on a **single NVIDIA L4 GPU** using **vLLM**. This project runs a controlled experiment matrix (C1-C8), collects trace-level evidence (TTFT/latency/tokens), and produces a final report that quantifies which serving optimisations matter under **realistic long-context workloads**.

---

## Why this project is important

Most “LLM performance” claims online are not trustworthy because they:
- change the workload between runs,
- ignore failure cases (HTTP 400/500 counted as “success”),
- hide context-overflow, batching, or warmup effects,
- report only averages (instead of p95/p99, which matter in production).

In production, you are judged on **SLOs**:
- how quickly the first token appears (**TTFT**) for a user,
- whether tail latency (**p95/p99**) stays stable under concurrency,
- how much throughput you can sustain per GPU (**tokens/sec**),
- whether the system is robust (no silent failures, no OOM surprises),
- and ultimately, cost per token.

This project builds a **credible, audit-friendly benchmark lab** so you can make decisions with evidence, not intuition.

---

## What this project achieves

### 1) Reproducible, audit-friendly benchmarking (not “demo numbers”)
- Fixed workload (`prompts.jsonl`) with **1800 requests** across:
  - **3 tasks**: summarisation / QA / dialogue  
  - **3 token buckets**: S / M / L  
- Each request is **context-safe by construction**:
  - `request_prompt_tokens + max_new_tokens <= 8192`
  - tokenisation uses **chat-template accounting** (same as the inference engine)
- Strict success criteria:
  - `http_status == 200`
  - `completion_tokens > 0`
  - `prompt_tokens_pre_send > 0`
  - valid timestamps (`end >= start`)
- Outputs include full trace evidence (`traces.jsonl`), GPU samples, and run manifests.

### 2) A controlled experiment matrix (C1-C8)
Each configuration changes **one serving factor** at a time. The suite quantifies deltas vs baseline for:
- **Throughput** (req/s, tokens/s)
- **TTFT** (p50/p95) for streaming responsiveness
- **End-to-end latency** (p50/p95/p99)
- **Failure modes** (HTTP 400, timeouts, parsing, OOM)
- **VRAM headroom** and utilisation behaviour

---

## Experiment Matrix (C1-C8)

> Note: exact flags are pinned in `server/vllm/configs/*.yaml` and recorded in `run_manifest.json`.

- **C1** - Baseline vLLM serving on L4 (reference)
- **C2** - Chunked prefill enabled (splits long-prompt prefill into chunks)
- **C3** - CUDA graphs disabled (`--enforce-eager`) (measures launch/capture impact)
- **C4** - Max sequences 512 (increases concurrent request capacity)
- **C5** - KV cache quantisation (FP8 KV) (reduces KV bandwidth)
- **C6** - Swap space 16GB (trades latency for capacity under pressure)
- **C7** - Prefix caching enabled (caches repeated prompt prefixes)
- **C8** - Multi-step scheduling (batches 4 decode steps together)

---

## Hardware / Environment

- **GPU**: NVIDIA L4 (24GB)
- **Instance**: GCP `g2-standard-8`
- **Backend**: vLLM (version pinned in manifest)
- **Model**: `mistralai/Mistral-7B-Instruct-v0.3` (FP16 baseline)
- **Max context**: 8192 tokens

All environment details are captured per run:
- machine type (GCP metadata)
- GPU/driver/CUDA (nvidia-smi)
- model id + tokeniser id
- git SHA
- exact CLI flags

---

## Dataset + Workload Construction

### Inputs (Kaggle-only; no scraping)
- long-form summarisation corpus
- QA corpus
- dialogue corpus

### Prompt packing (why we do it)
Many public datasets skew short. To produce controlled S/M/L buckets, we:
- keep a **PRIMARY** section that defines the task target (what the model should use)
- add **DISTRACTOR** context marked “ignore” to reach desired token lengths
- validate token lengths with the **same chat template** used at inference time

### Buckets (request-level tokens)
- **S**: ≤ 512 tokens  
- **M**: 513-2048 tokens
- **L**: 2049-8192 tokens (with per-task max_new_tokens constraints)

---

## Benchmark Results

### Test Environment
- **GPU**: NVIDIA L4 (24GB VRAM)
- **Model**: Mistral-7B-Instruct-v0.3 (BF16)
- **vLLM Version**: 0.10.1
- **Requests per Config**: 3,600 (1,800 streaming + 1,800 non-streaming)
- **Concurrency**: 4 parallel requests
- **Max New Tokens**: 200

### Overall Metrics (Streaming)

| Config | Description | Throughput | Latency P50 | Latency P95 | TTFT P50 | TTFT P95 |
|--------|-------------|------------|-------------|-------------|----------|----------|
| C1 | Baseline (default vLLM) | 48.8 t/s | 8.67s | 22.41s | 444ms | 2580ms |
| C2 | Chunked Prefill | 48.6 t/s | 8.68s | 22.59s | 438ms | 2621ms |
| C3 | CUDA Graphs OFF | 48.3 t/s | 8.71s | 22.76s | 441ms | 2619ms |
| C4 | Max Sequences 512 | 48.7 t/s | 8.67s | 22.49s | 455ms | 2599ms |
| C5 | FP8 KV Cache | 48.5 t/s | 8.80s | 21.95s | 444ms | 2488ms |
| C6 | Swap Space 16GB | 48.7 t/s | 8.68s | 22.51s | 441ms | 2623ms |
| C7 | Prefix Caching | 48.7 t/s | 8.69s | 22.40s | 441ms | 2582ms |
| C8 | Multi-step Scheduling | 48.3 t/s | 8.72s | 22.66s | 484ms | 2540ms |

### Relative Performance vs Baseline (C1)

| Config | Throughput Change | Latency P50 Change | TTFT P50 Change |
|--------|-------------------|--------------------| ----------------|
| C1 (Baseline) | - | - | - |
| C2 Chunked Prefill | -0.2% | +0.2% | -1.4% |
| C3 CUDA Graphs OFF | -0.9% | +0.5% | -0.5% |
| C4 Max Seqs 512 | -0.1% | +0.0% | +2.5% |
| C5 FP8 KV Cache | -0.4% | +1.5% | +0.0% |
| C6 Swap Space 16GB | -0.1% | +0.1% | -0.6% |
| C7 Prefix Caching | -0.1% | +0.2% | -0.7% |
| C8 Multi-step Sched | -0.8% | +0.6% | +9.0% |

### TTFT by Prompt Size (Streaming)

| Config | Small P50 | Small P95 | Medium P50 | Medium P95 | Large P50 | Large P95 |
|--------|-----------|-----------|------------|------------|-----------|-----------|
| C1 | 155ms | 284ms | 443ms | 740ms | 1029ms | 2615ms |
| C2 | 155ms | 251ms | 437ms | 652ms | 1092ms | 2663ms |
| C3 | 158ms | 260ms | 441ms | 655ms | 1081ms | 2666ms |
| C4 | 177ms | 298ms | 455ms | 914ms | 1054ms | 2661ms |
| C5 | 155ms | 234ms | 444ms | 740ms | 1004ms | 2501ms |
| C6 | 156ms | 258ms | 441ms | 655ms | 1096ms | 2675ms |
| C7 | 182ms | 269ms | 441ms | 798ms | 1108ms | 2617ms |
| C8 | 203ms | 384ms | 484ms | 1000ms | 1112ms | 2549ms |

### GPU Utilization

| Config | GPU Util Mean | GPU Util Max | Memory Used |
|--------|---------------|--------------|-------------|
| C1 | 97.4% | 100% | 20.7 GB |
| C2 | 97.5% | 100% | 20.7 GB |
| C3 | 97.4% | 100% | 20.3 GB |
| C4 | 97.5% | 100% | 20.7 GB |
| C5 | 97.5% | 100% | 20.6 GB |
| C6 | 97.5% | 100% | 20.7 GB |
| C7 | 97.5% | 100% | 20.7 GB |
| C8 | 99.2% | 100% | 20.8 GB |

### Key Findings

1. **All configs perform similarly** (within 1% of baseline)
   - vLLM 0.10+ already has good defaults
   - L4 GPU is compute-bound, not memory-bound

2. **TTFT scales with prompt size** (6.6x from Small to Large)
   - Expected: larger prompts need more prefill compute
   - No config significantly improves this ratio

3. **Multi-step scheduling (C8) hurts TTFT** (+9%)
   - Batching multiple decode steps delays first token
   - Only useful for throughput-focused batch workloads

4. **FP8 KV Cache (C5) shows best tail latency**
   - P95 latency improved by about 2%
   - Useful for latency-sensitive production deployments

5. **GPU is fully saturated** (97-99% utilization)
   - All configs hit 100% GPU util at peak
   - L4 compute is the bottleneck, not memory

---
