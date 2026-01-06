# L4 Inference Benchmark Lab (vLLM) — C1–C8 Optimisation Study

A reproducible benchmarking suite to measure **LLM inference performance trade-offs** on a **single NVIDIA L4 GPU** using **vLLM**. This project runs a controlled experiment matrix (C1–C8), collects trace-level evidence (TTFT/latency/tokens), and produces a final report that quantifies which serving optimisations matter under **realistic long-context workloads**.

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

### 2) A controlled experiment matrix (C1–C8)
Each configuration changes **one serving factor** at a time. The suite quantifies deltas vs baseline for:
- **Throughput** (req/s, tokens/s)
- **TTFT** (p50/p95) for streaming responsiveness
- **End-to-end latency** (p50/p95/p99)
- **Failure modes** (HTTP 400, timeouts, parsing, OOM)
- **VRAM headroom** and utilisation behaviour

---

## Experiment Matrix (C1–C8)

> Note: exact flags are pinned in `server/vllm/configs/*.yaml` and recorded in `run_manifest.json`.

- **C1** — Baseline vLLM serving on L4 (reference)
- **C2** — Chunked prefill enabled (optimises long-prompt prefill behaviour)
- **C3** — CUDA graphs disabled (`--enforce-eager`) (measures launch/capture impact)
- **C4** — Tokeniser pool enabled (reduces CPU-side tokenisation bottlenecks)
- **C5** — KV cache quantisation (FP8 KV) (improves capacity / reduces KV bandwidth)
- **C6** — Memory offload / swap-space (trades latency for capacity under pressure)
- **C7** — Speculative decoding (ngram-based) (improves decoding throughput)
- **C8** — FP8 weights model (reduces weight bandwidth / VRAM, improves kernels)

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
- **M**: 513–2048 tokens  
- **L**: 2049–8192 tokens (with per-task max_new_tokens constraints)

---
