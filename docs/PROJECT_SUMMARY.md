# LLM Inference Performance Lab on NVIDIA Ampere

A hands-on study of how to make small Large Language Models run faster on consumer NVIDIA hardware, built as a learning and demonstration project.

**Hardware:** NVIDIA RTX A1000 Laptop GPU (Ampere architecture, 6 GB VRAM)
**Model:** Meta Llama-3.2-1B-Instruct
**Status:** 2 of 8 planned optimization stages complete

---

## Plain-Language Summary

### What this project is

Modern AI chatbots (like ChatGPT, Claude, or Gemini) are powered by *Large Language Models*. These models are huge — billions of "parameters" — and running them takes serious computer hardware. Companies like NVIDIA spend enormous engineering effort making sure their hardware (GPUs) runs these models as fast and efficiently as possible.

This project is my hands-on study of *how* that performance work is actually done. I take an open-source AI model from Meta, run it on an ordinary laptop GPU, and then apply a series of optimization techniques one at a time — measuring how much faster the model runs after each change. The goal is to build deep, practical understanding of the same engineering work that NVIDIA's "Developer Technology" teams do for a living.

### What I've done so far

I picked a 1-billion-parameter Llama model from Meta. In its original, unoptimized form, it generates text at about **25 words per second** on my laptop's GPU. After applying just one optimization — a smarter way of representing numbers in memory — I doubled that speed to **47 words per second** while *also* cutting memory usage in half. Same model, same hardware, same answers (effectively); just a smarter way of using the chip.

### What I learned that's worth telling you

The most interesting finding wasn't the speedup itself — it's *why* different parts of the model sped up by different amounts. When the AI is "reading" your prompt, it works one way and got nearly **4× faster**. When it's "writing" its reply, word by word, it works a completely different way and got only **2× faster**. Same optimization, two very different results.

That distinction sounds like a technicality, but it's one of the most important ideas in modern AI performance engineering. It's the reason companies like NVIDIA design specialized hardware features and software libraries specifically for the "writing" phase — because that's the part that's slowest and that users actually wait for.

### What's still ahead

I have six more optimization stages planned, including techniques used in production AI systems at companies like OpenAI and Anthropic: a custom GPU "kernel" I'll write from scratch in a language called Triton, and comparisons against NVIDIA's own production-grade inference framework (TensorRT-LLM). Each stage adds a chapter to the story of how a model goes from "barely usable" to "genuinely fast."

---

## Technical Summary

### Project goals

1. Build a reproducible, layered benchmark of LLM inference optimizations on Ampere hardware
2. Quantify each optimization in isolation (one variable changed per stage)
3. Develop and demonstrate hands-on understanding of the GPU performance stack from PyTorch down to custom Triton kernels
4. Produce a body of work directly aligned with NVIDIA AI DevTech engineering practice

### Methodology

- **Greedy decoding, fixed seed** — eliminates sampling variance so stage-to-stage comparisons measure the optimization, not RNG
- **CUDA event timing** — async kernel launches make wall-clock timing unreliable; events record on the GPU stream itself
- **Explicit warmup runs** — first calls trigger cuDNN autotuning, kernel JIT, and allocator warmup; including them in the average is a classic perf-bench mistake
- **Per-prompt-tag decomposition** — short and long prompts exercise different regimes (decode-bound vs prefill-bound)
- **Median aggregation** — robust to thermal throttling outliers on a laptop GPU
- **Prefill vs. decode separation** — these are fundamentally different computational regimes; aggregating them hides the actual story

### Stage 01 — FP32 eager baseline

Deliberately naive starting point: float32 weights (Ampere's BF16 tensor cores idle for matmul), eager-mode PyTorch (no compilation), HuggingFace eager attention (no FlashAttention, no SDPA), batch size 1.

| Prompt | Tokens/sec | Prefill (ms) | Decode (ms/tok) | Peak VRAM |
|---|---|---|---|---|
| short (15 input tokens) | 25.7 | 97.6 | 38.9 | 4749 MB |
| long (~150 input tokens) | 24.5 | 187.4 | 40.8 | 4802 MB |

Weights resident: 4714 MB (1.24B params × 4 bytes ≈ 4.95 GB minus allocator alignment savings).

### Stage 02 — BF16 mixed precision

Single change vs. Stage 01: `dtype=torch.bfloat16`. Eager attention retained so the dtype effect is isolated. BF16 chosen over FP16 for inference because it preserves FP32's exponent range while delivering identical tensor-core throughput on SM 8.0+.

| Prompt | Tokens/sec | Prefill (ms) | Decode (ms/tok) | Peak VRAM |
|---|---|---|---|---|
| short | 47.5 | 24.1 | 21.1 | 2380 MB |
| long | 47.0 | 51.7 | 21.3 | 2406 MB |

Weights resident: 2358 MB (exact halving as expected).

### Key finding: prefill and decode respond differently to the same optimization

| Phase | Speedup | Why |
|---|---|---|
| Prefill | ~3.6–4.0× | Compute-bound; full prompt batched into a single matmul; Ampere's BF16 tensor cores engage where FP32 had no tensor-core path at all |
| Decode | ~1.85–1.92× | Memory-bandwidth-bound; single-token matmul against the full weight matrix means the GPU spends its time *reading weights from HBM*, not computing. Halving weight size halves bandwidth requirement, which roughly halves the time. |

This asymmetry — that decode speedup tracks the dtype size ratio while prefill speedup tracks tensor-core acceleration — is the central mental model for LLM inference optimization. Every subsequent optimization in this project's roadmap (FlashAttention, KV-cache batching, INT4 quantization, custom kernels) targets one or both of these regimes with knowledge of which one is the bottleneck.

### Variance observation

Stage 01 showed one outlier run (decode jumped from ~38 ms/tok median to 67 ms/tok on run 3) consistent with brief thermal throttling on a laptop GPU. Stage 02 was remarkably stable (all 5 runs within ~5% of median) because the lower-precision workload generates less heat. This is a real and underappreciated benefit of low-precision inference for thermally-constrained edge deployment scenarios.

Median-based aggregation absorbed the outlier without distorting the headline number.

### Roadmap (stages 3–8)

3. **`torch.compile`** with `mode="reduce-overhead"` — kernel fusion and CUDA graph capture; expected to particularly help decode by reducing per-step kernel launch overhead
4. **FlashAttention-2 / SDPA** — attention kernel optimization; bigger impact at longer contexts
5. **KV-cache batching** — throughput vs latency trade study across batch sizes 1, 2, 4, 8
6. **INT8 / INT4 quantization** — weight-only quantization via bitsandbytes; targets decode (the memory-bound regime) directly
7. **TensorRT-LLM** — production-grade inference engine as the upper-bound reference point
8. **Custom Triton kernel** — fused RMSNorm or quantized matmul written from scratch; the "novel algorithm leveraging NVIDIA hardware" capstone

Each stage will be accompanied by Nsight Systems traces showing kernel-level evidence of where the speedup comes from.

### Mapping to NVIDIA AI DevTech Engineering Manager role

| Job description requirement | This project demonstrates |
|---|---|
| Optimize end-to-end performance of real DL applications | Layered, quantified optimization of a real LLM with isolated single-variable changes |
| Develop novel algorithms leveraging NVIDIA hardware | Custom Triton kernel planned as Stage 8; Ampere-specific tensor-core utilization shown in Stage 2 |
| Influence next-gen HW/SW design | Findings documentation includes a "hardware wishlist" section — what specific architectural features would matter for next-generation small-model inference |
| Communicate technical solutions to internal and external collaborators | This document; per-stage results JSON; commit-by-commit narrative on GitHub |
| Drive technical initiatives to advance state-of-the-art | Project structured as a comparative study against TensorRT-LLM (the production state-of-the-art), with a custom-kernel capstone aiming to match or approach it on a single op |

---

## Repository

- **Code:** [github.com/amitb-gpu/ampere-llm-perf-lab](https://github.com/amitb-gpu/ampere-llm-perf-lab)
- **Stack:** PyTorch 2.6 + CUDA 12.4, transformers 5.7, Python 3.11, WSL2 on Ubuntu 24
- **Hardware:** NVIDIA RTX A1000 6GB Laptop GPU (Ampere SM 8.6)
