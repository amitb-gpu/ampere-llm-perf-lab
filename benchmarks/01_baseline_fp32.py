"""
Stage 1: Baseline FP32 inference.

Deliberately naive:
- FP32 weights (no Ampere tensor cores engaged for matmul)
- Eager mode (no torch.compile)
- Batch size 1
- No KV cache optimization beyond HF defaults
- No FlashAttention

This is the "worst case" we measure all subsequent optimizations against.

Outputs:
- results/raw/01_baseline_fp32.json   (full structured run data)
- results/benchmark_results.csv       (one row appended per run)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
STAGE_NAME = "01_baseline_fp32"

# Prompts chosen to exercise both prefill-heavy (long input) and
# decode-heavy (short input) regimes. Same prompt across all stages.
PROMPTS = {
    "short": "Explain what a GPU tensor core is in two sentences.",
    "long": (
        "You are a senior systems engineer writing technical documentation. "
        "Write a clear, detailed explanation of how mixed-precision training "
        "works on NVIDIA Ampere GPUs. Cover: (1) what FP16 and BF16 are, "
        "(2) why tensor cores accelerate them, (3) how loss scaling prevents "
        "underflow, and (4) the role of the master weights in FP32. "
        "Aim for a paragraph per topic, suitable for a developer audience "
        "already familiar with deep learning fundamentals but new to "
        "low-precision arithmetic. Begin your explanation now."
    ),
}

WARMUP_RUNS = 2
MEASURE_RUNS = 5
MAX_NEW_TOKENS = 128
SEED = 42

# Repo paths (script lives in benchmarks/, results live in results/)
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Single generation call's measurements."""
    prompt_tag: str
    input_tokens: int
    output_tokens: int
    prefill_ms: float          # time from start to first generated token
    total_ms: float            # full generation wall-clock (CUDA events)
    decode_ms_per_token: float # (total - prefill) / (output - 1)
    peak_vram_mb: float


@dataclass
class StageResult:
    """One full stage run with environment metadata."""
    stage: str
    model_id: str
    timestamp: str
    env: dict[str, Any]
    config: dict[str, Any]
    runs: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment capture (so results are reproducible / debuggable later)
# ---------------------------------------------------------------------------

def capture_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update({
            "cuda_built": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_mb": round(props.total_memory / 1024**2, 1),
            "sm_count": props.multi_processor_count,
        })
        # Driver version via nvidia-smi (best-effort, don't fail if absent)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip()
            env["driver"] = out
        except (subprocess.SubprocessError, FileNotFoundError):
            env["driver"] = "unknown"
    return env


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def time_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    prompt_tag: str,
    max_new_tokens: int,
) -> RunMetrics:
    """
    Measure a single greedy generation using CUDA events.

    Why CUDA events instead of time.time():
    - GPU work is asynchronous; CPU clock returns before kernels finish
    - cuda.synchronize() + time.time() works but is coarser
    - Events record on the stream itself, giving accurate device-side timing
    """
    # Apply chat template — Llama-3.2 instruct expects the proper format.
    # transformers 5.x returns a BatchEncoding (dict-like); we want the raw tensor.
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = enc["input_ids"].to(model.device)
    input_tokens = inputs.shape[1]

    # Reset peak memory tracker so we measure THIS run's footprint only
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Two events for prefill timing: prefill ends when first new token is generated.
    # We approximate by running a single forward pass first, then a separate
    # generate call. A more precise approach would use a custom logits processor
    # to mark the boundary, but this is accurate enough for stage comparisons.
    start_evt = torch.cuda.Event(enable_timing=True)
    prefill_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_evt.record()
        # Manual prefill: forward pass on the full input to populate KV cache
        out = model(inputs, use_cache=True)
        prefill_evt.record()
        past = out.past_key_values
        # Greedy first token
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated = [next_tok]

        # Decode loop — one token at a time, reusing KV cache
        for _ in range(max_new_tokens - 1):
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            generated.append(next_tok)
            # Stop on EOS
            if next_tok.item() == tokenizer.eos_token_id:
                break
        end_evt.record()

    torch.cuda.synchronize()
    output_tokens = len(generated)
    prefill_ms = start_evt.elapsed_time(prefill_evt)
    total_ms = start_evt.elapsed_time(end_evt)
    decode_ms = (total_ms - prefill_ms) / max(output_tokens - 1, 1)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    return RunMetrics(
        prompt_tag=prompt_tag,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prefill_ms=prefill_ms,
        total_ms=total_ms,
        decode_ms_per_token=decode_ms,
        peak_vram_mb=peak_vram_mb,
    )


def summarize(runs: list[RunMetrics]) -> dict[str, Any]:
    """Aggregate per-prompt-tag stats."""
    by_tag: dict[str, list[RunMetrics]] = {}
    for r in runs:
        by_tag.setdefault(r.prompt_tag, []).append(r)

    summary: dict[str, Any] = {}
    for tag, rs in by_tag.items():
        totals = [r.total_ms for r in rs]
        prefills = [r.prefill_ms for r in rs]
        decodes = [r.decode_ms_per_token for r in rs]
        peaks = [r.peak_vram_mb for r in rs]
        out_toks = [r.output_tokens for r in rs]
        # tokens/sec computed from decode phase only — that's the steady-state metric
        tok_per_sec = [(o - 1) / ((t - p) / 1000) for o, t, p in zip(out_toks, totals, prefills) if o > 1]

        summary[tag] = {
            "n_runs": len(rs),
            "total_ms_median": round(statistics.median(totals), 2),
            "prefill_ms_median": round(statistics.median(prefills), 2),
            "decode_ms_per_token_median": round(statistics.median(decodes), 3),
            "tokens_per_sec_median": round(statistics.median(tok_per_sec), 2) if tok_per_sec else None,
            "peak_vram_mb_max": round(max(peaks), 1),
            "input_tokens": rs[0].input_tokens,
            "output_tokens_median": int(statistics.median(out_toks)),
        }
    return summary


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def write_results(result: StageResult) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RAW_DIR / f"{result.stage}.json"
    json_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"  → wrote {json_path.relative_to(REPO_ROOT)}")

    # Append summary rows to the cross-stage CSV
    csv_new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if csv_new:
            writer.writerow([
                "stage", "prompt_tag", "input_tokens", "output_tokens",
                "prefill_ms", "decode_ms_per_token", "tokens_per_sec",
                "peak_vram_mb", "timestamp",
            ])
        for tag, s in result.summary.items():
            writer.writerow([
                result.stage, tag, s["input_tokens"], s["output_tokens_median"],
                s["prefill_ms_median"], s["decode_ms_per_token_median"],
                s["tokens_per_sec_median"], s["peak_vram_mb_max"],
                result.timestamp,
            ])
    print(f"  → appended to {CSV_PATH.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--measure-runs", type=int, default=MEASURE_RUNS)
    parser.add_argument("--warmup-runs", type=int, default=WARMUP_RUNS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available — this benchmark requires a GPU.")

    torch.manual_seed(SEED)

    print(f"Stage: {STAGE_NAME}")
    print(f"Model: {MODEL_ID}")
    env = capture_env()
    print(f"Device: {env['device_name']} (SM {env['compute_capability']}, {env['total_vram_mb']:.0f} MB)")

    print("\nLoading tokenizer + model (FP32)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,    # explicit: no Ampere tensor core path
        attn_implementation="eager",  # no FlashAttention, no SDPA fast path
    ).to("cuda")
    model.eval()

    # Report memory just from loading weights
    weights_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"Weights resident: {weights_mb:.1f} MB")

    # --- Warmup ---
    print(f"\nWarmup ({args.warmup_runs} runs per prompt)...")
    for _ in range(args.warmup_runs):
        for tag, prompt in PROMPTS.items():
            time_generation(model, tokenizer, prompt, tag, max_new_tokens=32)

    # --- Measure ---
    print(f"\nMeasuring ({args.measure_runs} runs per prompt, {args.max_new_tokens} new tokens)...")
    runs: list[RunMetrics] = []
    for i in range(args.measure_runs):
        for tag, prompt in PROMPTS.items():
            m = time_generation(model, tokenizer, prompt, tag, max_new_tokens=args.max_new_tokens)
            runs.append(m)
            print(f"  run {i+1} [{tag:5s}] "
                  f"prefill={m.prefill_ms:7.2f}ms  "
                  f"decode={m.decode_ms_per_token:6.2f}ms/tok  "
                  f"peak_vram={m.peak_vram_mb:7.1f}MB")

    summary = summarize(runs)

    result = StageResult(
        stage=STAGE_NAME,
        model_id=MODEL_ID,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        env=env,
        config={
            "dtype": "float32",
            "attn_implementation": "eager",
            "batch_size": 1,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "weights_mb": round(weights_mb, 1),
        },
        runs=[asdict(r) for r in runs],
        summary=summary,
    )

    print("\n=== Summary ===")
    for tag, s in summary.items():
        print(f"  [{tag}] tok/s={s['tokens_per_sec_median']:.1f}  "
              f"prefill={s['prefill_ms_median']:.1f}ms  "
              f"decode={s['decode_ms_per_token_median']:.2f}ms/tok  "
              f"peak_vram={s['peak_vram_mb_max']:.0f}MB")

    write_results(result)


if __name__ == "__main__":
    main()
