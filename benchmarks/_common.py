"""
Shared benchmark harness for the LLM perf lab.

Each stage script imports `run_stage(...)` and provides only what's
unique to that stage: a model loader and a stage name.

This file contains zero stage-specific logic — methodology fixes here
propagate to every benchmark consistently.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

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

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"


@dataclass
class RunMetrics:
    prompt_tag: str
    input_tokens: int
    output_tokens: int
    prefill_ms: float
    total_ms: float
    decode_ms_per_token: float
    peak_vram_mb: float


@dataclass
class StageResult:
    stage: str
    model_id: str
    timestamp: str
    env: dict
    config: dict
    runs: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def capture_env():
    env = {
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
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip()
            env["driver"] = out
        except (subprocess.SubprocessError, FileNotFoundError):
            env["driver"] = "unknown"
    return env


def time_generation(model, tokenizer, prompt, prompt_tag, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = enc["input_ids"].to(model.device)
    input_tokens = inputs.shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    prefill_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_evt.record()
        out = model(inputs, use_cache=True)
        prefill_evt.record()
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated = [next_tok]

        for _ in range(max_new_tokens - 1):
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            generated.append(next_tok)
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


def summarize(runs):
    by_tag = {}
    for r in runs:
        by_tag.setdefault(r.prompt_tag, []).append(r)

    summary = {}
    for tag, rs in by_tag.items():
        totals = [r.total_ms for r in rs]
        prefills = [r.prefill_ms for r in rs]
        decodes = [r.decode_ms_per_token for r in rs]
        peaks = [r.peak_vram_mb for r in rs]
        out_toks = [r.output_tokens for r in rs]
        tok_per_sec = [
            (o - 1) / ((t - p) / 1000)
            for o, t, p in zip(out_toks, totals, prefills) if o > 1
        ]

        summary[tag] = {
            "n_runs": len(rs),
            "total_ms_median": round(statistics.median(totals), 2),
            "prefill_ms_median": round(statistics.median(prefills), 2),
            "decode_ms_per_token_median": round(statistics.median(decodes), 3),
            "tokens_per_sec_median": (
                round(statistics.median(tok_per_sec), 2) if tok_per_sec else None
            ),
            "peak_vram_mb_max": round(max(peaks), 1),
            "input_tokens": rs[0].input_tokens,
            "output_tokens_median": int(statistics.median(out_toks)),
        }
    return summary


def write_results(result):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RAW_DIR / f"{result.stage}.json"
    json_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"  -> wrote {json_path.relative_to(REPO_ROOT)}")

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
    print(f"  -> appended to {CSV_PATH.relative_to(REPO_ROOT)}")


def run_stage(stage_name, model_loader, stage_config, *,
              warmup_runs=WARMUP_RUNS, measure_runs=MEASURE_RUNS,
              max_new_tokens=MAX_NEW_TOKENS):
    if not torch.cuda.is_available():
        sys.exit("CUDA not available - this benchmark requires a GPU.")

    torch.manual_seed(SEED)

    print(f"Stage: {stage_name}")
    print(f"Model: {MODEL_ID}")
    env = capture_env()
    print(f"Device: {env['device_name']} (SM {env['compute_capability']}, "
          f"{env['total_vram_mb']:.0f} MB)")

    print(f"\nLoading via stage loader...")
    model, tokenizer = model_loader()
    weights_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"Weights resident: {weights_mb:.1f} MB")

    print(f"\nWarmup ({warmup_runs} runs per prompt)...")
    for _ in range(warmup_runs):
        for tag, prompt in PROMPTS.items():
            time_generation(model, tokenizer, prompt, tag, max_new_tokens=32)

    print(f"\nMeasuring ({measure_runs} runs per prompt, {max_new_tokens} new tokens)...")
    runs = []
    for i in range(measure_runs):
        for tag, prompt in PROMPTS.items():
            m = time_generation(model, tokenizer, prompt, tag, max_new_tokens=max_new_tokens)
            runs.append(m)
            print(f"  run {i+1} [{tag:5s}] "
                  f"prefill={m.prefill_ms:7.2f}ms  "
                  f"decode={m.decode_ms_per_token:6.2f}ms/tok  "
                  f"peak_vram={m.peak_vram_mb:7.1f}MB")

    summary = summarize(runs)

    full_config = {
        **stage_config,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "weights_mb": round(weights_mb, 1),
    }

    result = StageResult(
        stage=stage_name,
        model_id=MODEL_ID,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        env=env,
        config=full_config,
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


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)
