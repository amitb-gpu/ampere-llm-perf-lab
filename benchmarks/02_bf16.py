"""
Stage 2: BF16 mixed precision inference.

Single change vs. stage 1: dtype = bfloat16.

Why BF16 over FP16 on Ampere:
- Same exponent range as FP32 (no underflow risk for activations)
- Same tensor-core throughput as FP16 on SM 8.0+
- Modern production default for LLM inference
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from _common import MODEL_ID, MAX_NEW_TOKENS, MEASURE_RUNS, WARMUP_RUNS, load_tokenizer, run_stage

STAGE_NAME = "02_bf16"


def load_model_and_tokenizer():
    tokenizer = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--measure-runs", type=int, default=MEASURE_RUNS)
    parser.add_argument("--warmup-runs", type=int, default=WARMUP_RUNS)
    args = parser.parse_args()

    run_stage(
        stage_name=STAGE_NAME,
        model_loader=load_model_and_tokenizer,
        stage_config={
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "batch_size": 1,
        },
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
