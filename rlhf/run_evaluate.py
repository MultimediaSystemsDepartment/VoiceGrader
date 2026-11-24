#!/usr/bin/env python
"""
run_evaluate.py -- Standalone WER/CER evaluation on any Whisper checkpoint.

Usage:
  py run_evaluate.py                        # evaluate base openai/whisper-medium
  py run_evaluate.py --lora path/to/ckpt    # evaluate with LoRA adapter
  py run_evaluate.py --model_id openai/whisper-large-v3
  py run_evaluate.py --max_samples 200      # quick sanity check
  py run_evaluate.py --report               # show full run history
"""

from __future__ import annotations

import argparse
import os
import sys
from paths import DEFAULT_PATHS


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate Whisper WER/CER")
    p.add_argument("--model_id",     default=DEFAULT_PATHS.whisper_model_id,
                   help="HuggingFace model ID for Whisper")
    p.add_argument("--lora",         default=None,
                   help="Path to LoRA checkpoint directory")
    p.add_argument("--device",       default="cuda",
                   help="Device: cuda or cpu")
    p.add_argument("--dtype",        default="float16",
                   choices=["float16", "bfloat16"])
    p.add_argument("--max_samples",  type=int,
                   default=int(os.environ.get("EVAL_MAX_SAMPLES", 0)) or None,
                   help="Max utterances per split (default: EVAL_MAX_SAMPLES env var or None=full set)")
    p.add_argument("--report",       action="store_true",
                   help="Print historical training run report from DB")
    p.add_argument("--markdown",     action="store_true",
                   help="Output report as Markdown (requires --report)")
    return p.parse_args()


def main():
    args = _parse_args()

    from paths import DEFAULT_PATHS
    from storage.feedback_store import FeedbackStore
    from evaluation.evaluator import WhisperEvaluator
    from evaluation.report import generate_text_report, generate_markdown_report
    from utils.log import setup_logging

    setup_logging()

    db_path = DEFAULT_PATHS.db_path
    store = FeedbackStore(db_path)

    if args.report:
        runs = store.get_training_runs()
        if args.markdown:
            print(generate_markdown_report(runs))
        else:
            print(generate_text_report(runs))
        return

    evaluator = WhisperEvaluator(
        model_id=args.model_id,
        paths=DEFAULT_PATHS,
        store=store,
        device=args.device,
        torch_dtype=args.dtype,
        lora_checkpoint=args.lora,
        max_samples=args.max_samples,
    )
    result = evaluator.evaluate()
    print(result)


if __name__ == "__main__":
    sys.exit(main())
