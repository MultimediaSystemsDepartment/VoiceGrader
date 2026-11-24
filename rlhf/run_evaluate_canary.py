#!/usr/bin/env python
"""Run or resume deterministic beam-5 evaluation on the frozen WER Monitor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _intermediate_sft_checkpoint_id(store, candidate: dict, previous_id: str):
    parent_id = candidate["parent_checkpoint_id"]
    if not parent_id or parent_id == previous_id:
        return None
    parent = store.get_model_checkpoint(parent_id)
    return parent_id if parent and parent["stage"] == "sft" else None


def _select_round_output(active_predictions, outputs):
    from evaluation.model_health import policy_promotion_from_predictions

    decisions = {
        checkpoint["id"]: policy_promotion_from_predictions(
            active_predictions, predictions
        )
        for checkpoint, predictions in outputs
    }
    eligible = [
        checkpoint
        for checkpoint, _ in outputs
        if decisions[checkpoint["id"]]["promote"]
        and decisions[checkpoint["id"]]["delta"] < 0
    ]
    selected = min(
        eligible,
        key=lambda checkpoint: decisions[checkpoint["id"]]["delta"],
        default=None,
    )
    return selected, decisions


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model health on the WER Monitor"
    )
    parser.add_argument(
        "--checkpoint-id", help="Evaluate only one registered checkpoint"
    )
    parser.add_argument("--canary-set-id", help="Evaluate this locked WER Monitor")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--if-ready",
        action="store_true",
        help="Exit successfully when no frozen WER Monitor exists",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help="Report evaluation failures without returning a failing exit code",
    )
    parser.add_argument(
        "--promote-pending",
        action="store_true",
        help="Evaluate and promote the pending policy only if it passes the WER gate",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from evaluation.canary import CanaryBackfill, discover_checkpoints
    from paths import DEFAULT_PATHS
    from storage.feedback_store import FeedbackStore
    from utils.log import setup_logging
    from utils.policy_checkpoint import record_active_policy_checkpoint

    setup_logging()
    store = FeedbackStore(DEFAULT_PATHS.db_path)
    selected_set = (
        store.get_canary_set(args.canary_set_id)
        if args.canary_set_id
        else store.get_promotion_gate_canary_set()
    )
    if args.if_ready and (selected_set is None or selected_set["status"] != "ready"):
        print("No frozen WER Monitor; health evaluation skipped.")
        return 0
    if args.promote_pending:
        if selected_set is None or selected_set["status"] != "ready":
            print("[REJECT] A frozen WER Monitor is required for policy promotion.")
            return 1
        candidate_id = store.get_setting("pending_policy_checkpoint_id", "")
        previous_id = store.get_setting("pending_policy_previous_checkpoint_id", "")
        if not candidate_id:
            print("No pending policy candidate.")
            return 1
        if not previous_id:
            base = next(
                row
                for row in discover_checkpoints(store, DEFAULT_PATHS)
                if row["stage"] == "base"
            )
            previous_id = base["id"]
        candidate = store.get_model_checkpoint(candidate_id)

        backfill = CanaryBackfill(
            store=store,
            paths=DEFAULT_PATHS,
            device=args.device,
            torch_dtype=args.dtype,
            batch_size=args.batch_size,
        )
        active_evaluation = backfill.run(
            checkpoint_id=previous_id, canary_set_id=selected_set["id"]
        )[0]
        intermediate_evaluation = None
        intermediate_id = _intermediate_sft_checkpoint_id(store, candidate, previous_id)
        if intermediate_id:
            intermediate_evaluation = backfill.run(
                checkpoint_id=intermediate_id, canary_set_id=selected_set["id"]
            )[0]
        candidate_evaluation = backfill.run(
            checkpoint_id=candidate_id, canary_set_id=selected_set["id"]
        )[0]
        required_evaluations = [active_evaluation, candidate_evaluation]
        if intermediate_evaluation is not None:
            required_evaluations.append(intermediate_evaluation)
        if any(row["status"] != "completed" for row in required_evaluations):
            store.set_setting("promotion_gate_status", "failed")
            print("[REJECT] WER Monitor evaluation did not complete.")
            return 1

        round_outputs = []
        if intermediate_evaluation is not None:
            round_outputs.append(
                (
                    store.get_model_checkpoint(intermediate_id),
                    store.get_canary_predictions(intermediate_evaluation["id"]),
                )
            )
        round_outputs.append(
            (
                candidate,
                store.get_canary_predictions(candidate_evaluation["id"]),
            )
        )
        selected, decisions = _select_round_output(
            store.get_canary_predictions(active_evaluation["id"]), round_outputs
        )
        best_output = min(
            (checkpoint for checkpoint, _ in round_outputs),
            key=lambda checkpoint: decisions[checkpoint["id"]]["delta"],
        )
        decision = decisions[selected["id"] if selected else best_output["id"]]
        gate_record = {
            **decision,
            "selected_checkpoint_id": selected["id"] if selected else None,
            "round_outputs": {
                checkpoint["id"]: {
                    "stage": checkpoint["stage"],
                    **decisions[checkpoint["id"]],
                }
                for checkpoint, _ in round_outputs
            },
        }
        store.set_setting(
            "promotion_gate_json", json.dumps(gate_record, sort_keys=True)
        )
        if selected is None:
            store.set_setting("promotion_gate_status", "rejected")
            if decision["changed_predictions"] < decision["min_changed_predictions"]:
                print(
                    "[REJECT] Candidate is behaviorally unchanged on the WER "
                    f"Monitor ({decision['changed_predictions']}/"
                    f"{decision['compared_predictions']} transcripts changed)."
                )
            else:
                print(
                    "[REJECT] Candidate vs active corpus WER delta "
                    f"95% CI upper={decision['ci_high']:.4f} exceeds "
                    f"{decision['max_wer_regression']:.4f}."
                )
            return 3

        model_version = (
            f"{DEFAULT_PATHS.whisper_model_id}+lora:"
            f"{Path(selected['checkpoint_path']).name}"
        )
        record_active_policy_checkpoint(
            store,
            Path(selected["checkpoint_path"]),
            Path(__file__).resolve().parent,
            reason="automatic",
            model_version=model_version,
            base_model_id=DEFAULT_PATHS.whisper_model_id,
        )
        store.set_setting("pending_policy_checkpoint_id", "")
        store.set_setting("pending_policy_previous_checkpoint_id", "")
        store.set_setting("pending_policy_model_version", "")
        store.set_setting("promotion_gate_status", "promoted")
        print(
            f"[PROMOTED] Best round output ({selected['stage']}) vs active "
            "corpus WER delta "
            f"95% CI upper={decision['ci_high']:.4f}."
        )
        return 0

    results = CanaryBackfill(
        store=store,
        paths=DEFAULT_PATHS,
        device=args.device,
        torch_dtype=args.dtype,
        batch_size=args.batch_size,
    ).run(
        checkpoint_id=args.checkpoint_id,
        canary_set_id=args.canary_set_id,
    )

    for result in results:
        print(
            f"{result['checkpoint_id']}  {result['status']}  "
            f"health={result['health_status']}"
        )
    succeeded = all(result["status"] == "completed" for result in results)
    return 0 if succeeded or args.non_blocking else 1


if __name__ == "__main__":
    sys.exit(main())
