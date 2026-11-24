"""Generate one round-scoped synthetic DPO pool without persisting labels in SQLite."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import HypothesisGenerator, InferenceConfig
from paths import DEFAULT_PATHS
from reward_model import AudioTextRewardModel, RewardModelConfig
from rlhf.config import RLHFConfig
from rlhf.preference_data import human_preference_rows
from storage import FeedbackStore
from utils.audio import load_audio
from utils.log import get_logger, setup_logging
from utils.policy_checkpoint import resolve_active_policy_checkpoint, validate_policy_model_id

log = get_logger(__name__)


def _latest_reward_checkpoint(base: Path) -> Path | None:
    candidates = list(base.glob("reward_model_v*")) if base.exists() else []
    return (
        max(candidates, key=lambda path: path.stat().st_mtime_ns)
        if candidates
        else None
    )


def _select_pair(
    candidates: Sequence[str],
    scores: Sequence[float],
    minimum_gap: float,
    minimum_token_edits: int = 1,
) -> tuple[str, str, float] | None:
    ranked = list(zip(candidates, scores))
    if len(ranked) < 2:
        return None
    chosen, chosen_score = max(ranked, key=lambda item: item[1])
    rejected, rejected_score = min(ranked, key=lambda item: item[1])
    gap = chosen_score - rejected_score
    if (
        chosen == rejected
        or gap < minimum_gap
        or _token_edit_distance(chosen, rejected) < minimum_token_edits
    ):
        return None
    return chosen, rejected, gap


def _token_edit_distance(first: str, second: str) -> int:
    left = first.split()
    right = second.split()
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, 1):
        current = [left_index]
        for right_index, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _pool_size(n_human: int, ratio: float, multiplier: float) -> int:
    target = max(1, math.ceil(n_human * ratio))
    return max(1, math.ceil(target * multiplier))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ephemeral synthetic DPO data"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round_id", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/rlhf_config.json"))
    parser.add_argument(
        "--inference_config", type=Path, default=Path("config/inference_config.json")
    )
    parser.add_argument(
        "--reward_config", type=Path, default=Path("config/reward_model_config.json")
    )
    parser.add_argument("--policy_checkpoint", type=Path)
    parser.add_argument("--rm_checkpoint", type=Path)
    parser.add_argument("--n", type=int)
    parser.add_argument("--n_candidates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_synthetic_ratio", type=float)
    parser.add_argument("--pool_multiplier", type=float)
    parser.add_argument("--minimum_reward_margin", type=float)
    parser.add_argument("--minimum_token_edits", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
    rlhf_config = RLHFConfig.from_json(args.config)
    inference_config = InferenceConfig.from_json(args.inference_config)
    reward_config = RewardModelConfig.from_json(args.reward_config)
    validate_policy_model_id(rlhf_config.whisper_model_id, DEFAULT_PATHS.whisper_model_id)
    validate_policy_model_id(inference_config.model_id, rlhf_config.whisper_model_id)
    n_candidates = args.n_candidates or rlhf_config.synthetic_n_candidates
    seed = args.seed if args.seed is not None else rlhf_config.synthetic_seed

    store = FeedbackStore(DEFAULT_PATHS.db_path)
    n_human = len(human_preference_rows(store))
    ratio = (
        args.max_synthetic_ratio
        if args.max_synthetic_ratio is not None
        else rlhf_config.max_synthetic_to_human_ratio
    )
    pool_size = args.n or (
        _pool_size(n_human, ratio, args.pool_multiplier)
        if args.pool_multiplier is not None
        else rlhf_config.synthetic_pool_size
    )
    minimum_reward_margin = (
        args.minimum_reward_margin
        if args.minimum_reward_margin is not None
        else reward_config.reward_gap_threshold
    )
    minimum_token_edits = args.minimum_token_edits or 1
    policy_checkpoint = args.policy_checkpoint or resolve_active_policy_checkpoint(
        store, Path(DEFAULT_PATHS.checkpoints_dir), Path(__file__).resolve().parent,
        base_model_id=inference_config.model_id,
    )
    reward_checkpoint = args.rm_checkpoint or _latest_reward_checkpoint(
        Path(DEFAULT_PATHS.reward_model_dir)
    )
    if reward_checkpoint is None:
        raise RuntimeError(
            "No reward-model checkpoint available for synthetic generation"
        )

    generator = HypothesisGenerator(
        store, inference_config, DEFAULT_PATHS, policy_checkpoint=policy_checkpoint
    )
    generator.load_audio_catalogue(
        datasets=inference_config.datasets, splits=inference_config.splits
    )
    audio_files = store.get_random_audio_files(pool_size, seed)
    if not audio_files:
        raise RuntimeError("No catalogue audio available for synthetic generation")
    policy_version = generator.model_version
    with generator:
        all_candidates = generator.generate_candidates(audio_files, n_candidates)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    inserted = 0
    reward_gap_sum = 0.0
    with AudioTextRewardModel(
        reward_config, checkpoint_path=reward_checkpoint
    ) as reward_model:
        with partial.open("w", encoding="utf-8") as handle:
            source = list(zip(audio_files, all_candidates))
            for start in range(0, len(source), reward_config.batch_size):
                entries = []
                for audio, candidates in source[
                    start : start + reward_config.batch_size
                ]:
                    unique = list(
                        dict.fromkeys(
                            candidate.strip()
                            for candidate in candidates
                            if candidate and candidate.strip()
                        )
                    )
                    if len(unique) < 2:
                        continue
                    try:
                        waveform, _ = load_audio(audio.file_path)
                        entries.append((audio, unique, waveform))
                    except Exception as exc:
                        log.warning(
                            "Synthetic scoring failed for %s: %s", audio.file_path, exc
                        )

                if not entries:
                    continue
                try:
                    score_groups = reward_model.score_candidate_batches(
                        [entry[2] for entry in entries],
                        [entry[1] for entry in entries],
                    )
                except Exception as exc:
                    log.warning(
                        "Batched synthetic scoring failed, retrying singly: %s", exc
                    )
                    score_groups = []
                    for audio, unique, waveform in entries:
                        try:
                            score_groups.append(
                                reward_model.score_candidates(waveform, unique)
                            )
                        except Exception as single_exc:
                            log.warning(
                                "Synthetic scoring failed for %s: %s",
                                audio.file_path,
                                single_exc,
                            )
                            score_groups.append([])

                for (audio, unique, _), scores in zip(entries, score_groups):
                    selected = _select_pair(
                        unique,
                        scores,
                        minimum_reward_margin,
                        minimum_token_edits,
                    )
                    if selected is None:
                        continue
                    chosen, rejected, gap = selected
                    row = {
                        "artifact_version": 1,
                        "round_id": args.round_id,
                        "policy_version": policy_version,
                        "reward_checkpoint": str(Path(reward_checkpoint).resolve()),
                        "audio_id": audio.id,
                        "file_path": audio.file_path,
                        "chosen_hyp": chosen,
                        "rejected_hyp": rejected,
                        "reward_gap": gap,
                        "seed": seed,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    inserted += 1
                    reward_gap_sum += gap

    if inserted == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError("Synthetic generation produced no confident DPO pairs")
    os.replace(partial, args.output)
    store.set_setting(
        "training_synthetic_diagnostics",
        json.dumps(
            {
                "human_pairs": n_human,
                "target_ratio": ratio,
                "pool_multiplier": args.pool_multiplier,
                "pool_size": pool_size,
                "generated_pairs": inserted,
                "mean_reward_margin": reward_gap_sum / inserted,
                "minimum_reward_margin": minimum_reward_margin,
                "minimum_token_edits": minimum_token_edits,
            }
        ),
    )
    log.info("Synthetic DPO artifact: %s (%d pairs)", args.output, inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
