"""
run_generate_pairs.py -- populate the hypothesis queue from audio CSV manifests.

Loads Whisper, transcribes audio files with two decoding strategies,
and inserts the resulting hypothesis pairs into the feedback DB as 'pending'.

N-best mode (--n_candidates N): generates N sampled transcriptions per audio,
scores all with the reward model (requires --rm_checkpoint), and inserts the
best/worst pair.  This is the RSO pattern described in docs/DPO_PRIMER.md §11.

Usage:
    py run_generate_pairs.py                         # default config
    py run_generate_pairs.py --n 100                 # generate 100 pairs
    py run_generate_pairs.py --n_candidates 6 --rm_checkpoint _outputs/reward_model/reward_model_v1
    py run_generate_pairs.py --config config/inference_config.json
    py run_generate_pairs.py --catalogue_only        # only ingest CSVs, no transcription
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import HypothesisGenerator, InferenceConfig
from paths import DEFAULT_PATHS
from reward_model import AudioTextRewardModel, RewardModelConfig
from storage import FeedbackStore
from storage.schema import HypothesisPair
from utils.log import get_logger, setup_logging
from utils.policy_checkpoint import (
    ensure_active_policy_model_version,
    resolve_active_policy_checkpoint,
    validate_policy_model_id,
)

log = get_logger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "inference_config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Whisper hypothesis pairs")
    p.add_argument(
        "--config", type=Path, default=None,
        help="Path to inference_config.json (default: config/inference_config.json if present)",
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Number of pairs to generate (overrides config.target_batch_size)",
    )
    p.add_argument(
        "--catalogue_only", action="store_true",
        help="Only ingest CSV manifests into the DB, do not run Whisper",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        choices=["admedvoice", "commonvoice"],
        help="Datasets to include (default: both)",
    )
    p.add_argument(
        "--splits", nargs="+", default=None,
        choices=["train", "val", "test"],
        help="Dataset splits to use (default: train)",
    )
    p.add_argument(
        "--n_candidates", type=int, default=6,
           help="Candidates per clip: one beam result plus temperature samples. "
               "Uses edit-distance pairing without --rm_checkpoint, then reward-model "
               "scoring once a checkpoint exists. Default 6.",
    )
    p.add_argument(
        "--rm_checkpoint", type=Path, default=None,
        help="Path to trained reward model checkpoint (required for --n_candidates > 2).",
    )
    p.add_argument(
        "--policy_checkpoint", type=Path, default=None,
        help="Whisper LoRA policy checkpoint. Defaults to the active/latest valid checkpoint.",
    )
    p.add_argument(
        "--audio_ids_file",
        type=Path,
        default=None,
        help="JSON array of exact audio IDs to generate, preserving file order.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the temperature-sampling pass for reproducible generation.",
    )
    p.add_argument(
        "--nbest_temperature",
        type=float,
        default=None,
        help="Override temperature sampling for this generation run.",
    )
    return p.parse_args()


def _load_audio_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("--audio_ids_file must contain a non-empty JSON array")
    if any(not isinstance(audio_id, str) or not audio_id.strip() for audio_id in payload):
        raise ValueError("Every scheduled audio ID must be a non-empty string")
    if len(set(payload)) != len(payload):
        raise ValueError("Scheduled audio IDs must be unique")
    return payload


def main() -> None:
    args = parse_args()

    setup_logging(logs_dir=DEFAULT_PATHS.logs_dir)
    log.info("=== run_generate_pairs ===")

    # Load inference config
    config_path = args.config or (_DEFAULT_CONFIG if _DEFAULT_CONFIG.exists() else None)
    if config_path:
        log.info("Loading inference config: %s", config_path)
        config = InferenceConfig.from_json(config_path)
    else:
        log.info("No config file found -- using defaults (model: %s)", DEFAULT_PATHS.whisper_model_id)
        config = InferenceConfig(model_id=DEFAULT_PATHS.whisper_model_id)

    # CLI overrides
    if args.datasets:
        config.datasets = args.datasets
    if args.splits:
        config.splits = args.splits
    if args.seed is not None:
        config.generation_seed = args.seed
    if args.nbest_temperature is not None:
        if args.nbest_temperature <= 0:
            raise ValueError("--nbest_temperature must be positive")
        config.nbest_temperature = args.nbest_temperature

    validate_policy_model_id(config.model_id, DEFAULT_PATHS.whisper_model_id)
    store = FeedbackStore(DEFAULT_PATHS.db_path)
    policy_checkpoint = args.policy_checkpoint
    if policy_checkpoint is not None:
        if not (policy_checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Invalid policy checkpoint: {policy_checkpoint}")
    else:
        policy_checkpoint = resolve_active_policy_checkpoint(
            store,
            Path(DEFAULT_PATHS.checkpoints_dir),
            Path(__file__).resolve().parent,
            base_model_id=config.model_id,
        )
    log.info("Generation policy: %s", policy_checkpoint or config.model_id)
    gen = HypothesisGenerator(
        store, config, DEFAULT_PATHS, policy_checkpoint=policy_checkpoint
    )
    ensure_active_policy_model_version(store, config.model_id, policy_checkpoint)

    # Step 1: ingest CSV manifests
    n_catalogued = gen.load_audio_catalogue(
        datasets=config.datasets,
        splits=config.splits,
    )
    n_total_files = store.count_audio_files()
    log.info("Audio catalogue: +%d new files  |  total: %d", n_catalogued, n_total_files)

    if n_total_files == 0:
        log.error(
            "Audio catalogue is empty -- no manifests were found on disk. '_data/' is "
            "gitignored and machine-local: place the source ZIP(s) in _data/ and run "
            "'docker compose run --rm prepare' (or 'python prepare_data.py') on this "
            "machine before generating pairs."
        )

    if args.catalogue_only:
        log.info("--catalogue_only set -- stopping before transcription")
        return

    if policy_checkpoint is not None:
        retired = store.retire_pending_pairs_except_model(gen.model_version)
        if retired:
            log.info("Retired %d pending pairs from older policies", retired)

    # Step 2: generate hypothesis pairs
    scheduled_audio_ids = (
        _load_audio_ids(args.audio_ids_file) if args.audio_ids_file else None
    )
    n_to_generate = len(scheduled_audio_ids) if scheduled_audio_ids else (
        args.n or config.target_batch_size
    )
    log.info("Pending pairs before: %d  |  requesting %d new",
             store.count_pending_pairs(), n_to_generate)

    with gen:
        n_generated = _generate_nbest(
            gen=gen,
            store=store,
            config=config,
            n_to_generate=n_to_generate,
            n_candidates=args.n_candidates,
            rm_checkpoint=args.rm_checkpoint,
            scheduled_audio_ids=scheduled_audio_ids,
        )

    log.info("Pending pairs after: %d  (+%d generated)",
             store.count_pending_pairs(), n_generated)
    log.info("Done.")


def _generate_nbest(
    gen: HypothesisGenerator,
    store: FeedbackStore,
    config: "InferenceConfig",
    n_to_generate: int,
    n_candidates: int,
    rm_checkpoint: "Optional[Path]",
    scheduled_audio_ids: "Optional[list[str]]" = None,
) -> int:
    """
    Two-pass N-best pair generation.

    Generates one beam hypothesis and n_candidates - 1 temperature samples,
    then selects the best/worst pair by
    one of two criteria:

    - **With reward model** (rm_checkpoint provided): AudioTextRewardModel scores
      every candidate; best/worst are chosen by maximum reward gap (RSO pattern,
      see docs/DPO_PRIMER.md §11).

        - **Cold-start / no reward model**: the deterministic beam result is paired
            with the most textually dissimilar sample, without blocking on a trained
            reward model.

    Returns the number of pairs inserted.
    """
    from pathlib import Path as _Path

    paired_ids: set[str] = set()
    if scheduled_audio_ids is not None:
        audio_files = store.get_eligible_audio_files_by_ids(
            scheduled_audio_ids, model_version=gen.model_version
        )
        selected_ids = {audio.id for audio in audio_files}
        paired_ids = store.get_paired_audio_ids_by_ids(
            scheduled_audio_ids, model_version=gen.model_version
        )
        unavailable = [
            audio_id
            for audio_id in scheduled_audio_ids
            if audio_id not in selected_ids and audio_id not in paired_ids
        ]
        if unavailable:
            raise ValueError(
                "Scheduled audio is missing or ineligible for policy "
                f"{gen.model_version}: {', '.join(unavailable)}"
            )
        if paired_ids:
            log.info(
                "Scheduled generation resuming with %d/%d pairs already present",
                len(paired_ids), len(scheduled_audio_ids),
            )
    else:
        audio_files = store.get_unqueued_audio_files(
            n_to_generate, model_version=gen.model_version
        )
    if not audio_files:
        log.info("No unqueued audio files -- nothing to generate")
        return 0

    from utils.origin_sampling import composition
    import json

    selected_composition = composition([dict(vars(audio), audio_id=audio.id) for audio in audio_files])
    if hasattr(store, "set_setting"):
        store.set_setting("queue_refill_composition", json.dumps(selected_composition))

    log.info(
        "N-best mode: %d audio files x %d candidates  rm_checkpoint=%s",
        len(audio_files), n_candidates, rm_checkpoint or "none (beam-rank fallback)",
    )

    # Step 1: generate one beam result plus temperature samples per file
    all_candidates = gen.generate_candidates(audio_files, n_candidates)

    if rm_checkpoint is None:
        # --- Cold-start fallback: beam-rank pairing (no reward model needed) ---
        log.info(
            "No reward model checkpoint -- pairing the beam result with the "
            "most dissimilar temperature sample."
        )
        inserted = 0
        skipped_identical = 0
        for af, candidates in zip(audio_files, all_candidates):
            valid = [c.strip() for c in candidates if c and c.strip()]
            seen: set = set()
            unique: list = []
            for c in valid:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            if len(unique) < 2:
                if not skipped_identical:
                    log.warning(
                        "No distinct hypotheses -- file: %s  candidates: %r",
                        af.file_path, candidates[:3],
                    )
                skipped_identical += 1
                continue
            # candidates[0] = beam search result (deterministic best)
            # candidates[1:] = temperature samples — pick most dissimilar to beam
            import difflib as _difflib
            best_hyp = unique[0]
            rest = unique[1:]
            worst_hyp = max(
                rest,
                key=lambda s: 1.0 - _difflib.SequenceMatcher(None, best_hyp, s).ratio(),
            )
            temperature = getattr(config, "nbest_temperature", 0.8)
            pair = HypothesisPair.new(
                audio_id=af.id,
                hyp_a=best_hyp,
                hyp_b=worst_hyp,
                strategy_a="beam_5_best",
                strategy_b=f"sample_t{temperature:.2f}_diverse",
                model_version=gen.model_version,
            )
            store.insert_hypothesis_pair(pair)
            inserted += 1
        if skipped_identical:
            log.warning(
                "N-best (beam+sample): %d/%d files had no distinct hypotheses "
                "(beam == all samples). Try increasing nbest_temperature in "
                "config/inference_config.json (current: %.2f).",
                skipped_identical, len(audio_files),
                getattr(config, "nbest_temperature", 0.8),
            )
        log.info("N-best (beam-rank): inserted %d pairs from %d audio files",
                 inserted, len(audio_files))
        _require_complete_schedule(
            scheduled_audio_ids,
            inserted + len(paired_ids),
            len(scheduled_audio_ids) if scheduled_audio_ids is not None else len(audio_files),
        )
        return inserted

    # --- Reward-model scoring: RSO best/worst by maximum reward gap ---
    rm_config = RewardModelConfig()
    rm_config_path = _Path("config") / "reward_model_config.json"
    if rm_config_path.exists():
        rm_config = RewardModelConfig.from_json(rm_config_path)

    threshold = rm_config.reward_gap_threshold
    inserted = 0

    with AudioTextRewardModel(rm_config, checkpoint_path=rm_checkpoint) as rm:
        from utils.audio import load_audio

        for af, candidates in zip(audio_files, all_candidates):
            valid = [c.strip() for c in candidates if c and c.strip()]
            # Deduplicate while preserving order
            seen: set = set()
            unique: list = []
            for c in valid:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            if len(unique) < 2:
                log.debug(
                    "Fewer than 2 unique candidates for %s -- skipping", af.file_path
                )
                continue

            try:
                waveform, _ = load_audio(af.file_path)
            except Exception as exc:
                log.warning("Audio load failed for %s: %s", af.file_path, exc)
                continue

            scores = []
            for hyp in unique:
                try:
                    scores.append(rm._score_one(waveform, hyp))
                except Exception as exc:
                    log.warning("Scoring failed for candidate: %s", exc)
                    scores.append(None)

            # Drop candidates where scoring failed
            valid_pairs = [
                (h, s) for h, s in zip(unique, scores) if s is not None
            ]
            if len(valid_pairs) < 2:
                continue

            best_hyp, best_score = max(valid_pairs, key=lambda x: x[1])
            worst_hyp, worst_score = min(valid_pairs, key=lambda x: x[1])

            if best_score - worst_score < threshold:
                log.debug(
                    "Reward gap %.3f below threshold %.3f -- skipping",
                    best_score - worst_score, threshold,
                )
                continue

            pair = HypothesisPair.new(
                audio_id=af.id,
                hyp_a=best_hyp,
                hyp_b=worst_hyp,
                strategy_a=f"nbest_winner_n{n_candidates}",
                strategy_b=f"nbest_loser_n{n_candidates}",
                model_version=gen.model_version,
            )
            store.insert_hypothesis_pair(pair)
            inserted += 1
            log.debug(
                "Inserted pair for %s  gap=%.3f", af.file_path, best_score - worst_score
            )

    log.info("N-best: inserted %d pairs from %d audio files", inserted, len(audio_files))
    _require_complete_schedule(
        scheduled_audio_ids,
        inserted + len(paired_ids),
        len(scheduled_audio_ids) if scheduled_audio_ids is not None else len(audio_files),
    )
    return inserted


def _require_complete_schedule(
    scheduled_audio_ids: "Optional[list[str]]",
    inserted: int,
    expected: int,
) -> None:
    if scheduled_audio_ids is not None and inserted != expected:
        raise RuntimeError(
            f"Scheduled generation produced {inserted}/{expected} pairs"
        )


if __name__ == "__main__":
    main()
