from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

from .admedvoice import catalogue, sample_records
from .config import RunConfig
from .metrics import gpu_snapshot
from .state import StateStore, StopRequested, check_stop, utc_now
from .training import generate_preferences, train_dpo, train_sft

log = logging.getLogger(__name__)


def _monitor_gpu(state: StateStore, finished: threading.Event) -> None:
    while not finished.wait(2):
        snapshot = gpu_snapshot()
        if snapshot is not None:
            state.update(gpu=snapshot)


def run(config: RunConfig, archive: Path, run_dir: Path) -> None:
    import torch

    state = StateStore(run_dir / "state.json")
    stop_file = run_dir / "stop.requested"
    finished = threading.Event()
    monitor = threading.Thread(target=_monitor_gpu, args=(state, finished), daemon=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the container")
    device_name = torch.cuda.get_device_name(0)
    log.info("CUDA ready: %s", device_name)
    state.update(status="running", stage="catalogue", gpu_name=device_name)
    monitor.start()
    current_checkpoint: Path | None = None
    try:
        all_records = catalogue(archive)
        log.info("ADMEDVOICE catalogue contains %d usable clips", len(all_records))
        for round_number in range(1, config.rounds + 1):
            check_stop(stop_file)
            round_dir = run_dir / f"round-{round_number:05d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            records = sample_records(
                all_records,
                config.samples,
                config.seed + round_number - 1,
            )
            state.update(
                status="running",
                stage="generation",
                round=round_number,
                rounds=config.rounds,
                progress_current=0,
                progress_total=len(records),
            )
            preference_path = round_dir / "preferences.jsonl"
            preferences = generate_preferences(
                archive,
                records,
                config,
                current_checkpoint,
                stop_file,
                preference_path,
                lambda current, total: state.update(
                    progress_current=current, progress_total=total
                ),
            )
            log.info("Generated %d usable oracle preference pairs", len(preferences))

            state.update(
                stage="sft",
                progress_current=0,
                progress_total=0,
            )
            sft_checkpoint = train_sft(
                archive,
                preferences,
                config,
                current_checkpoint,
                round_dir / "sft",
                stop_file,
                lambda current, total: state.update(
                    progress_current=current, progress_total=total
                ),
            )

            state.update(
                stage="dpo",
                progress_current=0,
                progress_total=0,
            )
            current_checkpoint = train_dpo(
                archive,
                preferences,
                config,
                sft_checkpoint,
                round_dir / "dpo",
                stop_file,
                lambda current, total: state.update(
                    progress_current=current, progress_total=total
                ),
            )
            summary = {
                "round": round_number,
                "preferences": len(preferences),
                "sft_checkpoint": str(sft_checkpoint),
                "dpo_checkpoint": str(current_checkpoint),
                "completed_at": utc_now(),
            }
            (round_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            state.update(last_checkpoint=str(current_checkpoint))
        state.update(status="completed", stage="complete", completed_at=utc_now())
    except StopRequested:
        log.info("Graceful stop completed")
        state.update(status="stopped", stage="stopped", completed_at=utc_now())
    except Exception as exc:
        log.exception("Training run failed")
        state.update(status="failed", error=str(exc), completed_at=utc_now())
        raise
    finally:
        finished.set()
        monitor.join(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ADMEDVOICE GPU stress loop")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    config = RunConfig.from_json(args.config)
    log.info("Starting profile %s: %s", config.profile, json.dumps(asdict(config)))
    run(config, args.archive, args.run_dir)


if __name__ == "__main__":
    main()
