from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from rlhf_loop_tests.surrogate_annotator import SurrogateAnnotator

from .runtime import subprocess_environment


class ArmWebServer:
    def __init__(
        self,
        port: int,
        environment: dict[str, str],
        project_root: Path,
        log_path: Path,
        startup_timeout: float = 30.0,
    ) -> None:
        self.port = port
        self.environment = environment
        self.project_root = project_root
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.base_url = f"http://127.0.0.1:{port}"
        self._process: subprocess.Popen | None = None
        self._log_handle = None

    def __enter__(self) -> str:
        _require_available_port(self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "web.app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.port),
                ],
                cwd=self.project_root,
                env=self.environment,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"Arm web service exited with code {self._process.returncode}; "
                        f"see {self.log_path}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"{self.base_url}/setup", timeout=1.0
                    ) as response:
                        if response.status == 200:
                            return self.base_url
                except OSError:
                    time.sleep(0.1)
            raise TimeoutError(f"Arm web service did not start on port {self.port}")
        except Exception:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()


def run_annotation_round(
    manifest_path: Path,
    replicate: int,
    round_number: int,
    command_runner: Callable[..., object] = subprocess.run,
    server_factory: Callable[..., object] = ArmWebServer,
    annotator_factory: Callable[..., SurrogateAnnotator] = SurrogateAnnotator,
) -> dict:
    manifest = _read_json(manifest_path)
    if not manifest.get("materialized_arms"):
        raise ValueError("Experiment arms must be materialized before annotation")
    existing = manifest.get("round_runs", [])
    if any(
        run["replicate"] == replicate and run["round"] == round_number
        for run in existing
    ):
        raise ValueError("The requested annotation round is already complete")

    replicate_plan = next(
        (item for item in manifest["replicates"] if item["replicate"] == replicate),
        None,
    )
    if replicate_plan is None:
        raise ValueError(f"Unknown replicate: {replicate}")
    round_files = replicate_plan["round_audio_id_files"]
    if not 1 <= round_number <= len(round_files):
        raise ValueError(f"Unknown round {round_number} for replicate {replicate}")
    schedule_path = Path(round_files[round_number - 1])
    scheduled_ids = _read_json(schedule_path)
    config = manifest["config"]
    project_root = Path(manifest["project_root"])
    round_seed = int(replicate_plan["seed"]) + round_number - 1
    arms = [arm for arm in manifest["arms"] if arm["replicate"] == replicate]
    expected_arms = {"sft", "rlhf", "rlhf_noisy"}
    if {arm["arm"] for arm in arms} != expected_arms:
        raise ValueError(f"Replicate {replicate} does not contain all experiment arms")

    arm_results = []
    for arm in arms:
        records = _run_arm_annotations(
            arm=arm,
            config=config,
            project_root=project_root,
            schedule_path=schedule_path,
            scheduled_ids=scheduled_ids,
            round_number=round_number,
            round_seed=round_seed,
            command_runner=command_runner,
            server_factory=server_factory,
            annotator_factory=annotator_factory,
        )
        arm_results.append(
            {
                "arm": arm["arm"],
                "annotations_file": str(
                    Path(arm["root"])
                    / f"round-{round_number:02d}"
                    / "annotations.jsonl"
                ),
                "annotation_count": len(records),
                "model_versions": sorted({record["model_version"] for record in records}),
            }
        )

    records_by_arm = {
        result["arm"]: _read_jsonl(Path(result["annotations_file"]))
        for result in arm_results
    }
    parity = _verify_parity(records_by_arm, scheduled_ids, config)
    parity_path = manifest_path.parent / f"replicate-{replicate:02d}" / (
        f"round-{round_number:02d}-parity.json"
    )
    _write_json_atomic(parity_path, parity)
    round_run = {
        "replicate": replicate,
        "round": round_number,
        "seed": round_seed,
        "schedule_file": str(schedule_path),
        "parity_file": str(parity_path),
        "arms": arm_results,
    }
    manifest.setdefault("round_runs", []).append(round_run)
    manifest["status"] = "annotation-round-complete"
    _write_json_atomic(manifest_path, manifest)
    return round_run


def _run_arm_annotations(
    arm: dict,
    config: dict,
    project_root: Path,
    schedule_path: Path,
    scheduled_ids: list[str],
    round_number: int,
    round_seed: int,
    command_runner: Callable[..., object],
    server_factory: Callable[..., object],
    annotator_factory: Callable[..., SurrogateAnnotator],
) -> list[dict]:
    round_root = Path(arm["root"]) / f"round-{round_number:02d}"
    state_path = round_root / "state.json"
    output_path = round_root / "annotations.jsonl"
    state = _read_json(state_path) if state_path.exists() else {"status": "pending"}
    expected_state = {
        "replicate": arm["replicate"],
        "arm": arm["arm"],
        "round": round_number,
        "seed": round_seed,
        "schedule_file": str(schedule_path),
    }
    for key, value in expected_state.items():
        if key in state and state[key] != value:
            raise ValueError(f"Round state mismatch for {arm['arm']}: {key}")
    state.update(expected_state)

    environment = subprocess_environment(arm["paths_file"], config["gpu"])
    environment["INLINE_COORDINATOR"] = "0"
    if state["status"] == "pending":
        command = [
            sys.executable,
            str(project_root / "run_generate_pairs.py"),
            "--audio_ids_file",
            str(schedule_path),
            "--seed",
            str(round_seed),
        ]
        initial_policy = config.get("initial_policy", "base")
        if round_number == 1 and initial_policy != "base":
            command.extend(("--policy_checkpoint", initial_policy))
        arm_paths = _read_json(Path(arm["paths_file"]))
        reward_root = Path(arm_paths["reward_model_dir"])
        reward_checkpoints = (
            list(reward_root.glob("reward_model_v*"))
            if reward_root.exists()
            else []
        )
        if reward_checkpoints:
            latest_reward = max(
                reward_checkpoints,
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
            command.extend(("--rm_checkpoint", str(latest_reward)))
        for temperature in (None, 0.9, 1.0, 1.1):
            attempt = list(command)
            if temperature is not None:
                attempt.extend(("--nbest_temperature", str(temperature)))
            try:
                command_runner(
                    attempt,
                    cwd=project_root,
                    env=environment,
                    check=True,
                )
                break
            except subprocess.CalledProcessError:
                if temperature == 1.1:
                    raise
        state["status"] = "generated"
        _write_json_atomic(state_path, state)

    records = _read_jsonl(output_path)
    if len(records) > len(scheduled_ids):
        raise ValueError(f"Too many annotations recorded for {arm['arm']}")
    if len(records) < len(scheduled_ids):
        with server_factory(
            arm["port"],
            environment,
            project_root,
            round_root / "web.log",
        ) as base_url:
            annotator = annotator_factory(
                base_url,
                f"surrogate-gt-r{arm['replicate']:02d}-{arm['arm']}",
                tie_threshold=config["tie_wer_threshold"],
                missing_gt=config["missing_gt"],
                submit_gt_correction=config["submit_gt_correction"],
                error_rate=float(arm.get("feedback_error_rate", 0.0)),
                seed=round_seed,
                rng_offset=sum(
                    record.get("oracle_preference", record.get("preference"))
                    in {"a", "b"}
                    for record in records
                ),
            )
            annotator.annotate(len(scheduled_ids) - len(records), output_path)
        records = _read_jsonl(output_path)
    _verify_arm_records(arm["arm"], records, scheduled_ids, config)
    state["status"] = "annotated"
    state["annotation_count"] = len(records)
    _write_json_atomic(state_path, state)
    return records


def _verify_arm_records(
    arm_name: str,
    records: list[dict],
    scheduled_ids: list[str],
    config: dict,
) -> None:
    actual_ids = [record.get("audio_id") for record in records]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(scheduled_ids):
        raise RuntimeError(
            f"{arm_name} annotations differ from the round schedule"
        )
    expected_corrections = [
        record.get("reference", "") if config["submit_gt_correction"] else ""
        for record in records
    ]
    submitted = [record.get("submitted_correction_text") for record in records]
    if submitted != expected_corrections:
        raise RuntimeError(f"{arm_name} did not submit the configured GT corrections")
    if not all(record.get("response_ok") for record in records):
        raise RuntimeError(f"{arm_name} contains a failed annotation submission")


def _verify_parity(
    records_by_arm: dict[str, list[dict]],
    scheduled_ids: list[str],
    config: dict,
) -> dict:
    for arm_name, records in records_by_arm.items():
        _verify_arm_records(arm_name, records, scheduled_ids, config)
    ordered = {
        arm_name: [
            {record["audio_id"]: record for record in records}[audio_id]
            for audio_id in scheduled_ids
        ]
        for arm_name, records in records_by_arm.items()
    }
    sft = ordered["sft"]
    gt_sft = [
        (record["audio_id"], record["reference"], record["submitted_correction_text"])
        for record in sft
    ]
    for arm_name, records in ordered.items():
        arm_gt = [
            (record["audio_id"], record["reference"], record["submitted_correction_text"])
            for record in records
        ]
        if gt_sft != arm_gt:
            raise RuntimeError(f"SFT and {arm_name} arms received different GT supervision")
    model_versions = {
        tuple(record["model_version"] for record in records)
        for records in ordered.values()
    }
    same_policy = len(model_versions) == 1
    pair_match = None
    if same_policy:
        signatures = {
            tuple(
                (
                    record["hyp_a"],
                    record["hyp_b"],
                    record.get("oracle_preference", record["preference"]),
                )
                for record in records
            )
            for records in ordered.values()
        }
        pair_match = len(signatures) == 1
        if not pair_match:
            raise RuntimeError("Same-policy arms produced different seeded pairs")
    return {
        "schema_version": 1,
        "audio_ids": scheduled_ids,
        "gt_supervision_match": True,
        "same_policy": same_policy,
        "seeded_pair_match": pair_match,
        "preference_flip_counts": {
            arm_name: sum(bool(record.get("preference_flipped")) for record in records)
            for arm_name, records in ordered.items()
        },
    }


def _require_available_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Experiment web port {port} is already in use") from exc


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
