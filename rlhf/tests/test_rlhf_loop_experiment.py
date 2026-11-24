from __future__ import annotations

import json
from pathlib import Path

import pytest

from paths import PathsConfig
from rlhf_loop_tests.experiment.config import ExperimentConfig
from rlhf_loop_tests.experiment.evaluation import compare_holdout_predictions
from rlhf_loop_tests.experiment.initialization import initialize_experiment
from rlhf_loop_tests.experiment.materialization import materialize_arm_states
from rlhf_loop_tests.experiment.orchestrator import run_experiment_to_completion
from rlhf_loop_tests.experiment.runtime import subprocess_environment
from rlhf_loop_tests.experiment.planning import plan_partitions
from rlhf_loop_tests.experiment.rounds import run_annotation_round
from rlhf_loop_tests.experiment.rounds import _verify_parity
from rlhf_loop_tests.experiment.reporting import build_experiment_report
from rlhf_loop_tests.experiment.schedule import build_partition_schedule
from rlhf_loop_tests.experiment.training import train_experiment_round
from rlhf_loop_tests.experiment.workspace import ExperimentWorkspace
from rlhf_loop_tests.surrogate_annotator import (
    SurrogateAnnotator,
    decide_from_ground_truth,
)
from storage.feedback_store import FeedbackStore
from tests.conftest import make_audio


def _config(tmp_path: Path, **overrides) -> ExperimentConfig:
    values = {
        "experiment_id": "unit",
        "seeds": [42, 137],
        "output_root": str(tmp_path / "experiments"),
    }
    values.update(overrides)
    config = ExperimentConfig(**values)
    config.validate()
    return config


def test_experiment_config_rejects_unsafe_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filesystem-safe"):
        _config(tmp_path, experiment_id="../escape")
    with pytest.raises(ValueError, match="gpu"):
        _config(tmp_path, gpu="cuda:0")
    with pytest.raises(ValueError, match="unique"):
        _config(tmp_path, seeds=[42, 42])
    with pytest.raises(ValueError, match="error_rate"):
        _config(tmp_path, noisy_feedback_error_rate=1.1)


def test_orchestrator_resumes_only_missing_stages(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "arms-materialized",
                "config": {"seeds": [42], "rounds": 2},
                "materialized_arms": [{"arm": "sft"}],
                "round_runs": [{"replicate": 1, "round": 1}],
                "training_runs": [{"replicate": 1, "round": 1}],
                "evaluation_runs": [],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def stage(name):
        def run(path, replicate, round_number):
            calls.append((name, replicate, round_number))
            manifest = json.loads(path.read_text(encoding="utf-8"))
            collection = {
                "annotate": "round_runs",
                "train": "training_runs",
                "evaluate": "evaluation_runs",
            }[name]
            manifest.setdefault(collection, []).append(
                {"replicate": replicate, "round": round_number}
            )
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return manifest

        return run

    result = run_experiment_to_completion(
        manifest_path,
        annotate=stage("annotate"),
        train=stage("train"),
        evaluate=stage("evaluate"),
        report=lambda path: {"complete": True},
    )

    assert result == {"complete": True}
    assert calls == [
        ("evaluate", 1, 1),
        ("annotate", 1, 2),
        ("train", 1, 2),
        ("evaluate", 1, 2),
    ]


def test_workspace_creates_isolated_arm_paths_and_manifest(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    production = project_root / "production"
    source_paths = PathsConfig(
        db_path=str(production / "feedback.sqlite"),
        checkpoints_dir=str(production / "checkpoints"),
        reward_model_dir=str(production / "reward_model"),
        latest_model_dir=str(production / "latest"),
        logs_dir=str(production / "logs"),
        output_dir=str(production),
    )
    config = _config(tmp_path)

    workspace = ExperimentWorkspace(config, source_paths, project_root)
    manifest_path = workspace.prepare()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["arms"]) == 6
    assert Path(manifest["initial"]["paths_file"]).exists()
    assert [arm["port"] for arm in manifest["arms"]] == [
        8180, 8181, 8182, 8183, 8184, 8185
    ]
    assert [arm["feedback_error_rate"] for arm in manifest["arms"][:3]] == [
        0.0, 0.0, 0.05
    ]
    for arm in manifest["arms"]:
        paths = json.loads(Path(arm["paths_file"]).read_text(encoding="utf-8"))
        for key in (
            "output_dir",
            "db_path",
            "checkpoints_dir",
            "latest_model_dir",
            "reward_model_dir",
            "logs_dir",
        ):
            assert workspace.root in Path(paths[key]).parents
            assert production not in Path(paths[key]).parents

    with pytest.raises(FileExistsError):
        workspace.prepare()
    assert workspace.prepare(resume=True) == manifest_path


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    source_paths = PathsConfig()
    first = ExperimentWorkspace(_config(tmp_path), source_paths, tmp_path)
    first.prepare()
    changed = ExperimentWorkspace(
        _config(tmp_path, tie_wer_threshold=0.1), source_paths, tmp_path
    )

    with pytest.raises(ValueError, match="different experiment configuration"):
        changed.prepare(resume=True)


def test_partition_schedule_is_deterministic_disjoint_and_round_aligned(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        rounds=2,
        annotations_per_round=3,
        interaction_samples=8,
        monitor_samples=2,
        holdout_samples=2,
    )
    audio_ids = [f"audio-{index}" for index in range(20)]
    audio_groups = {audio_id: f"session-{index}" for index, audio_id in enumerate(audio_ids)}

    first = build_partition_schedule(audio_ids, config, seed=42, audio_groups=audio_groups)
    second = build_partition_schedule(audio_ids, config, seed=42, audio_groups=audio_groups)

    assert first == second
    interaction = set(first["interaction_audio_ids"])
    monitor = set(first["promotion_monitor_audio_ids"])
    holdout = set(first["final_holdout_audio_ids"])
    assert interaction.isdisjoint(monitor | holdout)
    assert monitor.isdisjoint(holdout)
    assert [len(round_ids) for round_ids in first["rounds"]] == [3, 3]
    assert first["rounds"][0] == first["interaction_audio_ids"][:3]
    groups = first["partition_groups"]
    assert set(groups["interaction"]).isdisjoint(
        groups["promotion_monitor"] + groups["final_holdout"]
    )


def test_partition_schedule_keeps_recording_sessions_disjoint(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        seeds=[42],
        rounds=1,
        annotations_per_round=2,
        interaction_samples=2,
        monitor_samples=2,
        holdout_samples=2,
    )
    audio_ids = [f"audio-{index}" for index in range(9)]
    audio_groups = {
        audio_id: f"session-{index // 3}" for index, audio_id in enumerate(audio_ids)
    }

    schedule = build_partition_schedule(
        audio_ids, config, seed=42, audio_groups=audio_groups
    )

    assigned = schedule["partition_groups"]
    assert set(assigned["interaction"]).isdisjoint(assigned["promotion_monitor"])
    assert set(assigned["interaction"]).isdisjoint(assigned["final_holdout"])
    assert set(assigned["promotion_monitor"]).isdisjoint(assigned["final_holdout"])


def test_partition_schedule_rejects_insufficient_ground_truth(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="GT audio files"):
        build_partition_schedule(["only-one"], config, seed=42)


def test_partition_planner_emits_shared_round_files(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    config = _config(
        tmp_path,
        seeds=[42],
        rounds=2,
        annotations_per_round=2,
        interaction_samples=4,
        monitor_samples=2,
        holdout_samples=2,
    )
    workspace = ExperimentWorkspace(config, PathsConfig(), project_root)
    manifest_path = workspace.prepare()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_paths = json.loads(
        Path(manifest["initial"]["paths_file"]).read_text(encoding="utf-8")
    )
    store = FeedbackStore(initial_paths["db_path"])
    for index in range(10):
        make_audio(store, path=f"session-{index}/audio.wav")

    planned = plan_partitions(manifest_path)

    replicate = planned["replicates"][0]
    partitions = json.loads(
        Path(replicate["partitions_file"]).read_text(encoding="utf-8")
    )
    round_files = [Path(path) for path in replicate["round_audio_id_files"]]
    assert planned["status"] == "partitions-planned"
    assert [json.loads(path.read_text(encoding="utf-8")) for path in round_files] == (
        partitions["rounds"]
    )


def test_materialization_clones_identical_isolated_arm_states(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    config = _config(
        tmp_path,
        seeds=[42],
        rounds=1,
        annotations_per_round=2,
        interaction_samples=3,
        monitor_samples=2,
        holdout_samples=2,
    )
    manifest_path = ExperimentWorkspace(
        config, PathsConfig(), project_root
    ).prepare()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_paths = json.loads(
        Path(manifest["initial"]["paths_file"]).read_text(encoding="utf-8")
    )
    initial_store = FeedbackStore(initial_paths["db_path"])
    for index in range(8):
        make_audio(initial_store, path=f"session-{index}/audio.wav")
    plan_partitions(manifest_path)

    materialized = materialize_arm_states(manifest_path)

    assert materialized["status"] == "arms-materialized"
    assert len(materialized["materialized_arms"]) == 3
    fingerprints = {
        (
            arm["promotion_monitor"]["fingerprint"],
            arm["final_holdout"]["fingerprint"],
        )
        for arm in materialized["materialized_arms"]
    }
    assert len(fingerprints) == 1
    for arm in materialized["materialized_arms"]:
        store = FeedbackStore(arm["db_path"])
        assert store.count_audio_files() == 8
        assert store.get_active_policy_model_version() == "openai/whisper-large-v3"
        assert store.get_promotion_gate_canary_set()["id"] == (
            arm["promotion_monitor"]["id"]
        )
        assert len(store.get_ground_truth_audio_files()) == 4


def test_initialization_populates_only_the_isolated_initial_catalogue(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "run_generate_pairs.py").write_text("", encoding="utf-8")
    config = _config(
        tmp_path,
        seeds=[42],
        rounds=1,
        annotations_per_round=1,
        interaction_samples=2,
        monitor_samples=1,
        holdout_samples=1,
    )
    manifest_path = ExperimentWorkspace(
        config, PathsConfig(), project_root
    ).prepare()
    invocation = {}

    def fake_runner(command, **kwargs):
        invocation.update(command=command, **kwargs)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        initial_paths = json.loads(
            Path(manifest["initial"]["paths_file"]).read_text(encoding="utf-8")
        )
        store = FeedbackStore(initial_paths["db_path"])
        for index in range(5):
            make_audio(store, path=f"session-{index}/audio.wav")

    initialized = initialize_experiment(manifest_path, runner=fake_runner)

    assert initialized["status"] == "arms-materialized"
    assert invocation["command"][-1] == "--catalogue_only"
    assert invocation["env"]["RLHF_PATHS_FILE"] == (
        initialized["initial"]["paths_file"]
    )
    assert invocation["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert invocation["check"] is True


def test_annotation_round_uses_matched_schedule_seed_and_gt(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "run_generate_pairs.py").write_text("", encoding="utf-8")
    config = _config(
        tmp_path,
        seeds=[42],
        rounds=1,
        annotations_per_round=2,
        interaction_samples=2,
        monitor_samples=1,
        holdout_samples=1,
    )
    manifest_path = ExperimentWorkspace(
        config, PathsConfig(), project_root
    ).prepare()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_paths = json.loads(
        Path(manifest["initial"]["paths_file"]).read_text(encoding="utf-8")
    )
    initial_store = FeedbackStore(initial_paths["db_path"])
    for index in range(5):
        make_audio(initial_store, path=f"session-{index}/audio.wav")
    plan_partitions(manifest_path)
    materialize_arm_states(manifest_path)
    planned = json.loads(manifest_path.read_text(encoding="utf-8"))
    schedule_path = Path(planned["replicates"][0]["round_audio_id_files"][0])
    scheduled_ids = json.loads(schedule_path.read_text(encoding="utf-8"))
    invocations = []

    def fake_runner(command, **kwargs):
        invocations.append((command, kwargs))

    class FakeServer:
        def __init__(self, port, environment, project_root, log_path):
            self.port = port

        def __enter__(self):
            return f"http://arm-{self.port}"

        def __exit__(self, *args):
            return None

    class FakeAnnotator:
        def __init__(self, base_url, participant_id, **kwargs):
            self.participant_id = participant_id
            self.error_rate = kwargs["error_rate"]

        def annotate(self, budget, output_path):
            arm = self.participant_id.rsplit("-", 1)[-1]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as handle:
                for audio_id in scheduled_ids[-budget:]:
                    handle.write(
                        json.dumps(
                            {
                                "audio_id": audio_id,
                                "model_version": "openai/whisper-medium",
                                "hyp_a": f"beam-{audio_id}",
                                "hyp_b": f"sample-{audio_id}",
                                "oracle_preference": "a",
                                "preference": "b" if self.error_rate else "a",
                                "preference_flipped": bool(self.error_rate),
                                "reference": f"gt-{audio_id}",
                                "correction_text": f"gt-{audio_id}",
                                "submitted_correction_text": f"gt-{audio_id}",
                                "response_ok": True,
                                "arm": arm,
                            }
                        )
                        + "\n"
                    )

    result = run_annotation_round(
        manifest_path,
        replicate=1,
        round_number=1,
        command_runner=fake_runner,
        server_factory=FakeServer,
        annotator_factory=FakeAnnotator,
    )

    assert len(invocations) == 3
    assert all("--seed" in command for command, _ in invocations)
    assert all(command[command.index("--seed") + 1] == "42" for command, _ in invocations)
    assert all(
        command[command.index("--audio_ids_file") + 1] == str(schedule_path)
        for command, _ in invocations
    )
    assert len({kwargs["env"]["RLHF_PATHS_FILE"] for _, kwargs in invocations}) == 3
    parity = json.loads(Path(result["parity_file"]).read_text(encoding="utf-8"))
    assert parity["gt_supervision_match"] is True
    assert parity["seeded_pair_match"] is True
    assert parity["preference_flip_counts"]["rlhf_noisy"] == 2
    assert [arm["annotation_count"] for arm in result["arms"]] == [2, 2, 2]


def test_annotation_parity_accepts_retry_append_order() -> None:
    scheduled_ids = ["audio-1", "audio-2"]

    def record(audio_id: str) -> dict:
        return {
            "audio_id": audio_id,
            "model_version": "base:test",
            "hyp_a": f"beam-{audio_id}",
            "hyp_b": f"sample-{audio_id}",
            "preference": "a",
            "reference": f"gt-{audio_id}",
            "submitted_correction_text": f"gt-{audio_id}",
            "response_ok": True,
        }

    parity = _verify_parity(
        {
            "sft": [record("audio-2"), record("audio-1")],
            "rlhf": [record("audio-1"), record("audio-2")],
            "rlhf_noisy": [record("audio-2"), record("audio-1")],
        },
        scheduled_ids,
        {"submit_gt_correction": True},
    )

    assert parity["gt_supervision_match"] is True
    assert parity["seeded_pair_match"] is True


def test_training_round_runs_plain_sft_and_canonical_rlhf_in_isolated_arms(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    config = _config(
        tmp_path,
        seeds=[42],
        rounds=1,
        annotations_per_round=2,
        interaction_samples=2,
        monitor_samples=1,
        holdout_samples=1,
    )
    manifest_path = ExperimentWorkspace(
        config, PathsConfig(), project_root
    ).prepare()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_paths = json.loads(
        Path(manifest["initial"]["paths_file"]).read_text(encoding="utf-8")
    )
    initial_store = FeedbackStore(initial_paths["db_path"])
    for index in range(5):
        make_audio(initial_store, path=f"session-{index}/audio.wav")
    plan_partitions(manifest_path)
    materialize_arm_states(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["round_runs"] = [
        {
            "replicate": 1,
            "round": 1,
            "parity_file": str(tmp_path / "parity.json"),
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    invocations = []

    def fake_runner(command, **kwargs):
        invocations.append((command, kwargs))
        paths = json.loads(
            Path(kwargs["env"]["RLHF_PATHS_FILE"]).read_text(encoding="utf-8")
        )
        store = FeedbackStore(paths["db_path"])
        if Path(command[1]).name == "run_sft.py":
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir(parents=True)
            (output_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": paths["whisper_model_id"]}),
                encoding="utf-8",
            )
            store.set_setting(
                "training_sft_diagnostics", '{"optimizer_updates": 3}'
            )
        else:
            store.set_setting("training_status", "guarded")
            store.set_setting("promotion_gate_status", "rejected")

    result = train_experiment_round(
        manifest_path,
        replicate=1,
        round_number=1,
        command_runner=fake_runner,
    )

    assert [arm["outcome"] for arm in result["arms"]] == [
        "activated", "guarded", "guarded"
    ]
    assert Path(result["arms"][0]["active_checkpoint"]).is_dir()
    assert result["arms"][0]["diagnostics"] == {"optimizer_updates": 3}
    assert len(invocations) == 3
    assert Path(invocations[0][0][1]).name == "run_sft.py"
    assert invocations[1][0][-1] == "--once"
    assert invocations[2][0][-1] == "--once"
    assert len({call[1]["env"]["RLHF_PATHS_FILE"] for call in invocations}) == 3


def test_holdout_comparison_uses_paired_corpus_wer() -> None:
    sft = [
        {"audio_id": "a", "transcript": "jeden dwa trzy", "hypothesis": "jeden"},
        {"audio_id": "b", "transcript": "cztery", "hypothesis": "cztery"},
    ]
    rlhf = [
        {
            "audio_id": "a",
            "transcript": "jeden dwa trzy",
            "hypothesis": "jeden dwa trzy",
        },
        {"audio_id": "b", "transcript": "cztery", "hypothesis": "zle"},
    ]

    comparison = compare_holdout_predictions(sft, rlhf, seed=42)

    assert comparison["metric"] == "corpus_wer"
    assert comparison["delta"] == pytest.approx(-0.25)
    assert comparison["audio_ids"] == ["a", "b"]


def test_experiment_report_aggregates_rounds_and_seed_pairs(tmp_path: Path) -> None:
    evaluations = []
    for replicate, offset in ((1, 0.0), (2, 0.02)):
        for round_number in (1, 2):
            report_path = tmp_path / f"r{replicate}-round{round_number}.json"
            report_path.write_text(
                json.dumps(
                    {
                        "arms": {
                            "sft": {
                                "initial_metrics": {"wer": 0.5, "cer": 0.3},
                                "active_metrics": {
                                    "wer": 0.5 - 0.05 * round_number + offset,
                                    "cer": 0.3 - 0.02 * round_number,
                                },
                            },
                            "rlhf": {
                                "initial_metrics": {"wer": 0.5, "cer": 0.3},
                                "active_metrics": {
                                    "wer": 0.5 - 0.08 * round_number + offset,
                                    "cer": 0.3 - 0.03 * round_number,
                                },
                            },
                            "rlhf_noisy": {
                                "initial_metrics": {"wer": 0.5, "cer": 0.3},
                                "active_metrics": {
                                    "wer": 0.5 - 0.06 * round_number + offset,
                                    "cer": 0.3 - 0.025 * round_number,
                                },
                            },
                        },
                        "rlhf_minus_sft": {"delta": -0.03 * round_number},
                        "rlhf_noisy_minus_sft": {"delta": -0.01 * round_number},
                        "rlhf_noisy_minus_rlhf": {"delta": 0.02 * round_number},
                    }
                ),
                encoding="utf-8",
            )
            evaluations.append(
                {
                    "replicate": replicate,
                    "round": round_number,
                    "report_file": str(report_path),
                }
            )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {
                    "experiment_id": "report-test",
                    "seeds": [42, 137],
                    "rounds": 2,
                    "annotations_per_round": 10,
                },
                "evaluation_runs": evaluations,
            }
        ),
        encoding="utf-8",
    )

    report = build_experiment_report(manifest_path)

    assert report["replicate_count"] == 2
    assert report["reportable_seed_count"] is False
    assert report["across_seed_final_rlhf_minus_sft_wer"]["delta"] == pytest.approx(-0.06)
    assert report["replicates"][0]["arms"]["rlhf"][
        "normalized_wer_annotation_auc"
    ] == pytest.approx(0.42)
    assert report["replicates"][0]["arms"]["rlhf_noisy"][
        "normalized_wer_annotation_auc"
    ] == pytest.approx(0.44)
    assert report["across_seed_final_rlhf_noisy_minus_sft_wer"][
        "delta"
    ] == pytest.approx(-0.02)
    assert report["across_seed_final_rlhf_noisy_minus_rlhf_wer"][
        "delta"
    ] == pytest.approx(0.04)
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "convergence.csv").exists()
    svg = (tmp_path / "convergence.svg").read_text(encoding="utf-8")
    assert "Production RLHF" in svg
    assert "RLHF with 5% feedback errors" in svg


def test_gpu_selection_is_subprocess_local() -> None:
    gpu_env = subprocess_environment("arm.json", "2,3")
    cpu_env = subprocess_environment("arm.json", "cpu")

    assert gpu_env["RLHF_PATHS_FILE"] == "arm.json"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize(
    ("hypothesis_a", "hypothesis_b", "expected"),
    [
        ("ala ma kota", "ala kota", "a"),
        ("ala psa", "ala ma kota", "b"),
        ("Ala, ma kota!", "ala ma kota", "equal"),
    ],
)
def test_ground_truth_oracle_selects_lower_normalized_wer(
    hypothesis_a: str, hypothesis_b: str, expected: str
) -> None:
    decision = decide_from_ground_truth(
        "Ala ma kota", hypothesis_a, hypothesis_b, tie_threshold=0.05
    )

    assert decision.preference == expected
    assert decision.correction_text == "Ala ma kota"


def test_ground_truth_oracle_honors_tie_boundary() -> None:
    decision = decide_from_ground_truth(
        "jeden dwa trzy cztery",
        "jeden dwa trzy kot",
        "jeden dwa pies cztery",
        tie_threshold=0.0,
    )

    assert decision.preference == "equal"
    assert decision.wer_a == decision.wer_b


def test_ground_truth_oracle_rejects_missing_reference() -> None:
    with pytest.raises(ValueError, match="Ground-truth"):
        decide_from_ground_truth("  ", "a", "b")


def test_surrogate_submits_exact_oracle_decision(tmp_path: Path, monkeypatch) -> None:
    annotator = SurrogateAnnotator(
        "http://local",
        "surrogate",
        tie_threshold=1.0,
    )
    requests = []
    replies = iter(
        [
            b"setup",
            {"ok": True},
            {
                "done": False,
                "pair_id": "pair-1",
                "audio_id": "audio-1",
                "model_version": "base:test",
                "hyp_a": "dobry tekst",
                "hyp_b": "zly tekst",
                "reference": "dobry tekst",
            },
            {"ok": True},
        ]
    )

    def fake_request(path, form=None, expect_json=True):
        requests.append((path, form, expect_json))
        return next(replies)

    monkeypatch.setattr(annotator, "_request", fake_request)
    output = tmp_path / "annotations.jsonl"
    records = annotator.annotate(1, output)

    submit = next(form for path, form, _ in requests if path == "/api/submit")
    assert submit == {
        "pair_id": "pair-1",
        "preference": "equal",
        "correction_text": "dobry tekst",
        "strength": "10",
        "derive_preference": "false",
    }
    assert records[0]["model_version"] == "base:test"
    assert json.loads(output.read_text(encoding="utf-8"))["preference"] == "equal"


def test_surrogate_records_and_flips_noisy_actionable_decision(
    tmp_path: Path, monkeypatch
) -> None:
    annotator = SurrogateAnnotator(
        "http://local",
        "surrogate-noisy",
        tie_threshold=0.0,
        error_rate=1.0,
        seed=7,
    )
    replies = iter(
        [
            b"setup",
            {"ok": True},
            {
                "done": False,
                "pair_id": "pair-1",
                "audio_id": "audio-1",
                "model_version": "base:test",
                "hyp_a": "dobry tekst",
                "hyp_b": "zly tekst",
                "reference": "dobry tekst",
            },
            {"ok": True},
        ]
    )
    requests = []

    def fake_request(path, form=None, expect_json=True):
        requests.append((path, form, expect_json))
        return next(replies)

    monkeypatch.setattr(annotator, "_request", fake_request)
    records = annotator.annotate(1, tmp_path / "annotations.jsonl")

    submit = next(form for path, form, _ in requests if path == "/api/submit")
    assert submit["preference"] == "b"
    assert records[0]["oracle_preference"] == "a"
    assert records[0]["preference"] == "b"
    assert records[0]["preference_flipped"] is True


def test_surrogate_noise_rng_can_resume_after_recorded_decisions() -> None:
    uninterrupted = SurrogateAnnotator(
        "http://local", "all", error_rate=0.5, seed=17
    )
    first_two = [uninterrupted.rng.random() for _ in range(2)]
    expected_next = uninterrupted.rng.random()
    resumed = SurrogateAnnotator(
        "http://local", "resumed", error_rate=0.5, seed=17, rng_offset=2
    )

    assert first_two != [expected_next, expected_next]
    assert resumed.rng.random() == expected_next