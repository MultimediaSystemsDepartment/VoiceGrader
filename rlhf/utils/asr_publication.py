from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from evaluation.canary import evaluation_profile_hash
from evaluation.model_health import paired_bootstrap_corpus_wer
from utils.audio import load_audio
from utils.policy_checkpoint import validate_policy_checkpoint

REQUIRED_FILES = {"model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"}


def publication_root(paths) -> Path:
    return Path(paths.output_dir) / "asr_publications"


def checked_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or value in {"publication.json", "current.json"}
            or "\\" in value or ":" in value or path.is_absolute()
            or ".." in path.parts or path.as_posix() != value):
        raise ValueError("Unsafe publication file path")
    return value


def _adapter_digest(source: Path) -> str:
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        weights = source / filename
        if weights.is_symlink():
            raise ValueError("Adapter weights must not be a symlink")
        if weights.is_file():
            digest = hashlib.sha256()
            with weights.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
    raise ValueError("Adapter checkpoint has no weights")


def validate_manifest(manifest: dict) -> None:
    if (manifest.get("schema_version") != 1
            or manifest.get("format") != "ctranslate2-whisper"
            or manifest.get("quantization") != "float16"):
        raise ValueError("Unsupported ASR publication format")
    if str(UUID(manifest["publication_id"])) != manifest["publication_id"]:
        raise ValueError("Invalid publication identity")
    files = manifest["files"]
    if not isinstance(files, dict) or not REQUIRED_FILES <= files.keys():
        raise ValueError("Incomplete ASR publication")
    for name, info in files.items():
        checked_relative_path(name)
        if (not isinstance(info, dict) or not isinstance(info.get("size"), int)
                or info["size"] <= 0 or len(info.get("sha256", "")) != 64
                or any(c not in "0123456789abcdef" for c in info["sha256"])):
            raise ValueError("Invalid publication file metadata")


def read_current_publication(paths) -> dict | None:
    pointer = publication_root(paths) / "current.json"
    if not pointer.exists():
        return None
    identity = json.loads(pointer.read_text(encoding="utf-8"))["publication_id"]
    if str(UUID(identity)) != identity:
        raise ValueError("Invalid current publication identity")
    manifest = json.loads(
        (publication_root(paths) / identity / "publication.json").read_text(encoding="utf-8")
    )
    validate_manifest(manifest)
    if manifest["publication_id"] != identity:
        raise ValueError("Publication pointer does not match its manifest")
    return manifest


def approved_policy(store, paths) -> tuple[dict, dict, dict]:
    configured = store.get_setting("active_policy_checkpoint", "").strip()
    if not configured:
        raise ValueError("No deployed adapter is available for ASR publication")
    source = Path(configured)
    if not source.is_absolute():
        from paths import PROJECT_ROOT
        source = PROJECT_ROOT / source
    source = source.resolve()
    validate_policy_checkpoint(source, paths.whisper_model_id)
    checkpoint = store.get_model_checkpoint_by_path(str(source))
    if not checkpoint or not checkpoint["available"]:
        raise ValueError("Deployed adapter is absent from the checkpoint registry")
    if not any(row["new_checkpoint_id"] == checkpoint["id"]
               and row["reason"] == "automatic"
               for row in store.get_policy_activations()):
        raise ValueError("ASR publication requires an automatically gate-approved policy")
    monitor = store.get_promotion_gate_canary_set()
    if not monitor or monitor["status"] != "ready":
        raise ValueError("ASR publication requires a frozen WER Monitor")
    matches = [
        row for row in store.get_canary_evaluations(monitor["id"])
        if row["checkpoint_id"] == checkpoint["id"]
        and row["profile_hash"] == evaluation_profile_hash(monitor)
        and row["status"] == "completed" and row["health_status"] == "healthy"
    ]
    if not matches:
        raise ValueError("Deployed adapter lacks a healthy evaluation on this WER Monitor")
    return checkpoint, monitor, matches[-1]


def convert_adapter(source: Path, base_model_id: str, merged: Path, converted: Path) -> None:
    """Conversion is worker-only; the web process never imports model runtimes."""
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from ctranslate2.converters import TransformersConverter

    validate_policy_checkpoint(source, base_model_id)
    base = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.float16, device_map="cpu",
    )
    policy = PeftModel.from_pretrained(base, str(source))
    model = policy.merge_and_unload()
    model.save_pretrained(merged)
    processor = WhisperProcessor.from_pretrained(base_model_id)
    processor.save_pretrained(merged)
    processor.feature_extractor.save_pretrained(merged)
    del model, policy, base
    TransformersConverter(
        str(merged), copy_files=["tokenizer.json", "preprocessor_config.json"],
    ).convert(str(converted), quantization="float16")


def validate_conversion(store, monitor: dict, evaluation: dict, converted: Path) -> dict:
    from faster_whisper import WhisperModel

    items = store.get_canary_items(monitor["id"])
    predictions = {
        row["canary_item_id"]: row["hypothesis"]
        for row in store.get_canary_predictions(evaluation["id"])
    }
    if not items or any(item["id"] not in predictions for item in items):
        raise ValueError("Source evaluation is missing WER Monitor predictions")
    model = WhisperModel(str(converted), device="cuda", compute_type="float16",
                         local_files_only=True)
    references, source_hypotheses, converted_hypotheses = [], [], []
    for item in items:
        waveform, _ = load_audio(item["file_path"])
        segments, _ = model.transcribe(
            waveform, language="pl", task="transcribe", beam_size=5, temperature=0,
            condition_on_previous_text=False, vad_filter=False,
        )
        references.append(item["transcript"])
        source_hypotheses.append(predictions[item["id"]])
        converted_hypotheses.append(" ".join(segment.text.strip() for segment in segments))
    delta = paired_bootstrap_corpus_wer(
        references, source_hypotheses, converted_hypotheses,
    )
    if not math.isfinite(delta["ci_high"]) or delta["ci_high"] > 0.01:
        raise ValueError("Converted model failed the paired WER preservation check")
    return {"n_clips": len(items), "device": "cuda", "compute_type": "float16", **delta}


def publish_approved_model(store, paths) -> dict:
    checkpoint, monitor, evaluation = approved_policy(store, paths)
    source = Path(checkpoint["checkpoint_path"])
    source_digest = _adapter_digest(source)
    previous = read_current_publication(paths)
    if (previous and previous["checkpoint_id"] == checkpoint["id"]
            and previous["monitor_fingerprint"] == monitor["fingerprint"]):
        if previous.get("source_adapter_sha256") != source_digest:
            raise ValueError("Checkpoint weights changed after publication")
        return previous
    root = publication_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    identity = str(uuid4())
    staging = root / f".staging-{identity}"
    try:
        staging.mkdir()
        converted = staging / "converted"
        convert_adapter(source, paths.whisper_model_id,
                        staging / "merged", converted)
        validation = validate_conversion(store, monitor, evaluation, converted)
        if approved_policy(store, paths)[:2] != (checkpoint, monitor):
            raise RuntimeError("Deployed policy or WER Monitor changed during conversion")
        if _adapter_digest(source) != source_digest:
            raise RuntimeError("Source adapter weights changed during conversion")
        files = {}
        for path in sorted(converted.rglob("*")):
            if path.is_symlink():
                raise ValueError("Symlinks are forbidden in model publications")
            if path.is_file():
                name = checked_relative_path(path.relative_to(converted).as_posix())
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                files[name] = {"size": path.stat().st_size, "sha256": digest.hexdigest()}
        from datetime import datetime, timezone
        manifest = {
            "schema_version": 1, "format": "ctranslate2-whisper",
            "publication_id": identity, "checkpoint_id": checkpoint["id"],
            "source_adapter_sha256": source_digest,
            "base_model_id": paths.whisper_model_id, "quantization": "float16",
            "monitor_id": monitor["id"], "monitor_fingerprint": monitor["fingerprint"],
            "published_at": datetime.now(timezone.utc).isoformat(),
            "conversion_validation": validation, "files": files,
        }
        validate_manifest(manifest)
        (converted / "publication.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8",
        )
        converted.rename(root / identity)
        pointer = root / f".current-{identity}"
        try:
            with pointer.open("w", encoding="utf-8") as output:
                json.dump({"publication_id": identity}, output)
                output.flush()
                os.fsync(output.fileno())
            pointer.replace(root / "current.json")
        finally:
            pointer.unlink(missing_ok=True)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
