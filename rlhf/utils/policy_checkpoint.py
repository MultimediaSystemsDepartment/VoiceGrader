from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from storage.schema import ModelCheckpoint, PolicyActivation


ACTIVE_POLICY_SETTING = "active_policy_checkpoint"
ACTIVE_POLICY_MODEL_VERSION_SETTING = "active_policy_model_version"
log = logging.getLogger(__name__)


def is_lora_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").is_file()


def validate_policy_model_id(model_id: str, expected_model_id: str) -> None:
    if model_id != expected_model_id:
        raise ValueError(
            f"Policy model {model_id!r} does not match configured lineage {expected_model_id!r}; "
            "align the policy config and paths configuration"
        )


def validate_policy_checkpoint(path: Path, base_model_id: str) -> None:
    """Reject missing, malformed or incompatible adapters before model loading."""
    if not is_lora_checkpoint(path):
        raise ValueError(f"Invalid policy checkpoint: {path}")
    try:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read adapter configuration: {path}") from exc
    configured = config.get("base_model_name_or_path") if isinstance(config, dict) else None
    if configured != base_model_id:
        raise ValueError(
            f"Adapter base model {configured!r} does not match {base_model_id!r}: {path}"
        )


def find_latest_policy_checkpoint(
    checkpoints_dir: Path, base_model_id: Optional[str] = None
) -> Optional[Path]:
    if not checkpoints_dir.is_dir():
        return None

    candidates = [
        path
        for path in checkpoints_dir.iterdir()
        if is_lora_checkpoint(path)
        and (path.name == "sft" or path.name.startswith("whisper_dpo_"))
    ]
    if base_model_id is not None:
        compatible = []
        for path in candidates:
            try:
                validate_policy_checkpoint(path, base_model_id)
            except ValueError as exc:
                log.warning("Ignoring checkpoint during recovery: %s", exc)
            else:
                compatible.append(path)
        candidates = compatible
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def resolve_active_policy_checkpoint(
    store,
    checkpoints_dir: Path,
    project_root: Path,
    base_model_id: Optional[str] = None,
) -> Optional[Path]:
    configured = store.get_setting(ACTIVE_POLICY_SETTING, "").strip()
    deployed_version = store.get_setting(ACTIVE_POLICY_MODEL_VERSION_SETTING, "").strip()
    if not configured and deployed_version and "+lora:" not in deployed_version:
        if base_model_id is not None:
            validate_policy_model_id(deployed_version, base_model_id)
        return None
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = project_root / path
        if is_lora_checkpoint(path):
            if base_model_id is not None:
                validate_policy_checkpoint(path, base_model_id)
            return path

    return find_latest_policy_checkpoint(checkpoints_dir, base_model_id)


def policy_model_version(base_model_id: str, checkpoint: Optional[Path]) -> str:
    if checkpoint is None:
        return base_model_id
    return f"{base_model_id}+lora:{checkpoint.name}"


def ensure_active_policy_model_version(
    store,
    base_model_id: str,
    checkpoint: Optional[Path],
) -> str:
    """Initialize the deployed policy identity without replacing an activation."""
    if checkpoint is not None:
        validate_policy_checkpoint(checkpoint, base_model_id)
    configured = store.get_setting(ACTIVE_POLICY_MODEL_VERSION_SETTING, "").strip()
    if configured:
        if configured.split("+lora:", 1)[0] != base_model_id:
            raise ValueError(
                f"Active policy {configured!r} belongs to a different base model; "
                "use a fresh policy database instead of relabeling existing feedback"
            )
        return configured
    model_version = policy_model_version(base_model_id, checkpoint)
    store.set_setting(ACTIVE_POLICY_MODEL_VERSION_SETTING, model_version)
    return model_version


def record_active_policy_checkpoint(
    store,
    checkpoint: Path,
    project_root: Path,
    reason: str = "automatic",
    actor: str = "system",
    parent_checkpoint: Optional[Path] = None,
    model_version: Optional[str] = None,
    base_model_id: Optional[str] = None,
) -> int:
    if base_model_id is not None:
        validate_policy_checkpoint(checkpoint, base_model_id)
    registered, previous, value = register_policy_candidate(
        store,
        checkpoint,
        project_root,
        parent_checkpoint=parent_checkpoint,
    )
    if registered is None:
        store.set_setting(ACTIVE_POLICY_SETTING, value)
        return 0

    activation = PolicyActivation.new(
        new_checkpoint_id=registered["id"],
        previous_checkpoint_id=previous["id"] if previous else None,
        reason=reason,
        actor=actor,
        warning_present=registered["health_status"] == "warning",
    )
    return store.activate_model_checkpoint(
        activation,
        value,
        model_version=model_version,
        queue_log=(
            "New policy active; queue rollover committed"
            if model_version is not None
            else None
        ),
    )


def register_policy_candidate(
    store,
    checkpoint: Path,
    project_root: Path,
    parent_checkpoint: Optional[Path] = None,
) -> tuple[Optional[dict], Optional[dict], str]:
    checkpoint = checkpoint.resolve()
    try:
        value = str(checkpoint.relative_to(project_root.resolve()))
    except ValueError:
        value = str(checkpoint)
    if not all(
        hasattr(store, method)
        for method in (
            "register_model_checkpoint",
            "get_model_checkpoint_by_path",
            "activate_model_checkpoint",
        )
    ):
        return None, None, value

    previous = None
    configured = store.get_setting(ACTIVE_POLICY_SETTING, "").strip()
    if configured:
        previous_path = Path(configured)
        if not previous_path.is_absolute():
            previous_path = project_root / previous_path
        previous = store.get_model_checkpoint_by_path(str(previous_path.resolve()))

    parent = previous
    if parent_checkpoint is not None:
        parent_path = parent_checkpoint.resolve()
        parent = store.get_model_checkpoint_by_path(str(parent_path))
        if parent is None:
            parent_stage = (
                "sft"
                if parent_path.name == "sft" or parent_path.name.startswith("sft_")
                else "dpo"
            )
            parent = store.register_model_checkpoint(
                ModelCheckpoint.new(
                    model_key=f"{parent_stage}:{parent_path}",
                    stage=parent_stage,
                    checkpoint_path=str(parent_path),
                    parent_checkpoint_id=previous["id"] if previous else None,
                    provenance="pipeline",
                )
            )

    stage = "sft" if checkpoint.name == "sft" or checkpoint.name.startswith("sft_") else "dpo"
    registered = store.register_model_checkpoint(
        ModelCheckpoint.new(
            model_key=f"{stage}:{checkpoint}",
            stage=stage,
            checkpoint_path=str(checkpoint),
            parent_checkpoint_id=parent["id"] if parent else None,
            provenance="pipeline",
        )
    )
    return registered, previous, value


def rollback_active_policy_checkpoint(
    store,
    checkpoint_id: str,
    project_root: Path,
    actor: str = "admin",
    model_version: Optional[str] = None,
    base_model_id: Optional[str] = None,
) -> Path:
    checkpoint = store.get_model_checkpoint(checkpoint_id)
    if checkpoint is None or not checkpoint["available"]:
        raise ValueError("Checkpoint is not available for rollback")
    if checkpoint["stage"] == "base" or not checkpoint["checkpoint_path"]:
        raise ValueError("Base-model rollback is not supported by the adapter policy resolver")
    path = Path(checkpoint["checkpoint_path"])
    if not is_lora_checkpoint(path):
        raise ValueError("Checkpoint adapter files are missing")
    record_active_policy_checkpoint(
        store,
        path,
        project_root,
        reason="rollback",
        actor=actor,
        model_version=model_version,
        base_model_id=base_model_id,
    )
    return path