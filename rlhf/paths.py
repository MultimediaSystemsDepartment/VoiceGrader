"""
Path resolution for the RLHF-ASR project.

Edit default_paths.json (project root) once per machine to point at your
local ADMEDVOICE data directory. All other paths default to subdirectories
of _outputs/ inside the project root and require no editing.

Usage anywhere in the project:
    from paths import DEFAULT_PATHS
    store = FeedbackStore(DEFAULT_PATHS.db_path)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
_OUTPUTS = PROJECT_ROOT / "_outputs"
DEFAULT_WHISPER_MODEL_ID = "openai/whisper-large-v3"
_POLICY_OUTPUTS = _OUTPUTS / "large-v3"
_DEFAULT_PATHS_FILE = PROJECT_ROOT / "default_paths.json"
_PATHS_FILE_ENV = "RLHF_PATHS_FILE"


@dataclass
class PathsConfig:
    # -- Dataset metadata CSVs (';' delimiter, same format as source datasets) --
    admedvoice_train_csv: str = ""
    admedvoice_val_csv: str = ""
    admedvoice_test_csv: str = ""
    commonvoice_train_csv: str = ""
    commonvoice_val_csv: str = ""
    commonvoice_test_csv: str = ""
    csv_delimiter: str = ";"

    # -- Audio root directories (set per machine in default_paths.json) ---------
    admedvoice_audio_dir: str = ""
    commonvoice_audio_dir: str = ""

    # -- Output directories (defaults are relative to project root) -------------
    output_dir: str = str(_POLICY_OUTPUTS)
    db_path: str = str(_POLICY_OUTPUTS / "feedback.sqlite")
    checkpoints_dir: str = str(_POLICY_OUTPUTS / "checkpoints")
    latest_model_dir: str = str(_POLICY_OUTPUTS / "latest")
    reward_model_dir: str = str(_POLICY_OUTPUTS / "reward_model")
    logs_dir: str = str(_POLICY_OUTPUTS / "logs")

    # -- Model identifiers (HuggingFace Hub IDs or absolute local paths) --------
    whisper_model_id: str = DEFAULT_WHISPER_MODEL_ID
    bert_model_id: str = "dkleczek/bert-base-polish-uncased-v1"


def _resolve_path(value: str, base_dir: Path = PROJECT_ROOT) -> str:
    """
    Resolve a path value from default_paths.json.

    If the value is an absolute path, return it unchanged.
    If it is a relative path (or starts with a known relative prefix), resolve
    it against PROJECT_ROOT so the config works correctly regardless of the
    working directory the caller was launched from.
    Comments (keys starting with '_') are never path values and are skipped
    before this function is called.
    """
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return value
    return str(base_dir / p)


def load_paths(override_file: Optional[Path] = None) -> PathsConfig:
    """
    Load PathsConfig from default_paths.json if it exists, else return defaults.
    Unknown keys in the JSON file are silently ignored (forward-compatible).
    Relative path values are resolved against PROJECT_ROOT automatically, so
    default_paths.json can use repo-relative paths like '_data/manifests/foo.csv'
    and they will work from any working directory.
    """
    configured_file = os.environ.get(_PATHS_FILE_ENV)
    target = Path(override_file or configured_file or _DEFAULT_PATHS_FILE).resolve()
    base_dir = target.parent if configured_file and override_file is None else PROJECT_ROOT
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        # Path fields: any key that ends in _csv, _dir, _path, or _file
        _path_suffixes = ("_csv", "_dir", "_path", "_file")
        known = {}
        for k, v in data.items():
            if not hasattr(PathsConfig, k):
                continue
            if isinstance(v, str) and any(k.endswith(s) for s in _path_suffixes):
                known[k] = _resolve_path(v, base_dir)
            else:
                known[k] = v
        return PathsConfig(**known)
    return PathsConfig()


def save_paths(config: PathsConfig, target: Optional[Path] = None) -> None:
    """Persist a PathsConfig back to default_paths.json (or a custom path)."""
    dest = target or _DEFAULT_PATHS_FILE
    dest.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -- Module-level singleton -- import this everywhere ---------------------------
DEFAULT_PATHS: PathsConfig = load_paths()
