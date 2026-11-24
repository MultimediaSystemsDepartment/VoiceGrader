from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

from paths import DEFAULT_PATHS


def token_path() -> Path:
    return Path(os.environ.get(
        "VOICEGRADER_BRIDGE_TOKEN_FILE",
        str(Path(DEFAULT_PATHS.output_dir).parent / "bridge_token"),
    ))


def read_bridge_token() -> str:
    value = token_path().read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid model-bridge credential file")
    return value


def ensure_bridge_token() -> None:
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        read_bridge_token()
        return
    with os.fdopen(fd, "w", encoding="ascii") as output:
        output.write(secrets.token_hex(32))
        output.flush()
        os.fsync(output.fileno())
    owner = os.environ.get("VOICEGRADER_BRIDGE_TOKEN_UID")
    if owner and os.name != "nt":
        os.chown(path, int(owner), int(owner))


if __name__ == "__main__":
    ensure_bridge_token()
    print("Model-bridge credential configured; transfer this file securely to ASR.")
