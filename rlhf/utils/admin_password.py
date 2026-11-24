"""Persistent admin password configuration shared by web and local tools."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import secrets
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from paths import PROJECT_ROOT

BOOTSTRAP_PASSWORD = "change-me"  # Public installation default, not a secret.
ITERATIONS = 260_000
MIN_PASSWORD_LENGTH = 6


@dataclass(frozen=True)
class PasswordConfig:
    iterations: int
    salt: str
    hash: str


def password_config_path() -> Path:
    configured = os.environ.get("ADMIN_PASSWORD_CFG")
    return Path(configured) if configured else PROJECT_ROOT / "_outputs" / "admin_password.json"


def load_password_config(path: Path) -> PasswordConfig | None:
    try:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Admin password configuration must be a JSON object")
    iterations = payload.get("iterations", ITERATIONS)
    salt = payload.get("salt")
    digest = payload.get("hash")
    if (
        payload.get("algorithm", "pbkdf2-hmac-sha256") != "pbkdf2-hmac-sha256"
        or not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1
        or not isinstance(salt, str) or not re.fullmatch(r"[0-9a-fA-F]+", salt)
        or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
    ):
        raise ValueError("Invalid admin password configuration")
    return PasswordConfig(iterations, salt, digest)


def verify_password(password: str, config: PasswordConfig | None) -> bool:
    if config is None:
        return secrets.compare_digest(password.encode("utf-8"), BOOTSTRAP_PASSWORD.encode("utf-8"))
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), config.salt.encode("utf-8"), config.iterations
    ).hex()
    return secrets.compare_digest(candidate, config.hash)


def password_fingerprint(config: PasswordConfig | None) -> str:
    encoded = json.dumps(asdict(config) if config else None, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_password_config(path: Path, password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH or not password.strip():
        raise ValueError(f"Password must contain at least {MIN_PASSWORD_LENGTH} characters")
    salt = os.urandom(32).hex()
    config = PasswordConfig(
        ITERATIONS, salt,
        hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), ITERATIONS).hex(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(dict(asdict(config), algorithm="pbkdf2-hmac-sha256"), output, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Set or reset the admin password without a GUI")
    parser.add_argument("--config", type=Path, default=password_config_path())
    args = parser.parse_args()
    password = getpass.getpass("New admin password: ")
    if password != getpass.getpass("Confirm password: "):
        parser.error("Passwords do not match")
    try:
        save_password_config(args.config, password)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Admin password saved. Existing admin sessions are invalidated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
