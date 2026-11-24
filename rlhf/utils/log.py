"""
Structured logging for the RLHF-ASR project.

All modules should obtain their logger via get_logger(__name__).
Log files land in PathsConfig.logs_dir / <date>.log.
Console output is always enabled; file output is optional (call setup_logging).

Usage:
    from utils.log import get_logger
    log = get_logger(__name__)
    log.info("Loaded %d audio files", n)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def setup_logging(
    logs_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> None:
    """
    Configure root logger once per process.
    Call this from each run_*.py entry point before anything else.
    Subsequent calls are no-ops.
    """
    global _root_configured
    if _root_configured:
        return

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (optional)
    if logs_dir is not None:
        logs_path = Path(logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        log_file = logs_path / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Quiet noisy third-party loggers
    for noisy in ("transformers", "datasets", "accelerate", "peft", "trl",
                  "torch", "filelock", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. setup_logging() need not have been called first."""
    return logging.getLogger(name)
