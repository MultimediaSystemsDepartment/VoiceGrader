"""Validate local recorded-audio manifests and import inactive corpora."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from paths import DEFAULT_PATHS, PROJECT_ROOT
from storage.feedback_store import FeedbackStore
from storage.schema import AudioFile, Corpus, normalize_utterance_text
from utils.audio import MAX_DURATION_S, MIN_DURATION_S, load_audio
from utils.log import get_logger

log = get_logger(__name__)


@dataclass
class ManifestReport:
    rows: int = 0
    unique_files: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


def _stored_path(path: Path) -> str:
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def _resolve_audio_path(raw: str, root: Path) -> tuple[str, str]:
    if raw.startswith("zip:"):
        archive_name, separator, member = raw[4:].partition("!")
        if (
            not separator or not archive_name or not member
            or PurePosixPath(member).is_absolute()
            or ".." in PurePosixPath(member).parts
            or "\\" in member
        ):
            raise ValueError("Expected zip:archive.zip!relative/member.wav")
        archive = Path(archive_name)
        if not archive.is_absolute():
            archive = root / archive
        archive = archive.resolve(strict=True)
        member = str(PurePosixPath(member))
        return f"zip:{archive}!{member}", f"zip:{_stored_path(archive)}!{member}"
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Not an audio file: {path}")
    return str(path), _stored_path(path)


def _inspect_audio(path: str) -> tuple[float, int]:
    import numpy as np
    import soundfile as sf

    if path.startswith("zip:"):
        archive, member = path[4:].split("!", 1)
        with zipfile.ZipFile(archive) as zipped:
            with zipped.open(member) as source:
                info = sf.info(source)
    else:
        info = sf.info(path)
    if not MIN_DURATION_S <= info.duration <= MAX_DURATION_S:
        raise ValueError(
            f"Duration {info.duration:.6f}s is outside "
            f"{MIN_DURATION_S}-{MAX_DURATION_S}s; segment long recordings first"
        )
    waveform, _ = load_audio(path)
    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains non-finite samples")
    return info.duration, info.samplerate


def validate_manifest(
    manifest: Path, corpus_name: str, audio_root: Path | None = None,
    delimiter: str = ";",
) -> tuple[list[AudioFile], ManifestReport]:
    if len(delimiter) != 1:
        raise ValueError("CSV delimiter must be one character")
    manifest = manifest.resolve(strict=True)
    root = (audio_root or manifest.parent).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Audio root must be a directory")
    report = ManifestReport()
    audio_files: list[AudioFile] = []
    references: dict[str, str | None] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter, strict=True)
        columns = reader.fieldnames
        if (
            not columns or "file_path" not in columns
            or len(columns) != len(set(columns))
            or set(columns) - {"file_path", "transcript"}
        ):
            raise ValueError("Expected unique file_path and optional transcript CSV columns")
        for row in reader:
            report.rows += 1
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Row does not match the CSV header")
                raw = (row["file_path"] or "").strip()
                if not raw:
                    raise ValueError("file_path is required")
                resolved, stored = _resolve_audio_path(raw, root)
                transcript = (row.get("transcript") or "").strip() or None
                normalized = normalize_utterance_text(transcript)
                if stored in references:
                    if normalized != references[stored]:
                        raise ValueError(f"Duplicate path has conflicting transcripts: {raw}")
                    report.duplicates += 1
                    continue
                duration, sample_rate = _inspect_audio(resolved)
                audio_files.append(AudioFile.new(
                    stored, corpus_name, transcript, duration, sample_rate,
                    provenance_json=json.dumps({
                        "import_manifest": _stored_path(manifest),
                        "manifest_row": reader.line_num,
                    }),
                ))
                references[stored] = normalized
            except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
                report.errors.append(f"Row {reader.line_num}: {exc}")
    report.unique_files = len(audio_files)
    if not report.rows:
        report.errors.append("Manifest contains no audio rows")
    return audio_files, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(DEFAULT_PATHS.db_path))
    commands = parser.add_subparsers(dest="command", required=True)
    importing = commands.add_parser("import", help="Validate and register an inactive corpus")
    importing.add_argument("--name", required=True)
    importing.add_argument("--manifest", type=Path, required=True)
    importing.add_argument("--audio-root", type=Path)
    importing.add_argument("--delimiter", default=";")
    importing.add_argument("--language", default="pl")
    importing.add_argument("--dry-run", action="store_true", help="Validate files without DB access")
    commands.add_parser("list", help="List corpora and their audio counts")
    for action in ("activate", "deactivate"):
        command = commands.add_parser(action)
        command.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "import":
            corpus = Corpus.new(args.name, language=args.language)
            corpus.active = False
            audio_files, report = validate_manifest(
                args.manifest, corpus.name, args.audio_root, args.delimiter
            )
            result = asdict(report)
            if report.errors:
                result["status"] = "rejected"
                print(json.dumps(result, ensure_ascii=False, indent=2))
                log.error("Import rejected; no corpus or audio records were written")
                return 1
            if args.dry_run:
                result["status"] = "validated"
            else:
                result.update(FeedbackStore(args.db).import_corpus_audio(corpus, audio_files))
                result["status"] = "imported"
        else:
            if not args.db.is_file():
                raise FileNotFoundError(f"Catalogue database does not exist: {args.db}")
            store = FeedbackStore(args.db)
            if args.command == "list":
                result = {
                    "corpora": [
                        dict(corpus, audio_files=store.count_audio_files(corpus["name"]))
                        for corpus in store.get_corpora()
                    ]
                }
            else:
                result = store.set_corpus_active(args.name.strip(), args.command == "activate")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, csv.Error, sqlite3.Error) as exc:
        print(json.dumps({"status": "failed", "errors": [str(exc)]}, ensure_ascii=False, indent=2))
        log.error("Corpus operation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
