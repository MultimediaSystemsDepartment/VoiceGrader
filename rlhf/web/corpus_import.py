"""Bounded local uploads, staged validation and explicit catalogue acceptance."""

from __future__ import annotations

import json
import asyncio
import csv
import shutil
import uuid
from pathlib import Path

from starlette.datastructures import FormData, UploadFile

from run_import_corpus import validate_manifest
from storage.feedback_store import FeedbackStore
from storage.schema import AudioFile, Corpus

MAX_AUDIO_FILES = 200
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


def stage_path(root: Path, stage_id: str) -> Path:
    if str(uuid.UUID(stage_id)) != stage_id:
        raise ValueError("Invalid import identifier")
    return root / stage_id


def _filename(name: str | None) -> str:
    if not name or name in {".", ".."} or any(c in name for c in '/\\:\x00'):
        raise ValueError("Upload names must be plain filenames, without directories")
    if name.startswith(".") or name.lower() in {"manifest.csv", "stage.json", "accept.lock"}:
        raise ValueError(f"Reserved upload filename: {name}")
    return name


async def stage_upload(root: Path, form: FormData, store: FeedbackStore) -> str:
    purpose = str(form.get("purpose", ""))
    corpus = Corpus.new(str(form.get("name", "")), purpose=purpose)
    corpus.active = False
    if len(corpus.name) > 180:
        raise ValueError("Corpus name exceeds 180 characters")
    manifest = form.get("manifest")
    audio = form.getlist("audio")
    if not isinstance(manifest, UploadFile) or not audio or len(audio) > MAX_AUDIO_FILES:
        raise ValueError("Upload a CSV manifest and 1-200 audio files")
    stage_id = str(uuid.uuid4())
    directory = stage_path(root, stage_id)
    directory.mkdir(parents=True)
    try:
        names = set()
        total = 0
        for upload in [manifest, *audio]:
            if not isinstance(upload, UploadFile):
                raise ValueError("Expected uploaded files")
            name = "manifest.csv" if upload is manifest else _filename(upload.filename)
            if name.casefold() in names:
                raise ValueError(f"Duplicate upload filename: {name}")
            names.add(name.casefold())
            size = 0
            with (directory / name).open("xb") as target:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES or (upload is manifest and size > MAX_MANIFEST_BYTES):
                        raise ValueError("Upload exceeds 500 MiB total or 1 MiB manifest limit")
                    target.write(chunk)
        # Never allow the shared CLI validator to resolve paths outside this upload.
        with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter=";")
            referenced = set()
            for row in reader:
                name = _filename(row.get("file_path"))
                if name.casefold() not in names or not (directory / name).is_file():
                    raise ValueError(f"Row {reader.line_num}: audio was not uploaded: {name}")
                referenced.add(name.casefold())
            if referenced != names - {"manifest.csv"}:
                raise ValueError("Every uploaded audio file must appear in the manifest")
        files, report = await asyncio.to_thread(
            validate_manifest, directory / "manifest.csv", corpus.name, directory
        )
        if report.errors or report.duplicates:
            raise ValueError("; ".join(report.errors) or "Duplicate manifest rows are not allowed")
        await asyncio.to_thread(store.validate_uploaded_corpus, corpus, files)
        (directory / "stage.json").write_text(json.dumps({
            "name": corpus.name, "purpose": purpose, "status": "validated",
        }), encoding="utf-8")
        return stage_id
    except BaseException:
        shutil.rmtree(directory)
        raise


def preview_stage(root: Path, stage_id: str) -> tuple[dict, list[AudioFile]]:
    directory = stage_path(root, stage_id)
    metadata = json.loads((directory / "stage.json").read_text(encoding="utf-8"))
    files, report = validate_manifest(directory / "manifest.csv", metadata["name"], directory)
    if report.errors:
        raise ValueError("; ".join(report.errors))
    return metadata, files


def accept_stage(root: Path, stage_id: str, store: FeedbackStore) -> dict:
    directory = stage_path(root, stage_id)
    with (directory / "accept.lock").open("x"):
        pass
    try:
        metadata, files = preview_stage(root, stage_id)
        if metadata["status"] != "validated":
            raise ValueError("Import has already been accepted")
        corpus = Corpus.new(metadata["name"], purpose=metadata["purpose"])
        corpus.active = False
        result = store.import_uploaded_corpus(corpus, files)
        metadata["status"] = "accepted"
        (directory / "stage.json").write_text(json.dumps(metadata), encoding="utf-8")
        return result
    finally:
        (directory / "accept.lock").unlink()


def cancel_stage(root: Path, stage_id: str, store: FeedbackStore) -> None:
    directory = stage_path(root, stage_id)
    with (directory / "accept.lock").open("x"):
        pass
    try:
        metadata, files = preview_stage(root, stage_id)
        registered = store.get_all_audio_paths()
        if metadata["status"] != "validated" or any(audio.file_path in registered for audio in files):
            raise ValueError("Accepted recordings cannot be deleted through import cancellation")
        shutil.rmtree(directory)
    finally:
        (directory / "accept.lock").unlink(missing_ok=True)
