#!/usr/bin/env python
"""
prepare_data.py -- Build flat manifest CSVs from the zipped datasets.

Reads audio and transcripts directly from the ZIP archives in _data/,
writes manifest CSVs with zip:-prefixed paths (no extraction needed),
and updates default_paths.json automatically.

Usage:
    py prepare_data.py                                     # process both datasets
    py prepare_data.py --datasets admedvoice
    py prepare_data.py --datasets commonvoice
    py prepare_data.py --test_ratio 0.1                    # ADMEDVOICE train/test split ratio
    py prepare_data.py --admed_zip /data/archive.zip       # explicit ADMEDVOICE zip path
    py prepare_data.py --cv_zip /data/cv-corpus-17.zip     # explicit CommonVoice zip path

CommonVoice ZIP auto-detection: if --cv_zip is not given, the script looks for
any file matching 'cv-corpus-*.zip' in _data/. If multiple are found, the newest
(alphabetically last) is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # type: ignore
        return it

# -- Paths ----------------------------------------------------------------------

ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "_data"
ADMED_ZIP     = DATA_DIR / "archive.zip"   # default; override with --admed_zip
OUTPUT_DIR    = DATA_DIR / "manifests"
PATHS_JSON    = ROOT / "default_paths.json"

DELIMITER     = ";"


def _find_cv_zip() -> Path | None:
    """
    Auto-detect the CommonVoice ZIP in _data/.
    Returns the alphabetically-last match (newest version), or None if not found.
    """
    candidates = sorted(DATA_DIR.glob("cv-corpus-*.zip"))
    return candidates[-1] if candidates else None


def _zip_path(zip_file: Path, inner: str) -> str:
    """Build a zip:-prefixed path reference, relative to the project root.

    Relative paths work on any machine (Windows dev, Docker, CI) as long as
    scripts are run from the project root -- which coordinator.py enforces via
    cwd=_PROJECT_ROOT. Absolute paths from a specific machine would break
    portability (e.g. OneDrive path on Windows vs /app/_data in Docker).
    """
    try:
        rel = zip_file.relative_to(ROOT)
        return f"zip:{rel.as_posix()}!{inner}"
    except ValueError:
        # zip_file is outside the project root -- use absolute as fallback
        return f"zip:{zip_file.as_posix()}!{inner}"


# -- ADMEDVOICE -----------------------------------------------------------------

def prepare_admedvoice(test_ratio: float = 0.1, seed: int = 42) -> dict:
    """
    Merge all per-session CSVs from archive.zip into train + test manifests.

    Each session CSV has columns: filename;id;phrase;comments;mos_pred;grade
    Audio files live in the same zip directory as their CSV.

    Returns dict with output paths.
    """
    print(f"Reading ADMEDVOICE from {ADMED_ZIP} ...")
    rows = []

    with zipfile.ZipFile(ADMED_ZIP, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        print(f"  Found {len(csv_names)} session CSVs")

        for csv_name in tqdm(csv_names, desc="ADMEDVOICE sessions", unit="csv"):
            session_dir = csv_name.rsplit("/", 1)[0]  # directory containing the CSV
            with zf.open(csv_name) as f:
                content = f.read().decode("utf-8", errors="replace")

            reader = csv.DictReader(
                content.splitlines(),
                delimiter=";",
                quotechar='"',
            )
            for row in reader:
                filename   = (row.get("filename") or "").strip().strip('"')
                transcript = (row.get("phrase")   or "").strip().strip('"')
                if not filename or not transcript:
                    continue
                inner_path = f"{session_dir}/{filename}"
                # verify file exists in zip
                if inner_path not in zf.namelist():
                    continue
                rows.append({
                    "file_path":  _zip_path(ADMED_ZIP, inner_path),
                    "transcript": transcript,
                })

    print(f"  Total valid rows: {len(rows)}")

    # Train / test split by row (shuffle then split)
    random.seed(seed)
    random.shuffle(rows)
    split = max(1, int(len(rows) * (1 - test_ratio)))
    train_rows = rows[:split]
    test_rows  = rows[split:]
    print(f"  Train: {len(train_rows)}  Test: {len(test_rows)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "admedvoice_train.csv"
    test_path  = OUTPUT_DIR / "admedvoice_test.csv"

    _write_manifest(train_path, train_rows)
    _write_manifest(test_path,  test_rows)
    print(f"  Written: {train_path}")
    print(f"  Written: {test_path}")

    return {
        "admedvoice_train_csv": str(train_path),
        "admedvoice_test_csv":  str(test_path),
        "admedvoice_audio_dir": "",   # not needed -- paths are zip:-prefixed
    }


# -- CommonVoice ----------------------------------------------------------------

def prepare_commonvoice(cv_zip: Path) -> dict:
    """
    Convert CommonVoice TSV splits (train, dev, test) to manifest CSVs.

    TSV columns: client_id, path, sentence_id, sentence, ...
    Audio files live at <corpus_root>/pl/clips/<path>.
    The corpus root directory is inferred from the ZIP's top-level directory.
    """
    print(f"Reading CommonVoice from {cv_zip} ...")

    # Derive the corpus prefix from the ZIP's top-level directory name.
    with zipfile.ZipFile(cv_zip, "r") as _zf:
        top_level = {n.split("/")[0] for n in _zf.namelist() if "/" in n}
    # The corpus root is the only top-level directory (e.g. 'cv-corpus-25.0-2026-03-09')
    corpus_root = sorted(top_level)[0] if top_level else cv_zip.stem
    cv_prefix   = f"{corpus_root}/pl"

    splits = {
        "train": (f"{cv_prefix}/train.tsv", OUTPUT_DIR / "commonvoice_train.csv"),
        "dev":   (f"{cv_prefix}/dev.tsv",   OUTPUT_DIR / "commonvoice_val.csv"),
        "test":  (f"{cv_prefix}/test.tsv",  OUTPUT_DIR / "commonvoice_test.csv"),
    }

    with zipfile.ZipFile(cv_zip, "r") as zf:
        all_clips = {n for n in zf.namelist() if "/clips/" in n}

        result = {}
        for split_name, (tsv_inner, out_path) in splits.items():
            if tsv_inner not in zf.namelist():
                print(f"  {split_name}.tsv not found in zip -- skipping")
                continue

            with zf.open(tsv_inner) as f:
                content = f.read().decode("utf-8", errors="replace")

            reader = csv.DictReader(content.splitlines(), delimiter="\t")
            rows = []
            for row in tqdm(reader, desc=f"CommonVoice {split_name}", unit="rows", leave=False):
                clip     = (row.get("path") or "").strip()
                sentence = (row.get("sentence") or "").strip()
                if not clip or not sentence:
                    continue
                inner = f"{cv_prefix}/clips/{clip}"
                if inner not in all_clips:
                    continue
                rows.append({
                    "file_path":  _zip_path(cv_zip, inner),
                    "transcript": sentence,
                })

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            _write_manifest(out_path, rows)
            print(f"  {split_name}: {len(rows)} rows -> {out_path}")
            result[split_name] = str(out_path)

    return {
        "commonvoice_train_csv": result.get("train", ""),
        "commonvoice_val_csv":   result.get("dev",   ""),
        "commonvoice_test_csv":  result.get("test",  ""),
        "commonvoice_audio_dir": "",  # not needed -- paths are zip:-prefixed
    }


# -- Manifest writer ------------------------------------------------------------

def _write_manifest(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_path", "transcript"], delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(rows)


# -- default_paths.json updater -------------------------------------------------

def update_default_paths(updates: dict) -> None:
    data = json.loads(PATHS_JSON.read_text(encoding="utf-8"))
    data.update({k: v for k, v in updates.items() if v})  # only non-empty values
    PATHS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {PATHS_JSON}")


# -- Entry point ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset manifests from ZIP archives")
    parser.add_argument("--datasets", nargs="+", default=["admedvoice", "commonvoice"],
                        choices=["admedvoice", "commonvoice"])
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction of ADMEDVOICE held out for test (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--admed_zip", type=Path, default=None,
        help="Path to ADMEDVOICE archive.zip (default: _data/archive.zip)",
    )
    parser.add_argument(
        "--cv_zip", type=Path, default=None,
        help=(
            "Path to CommonVoice ZIP (default: auto-detect cv-corpus-*.zip in _data/). "
            "Use this when you have a specific corpus version or non-standard path."
        ),
    )
    args = parser.parse_args()

    admed_zip = args.admed_zip or ADMED_ZIP

    # CommonVoice ZIP: explicit arg > auto-detect > error
    cv_zip: Path | None = args.cv_zip
    if cv_zip is None and "commonvoice" in args.datasets:
        cv_zip = _find_cv_zip()
        if cv_zip is not None:
            print(f"Auto-detected CommonVoice ZIP: {cv_zip}")

    updates = {}

    if "admedvoice" in args.datasets:
        if not admed_zip.exists():
            print(f"ADMEDVOICE zip not found: {admed_zip}")
            print("  Place the archive at _data/archive.zip or pass --admed_zip <path>")
        else:
            updates.update(prepare_admedvoice(args.test_ratio, args.seed))

    if "commonvoice" in args.datasets:
        if cv_zip is None or not cv_zip.exists():
            print("CommonVoice zip not found in _data/. "
                  "Download from https://commonvoice.mozilla.org/en/datasets "
                  "and pass --cv_zip <path>.")
        else:
            updates.update(prepare_commonvoice(cv_zip))

    if updates:
        update_default_paths(updates)
        print("\nAll done. Run:")
        print("  py run_generate_pairs.py --catalogue_only")


if __name__ == "__main__":
    main()
