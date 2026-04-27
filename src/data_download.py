"""Download and locate the UCI diabetes readmission dataset."""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .config import (
    DATASET_CSV,
    IDS_MAPPING_CSV,
    RAW_DATA_DIR,
    UCI_DATASET_PAGE,
    UCI_ZIP_URLS,
    ensure_directories,
)


def find_raw_csv(raw_dir: Path = RAW_DATA_DIR) -> Path | None:
    """Return the first matching raw CSV path, if present."""
    candidates = [
        raw_dir / DATASET_CSV,
        raw_dir / "dataset_diabetes" / DATASET_CSV,
        raw_dir / "diabetes+130-us+hospitals+for+years+1999-2008" / DATASET_CSV,
    ]
    candidates.extend(raw_dir.rglob(DATASET_CSV) if raw_dir.exists() else [])
    for path in candidates:
        if path.exists():
            return path
    return None


def _extract_zip(zip_path: Path, raw_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    found = find_raw_csv(raw_dir)
    if found and found != raw_dir / DATASET_CSV:
        shutil.copy2(found, raw_dir / DATASET_CSV)

    for mapping in raw_dir.rglob(IDS_MAPPING_CSV):
        if mapping != raw_dir / IDS_MAPPING_CSV:
            shutil.copy2(mapping, raw_dir / IDS_MAPPING_CSV)
            break


def download_dataset(raw_dir: Path = RAW_DATA_DIR, force: bool = False) -> Path:
    """Download the UCI zip archive and return the raw `diabetic_data.csv` path.

    The UCI archive URL has changed over time, so the function tries multiple
    known URLs and raises a clear manual-upload message if all fail.
    """
    ensure_directories()
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = find_raw_csv(raw_dir)
    if existing and not force:
        return existing

    errors: list[str] = []
    zip_path = raw_dir / "uci_diabetes_130_hospitals.zip"

    for url in UCI_ZIP_URLS:
        try:
            print(f"Downloading dataset from {url}")
            urllib.request.urlretrieve(url, zip_path)
            _extract_zip(zip_path, raw_dir)
            csv_path = find_raw_csv(raw_dir)
            if csv_path:
                return csv_path
        except (urllib.error.URLError, urllib.error.HTTPError, zipfile.BadZipFile, OSError) as exc:
            errors.append(f"{url}: {exc}")

    message = (
        "Could not download the UCI diabetes dataset automatically.\n"
        f"Open {UCI_DATASET_PAGE}, download the dataset zip or diabetic_data.csv, "
        f"and place diabetic_data.csv in {raw_dir}.\n"
        "Errors:\n- " + "\n- ".join(errors)
    )
    raise RuntimeError(message)


def copy_uploaded_csv(uploaded_path: str | Path, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Copy a manually uploaded CSV into `data/raw/` and return its path."""
    ensure_directories()
    uploaded_path = Path(uploaded_path)
    if not uploaded_path.exists():
        raise FileNotFoundError(f"Uploaded CSV not found: {uploaded_path}")
    destination = raw_dir / DATASET_CSV
    shutil.copy2(uploaded_path, destination)
    return destination


def colab_upload_dataset(raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Upload `diabetic_data.csv` in Google Colab.

    This function imports Colab only when called, so local runs do not depend on
    `google.colab`.
    """
    try:
        from google.colab import files  # type: ignore
    except ImportError as exc:
        raise RuntimeError("This helper is only available inside Google Colab.") from exc

    uploaded = files.upload()
    if DATASET_CSV not in uploaded:
        raise RuntimeError(f"Please upload a file named {DATASET_CSV}.")

    return copy_uploaded_csv(DATASET_CSV, raw_dir=raw_dir)


if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset ready: {path}")
