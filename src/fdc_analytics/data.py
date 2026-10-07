from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_SECOM_ZIP = "https://archive.ics.uci.edu/static/public/179/secom.zip"
UCI_LEGACY_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"


def download_secom(data_dir: str | Path, force: bool = False) -> tuple[Path, Path]:
    """UCI SECOM 데이터를 내려받아 data_dir에 저장합니다."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "secom.data"
    labels_path = data_dir / "secom_labels.data"
    if data_path.exists() and labels_path.exists() and not force:
        return data_path, labels_path

    archive = data_dir / "secom.zip"
    zip_error = None
    try:
        urllib.request.urlretrieve(UCI_SECOM_ZIP, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
    except Exception as exc:
        zip_error = exc
        for filename in ("secom.data", "secom_labels.data"):
            try:
                urllib.request.urlretrieve(f"{UCI_LEGACY_BASE}/{filename}", data_dir / filename)
            except Exception:
                pass
    finally:
        if archive.exists():
            archive.unlink()

    if not data_path.exists() or not labels_path.exists():
        found_data = next(data_dir.rglob("secom.data"), None)
        found_labels = next(data_dir.rglob("secom_labels.data"), None)
        if found_data and found_data != data_path:
            shutil.copy2(found_data, data_path)
        if found_labels and found_labels != labels_path:
            shutil.copy2(found_labels, labels_path)

    if not data_path.exists() or not labels_path.exists():
        detail = f" Zip download error: {zip_error}" if zip_error else ""
        raise FileNotFoundError(
            "SECOM 데이터를 내려받지 못했습니다. "
            "secom.data와 secom_labels.data를 data/raw/에 넣고 다시 실행하세요."
            + detail
        )
    return data_path, labels_path


def load_secom(data_dir: str | Path, auto_download: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data_dir = Path(data_dir)
    data_path = data_dir / "secom.data"
    labels_path = data_dir / "secom_labels.data"
    if auto_download and (not data_path.exists() or not labels_path.exists()):
        data_path, labels_path = download_secom(data_dir)

    X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"signal_{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(labels_path, sep=r"\s+", header=None, dtype=str)
    y = pd.to_numeric(labels.iloc[:, 0], errors="raise").astype(int)
    timestamp_text = labels.iloc[:, 1:].fillna("").agg(" ".join, axis=1).str.strip()
    timestamp = pd.to_datetime(timestamp_text, dayfirst=True, errors="coerce")
    if timestamp.isna().any():
        raise ValueError("SECOM timestamp를 읽지 못했습니다.")

    order = timestamp.sort_values().index
    return X.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True), timestamp.loc[order].reset_index(drop=True)
