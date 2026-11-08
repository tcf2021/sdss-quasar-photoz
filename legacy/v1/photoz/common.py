from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

BANDS = ["u", "g", "r", "i", "z"]
FEATURES = {
    "mag": [f"mag0_{b}" for b in BANDS],
    "rich": [f"mag0_{b}" for b in BANDS]
    + ["u-g", "g-r", "r-i", "i-z"] + [f"magerr_{b}" for b in BANDS],
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment():
    versions = {}
    for name in ["numpy", "scipy", "scikit-learn", "astropy", "xgboost", "matplotlib", "torch", "joblib"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def source_digest():
    root = Path(__file__).parent
    return digest({p.name: sha256(p) for p in sorted(root.glob("*.py"))})


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("Metrics require nonempty equal-length 1D arrays")
    if not np.isfinite(y).all() or not np.isfinite(pred).all() or np.any(y <= -1):
        raise ValueError("Non-finite predictions/labels or invalid redshift")
    delta = pred - y
    norm = delta / (1 + y)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "bias_norm": float(np.mean(norm)),
        "mae_norm": float(np.mean(np.abs(norm))),
        "rmse_norm": float(np.sqrt(np.mean(norm ** 2))),
        # Explicit convention: median absolute deviation of normalized residuals.
        "nmad": float(1.4826 * np.median(np.abs(norm - np.median(norm)))),
        "outlier_rate_015": float(np.mean(np.abs(norm) > 0.15)),
    }


def read_split(data_dir, split, limit=None):
    # Training calls this only for train/val; test is a separate file.
    with np.load(Path(data_dir) / f"{split}.npz", allow_pickle=False) as z:
        return {k: z[k][:limit].copy() for k in z.files}


def verify_data_files(data_dir, splits):
    manifest = read_json(Path(data_dir) / "manifest.json")
    if manifest["dataset_id"] != digest({k: v for k, v in manifest.items() if k != "dataset_id"}):
        raise RuntimeError("Dataset manifest was changed; use the original prepared data")
    for split in splits:
        if sha256(Path(data_dir) / f"{split}.npz") != manifest["files"][split]:
            raise RuntimeError(f"{split}.npz hash differs from manifest")
    return manifest
