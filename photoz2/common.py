from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

BANDS = ["u", "g", "r", "i", "z"]
COLORS = ["u-g", "g-r", "r-i", "i-z"]
FEATURES = {
    "mag": [f"mag0_{b}" for b in BANDS],
    "colors": COLORS,
    "colors_i": COLORS + ["mag0_i"],
    "colors_i_err": COLORS + ["mag0_i"] + [f"magerr_{b}" for b in BANDS],
}
FAMILIES = ["knn", "rf", "xgb", "mlp"]
METRICS = ["mae", "rmse", "mae_norm", "rmse_norm", "bias_norm", "nmad", "outlier_rate_015"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def seal(value, key="id"):
    value = dict(value)
    value[key] = digest(value)
    return value


def read_sealed(path, key="id"):
    value = read_json(path)
    expected = value.pop(key)
    if digest(value) != expected:
        raise ValueError(f"Changed or invalid manifest: {path}")
    value[key] = expected
    return value


def save_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if y.ndim != 1 or pred.shape != y.shape or not len(y):
        raise ValueError("Metrics require nonempty equal-length vectors")
    if not np.isfinite(y).all() or not np.isfinite(pred).all() or np.any(y <= -1):
        raise ValueError("Invalid label or prediction")
    delta = pred - y
    norm = delta / (1 + y)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae_norm": float(np.mean(abs(norm))),
        "rmse_norm": float(np.sqrt(np.mean(norm**2))),
        "bias_norm": float(np.mean(norm)),
        "nmad": float(1.4826 * np.median(abs(norm - np.median(norm)))),
        "outlier_rate_015": float(np.mean(abs(norm) > .15)),
    }


def environment():
    names = ["numpy", "scipy", "scikit-learn", "astropy", "xgboost", "matplotlib", "torch", "joblib"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def source_digest():
    root = Path(__file__).resolve().parent
    files = list(root.glob("*.py")) + [root.parent / "v2.py"]
    return digest({str(p.relative_to(root.parent)): sha256(p) for p in files})


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def subset(data, ids):
    return {k: v[ids] for k, v in data.items()}


def features(data, name):
    mag = np.asarray(data["mag"], dtype=np.float32)
    colors = mag[:, :-1] - mag[:, 1:]
    result = {
        "mag": mag,
        "colors": colors,
        "colors_i": np.column_stack([colors, mag[:, 3]]),
        "colors_i_err": np.column_stack([colors, mag[:, 3], data["err"]]),
    }[name]
    return result.astype(np.float32, copy=False)


def artifact_hashes(directory):
    directory = Path(directory)
    return {p.name: sha256(p) for p in sorted(directory.iterdir())
            if p.is_file() and p.name not in {"complete.json", "job.json"} and not p.name.endswith(".tmp")}


def verify_artifacts(directory, hashes):
    directory = Path(directory)
    for name, expected in hashes.items():
        if not (directory / name).is_file() or sha256(directory / name) != expected:
            raise ValueError(f"Missing or changed artifact: {directory / name}")


def dataset_manifest(data_dir, splits=("dev",)):
    root = Path(data_dir)
    manifest = read_sealed(root / "manifest.json", "dataset_id")
    for split in splits:
        if sha256(root / f"{split}.npz") != manifest["files"][split]:
            raise ValueError(f"Changed dataset: {split}")
    return manifest


def validate_config(cfg):
    if cfg.get("schema_version") != 2 or cfg.get("folds") != 3:
        raise ValueError("V2 requires schema_version=2 and exactly three folds")
    if not cfg.get("features") or not set(cfg["features"]) <= set(FEATURES):
        raise ValueError("Invalid feature groups")
    if not cfg.get("models") or not set(cfg["models"]) <= set(FAMILIES):
        raise ValueError("Invalid model families")
    if len(set(cfg["features"])) != len(cfg["features"]):
        raise ValueError("Duplicate feature groups")
    if not .05 <= cfg["early_stop_fraction"] <= .3:
        raise ValueError("early_stop_fraction must lie in [0.05, 0.3]")
    for family, candidates in cfg["models"].items():
        if not candidates or len({digest(c) for c in candidates}) != len(candidates):
            raise ValueError(f"Empty or duplicate candidates: {family}")
    if cfg["shift"]["feature"] not in cfg["features"]:
        raise ValueError("Shift feature must have a CV comparison")
    if cfg["cv_max_samples"] is not None and cfg["cv_max_samples"] < 90:
        raise ValueError("cv_max_samples must be null or at least 90")
    if cfg.get("allow_final_test") and set(cfg["models"]) != set(FAMILIES):
        raise ValueError("Formal V2 comparisons require all four model families")
    return cfg
