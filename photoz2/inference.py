from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np

from .common import BANDS, features, read_sealed, save_csv, sha256
from .data import clean_photometry
from .experiment import verify_model_entry
from .models import load_bundle, predict_bundle


def predict_csv(run_dir, group, input_file, output_file):
    run_dir = Path(run_dir)
    frozen = read_sealed(run_dir / "frozen.json", "freeze_id")
    if sha256(run_dir / "final_models.json") != frozen["files"]["final_models.json"]:
        raise ValueError("Frozen model selection changed")
    selected = read_sealed(run_dir / "final_models.json", "models_id")
    if group == "primary":
        group = frozen["overall_best_group"]
    if group not in selected["models"]:
        raise ValueError(f"Unknown model {group}. Available: {list(selected['models'])}")
    with open(input_file, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {f"{prefix}_{b}" for prefix in ["psfmag", "extinction"] for b in BANDS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required-set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("No input objects")
    def number(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    arrays = [np.array([[number(r.get(f"{prefix}_{b}")) for b in BANDS] for r in rows])
              for prefix in ["psfmag", "psfmagerr", "extinction"]]
    mag, err, valid = clean_photometry(*arrays)
    entry = selected["models"][group]
    model = load_bundle(verify_model_entry(run_dir, entry))
    pred = np.full(len(rows), np.nan)
    if valid.any():
        x = mag if entry["feature"] == "none" else features({"mag": mag, "err": err}, entry["feature"])
        pred[valid] = predict_bundle(model, x[valid])
    save_csv(output_file, [
        {"object_id": row.get("object_id", str(i)), "model": group,
         "prediction": float(pred[i]) if valid[i] else "", "status": "ok" if valid[i] else "fewer_than_3_valid_bands"}
        for i, row in enumerate(rows)])
    print(f"Saved {len(rows)} predictions to {output_file}")


def bundle(data_dir, run_dir, output):
    from .experiment import frozen_plan
    from .common import read_json
    data_dir, run_dir, output = Path(data_dir).resolve(), Path(run_dir).resolve(), Path(output).resolve()
    frozen = frozen_plan(data_dir, run_dir)
    check = read_json(run_dir / "report" / "reload_check.json")
    if not check.get("passed") or check["freeze_id"] != frozen["freeze_id"]:
        raise ValueError("Run verify successfully before packaging")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.is_relative_to(data_dir) or output.is_relative_to(run_dir):
        raise ValueError("Put the output ZIP outside the data and run directories")
    root = Path(__file__).resolve().parents[1]
    code = [root / name for name in ["v2.py", "run_windows.ps1", "requirements.txt", "README.md", ".gitignore"]]
    code += [p for folder in ["photoz2", "configs", "tests"] for p in (root / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
        for path in code:
            z.write(path, "sdss-photoz-v2/" + path.relative_to(root).as_posix())
        for path in data_dir.iterdir():
            if path.is_file() and path.suffix in {".npz", ".json"}:
                z.write(path, "sdss-photoz-v2/data/v2/" + path.name)
        for path in run_dir.rglob("*"):
            if path.is_file():
                z.write(path, "sdss-photoz-v2/runs/exported/" + path.relative_to(run_dir).as_posix())
    print(f"Saved complete experiment (Zip64): {output}")
