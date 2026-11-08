from __future__ import annotations

import time
from pathlib import Path

from .common import digest, environment, read_json, read_split, save_csv, save_json, sha256, source_digest, verify_data_files
from .models import train_model


def artifact_hashes(directory):
    return {p.name: sha256(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.name != "complete.json"}


def verify_artifacts(directory, expected):
    for name, hash_value in expected.items():
        p = Path(directory) / name
        if not p.is_file() or sha256(p) != hash_value:
            raise RuntimeError(f"Model artifact changed or missing: {p}")


def train(data_dir, run_dir, config_file, device="cuda", threads=8):
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    if (data_dir / "TEST_OPENED.json").exists():
        raise RuntimeError("The held-out test has been opened. Training on this experiment is frozen; use verify for reload checks.")
    cfg = read_json(config_file)
    manifest = verify_data_files(data_dir, ["train", "val"])
    train_data = read_split(data_dir, "train", cfg.get("train_limit"))
    val = read_split(data_dir, "val", cfg.get("val_limit"))
    if cfg.get("allow_test", False) and (len(train_data["y"]) != manifest["split_counts"]["train"] or len(val["y"]) != manifest["split_counts"]["val"]):
        raise ValueError("A test-eligible run must use the entire prepared training/validation sets")
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = {"config": cfg, "dataset_id": manifest["dataset_id"], "source_digest": source_digest(),
            "environment": environment(), "device_requested": device, "threads": threads}
    run_id = digest(spec)
    if (run_dir / "run.json").exists():
        if read_json(run_dir / "run.json")["run_id"] != run_id:
            raise RuntimeError("Run directory belongs to a different config/data/code/environment. Choose a new --run directory.")
    else:
        save_json(run_dir / "run.json", {**spec, "run_id": run_id, "started_at_unix": time.time()})
    tasks = [("constant_median", "constant", "none", 0, {"statistic": "median"}),
             ("constant_mean", "constant", "none", 0, {"statistic": "mean"})]
    for family in ["xgb", "mlp"]:
        for feature in cfg["features"]:
            for index, params in enumerate(cfg[family]):
                tasks.append((f"{family}_{feature}", family, feature, index+1, params))
    leaderboard, best = [], {}
    for group, family, feature, index, params in tasks:
        name = group if family == "constant" else f"{group}_{index:02d}"
        directory = run_dir / name
        marker = directory / "complete.json"
        if marker.exists():
            done = read_json(marker)
            verify_artifacts(directory, done["artifacts"])
            score, meta = read_json(directory / "validation_metrics.json"), read_json(directory / "meta.json")
            print(f"Reusing completed trial {name}: val_mae={score['mae']:.6f}", flush=True)
        else:
            print(f"Training {name}", flush=True)
            # Same seed across ablations; initialization still differs with input dimension.
            score, meta = train_model(directory, family, feature, params, train_data, val, cfg["seed"], device, threads)
            save_json(marker, {"run_id": run_id, "artifacts": artifact_hashes(directory)})
        row = {"group": group, "trial": name, **score, "train_seconds": meta["train_seconds"]}
        leaderboard.append(row)
        if group not in best or score["mae"] < best[group]["val_mae"]:
            best[group] = {"trial": name, "val_mae": score["mae"], "artifacts": read_json(marker)["artifacts"]}
        save_csv(run_dir / "validation_leaderboard.csv", leaderboard)
    selection = {"run_id": run_id, "dataset_id": manifest["dataset_id"], "allow_test": cfg.get("allow_test", False),
                 "purpose": manifest["purpose"], "selection_metric": "validation MAE in original redshift units",
                 "selected": best, "overall_best_group": min(best, key=lambda k: best[k]["val_mae"])}
    selection["selection_id"] = digest(selection)
    save_json(run_dir / "selection.json", selection)
    print(f"Frozen validation selection: {run_dir / 'selection.json'}", flush=True)
    return selection
