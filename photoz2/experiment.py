from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from .common import (METRICS, dataset_manifest, digest, environment, load_npz, read_json,
                     read_sealed, save_csv, save_json, seal, sha256, source_digest,
                     validate_config, verify_artifacts)


def ensure_trainable(data_dir, run_dir):
    if (Path(data_dir) / "TEST_OPENED.json").exists():
        raise RuntimeError("The V2 holdout has been opened. Training is frozen; use evaluate/verify on its frozen run.")
    if (Path(run_dir) / "frozen.json").exists():
        raise RuntimeError("This run is frozen. Continue with evaluate/verify; do not change its models.")


def initialize(data_dir, run_dir, config_path, device, threads):
    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    ensure_trainable(data_dir, run_dir)
    manifest = dataset_manifest(data_dir)
    cfg = validate_config(read_json(config_path))
    spec = {"config": cfg, "dataset_id": manifest["dataset_id"], "source_digest": source_digest(),
            "environment": environment(), "device": device, "threads": threads}
    run = seal(spec, "run_id")
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run.json").exists():
        current = read_sealed(run_dir / "run.json", "run_id")
        if current["run_id"] != run["run_id"]:
            raise RuntimeError("Run config/data/code/environment/device changed. Choose a new --run directory.")
    else:
        save_json(run_dir / "run.json", run)
    return data_dir, run_dir, manifest, run


def get_run(data_dir, run_dir):
    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    run = read_sealed(run_dir / "run.json", "run_id")
    manifest = dataset_manifest(data_dir)
    if run["dataset_id"] != manifest["dataset_id"] or run["source_digest"] != source_digest():
        raise ValueError("Run data or source code changed")
    return data_dir, run_dir, manifest, run


def make_folds(n, limit, seed, fraction):
    ids = np.random.default_rng(seed).permutation(n)
    if limit is not None:
        ids = ids[:min(limit, n)]
    if len(ids) < 90:
        raise ValueError("Too few development samples for this experiment")
    result = {"cv_population": ids}
    for fold, (a, b) in enumerate(KFold(3, shuffle=True, random_state=seed).split(ids)):
        outer_train, outer_val = ids[a], ids[b]
        it, iv = train_test_split(outer_train, test_size=fraction, random_state=seed+100+fold)
        result.update({f"f{fold}_train": outer_train, f"f{fold}_eval": outer_val,
                       f"f{fold}_inner_train": it, f"f{fold}_inner_val": iv})
    return result


def persist_indices(path, arrays):
    if path.exists():
        actual = load_npz(path)
        if set(actual) != set(arrays) or any(not np.array_equal(actual[k], arrays[k]) for k in arrays):
            raise ValueError(f"Changed split indices: {path}")
    else:
        np.savez_compressed(path, **arrays)


def run_job(data_dir, run_dir, manifest, run, name, indices_path, **arguments):
    job_dir = run_dir / "trials" / name
    job_dir.mkdir(parents=True, exist_ok=True)
    job = seal({
        "run_id": run["run_id"], "source_digest": run["source_digest"],
        "dev_file": str(data_dir / "dev.npz"), "dev_sha256": manifest["files"]["dev"],
        "indices_file": str(indices_path), "indices_sha256": sha256(indices_path),
        "device": run["device"], "threads": run["threads"], **arguments,
    }, "job_id")
    job_path = job_dir / "job.json"
    if job_path.exists() and read_sealed(job_path, "job_id")["job_id"] != job["job_id"]:
        raise ValueError(f"Trial settings changed: {name}")
    save_json(job_path, job)
    if (job_dir / "complete.json").exists():
        done = read_sealed(job_dir / "complete.json")
        if done["job_id"] != job["job_id"]:
            raise ValueError(f"Wrong completed trial: {name}")
        verify_artifacts(job_dir, done["artifacts"])
        print(f"Reusing completed trial: {name}", flush=True)
    else:
        print(f"Starting trial: {name}", flush=True)
        logs = run_dir / "logs"
        logs.mkdir(exist_ok=True)
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
            env[key] = str(run["threads"])
        cmd = [sys.executable, "-u", "-X", "faulthandler", str(Path(__file__).resolve().parents[1] / "v2.py"),
               "_fit", "--job", str(job_path)]
        with (logs / f"{name}.log").open("a", encoding="utf-8") as log:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", env=env) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        if code != 0:
            raise RuntimeError(f"Trial {name} failed (exit {code}). See {logs / (name+'.log')}. "
                               "Completed trials are retained; no automatic device change or retry.")
        done = read_sealed(job_dir / "complete.json")
        verify_artifacts(job_dir, done["artifacts"])
    return read_json(job_dir / "result.json"), {
        "directory": str(job_dir.relative_to(run_dir)), "complete_sha256": sha256(job_dir / "complete.json")}


def verify_model_entry(run_dir, entry):
    directory = Path(run_dir) / entry["directory"]
    if sha256(directory / "complete.json") != entry["complete_sha256"]:
        raise ValueError(f"Changed completion record: {directory}")
    done = read_sealed(directory / "complete.json")
    verify_artifacts(directory, done["artifacts"])
    return directory


def cv(data_dir, run_dir, config, device="cuda", threads=8):
    data_dir, run_dir, manifest, run = initialize(data_dir, run_dir, config, device, threads)
    cfg = run["config"]
    fold_path = run_dir / "cv_indices.npz"
    arrays = make_folds(manifest["counts"]["dev"], cfg["cv_max_samples"], cfg["seed"], cfg["early_stop_fraction"])
    persist_indices(fold_path, arrays)
    groups = [(f, "none", [{}]) for f in ["median", "mean"]]
    groups += [(f, feature, candidates) for f, candidates in cfg["models"].items() for feature in cfg["features"]]
    rows, summary, selected = [], [], {}
    for family, feature, candidates in groups:
        group = family if feature == "none" else f"{family}_{feature}"
        group_candidates = []
        for index, params in enumerate(candidates, 1):
            records, entries = [], []
            for fold in range(3):
                trial = f"cv_{group}_c{index:02d}_f{fold+1}"
                result, entry = run_job(data_dir, run_dir, manifest, run, trial, fold_path,
                    stage="cv", family=family, feature=feature, params=params, seed=cfg["seed"]+fold,
                    train_key=f"f{fold}_train", eval_key=f"f{fold}_eval",
                    inner_train_key=f"f{fold}_inner_train", inner_val_key=f"f{fold}_inner_val")
                record = {"group": group, "family": family, "feature": feature,
                          "candidate": index, "fold": fold+1, **result}
                rows.append(record)
                records.append(record)
                entries.append(entry)
                save_csv(run_dir / "cv_folds.csv", rows)
            row = {"group": group, "family": family, "feature": feature, "candidate": index,
                   "cv_objects": len(arrays["cv_population"]), "folds": 3}
            for metric in METRICS:
                scores = [r[metric] for r in records]
                row[metric+"_mean"] = float(np.mean(scores))
                row[metric+"_std"] = float(np.std(scores, ddof=1))
            row["fit_seconds_mean"] = float(np.mean([r["fit_seconds"] for r in records]))
            row["predict_seconds_mean"] = float(np.mean([r["predict_seconds"] for r in records]))
            budgets = [r["budget"] for r in records if r["budget"] is not None]
            budget = max(1, int(np.median(budgets))) if budgets else None
            row["refit_budget"] = budget
            summary.append(row)
            group_candidates.append({"params": params, "family": family, "feature": feature,
                                     "candidate": index, "cv": row, "budget": budget, "fold_trials": entries})
            save_csv(run_dir / "cv_summary.csv", summary)
        selected[group] = min(group_candidates, key=lambda c: (c["cv"]["mae_mean"], c["candidate"]))
    overall = min(selected, key=lambda k: selected[k]["cv"]["mae_mean"])
    selection = seal({"run_id": run["run_id"], "dataset_id": manifest["dataset_id"],
                      "cv_indices_sha256": sha256(fold_path), "cv_summary_sha256": sha256(run_dir / "cv_summary.csv"),
                      "selected": selected, "overall_best_group": overall,
                      "selection_metric": "Mean outer-fold MAE; these are development selection scores, not final-test estimates",
                      "early_stopping": "Inner split only; refit all outer training objects for the chosen number of steps"},
                     "selection_id")
    save_json(run_dir / "cv_selection.json", selection)
    print(f"CV completed. Selected by development CV: {overall}", flush=True)
    return selection


def checked_selection(run_dir):
    selection = read_sealed(Path(run_dir) / "cv_selection.json", "selection_id")
    if sha256(Path(run_dir) / "cv_indices.npz") != selection["cv_indices_sha256"]:
        raise ValueError("CV indices changed")
    if sha256(Path(run_dir) / "cv_summary.csv") != selection["cv_summary_sha256"]:
        raise ValueError("CV summary changed")
    return selection


def fit(data_dir, run_dir):
    data_dir, run_dir, manifest, run = get_run(data_dir, run_dir)
    ensure_trainable(data_dir, run_dir)
    selection = checked_selection(run_dir)
    indices = run_dir / "final_indices.npz"
    persist_indices(indices, {"train": np.arange(manifest["counts"]["dev"])})
    entries = {}
    for group, item in selection["selected"].items():
        _, entry = run_job(data_dir, run_dir, manifest, run, "final_"+group, indices,
            stage="final", family=item["family"], feature=item["feature"], params=item["params"],
            seed=run["config"]["seed"], train_key="train", fixed_steps=item["budget"])
        entries[group] = {**entry, "family": item["family"], "feature": item["feature"],
                          "candidate": item["candidate"], "budget": item["budget"]}
    save_json(run_dir / "final_models.json", seal({
        "selection_id": selection["selection_id"], "models": entries}, "models_id"))
    print("All selected models refitted on the full development set.", flush=True)


def make_shift_indices(data, threshold, max_samples, seed):
    i = data["mag"][:, 3]
    pool = np.flatnonzero(np.isfinite(i))
    bright = np.flatnonzero(np.isfinite(i) & (i < threshold))
    n = min(int(max_samples), len(bright))
    if n < 20 or not np.any(i[pool] >= threshold):
        raise ValueError("Need at least 20 bright training objects and a faint development population")
    rng = np.random.default_rng(seed)
    return {"bright": rng.permutation(bright)[:n], "mixed": rng.permutation(pool)[:n]}


def shift_fit(data_dir, run_dir):
    data_dir, run_dir, manifest, run = get_run(data_dir, run_dir)
    ensure_trainable(data_dir, run_dir)
    selection = checked_selection(run_dir)
    cfg = run["config"]["shift"]
    data = load_npz(data_dir / "dev.npz")
    arrays = make_shift_indices(data, cfg["i_threshold"], cfg["max_train_samples"], run["config"]["seed"]+999)
    indices = run_dir / "shift_indices.npz"
    persist_indices(indices, arrays)
    entries = {}
    for family in run["config"]["models"]:
        item = selection["selected"][f"{family}_{cfg['feature']}"]
        for arm in ["bright", "mixed"]:
            name = f"shift_{family}_{arm}"
            _, entry = run_job(data_dir, run_dir, manifest, run, name, indices,
                stage="shift", family=family, feature=cfg["feature"], params=item["params"],
                seed=run["config"]["seed"], train_key=arm, fixed_steps=item["budget"])
            entries[f"{family}_{arm}"] = {**entry, "family": family, "arm": arm, "feature": cfg["feature"]}
    save_json(run_dir / "shift_models.json", seal({
        "selection_id": selection["selection_id"], "models": entries,
        "indices_sha256": sha256(indices), "i_threshold": cfg["i_threshold"],
        "feature": cfg["feature"], "train_n_per_arm": len(arrays["bright"]),
        "mixed_train_faint_count": int(np.sum(data["mag"][arrays["mixed"], 3] >= cfg["i_threshold"])),
        "evaluation": "Both arms evaluated on exactly the same finite-i faint subset of the NEW holdout",
        "interpretation": "Controlled brightness-selection shift; labels and other covariates may also shift. Not cross-survey validation.",
    }, "shift_id"))
    print(f"Shift models ready: {len(arrays['bright']):,} training objects per arm.", flush=True)


def freeze(data_dir, run_dir):
    data_dir, run_dir, manifest, run = get_run(data_dir, run_dir)
    if (run_dir / "frozen.json").exists():
        return frozen_plan(data_dir, run_dir)
    ensure_trainable(data_dir, run_dir)
    if not run["config"]["allow_final_test"]:
        raise RuntimeError("Smoke runs cannot access the final test")
    selection = checked_selection(run_dir)
    final = read_sealed(run_dir / "final_models.json", "models_id")
    shift = read_sealed(run_dir / "shift_models.json", "shift_id")
    if final["selection_id"] != selection["selection_id"] or shift["selection_id"] != selection["selection_id"]:
        raise ValueError("Inconsistent selection")
    if set(final["models"]) != set(selection["selected"]):
        raise ValueError("Not all selected models were fitted")
    expected_shift = {f"{family}_{arm}" for family in run["config"]["models"] for arm in ["bright", "mixed"]}
    if set(shift["models"]) != expected_shift:
        raise ValueError("Missing shift comparison models")
    for entry in list(final["models"].values())+list(shift["models"].values()):
        verify_model_entry(run_dir, entry)
    freeze_files = ["cv_selection.json", "cv_summary.csv", "cv_folds.csv", "cv_indices.npz",
                    "final_models.json", "final_indices.npz", "shift_models.json", "shift_indices.npz"]
    frozen = seal({"run_id": run["run_id"], "dataset_id": manifest["dataset_id"],
                   "selection_id": selection["selection_id"], "purpose": manifest["purpose"],
                   "files": {name: sha256(run_dir / name) for name in freeze_files},
                   "overall_best_group": selection["overall_best_group"]}, "freeze_id")
    save_json(run_dir / "frozen.json", frozen)
    print(f"Frozen V2 selection: {run_dir / 'frozen.json'}", flush=True)
    return frozen


def frozen_plan(data_dir, run_dir):
    data_dir, run_dir, manifest, run = get_run(data_dir, run_dir)
    frozen = read_sealed(run_dir / "frozen.json", "freeze_id")
    if frozen["run_id"] != run["run_id"] or frozen["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Wrong frozen experiment")
    for name, expected in frozen["files"].items():
        if sha256(run_dir / name) != expected:
            raise ValueError(f"Frozen experiment file changed: {name}")
    for name, key in [("final_models.json", "models_id"), ("shift_models.json", "shift_id")]:
        for entry in read_sealed(run_dir / name, key)["models"].values():
            verify_model_entry(run_dir, entry)
    return frozen
