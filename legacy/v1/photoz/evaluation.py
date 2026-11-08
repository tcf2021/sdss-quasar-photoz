from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import BANDS, digest, metrics, read_json, read_split, save_csv, save_json, sha256, verify_data_files
from .data import photometry
from .experiments import verify_artifacts
from .models import predict_model


def frozen_selection(run_dir):
    selection = read_json(Path(run_dir) / "selection.json")
    if digest({k: v for k, v in selection.items() if k != "selection_id"}) != selection["selection_id"]:
        raise RuntimeError("Selection file has changed")
    for winner in selection["selected"].values():
        verify_artifacts(Path(run_dir) / winner["trial"], winner["artifacts"])
    return selection


def plot_history(directory, output, name):
    path = Path(directory) / "history.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7, 4))
    step = [float(x["step"]) for x in rows]
    for key in rows[0]:
        if key != "step":
            label = key.replace("_", " ")
            if key == "train_rmse":
                label += " (training mode; dropout on)"
            ax.plot(step, [float(r[key]) for r in rows], label=label)
    ax.set(xlabel="Epoch / boosting round", ylabel="Redshift error", title=f"{name}: training history")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(output) / f"{name}_training.png", dpi=160)
    plt.close(fig)


def plot_predictions(y, pred, name, output, purpose):
    delta = (pred-y)/(1+y)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    h = ax[0].hexbin(y, pred, gridsize=65, bins="log", mincnt=1, cmap="viridis")
    lower, upper = min(float(y.min()), float(pred.min()), 0), max(float(y.max()), float(pred.max()))
    ax[0].plot([lower, upper], [lower, upper], "r--", lw=1)
    ax[0].set(xlabel="Spectroscopic redshift Z", ylabel="Predicted photometric redshift")
    fig.colorbar(h, ax=ax[0], label="Count (log color)")
    ax[1].hist(delta, bins=100, color="#245c8c")
    ax[1].axvline(-.15, color="red", ls="--")
    ax[1].axvline(.15, color="red", ls="--")
    ax[1].set(xlabel="(z_pred - Z) / (1 + Z)", ylabel="Count")
    ax[2].hexbin(y, delta, gridsize=65, bins="log", mincnt=1, cmap="viridis")
    ax[2].axhline(0, color="black", lw=1)
    ax[2].axhline(.15, color="red", ls="--")
    ax[2].axhline(-.15, color="red", ls="--")
    ax[2].set(xlabel="Spectroscopic redshift Z", ylabel="Normalized residual")
    fig.suptitle(f"{name} | {purpose}", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(output) / f"{name}_predictions.png", dpi=160)
    plt.close(fig)


def evaluate(data_dir, run_dir, device="cpu"):
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    selection = frozen_selection(run_dir)
    if not selection["allow_test"]:
        raise RuntimeError("Pilot/smoke runs cannot evaluate held-out test data. Complete configs/full.json first.")
    manifest = verify_data_files(data_dir, ["test"])
    if manifest["dataset_id"] != selection["dataset_id"]:
        raise ValueError("Training and test datasets do not match")
    marker = data_dir / "TEST_OPENED.json"
    if marker.exists() and read_json(marker)["selection_id"] != selection["selection_id"]:
        raise RuntimeError("Test already evaluated with another selection; further model selection would contaminate this holdout")
    if not marker.exists():
        save_json(marker, {"selection_id": selection["selection_id"], "opened_at_unix": time.time(),
                           "note": "Training is now frozen; repeated evaluation/reload of this exact selection is allowed."})
    test = read_split(data_dir, "test")
    report = run_dir / "report"
    report.mkdir(exist_ok=True)
    y, rows, binned, predictions = test["y"], [], [], {}
    for group, selected in selection["selected"].items():
        directory = run_dir / selected["trial"]
        pred = predict_model(directory, test, device=device)
        predictions[group] = pred
        row = {"model": group, "trial": selected["trial"], "validation_mae": selected["val_mae"], **metrics(y, pred)}
        rows.append(row)
        for axis, values, bins in [("z_spec", y, [0, .5, 1, 1.5, 2, 2.5, 3, 4, 7]),
                                   ("i_mag0", test["mag"][:, 3], [0, 18, 19, 20, 21, 22, 40])]:
            for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                mask = (values >= low) & ((values <= high) if i == len(bins)-2 else (values < high))
                if mask.any():
                    binned.append({"model": group, "axis": axis, "low": low, "high": high, **metrics(y[mask], pred[mask])})
        if not group.startswith("constant"):
            plot_predictions(y, pred, group, report, manifest["purpose"])
            plot_history(directory, report, group)
        print(f"{group:18s} MAE={row['mae']:.5f} RMSE={row['rmse']:.5f} outlier={100*row['outlier_rate_015']:.2f}%", flush=True)
    save_csv(report / "metrics_test.csv", rows)
    save_csv(report / "metrics_by_bin.csv", binned)
    save_json(report / "metrics_test.json", rows)
    np.savez_compressed(report / "predictions.npz", y=y, row=test["row"], **predictions)
    save_csv(report / "predictions.csv", (
        {"sdss_name": str(test["names"][i]), "catalog_row": int(test["row"][i]), "z_spec": float(y[i]),
         **{k: float(v[i]) for k, v in predictions.items()}} for i in range(len(y))
    ))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    learned = [r for r in rows if not r["model"].startswith("constant")]
    for a, metric, title in [(ax[0], "mae", "Test MAE"), (ax[1], "outlier_rate_015", "Outlier fraction |delta| > 0.15")]:
        a.bar([r["model"] for r in learned], [r[metric] for r in learned], color=["#466ca1", "#25466b", "#d78b47", "#a55b26"][:len(learned)])
        a.set_title(title)
        a.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(report / "comparison.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for group in predictions:
        if group.startswith("constant"):
            continue
        r = [x for x in binned if x["model"] == group and x["axis"] == "z_spec"]
        ax.plot([(x["low"]+x["high"])/2 for x in r], [x["mae"] for x in r], "o-", label=group)
    ax.set(xlabel="Spectroscopic redshift bin center", ylabel="MAE", title="Test error by redshift (counts in metrics_by_bin.csv)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report / "error_by_redshift.png", dpi=170)
    plt.close(fig)
    body = ["# SDSS DR16Q photometric redshift experiment", "", f"Purpose: **{manifest['purpose']}**", "",
            f"Prepared objects: {manifest['n_selected']:,}; split: {manifest['split_counts']}.",
            f"Dataset ID: `{manifest['dataset_id']}`.",
            f"Selected by validation MAE; overall validation winner: `{selection['overall_best_group']}`.", "",
            "| Model | Validation MAE | Test MAE | Test RMSE | NMAD | Outlier fraction |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        body.append(f"| {r['model']} | {r['validation_mae']:.5f} | {r['mae']:.5f} | {r['rmse']:.5f} | {r['nmad']:.5f} | {r['outlier_rate_015']:.3%} |")
    body.extend(["", "Normalized residual: delta = (z_pred - Z)/(1 + Z).",
                 "NMAD = 1.4826 * median(abs(delta - median(delta))). Outliers: abs(delta) > 0.15.",
                 "Predictions are evaluated without clipping. Empty bins are omitted from the binned CSV.",
                 "Magnitude bins exclude missing i-band magnitudes; redshift bins include the entire test set.",
                 "", "![Comparison](comparison.png)", "", "![Error by redshift](error_by_redshift.png)", "",
                 "These are random within-catalog holdout results for selected, spectroscopically confirmed quasars with 0 < Z <= 7.",
                 "They do not demonstrate generalization to another survey or arbitrary unconfirmed photometric objects.",
                 "Single-seed results are descriptive; no statistical-significance claim is made.",
                 "The rich-feature ablation jointly adds colors and errors; it does not isolate their individual effects.",
                 "Inspect metrics_by_bin.csv counts before discussing rare high-redshift objects.", ""])
    (report / "RESULTS.md").write_text("\n".join(body), encoding="utf-8")
    save_json(report / "evaluation.json", {"selection_id": selection["selection_id"], "purpose": manifest["purpose"],
        "device": device, "predictions_sha256": sha256(report / "predictions.npz"),
        "metrics_sha256": sha256(report / "metrics_test.json"), "missing_i_mag": int(np.isnan(test["mag"][:, 3]).sum())})
    return rows


def verify(data_dir, run_dir, device="cpu"):
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    selection = frozen_selection(run_dir)
    manifest = verify_data_files(data_dir, ["test"])
    if manifest["dataset_id"] != selection["dataset_id"]:
        raise ValueError("Dataset mismatch")
    marker = read_json(data_dir / "TEST_OPENED.json")
    if marker["selection_id"] != selection["selection_id"]:
        raise ValueError("Test selection mismatch")
    report = run_dir / "report"
    eval_meta = read_json(report / "evaluation.json")
    if eval_meta["selection_id"] != selection["selection_id"]:
        raise ValueError("Evaluation does not match selected models")
    for f, key in [("predictions.npz", "predictions_sha256"), ("metrics_test.json", "metrics_sha256")]:
        if sha256(report / f) != eval_meta[key]:
            raise ValueError(f"Stored {f} has changed")
    test = read_split(data_dir, "test")
    old_metrics = {x["model"]: x for x in read_json(report / "metrics_test.json")}
    checks = {}
    with np.load(report / "predictions.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(test["row"], old["row"])
        np.testing.assert_array_equal(test["y"], old["y"])
        for group, selected in selection["selected"].items():
            pred = predict_model(run_dir / selected["trial"], test, device)
            np.testing.assert_allclose(pred, old[group], rtol=1e-5, atol=1e-4)
            m = metrics(test["y"], pred)
            # Thresholded outlier counts can differ by one at a floating point boundary.
            for k in ["mae", "rmse", "mae_norm", "rmse_norm", "bias_norm", "nmad"]:
                np.testing.assert_allclose(m[k], old_metrics[group][k], rtol=1e-5, atol=1e-4)
            checks[group] = {"max_absolute_difference": float(np.max(np.abs(pred-old[group]))),
                             "outlier_fraction_difference": float(m["outlier_rate_015"]-old_metrics[group]["outlier_rate_015"])}
    save_json(report / "reload_check.json", {"passed": True, "device": device, "models": checks})
    print("PASS: saved models reproduce predictions and metrics within numerical tolerance", flush=True)


def predict_csv(run_dir, group, input_csv, output_csv, device="cpu"):
    run_dir = Path(run_dir)
    selection = frozen_selection(run_dir)
    if group not in selection["selected"]:
        raise ValueError(f"Unknown model {group}; choose from {list(selection['selected'])}")
    with open(input_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {f"{p}_{b}" for p in ["psfmag", "extinction"] for b in BANDS}
        if not required.issubset(fields):
            raise ValueError(f"Missing CSV columns: {sorted(required-fields)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV is empty")
    def number(x):
        return float(x) if x is not None and str(x).strip() else np.nan
    arrays = [np.array([[number(r.get(f"{p}_{b}")) for b in BANDS] for r in rows])
              for p in ["psfmag", "psfmagerr", "extinction"]]
    data, valid = photometry(*arrays)
    pred = np.full(len(rows), np.nan)
    if valid.any():
        pred[valid] = predict_model(run_dir / selection["selected"][group]["trial"], {k: v[valid] for k, v in data.items()}, device)
    save_csv(output_csv, [{"object_id": r.get("object_id", str(i)), "z_phot": float(pred[i]) if valid[i] else "",
                          "status": "ok" if valid[i] else "fewer_than_3_valid_bands"} for i, r in enumerate(rows)])
    print(f"Predicted {int(valid.sum())}/{len(rows)} rows -> {output_csv}", flush=True)
