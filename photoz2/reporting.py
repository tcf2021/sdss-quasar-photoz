from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .common import BANDS, COLORS, FEATURES, features, save_csv
from .models import make_preprocessor


def plots(report, test, predictions, rows, bins, selection):
    report = Path(report)
    groups = [k for k in predictions if k not in {"mean", "median"}]
    feature_names = list(dict.fromkeys(selection["selected"][k]["feature"] for k in groups))
    families = list(dict.fromkeys(selection["selected"][k]["family"] for k in groups))
    lookup = {r["model"]: r for r in rows}
    x = np.arange(len(feature_names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = .8 / len(families)
    for j, family in enumerate(families):
        for ax, metric in zip(axes, ["mae", "outlier_rate_015"]):
            vals = [lookup[f"{family}_{f}"][metric] for f in feature_names]
            ax.bar(x-.4+(j+.5)*width, vals, width, label=family)
            ax.set_xticks(x, feature_names, rotation=15)
            ax.set_title({"mae": "New holdout MAE", "outlier_rate_015": "Outlier fraction |delta| > 0.15"}[metric])
            ax.legend()
    fig.tight_layout()
    fig.savefig(report / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    labels = list(selection["selected"])
    ax.errorbar(np.arange(len(labels)),
                [selection["selected"][g]["cv"]["mae_mean"] for g in labels],
                yerr=[selection["selected"][g]["cv"]["mae_std"] for g in labels], fmt="o", capsize=3)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
    ax.set(ylabel="Outer-fold MAE (mean +/- sample SD)", title="Development CV selection scores (not final-test estimates)")
    fig.tight_layout()
    fig.savefig(report / "cv_comparison.png", dpi=160)
    plt.close(fig)
    primary = selection["overall_best_group"]
    y, pred = test["y"], predictions[primary]
    delta = (pred-y)/(1+y)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    h = ax[0].hexbin(y, pred, gridsize=65, bins="log", mincnt=1)
    low, high = min(0, float(pred.min())), max(float(y.max()), float(pred.max()))
    ax[0].plot([low, high], [low, high], "r--", lw=1)
    ax[0].set(xlabel="Spectroscopic Z", ylabel="Predicted redshift")
    fig.colorbar(h, ax=ax[0], label="Object count (log color)")
    ax[1].hist(delta, bins=80)
    ax[1].set(xlabel="(prediction - Z)/(1 + Z)", ylabel="Objects")
    ax[2].hexbin(y, delta, gridsize=65, bins="log", mincnt=1)
    ax[2].set(xlabel="Spectroscopic Z", ylabel="Normalized residual")
    for threshold in [-.15, .15]:
        ax[1].axvline(threshold, color="red", ls="--")
        ax[2].axhline(threshold, color="red", ls="--")
    fig.suptitle(f"{primary}: selected before final test")
    fig.tight_layout()
    fig.savefig(report / "best_predictions.png", dpi=160)
    plt.close(fig)
    selected_feature = "colors_i_err" if "colors_i_err" in feature_names else feature_names[-1]
    for axis, xlabel in [("z_spec", "Spectroscopic redshift"), ("i_mag0", "Extinction-corrected i magnitude")]:
        fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for family in families:
            values = [b for b in bins if b["model"] == f"{family}_{selected_feature}" and b["axis"] == axis]
            centers = [(b["low"]+b["high"])/2 for b in values]
            ax[0].plot(centers, [b["mae"] for b in values], "o-", label=family)
        if values:
            ax[1].bar(centers, [b["n"] for b in values], width=[(b["high"]-b["low"])*.8 for b in values])
        ax[0].set(ylabel="MAE", title=f"{selected_feature}: new holdout performance")
        ax[0].legend()
        ax[1].set(xlabel=xlabel, ylabel="Objects")
        fig.tight_layout()
        fig.savefig(report / f"error_by_{axis}.png", dpi=160)
        plt.close(fig)


def history_plots(run_dir, report, selection):
    run_dir, report = Path(run_dir), Path(report)
    for group, item in selection["selected"].items():
        if item["family"] not in {"mlp", "xgb"}:
            continue
        path = run_dir / item["fold_trials"][0]["directory"] / "inner_history.csv"
        with path.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        train_key, val_key = (("train_rmse_dropout_on", "inner_val_rmse") if item["family"] == "mlp"
                               else ("train_mae", "inner_val_mae"))
        fig, ax = plt.subplots(figsize=(7, 4))
        for key in [train_key, val_key]:
            ax.plot([int(r["step"]) for r in rows], [float(r[key]) for r in rows], label=key)
        ax.legend()
        ax.set(xlabel="Epoch / boosting round", ylabel="RMSE" if item["family"] == "mlp" else "MAE",
               title=f"{group}: fold 1 INNER early stopping")
        fig.tight_layout()
        fig.savefig(report / f"{group}_inner_training.png", dpi=150)
        plt.close(fig)


def pick_cases(y, pred, count):
    errors = abs((pred-y)/(1+y))
    # Predeclared strata prevent the list from containing only one extreme regime.
    intervals, shares = [(0, 1), (1, 2), (2, 4), (4, 7.00001)], [.4, .2, .2, .2]
    selected = []
    for (low, high), share in zip(intervals, shares):
        ids = np.flatnonzero((y >= low) & (y < high) & (errors > .15))
        selected.extend(ids[np.argsort(-errors[ids], kind="stable")[:int(count*share)]].tolist())
    pool = np.flatnonzero(errors > .15)
    for i in pool[np.argsort(-errors[pool], kind="stable")]:
        if len(selected) >= count:
            break
        if int(i) not in selected:
            selected.append(int(i))
    return np.array(selected[:count], dtype=np.int64)


def case_audit(report, dev, test, predictions, primary, count=20, neighbors=15, threads=4):
    report = Path(report)
    y, pred = test["y"], predictions[primary]
    outliers = pick_cases(y, pred, count)
    delta = abs((pred-y)/(1+y))
    controls, used = [], set()
    for i in outliers:
        # Match one reasonably well-predicted control in the same coarse redshift band,
        # choosing its closest i magnitude. This is descriptive case selection.
        low, high = next((lo, hi) for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 7.00001)] if lo <= y[i] < hi)
        candidates = np.flatnonzero((y >= low) & (y < high) & (delta <= .15))
        candidates = np.array([j for j in candidates if int(j) not in used], dtype=np.int64)
        if len(candidates):
            brightness_distance = abs(test["mag"][candidates, 3] - test["mag"][i, 3])
            brightness_distance = np.where(np.isfinite(brightness_distance), brightness_distance, np.inf)
            j = int(candidates[np.argmin(brightness_distance)])
            controls.append((j, int(i)))
            used.add(j)
    case_ids = list(outliers) + [c[0] for c in controls]
    if not case_ids:
        save_csv(report / "cases.csv", [], ["case_id", "role", "sdss_name"])
        save_csv(report / "case_neighbors.csv", [], ["case_id", "neighbor_rank"])
        return {"outlier_cases": 0, "controls": 0}
    prep = make_preprocessor("knn")
    xd = prep.fit_transform(features(dev, "colors")).astype(np.float32)
    xt = prep.transform(features(test, "colors")[case_ids]).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=min(neighbors, len(xd)), n_jobs=threads).fit(xd)
    distances, near = nn.kneighbors(xt)
    rows, neighbors_rows = [], []
    for position, i in enumerate(case_ids):
        is_outlier = position < len(outliers)
        case_id = f"case_{position+1:02d}"
        znear = dev["y"][near[position]]
        row = {"case_id": case_id, "role": "outlier" if is_outlier else "matched_control",
               "matched_outlier_catalog_row": "" if is_outlier else int(test["row"][controls[position-len(outliers)][1]]),
               "sdss_name": str(test["names"][i]), "catalog_row": int(test["row"][i]),
               "z_spec": float(y[i]), "primary_prediction": float(pred[i]), "normalized_residual": float((pred[i]-y[i])/(1+y[i])),
               "source_z": str(test["source_z"][i]), "z_warning": int(test["z_warning"][i]),
               "missing_mag_count": int(np.isnan(test["mag"][i]).sum()),
               "missing_error_count": int(np.isnan(test["err"][i]).sum()),
               "neighbor_z_median": float(np.median(znear)), "neighbor_z_std": float(np.std(znear)),
               "nearest_color_distance": float(distances[position, 0])}
        row.update({f"mag0_{b}": float(test["mag"][i, j]) for j, b in enumerate(BANDS)})
        row.update({f"color_{c}": float(features(test, "colors")[i, j]) for j, c in enumerate(COLORS)})
        row.update({f"magerr_{b}": float(test["err"][i, j]) for j, b in enumerate(BANDS)})
        row.update({f"prediction_{name}": float(values[i]) for name, values in predictions.items()})
        rows.append(row)
        for rank, (j, distance) in enumerate(zip(near[position], distances[position]), 1):
            neighbors_rows.append({"case_id": case_id, "neighbor_rank": rank,
                                   "training_sdss_name": str(dev["names"][j]), "training_catalog_row": int(dev["row"][j]),
                                   "training_z_spec": float(dev["y"][j]), "standardized_color_distance": float(distance)})
    save_csv(report / "cases.csv", rows)
    save_csv(report / "case_neighbors.csv", neighbors_rows)
    if len(outliers):
        ncols, nrows = 4, (len(outliers)+3)//4
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows*2.6), squeeze=False)
        for position, ax in enumerate(axes.ravel()):
            if position >= len(outliers):
                ax.axis("off")
                continue
            i = outliers[position]
            ax.hist(dev["y"][near[position]], bins=8, color="#4d7595", alpha=.8)
            ax.axvline(y[i], color="black", label="true Z")
            ax.axvline(pred[i], color="red", ls="--", label="prediction")
            ax.set_title(f"case_{position+1:02d}: Z={y[i]:.2f}, pred={pred[i]:.2f}", fontsize=9)
            ax.set_xlabel("Training-neighbor redshift", fontsize=8)
            if position == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Color neighbors from DEVELOPMENT only; descriptive, not causal diagnoses")
        fig.tight_layout()
        fig.savefig(report / "outlier_neighbors.png", dpi=160)
        plt.close(fig)
    return {"outlier_cases": len(outliers), "controls": len(controls),
            "neighbor_features": "four colors, development-only imputation and scaling",
            "interpretation": "Large neighbor redshift dispersion suggests ambiguity to investigate; it does not prove its physical cause"}


def color_diagram(report, dev):
    colors = features(dev, "colors")
    edges = np.arange(0, 7.0001, .25)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        centers, median, low, high = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (dev["y"] >= a) & (dev["y"] < b) & np.isfinite(colors[:, j])
            if mask.sum() >= 20:
                centers.append((a+b)/2)
                q = np.quantile(colors[mask, j], [.25, .5, .75])
                low.append(q[0]); median.append(q[1]); high.append(q[2])
        ax.plot(centers, median)
        ax.fill_between(centers, low, high, alpha=.25)
        ax.set(xlabel="Development spectroscopic Z", ylabel=COLORS[j])
    fig.suptitle("Development colors: median and interquartile range; bins with >=20 valid objects")
    fig.tight_layout()
    fig.savefig(Path(report) / "development_color_redshift.png", dpi=160)
    plt.close(fig)
