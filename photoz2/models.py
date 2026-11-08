from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (FEATURES, artifact_hashes, features, load_npz, metrics, read_json,
                     read_sealed, save_csv, save_json, seal, sha256, source_digest)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def make_preprocessor(family):
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if family in {"knn", "mlp"}:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def torch_module(input_dim, hidden, dropout):
    import torch
    layers = []
    for width in hidden:
        layers.extend([torch.nn.Linear(input_dim, width), torch.nn.ReLU(), torch.nn.Dropout(dropout)])
        input_dim = width
    layers.append(torch.nn.Linear(input_dim, 1))
    return torch.nn.Sequential(*layers)


def _mlp_predict(model, x, meta, device, batch=8192):
    import torch
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=device)
            values.append((model(xb).squeeze(-1) * meta["ystd"] + meta["ymean"]).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def train_arrays(family, x, y, params, seed, threads, device, validation=None, fixed_steps=None):
    """No outer-fold labels enter this function."""
    prep = make_preprocessor(family)
    x = prep.fit_transform(x).astype(np.float32)
    xv, yv = (None, None) if validation is None else (
        prep.transform(validation[0]).astype(np.float32), validation[1])
    meta = {"family": family, "params": params, "seed": seed, "input_dim": x.shape[1],
            "fit_n": len(y), "prediction_clipping": "none"}
    history = []
    if family == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        if params["n_neighbors"] > len(y):
            raise ValueError("n_neighbors exceeds training sample count")
        model = KNeighborsRegressor(n_jobs=threads, **params).fit(x, y)
        meta.update(device="cpu", budget=None)
    elif family == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=seed, n_jobs=threads, **params).fit(x, y)
        meta.update(device="cpu", budget=None)
    elif family == "xgb":
        from xgboost import XGBRegressor
        options = {k: v for k, v in params.items() if k not in {"max_rounds", "patience"}}
        if validation is not None:
            options["early_stopping_rounds"] = params["patience"]
        model = XGBRegressor(
            objective="reg:squarederror", eval_metric="mae", tree_method="hist",
            device="cpu", n_jobs=threads, random_state=seed,
            n_estimators=int(fixed_steps or params["max_rounds"]), **options)
        evals = [(x, y)] + ([(xv, yv)] if validation is not None else [])
        model.fit(x, y, eval_set=evals, verbose=False)
        values = model.evals_result()
        for i, value in enumerate(values["validation_0"]["mae"]):
            row = {"step": i + 1, "train_mae": value}
            if validation is not None:
                row["inner_val_mae"] = values["validation_1"]["mae"][i]
            history.append(row)
        budget = model.best_iteration + 1 if validation is not None else model.n_estimators
        meta.update(device="cpu", budget=int(budget))
    elif family == "mlp":
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Choose --device cpu with a new run directory.")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(max(1, threads))
        torch.use_deterministic_algorithms(True)
        model = torch_module(x.shape[1], params["hidden"], params["dropout"]).to(device)
        meta.update(device=device, ymean=float(np.mean(y)), ystd=max(float(np.std(y)), 1e-6))
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        xt = torch.as_tensor(x, device=device)
        yt = torch.as_tensor((y - meta["ymean"]) / meta["ystd"], dtype=torch.float32, device=device)
        best, stale, best_state, best_epoch = float("inf"), 0, None, 0
        epochs = int(fixed_steps or params["max_epochs"])
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(y), device=device)
            total = 0.
            for ids in order.split(params["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(xt[ids]).squeeze(-1), yt[ids])
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite MLP loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += loss.item() * len(ids)
            row = {"step": epoch, "train_rmse_dropout_on": (total / len(y))**.5 * meta["ystd"]}
            if validation is not None:
                pred = _mlp_predict(model, xv, meta, device)
                score = metrics(yv, pred)
                row.update(inner_val_mae=score["mae"], inner_val_rmse=score["rmse"])
                if score["mae"] < best:
                    best, stale, best_epoch = score["mae"], 0, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
            history.append(row)
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"  epoch={epoch:03d} train_rmse={row['train_rmse_dropout_on']:.5f}"
                      + (f" inner_val_mae={row['inner_val_mae']:.5f}" if validation is not None else ""), flush=True)
            if validation is not None and stale >= params["patience"]:
                break
        if validation is not None:
            if best_state is None:
                raise RuntimeError("No finite inner-validation checkpoint")
            model.load_state_dict(best_state)
        meta["budget"] = int(best_epoch if validation is not None else epochs)
    else:
        raise ValueError(f"Unknown family {family}")
    return {"prep": prep, "model": model, "meta": meta}, history


def predict_bundle(bundle, x):
    meta = bundle["meta"]
    if meta["family"] in {"mean", "median"}:
        return np.full(len(x), meta["value"], dtype=np.float64)
    x = bundle["prep"].transform(x).astype(np.float32)
    if meta["family"] == "mlp":
        return _mlp_predict(bundle["model"], x, meta, meta["device"])
    # Bounded prediction batches also bound kNN's working memory.
    return np.concatenate([bundle["model"].predict(x[i:i + 2048]) for i in range(0, len(x), 2048)])


def save_bundle(directory, bundle):
    directory = Path(directory)
    meta = bundle["meta"]
    save_json(directory / "meta.json", meta)
    if meta["family"] in {"mean", "median"}:
        return
    joblib.dump(bundle["prep"], directory / "preprocessor.joblib", compress=3)
    if meta["family"] == "mlp":
        import torch
        state = {k: v.detach().cpu() for k, v in bundle["model"].state_dict().items()}
        temp = directory / "model.pt.tmp"
        torch.save(state, temp)
        temp.replace(directory / "model.pt")
    elif meta["family"] == "xgb":
        bundle["model"].save_model(directory / "model.ubj")
    else:
        joblib.dump(bundle["model"], directory / "model.joblib", compress=3)


def load_bundle(directory, device="cpu", threads=4):
    directory = Path(directory)
    meta = read_json(directory / "meta.json")
    if meta["family"] in {"mean", "median"}:
        return {"meta": meta}
    if meta["feature_names"] != FEATURES[meta["feature"]]:
        raise ValueError("Model feature schema changed")
    prep = joblib.load(directory / "preprocessor.joblib")
    if meta["family"] == "mlp":
        import torch
        torch.set_num_threads(max(1, threads))
        model = torch_module(meta["input_dim"], meta["params"]["hidden"], meta["params"]["dropout"])
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        model.to(device)
        meta["device"] = device
    elif meta["family"] == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(directory / "model.ubj")
        model.set_params(device="cpu", n_jobs=threads)
    else:
        model = joblib.load(directory / "model.joblib")
        model.set_params(n_jobs=threads)
    return {"prep": prep, "model": model, "meta": meta}


def execute_job(job_path):
    job_path = Path(job_path)
    job = read_sealed(job_path, "job_id")
    directory = job_path.parent
    if source_digest() != job["source_digest"]:
        raise ValueError("Source code changed after job creation")
    if sha256(job["dev_file"]) != job["dev_sha256"] or sha256(job["indices_file"]) != job["indices_sha256"]:
        raise ValueError("Training data or fold indices changed")
    data = load_npz(job["dev_file"])  # This worker never reads the held-out test file.
    indices = load_npz(job["indices_file"])
    train_ids = indices[job["train_key"]]
    eval_ids = indices[job["eval_key"]] if job.get("eval_key") else None
    if eval_ids is not None and np.intersect1d(train_ids, eval_ids).size:
        raise ValueError("Outer training and validation overlap")
    family, feature, params = job["family"], job["feature"], job["params"]
    x = features(data, feature) if feature != "none" else data["mag"]
    y = data["y"]
    started = time.perf_counter()
    budget = job.get("fixed_steps")
    if family in {"xgb", "mlp"} and job.get("inner_train_key"):
        it, iv = indices[job["inner_train_key"]], indices[job["inner_val_key"]]
        if np.intersect1d(it, iv).size or not np.array_equal(np.sort(np.concatenate([it, iv])), np.sort(train_ids)):
            raise ValueError("Invalid inner early-stopping split")
        selected, history = train_arrays(family, x[it], y[it], params, job["seed"], job["threads"],
                                         job["device"], validation=(x[iv], y[iv]))
        budget = selected["meta"]["budget"]
        save_csv(directory / "inner_history.csv", history)
        del selected
        gc.collect()
        if family == "mlp" and job["device"] == "cuda":
            import torch
            torch.cuda.empty_cache()
    if family in {"mean", "median"}:
        value = float(np.mean(y[train_ids]) if family == "mean" else np.median(y[train_ids]))
        bundle = {"meta": {"family": family, "value": value, "budget": None, "device": "cpu",
                           "fit_n": len(train_ids), "seed": job["seed"], "params": params}}
        history = []
    else:
        bundle, history = train_arrays(family, x[train_ids], y[train_ids], params,
                                        job["seed"], job["threads"], job["device"], fixed_steps=budget)
    bundle["meta"].update(feature=feature, feature_names=FEATURES.get(feature, []), stage=job["stage"],
                           job_id=job["job_id"], training_index_digest=sha256(job["indices_file"]))
    fit_seconds = time.perf_counter() - started
    if history:
        save_csv(directory / "training_history.csv", history)
    probe_ids = train_ids[:min(128, len(train_ids))]
    reference = predict_bundle(bundle, x[probe_ids])
    save_bundle(directory, bundle)
    del bundle
    gc.collect()
    restored = load_bundle(directory, "cpu", job["threads"])
    actual = predict_bundle(restored, x[probe_ids])
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-4)
    save_json(directory / "reload_training_probe.json", {
        "passed": True, "purpose": "Serialization check on training objects, not an evaluation score",
        "n": len(probe_ids), "max_absolute_difference": float(np.max(abs(actual-reference)))})
    result = {"fit_seconds": fit_seconds, "budget": restored["meta"].get("budget"), "train_n": len(train_ids)}
    if eval_ids is not None:
        start_pred = time.perf_counter()
        pred = predict_bundle(restored, x[eval_ids])
        result.update(predict_seconds=time.perf_counter()-start_pred, **metrics(y[eval_ids], pred))
        np.savez_compressed(directory / "outer_predictions.npz", row=data["row"][eval_ids], y=y[eval_ids], pred=pred)
    if job["stage"] == "cv":
        # CV keeps auditable predictions/history, not 156 redundant serialized models.
        del restored
        gc.collect()
        for filename in ["model.pt", "model.ubj", "model.joblib", "preprocessor.joblib"]:
            (directory / filename).unlink(missing_ok=True)
    save_json(directory / "result.json", result)
    save_json(directory / "complete.json", seal({"job_id": job["job_id"], "artifacts": artifact_hashes(directory)}))
    print(f"Completed {directory.name}: fit={fit_seconds:.1f}s"
          + (f" outer_MAE={result['mae']:.5f}" if "mae" in result else ""), flush=True)


def check_environment(device="cuda"):
    from .common import environment
    rng = np.random.default_rng(42)
    x = rng.normal(size=(256, 10)).astype(np.float32)
    y = (x[:, 0] + .5 * x[:, 1] + 2).astype(np.float32)
    params = {"hidden": [32, 16], "dropout": 0., "lr": .01, "weight_decay": .0001,
              "batch_size": 128, "max_epochs": 15, "patience": 3}
    bundle, history = train_arrays("mlp", x, y, params, 42, 4, device, fixed_steps=15)
    if history[-1]["train_rmse_dropout_on"] >= history[0]["train_rmse_dropout_on"]:
        raise RuntimeError("Synthetic training did not reduce loss")
    print(__import__("json").dumps({"environment": environment(), "device": device,
          "synthetic_training_passed": True, "final_rmse": metrics(y, predict_bundle(bundle, x))["rmse"]}, indent=2))
