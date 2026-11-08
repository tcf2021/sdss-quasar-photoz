from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBRegressor

from .common import FEATURES, metrics, read_json, save_csv, save_json


def device_for(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False; run check-env")
    return requested


def set_seed(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def torch_predict(model, x, device, ymean, ystd, batch=8192):
    model.eval()
    pred = []
    for start in range(0, len(x), batch):
        xb = torch.as_tensor(x[start:start+batch], dtype=torch.float32, device=device)
        pred.append((model(xb) * ystd + ymean).cpu().numpy())
    return np.concatenate(pred) if pred else np.empty(0, dtype=np.float32)


def train_model(directory, family, feature, params, train, val, seed, device, threads=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(seed, min(threads, 8))
    start = time.perf_counter()
    ytrain, yval = train["y"], val["y"]
    meta = {"family": family, "feature": feature, "params": params, "seed": seed,
            "feature_names": FEATURES.get(feature, []), "prediction_clipping": "none"}
    history = []
    if family == "constant":
        statistic = params["statistic"]
        value = float(np.median(ytrain) if statistic == "median" else np.mean(ytrain))
        meta["value"] = value
        pred = np.full(len(yval), value, dtype=np.float32)
        meta["device"] = "cpu"
    else:
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if family == "mlp":
            steps.append(("scaler", StandardScaler()))
        prep = Pipeline(steps)
        xtrain = prep.fit_transform(train[feature]).astype(np.float32)
        xval = prep.transform(val[feature]).astype(np.float32)
        joblib.dump(prep, directory / "preprocessor.joblib")
        if family == "xgb":
            model = XGBRegressor(
                objective="reg:squarederror", eval_metric="mae", tree_method="hist", device="cpu",
                n_jobs=threads, random_state=seed, **params,
            )
            model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xval, yval)], verbose=100)
            # sklearn predict automatically respects best_iteration, including after load_model.
            pred = model.predict(xval)
            model.save_model(directory / "model.ubj")
            values = model.evals_result()
            for i, (t, v) in enumerate(zip(values["validation_0"]["mae"], values["validation_1"]["mae"])):
                history.append({"step": i+1, "train_mae": t, "val_mae": v})
            meta.update(best_iteration=int(model.best_iteration), device="cpu")
            save_csv(directory / "feature_importance.csv", [
                {"feature": f, "importance_gain": float(v)}
                for f, v in sorted(zip(FEATURES[feature], model.feature_importances_), key=lambda x: -x[1])
            ])
        elif family == "mlp":
            device = device_for(device)
            ymean, ystd = float(ytrain.mean()), max(float(ytrain.std()), 1e-6)
            meta.update(ymean=ymean, ystd=ystd, input_dim=xtrain.shape[1], device=device)
            model = MLP(xtrain.shape[1], params["hidden"], params["dropout"]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
            # This tabular dataset fits easily in memory. Explicit tensor batching avoids DataLoader overhead on WSL.
            xt = torch.as_tensor(xtrain, device=device)
            yt = torch.as_tensor((ytrain-ymean)/ystd, device=device)
            best, best_state, stale, best_epoch = float("inf"), None, 0, 0
            for epoch in range(1, params["epochs"]+1):
                model.train()
                order = torch.randperm(len(yt), device=device)
                total = 0.0
                for idx in order.split(params["batch_size"]):
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite MLP loss; inspect input data and learning rate")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total += loss.item() * len(idx)
                pred = torch_predict(model, xval, device, ymean, ystd)
                score = metrics(yval, pred)["mae"]
                history.append({"step": epoch, "train_rmse": (total/len(yt))**.5 * ystd, "val_mae": score})
                # No test access and no test-based checkpoint selection.
                if score < best:
                    best, stale, best_epoch = score, 0, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(best_state, directory / "model.pt")
                else:
                    stale += 1
                print(f"epoch={epoch:03d} train_rmse={history[-1]['train_rmse']:.5f} val_mae={score:.5f} best={best:.5f}", flush=True)
                save_csv(directory / "history.csv", history)
                if stale >= params["patience"]:
                    break
            if best_state is None:
                raise RuntimeError("MLP failed to produce a checkpoint")
            model.load_state_dict(best_state)
            pred = torch_predict(model, xval, device, ymean, ystd)
            meta["best_epoch"] = best_epoch
        else:
            raise ValueError(f"Unknown model family {family}")
    meta["train_seconds"] = time.perf_counter() - start
    meta["train_n"], meta["val_n"] = int(len(ytrain)), int(len(yval))
    save_json(directory / "meta.json", meta)
    if history:
        save_csv(directory / "history.csv", history)
    score = metrics(yval, pred)
    save_json(directory / "validation_metrics.json", score)
    np.savez_compressed(directory / "validation_predictions.npz", row=val["row"], y=yval, pred=pred)
    # Validate the saved artifacts, rather than relying only on in-memory objects.
    reloaded = predict_model(directory, val, device="cpu")
    np.testing.assert_allclose(pred, reloaded, rtol=1e-5, atol=1e-4)
    save_json(directory / "reload_validation.json", {"passed": True, "max_absolute_difference": float(np.max(np.abs(pred-reloaded)))})
    return score, meta


def predict_model(directory, data, device="cpu"):
    directory = Path(directory)
    meta = read_json(directory / "meta.json")
    family = meta["family"]
    n = len(data["mag"])
    if family == "constant":
        return np.full(n, meta["value"], dtype=np.float32)
    feature = meta["feature"]
    if meta["feature_names"] != FEATURES[feature]:
        raise ValueError("Feature schema mismatch")
    prep = joblib.load(directory / "preprocessor.joblib")
    x = prep.transform(data[feature]).astype(np.float32)
    if family == "xgb":
        model = XGBRegressor()
        model.load_model(directory / "model.ubj")
        model.set_params(device="cpu", n_jobs=4)
        return model.predict(x)
    if family == "mlp":
        device = device_for(device)
        model = MLP(meta["input_dim"], meta["params"]["hidden"], meta["params"]["dropout"])
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        model.to(device)
        return torch_predict(model, x, device, meta["ymean"], meta["ystd"])
    raise ValueError(f"Unknown family {family}")


def check_environment(requested="cuda"):
    from .common import environment
    device = device_for(requested)
    set_seed(42)
    model = MLP(14, [32, 16], 0).to(device)
    x = torch.randn(256, 14, device=device)
    y = x[:, 0] * .5 + x[:, 1]
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    initial = nn.functional.mse_loss(model(x), y).item()
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    final = nn.functional.mse_loss(model(x), y).item()
    if not np.isfinite(final) or final >= initial:
        raise RuntimeError("Actual optimizer-step check did not decrease the synthetic loss")
    result = {**environment(), "device": device, "torch_cuda_build": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "synthetic_optimizer_check": {"initial_loss": initial, "final_loss": final, "passed": True}}
    print(__import__("json").dumps(result, indent=2), flush=True)
    return result
