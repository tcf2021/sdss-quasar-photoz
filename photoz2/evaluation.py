from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np

from .common import (METRICS, dataset_manifest, environment, features, load_npz, metrics,
                     read_json, read_sealed, save_csv, save_json, sha256)
from .experiment import frozen_plan, get_run, verify_model_entry
from .models import load_bundle, predict_bundle


def bin_metrics(test, pred, name):
    rows = []
    for axis, values, edges in [
        ("z_spec", test["y"], [0, .5, 1, 1.5, 2, 2.5, 3, 4, 7]),
        ("i_mag0", test["mag"][:, 3], [0, 18, 19, 20, 21, 22, 40]),
    ]:
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            mask = np.isfinite(values) & (values >= low) & ((values <= high) if index == len(edges)-2 else (values < high))
            if mask.any():
                rows.append({"model": name, "axis": axis, "low": low, "high": high, **metrics(test["y"][mask], pred[mask])})
    return rows


def evaluate(data_dir, run_dir):
    data_dir, run_dir, manifest, run = get_run(data_dir, run_dir)
    frozen = frozen_plan(data_dir, run_dir)
    marker = data_dir / "TEST_OPENED.json"
    if marker.exists() and read_json(marker)["freeze_id"] != frozen["freeze_id"]:
        raise RuntimeError("The new test was already opened by another frozen selection")
    if not marker.exists():
        save_json(marker, {"freeze_id": frozen["freeze_id"], "run_id": run["run_id"],
                          "note": "All V2 training is now frozen. Only this selection may be reevaluated."})
    dataset_manifest(data_dir, ("test",))
    test = load_npz(data_dir / "test.npz")
    report = run_dir / "report"
    report.mkdir(exist_ok=True)
    selected = read_sealed(run_dir / "cv_selection.json", "selection_id")
    final = read_sealed(run_dir / "final_models.json", "models_id")
    shift = read_sealed(run_dir / "shift_models.json", "shift_id")
    predictions, rows, binned = {}, [], []
    for name, entry in final["models"].items():
        directory = verify_model_entry(run_dir, entry)
        bundle = load_bundle(directory, "cpu", run["threads"])
        x = test["mag"] if entry["feature"] == "none" else features(test, entry["feature"])
        pred = predict_bundle(bundle, x)
        predictions[name] = pred
        rows.append({"model": name, "family": entry["family"], "feature": entry["feature"],
                     "cv_mae_mean": selected["selected"][name]["cv"]["mae_mean"],
                     "cv_mae_std": selected["selected"][name]["cv"]["mae_std"],
                     "negative_prediction_count": int(np.sum(pred < 0)), **metrics(test["y"], pred)})
        binned.extend(bin_metrics(test, pred, name))
        print(f"{name:24s} MAE={rows[-1]['mae']:.5f} outliers={rows[-1]['outlier_rate_015']:.2%}", flush=True)
        del bundle
        gc.collect()
    save_csv(report / "metrics_test.csv", rows)
    save_json(report / "metrics_test.json", rows)
    save_csv(report / "metrics_by_bin.csv", binned)
    save_csv(report / "predictions.csv", [
        {"sdss_name": str(test["names"][i]), "catalog_row": int(test["row"][i]), "z_spec": float(test["y"][i]),
         **{name: float(pred[i]) for name, pred in predictions.items()}} for i in range(len(test["y"]))])
    np.savez_compressed(report / "predictions.npz", row=test["row"], y=test["y"], **predictions)
    faint = np.isfinite(test["mag"][:, 3]) & (test["mag"][:, 3] >= shift["i_threshold"])
    shift_preds, shift_rows = {}, []
    if faint.sum() < 1:
        raise ValueError("No faint test objects; cannot evaluate the predeclared shift experiment")
    for name, entry in shift["models"].items():
        bundle = load_bundle(verify_model_entry(run_dir, entry), "cpu", run["threads"])
        pred = predict_bundle(bundle, features(test, entry["feature"])[faint])
        shift_preds[name] = pred
        shift_rows.append({"model": name, "family": entry["family"], "arm": entry["arm"],
                           "train_n": shift["train_n_per_arm"], "i_threshold": shift["i_threshold"],
                           **metrics(test["y"][faint], pred)})
        del bundle
        gc.collect()
    save_csv(report / "shift_metrics.csv", shift_rows)
    np.savez_compressed(report / "shift_predictions.npz", row=test["row"][faint], y=test["y"][faint], **shift_preds)
    shift_comparison = []
    for family in run["config"]["models"]:
        bright = next(r for r in shift_rows if r["family"] == family and r["arm"] == "bright")
        mixed = next(r for r in shift_rows if r["family"] == family and r["arm"] == "mixed")
        shift_comparison.append({"family": family, "train_n_per_arm": shift["train_n_per_arm"],
                                 "same_faint_test_n": int(faint.sum()), "bright_only_mae": bright["mae"],
                                 "mixed_train_mae": mixed["mae"], "mae_gap_bright_minus_mixed": bright["mae"]-mixed["mae"],
                                 "outlier_gap_percentage_points": 100*(bright["outlier_rate_015"]-mixed["outlier_rate_015"])})
    save_csv(report / "shift_comparison.csv", shift_comparison)
    ablation = []
    for family in run["config"]["models"]:
        baseline = next((r for r in rows if r["family"] == family and r["feature"] == "mag"), None)
        for row in rows:
            if row["family"] == family:
                ablation.append({**row, "mae_change_vs_mag": row["mae"]-baseline["mae"] if baseline else "",
                                 "mae_relative_reduction_vs_mag": 1-row["mae"]/baseline["mae"] if baseline else ""})
    save_csv(report / "feature_ablation.csv", ablation)
    from .reporting import plots, history_plots, case_audit, color_diagram
    plots(report, test, predictions, rows, binned, selected)
    history_plots(run_dir, report, selected)
    dev = load_npz(data_dir / "dev.npz")
    audit = case_audit(report, dev, test, predictions, frozen["overall_best_group"],
                       count=run["config"]["audit_cases"], threads=run["threads"])
    color_diagram(report, dev)
    body = [
        "# SDSS DR16Q V2 实验结果", "",
        f"实验用途：{manifest['purpose']}。",
        f"开发集 {manifest['counts']['dev']:,} 个对象；新的保留测试集 {manifest['counts']['test']:,} 个对象。",
        f"新测试集排除了 V1 的全部 {manifest['v1_all_objects_excluded_from_new_test']:,} 个对象及关联重复身份。",
        f"三折交叉验证使用固定开发子集：{next(iter(selected['selected'].values()))['cv']['cv_objects']:,} 条。",
        f"测试前按开发集 CV MAE 选定的主模型：{frozen['overall_best_group']}。",
        "CV 分数用于开发选模；三折标准差不是置信区间，也不是多次独立训练种子的误差估计。",
        "早停只使用各外层训练折的内部划分；随后按选出的步数在整个外层训练折重训。",
        "最终模型使用全部开发数据，训练步数取入选配置三个折内最佳步数的中位数。", "",
        "| 模型 | CV MAE 均值 | CV 标准差 | 新测试 MAE | RMSE | NMAD | 离群率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        body.append(f"| {row['model']} | {row['cv_mae_mean']:.5f} | {row['cv_mae_std']:.5f} | "
                    f"{row['mae']:.5f} | {row['rmse']:.5f} | {row['nmad']:.5f} | {row['outlier_rate_015']:.2%} |")
    body += ["", "归一化残差 delta=(预测红移−光谱红移)/(1+光谱红移)。离群：|delta|>0.15。",
             "NMAD=1.4826×median(|delta−median(delta)|)。预测未裁剪。",
             "四颜色+i 星等在完整测光时与五星等包含相同信息；表示方式不同。",
             "存在缺失值时，派生特征后分别填补可能破坏这种精确等价关系。", "",
             "## 亮度偏移对照", "",
             f"阈值为消光改正后的 i={shift['i_threshold']}；每个训练臂 {shift['train_n_per_arm']:,} 个对象。",
             f"两臂在完全相同的 {int(faint.sum()):,} 个暗源测试对象上评估。",
             "| 模型 | 仅亮源训练 MAE | 混合亮度训练 MAE | 前者减后者 |", "|---|---:|---:|---:|"]
    for row in shift_comparison:
        body.append(f"| {row['family']} | {row['bright_only_mae']:.5f} | {row['mixed_train_mae']:.5f} | "
                    f"{row['mae_gap_bright_minus_mixed']:.5f} |")
    body += ["", "训练样本量、特征、超参数和最终评估对象保持一致。该实验改变亮度选择，同时也可能改变红移等分布。",
             "## 异常对象", "",
             f"抽取 {audit['outlier_cases']} 个离群案例和 {audit['controls']} 个正常对照。",
             "cases.csv 包含测光、测量误差、标签来源、各模型预测与开发集颜色近邻摘要。",
             "case_neighbors.csv 只包含开发集近邻。邻居红移分散是待调查线索，不是物理原因的证明。",
             "ZWARNING 只作为诊断元数据；不能据此自动否定可能经过人工修正的主红移。", "",
             "## 范围", "",
             "结果仅适用于本次筛选出的同目录类星体分布，不证明跨巡天泛化。",
             "V1 测试结果与 V2 的新测试结果来自不同对象，不能把二者差值当作纯模型提升。",
             "模型文件与预处理器在上级 trials/final_* 和 trials/shift_* 中；报告目录本身不含权重。",
             "reload_check.json 由独立 verify 命令生成；生成报告不等于该命令已执行。", "",
             "![模型比较](comparison.png)", "![主模型预测](best_predictions.png)",
             "![颜色与红移](development_color_redshift.png)"]
    (report / "RESULTS.md").write_text("\n".join(body), encoding="utf-8")
    save_json(report / "evaluation.json", {
        "freeze_id": frozen["freeze_id"], "purpose": manifest["purpose"], "prediction_device": "cpu",
        "environment": environment(), "missing_i_mag": int(np.isnan(test["mag"][:, 3]).sum()), "audit": audit,
        "files": {f: sha256(report / f) for f in ["predictions.npz", "metrics_test.json", "shift_predictions.npz"]}})
    print(f"Evaluation completed: {report}", flush=True)


def verify(data_dir, run_dir):
    data_dir, run_dir, manifest, run = get_run(data_dir, run_dir)
    frozen = frozen_plan(data_dir, run_dir)
    if read_json(data_dir / "TEST_OPENED.json")["freeze_id"] != frozen["freeze_id"]:
        raise ValueError("Wrong test marker")
    report = run_dir / "report"
    evaluation = read_json(report / "evaluation.json")
    if evaluation["freeze_id"] != frozen["freeze_id"]:
        raise ValueError("Wrong evaluation")
    for name, expected in evaluation["files"].items():
        if sha256(report / name) != expected:
            raise ValueError(f"Changed saved evaluation: {name}")
    dataset_manifest(data_dir, ("test",))
    test = load_npz(data_dir / "test.npz")
    final = read_sealed(run_dir / "final_models.json", "models_id")
    shift = read_sealed(run_dir / "shift_models.json", "shift_id")
    final_metrics = {row["model"]: row for row in read_json(report / "metrics_test.json")}
    checks = {}
    for stage, records, mask, filename in [
        ("final", final["models"], np.ones(len(test["y"]), dtype=bool), "predictions.npz"),
        ("shift", shift["models"], np.isfinite(test["mag"][:, 3]) & (test["mag"][:, 3] >= shift["i_threshold"]), "shift_predictions.npz"),
    ]:
        with np.load(report / filename, allow_pickle=False) as old:
            np.testing.assert_array_equal(old["row"], test["row"][mask])
            np.testing.assert_array_equal(old["y"], test["y"][mask])
            for name, entry in records.items():
                bundle = load_bundle(verify_model_entry(run_dir, entry), "cpu", run["threads"])
                x = test["mag"] if entry["feature"] == "none" else features(test, entry["feature"])
                pred = predict_bundle(bundle, x[mask])
                np.testing.assert_allclose(pred, old[name], rtol=1e-5, atol=1e-4)
                actual = metrics(test["y"][mask], pred)
                expected = final_metrics[name] if stage == "final" else metrics(old["y"], old[name])
                for key in METRICS:
                    tolerance = 1/len(pred)+1e-12 if key == "outlier_rate_015" else 1e-5
                    np.testing.assert_allclose(actual[key], expected[key], rtol=1e-5, atol=tolerance)
                checks[f"{stage}_{name}"] = {"max_absolute_difference": float(np.max(abs(pred-old[name]))), "passed": True}
                del bundle
                gc.collect()
    save_json(report / "reload_check.json", {"passed": True, "freeze_id": frozen["freeze_id"], "models": checks})
    print("PASS: all final and shift models reproduce saved predictions and metrics.", flush=True)
