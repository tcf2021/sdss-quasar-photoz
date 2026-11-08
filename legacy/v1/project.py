#!/usr/bin/env python3
"""Run from the project root. python project.py --help"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="SDSS DR16Q photometric-redshift pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    env = sub.add_parser("check-env", help="Run actual synthetic optimizer steps on the requested device")
    env.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    dl = sub.add_parser("download", help="Download the official DR16Q_v4 with resume and integrity checks")
    dl.add_argument("--output", default="data/raw/DR16Q_v4.fits")
    prep = sub.add_parser("prepare", help="Clean, deduplicate, sample, and freeze the split")
    prep.add_argument("--raw", default="data/raw/DR16Q_v4.fits")
    prep.add_argument("--output", default="data/processed")
    prep.add_argument("--max-samples", type=int, default=200000)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--technical-check", action="store_true", help="Allow non-full catalog ONLY for technical tests; marks output accordingly")
    train = sub.add_parser("train", help="Train on train/val only; select configurations by validation MAE")
    train.add_argument("--config", default="configs/full.json")
    train.add_argument("--threads", type=int, default=8)
    for name in ["evaluate", "verify"]:
        sub.add_parser(name, help={"evaluate": "Evaluate frozen selected models on test and produce figures", "verify": "Reload models and compare to saved test predictions"}[name])
    for name in ["train", "evaluate", "verify"]:
        cmd = sub.choices[name]
        cmd.add_argument("--data", default="data/processed")
        cmd.add_argument("--run", default="runs/full")
        cmd.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda" if name == "train" else "cpu")
    pred = sub.add_parser("predict", help="Predict new objects from raw photometry CSV")
    pred.add_argument("--run", default="runs/full")
    pred.add_argument("--model", default="mlp_rich")
    pred.add_argument("--input", required=True)
    pred.add_argument("--output", default="runs/new_predictions.csv")
    pred.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cpu")
    a = p.parse_args()
    if a.command == "download":
        from photoz.data import download
        download(a.output)
    elif a.command == "prepare":
        from photoz.data import prepare
        prepare(a.raw, a.output, a.max_samples, a.seed, a.technical_check)
    elif a.command == "check-env":
        from photoz.models import check_environment
        check_environment(a.device)
    elif a.command == "train":
        from photoz.experiments import train
        train(a.data, a.run, a.config, a.device, a.threads)
    else:
        from photoz.evaluation import evaluate, verify, predict_csv
        if a.command == "evaluate":
            evaluate(a.data, a.run, a.device)
        elif a.command == "verify":
            verify(a.data, a.run, a.device)
        elif a.command == "predict":
            predict_csv(a.run, a.model, a.input, a.output, a.device)


if __name__ == "__main__":
    main()
