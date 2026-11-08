#!/usr/bin/env python3
"""SDSS photo-z V2 entry point. All model training is performed in child processes."""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--v1-data", required=True)
    prepare.add_argument("--raw", required=True)
    prepare.add_argument("--output", default="data/v2")
    prepare.add_argument("--test-size", type=int, default=30000)
    prepare.add_argument("--seed", type=int, default=20261107)
    prepare.add_argument("--technical-check", action="store_true")
    check = commands.add_parser("check-env")
    check.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    for name in ["cv", "train", "fit", "shift-fit", "freeze", "evaluate", "verify", "bundle"]:
        command = commands.add_parser(name)
        command.add_argument("--data", default="data/v2")
        command.add_argument("--run", default="runs/full_cuda")
        if name in ["cv", "train"]:
            command.add_argument("--config", default="configs/full.json")
            command.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
            command.add_argument("--threads", type=int, default=8)
        if name == "bundle":
            command.add_argument("--output", default="sdss-photoz-v2-results.zip")
    predict = commands.add_parser("predict")
    predict.add_argument("--run", default="runs/full_cuda")
    predict.add_argument("--model", default="primary")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", default="predictions_new.csv")
    worker = commands.add_parser("_fit", help=argparse.SUPPRESS)
    worker.add_argument("--job", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from photoz2.data import prepare
        prepare(args.v1_data, args.raw, args.output, args.test_size, args.seed, args.technical_check)
    elif args.command == "check-env":
        from photoz2.models import check_environment
        check_environment(args.device)
    elif args.command == "_fit":
        from photoz2.models import execute_job
        execute_job(args.job)
    elif args.command in ["cv", "train"]:
        from photoz2.experiment import cv, fit, shift_fit, freeze, frozen_plan
        if args.command == "train" and (Path(args.run) / "frozen.json").exists():
            from photoz2.common import read_json, read_sealed
            previous = read_sealed(Path(args.run) / "run.json", "run_id")
            if previous["config"] != read_json(args.config) or previous["device"] != args.device or previous["threads"] != args.threads:
                raise RuntimeError("Requested settings differ from the frozen run; nothing has been retrained.")
            frozen_plan(args.data, args.run)
            print("Training already complete and frozen; continue with evaluate / verify.")
            return
        cv(args.data, args.run, args.config, args.device, args.threads)
        if args.command == "train":
            fit(args.data, args.run)
            shift_fit(args.data, args.run)
            freeze(args.data, args.run)
    elif args.command in ["fit", "shift-fit", "freeze"]:
        from photoz2 import experiment
        getattr(experiment, args.command.replace("-", "_"))(args.data, args.run)
    elif args.command in ["evaluate", "verify"]:
        from photoz2 import evaluation
        getattr(evaluation, args.command)(args.data, args.run)
    elif args.command == "predict":
        from photoz2.inference import predict_csv
        predict_csv(args.run, args.model, args.input, args.output)
    elif args.command == "bundle":
        from photoz2.inference import bundle
        bundle(args.data, args.run, args.output)


if __name__ == "__main__":
    main()
