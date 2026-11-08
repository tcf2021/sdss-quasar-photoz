# Reproducing the experiment

## Scope

The recorded formal run used Python 3.11.18 on Windows, PyTorch 2.9.1+cu130 and the exact package versions in results/run.json. Preserve the source and configurations when comparing with that run. Floating-point equality across devices is not guaranteed. A fresh training run is intentionally not performed while packaging this repository.

This repository has no pretrained weights or processed datasets. Reading the included reports and running unit tests needs neither. Full reproduction downloads a 1.7 GB public FITS catalog and trains all models; allow substantial disk space and compute time.

## Install an isolated environment

Use Python 3.11. Install requirements.txt in an isolated virtual environment. Install the PyTorch build appropriate to your device from the official PyTorch installation instructions. For exact environment matching, use the recorded 2.9.1+cu130 build where supported. The original run_windows.ps1 can reuse an existing environment via -Python; it does not install Torch.

From the repository root, with that environment active:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python v2.py check-env --device cuda
```

Use --device cpu if appropriate, with a separate run directory. check-env performs actual optimizer updates on synthetic data; it is not a scientific result.

## Option A: Reuse the original local V1 artifacts

Keep the original training directories intact. Pass the existing V1 root and Python explicitly to the wrapper, for example using variables for their actual paths:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage prepare -V1Root $V1Root -Python $Python
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage smoke -Device cuda -V1Root $V1Root -Python $Python
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Stage all -Device cuda -V1Root $V1Root -Python $Python
```

Set $V1Root to the original V1 project directory and $Python to its usable python.exe. Do not point V1Root at the V2 directory or at an empty copy. The wrapper's historical default expects a sibling named sdss-photoz, so do not rely on that default after cloning this repository.

## Option B: Rebuild from public data

The legacy/v1 directory contains the actual V1 code needed to recreate the original 200,000-object split. No V1 model training or V1 test evaluation is necessary. Run the following from the repository root with your environment active:

```powershell
python legacy/v1/project.py download --output data/raw/DR16Q_v4.fits
python legacy/v1/project.py prepare --raw data/raw/DR16Q_v4.fits --output data/v1 --max-samples 200000 --seed 42
python v2.py prepare --v1-data data/v1 --raw data/raw/DR16Q_v4.fits --output data/v2 --test-size 30000 --seed 20261107
python v2.py cv --data data/v2 --run runs/smoke_cuda --config configs/smoke.json --device cuda --threads 8
python v2.py train --data data/v2 --run runs/full_cuda --config configs/full.json --device cuda --threads 8
python v2.py evaluate --data data/v2 --run runs/full_cuda
python v2.py verify --data data/v2 --run runs/full_cuda
```

The raw catalog SHA256 recorded in the actual experiment is:

```text
8dfc18a580dc137ca9a729ea7febc491aff61eea4953b91fc480ef7e85b70f6e
```

Compare the generated manifests with results/v1_dataset_manifest.json and results/dataset_manifest.json. NPZ byte hashes can depend on serialization details; distinguish byte identity from equality of object IDs, feature arrays and splits. Never edit frozen manifests to force a match.

Once evaluate opens the new holdout, do not tune further against it or remove TEST_OPENED.json. A resumed identical frozen experiment may be evaluated or verified again without retraining.

## Inference

After training or restoring a complete trusted result bundle, use:

```powershell
python v2.py predict --run runs/full_cuda --model primary --input new_objects.csv --output predictions_new.csv
```

The CSV contract is documented in photoz2/inference.py and the original experiment guide. The repository's results directory is a small reporting snapshot, not a substitute for runs/full_cuda with its weights and preprocessors. Load joblib/PyTorch artifacts only from trusted sources.
