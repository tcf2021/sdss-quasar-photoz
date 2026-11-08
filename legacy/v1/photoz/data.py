from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from astropy.io import fits

from .common import BANDS, FEATURES, digest, save_csv, save_json, sha256

URL = "https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits"
OFFICIAL_BYTES = 1718484480
REQUIRED = ["SDSS_NAME", "THING_ID", "Z", "IS_QSO_FINAL", "PSFMAG", "PSFMAGERR", "EXTINCTION"]


def validate_fits(path, official=False):
    path = Path(path)
    if official and path.stat().st_size != OFFICIAL_BYTES:
        raise ValueError(f"Expected {OFFICIAL_BYTES} bytes; got {path.stat().st_size}")
    with fits.open(path, memmap=True) as h:
        h.verify("exception")
        if len(h) < 2 or not isinstance(h[1], fits.BinTableHDU):
            raise ValueError("Expected a FITS binary table in HDU 1")
        missing = sorted(set(REQUIRED) - set(h[1].columns.names))
        if missing:
            raise ValueError(f"Missing DR16Q columns: {missing}")
        if official and h[1].header["NAXIS2"] != 750414:
            raise ValueError("Not the expected complete DR16Q_v4 catalog")
        for name in ["PSFMAG", "PSFMAGERR", "EXTINCTION"]:
            if h[1].data[name].shape[1:] != (5,):
                raise ValueError(f"{name} must have five entries in u,g,r,i,z order")
        if len(h[1].data):
            _ = h[1].data["Z"][-1]  # Catch a truncated data section.


def download(output):
    import requests
    from tqdm import tqdm

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        validate_fits(output, official=True)
        print(f"Complete FITS already exists: {output}", flush=True)
    else:
        part = output.with_suffix(output.suffix + ".part")
        start = part.stat().st_size if part.exists() else 0
        if start > OFFICIAL_BYTES:
            raise ValueError(f"Partial file is too large. Move/remove {part} and retry")
        if start < OFFICIAL_BYTES:
            headers = {"Accept-Encoding": "identity"}
            if start:
                headers["Range"] = f"bytes={start}-"
            with requests.get(URL, headers=headers, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                if r.status_code == 206:
                    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", ""))
                    if not m or int(m[1]) != start or int(m[3]) != OFFICIAL_BYTES:
                        raise RuntimeError("Unexpected Content-Range; partial file preserved")
                elif r.status_code == 200:
                    start = 0  # A server ignoring Range must be overwritten, never appended.
                else:
                    raise RuntimeError(f"Unexpected HTTP status {r.status_code}")
                mode = "ab" if start else "wb"
                with part.open(mode) as f, tqdm(total=OFFICIAL_BYTES, initial=start, unit="B", unit_scale=True) as bar:
                    for block in r.iter_content(1024 * 1024):
                        if block:
                            f.write(block)
                            bar.update(len(block))
        validate_fits(part, official=True)
        part.replace(output)
    local_hash = sha256(output)
    save_json(output.with_suffix(".download.json"), {
        "url": URL, "bytes": output.stat().st_size, "sha256": local_hash,
        "sha256_note": "Locally computed provenance hash, not an independently published SDSS checksum.",
    })
    print(f"Validated catalog; SHA256={local_hash}", flush=True)


def photometry(mag, err, extinction):
    """Shared, label-independent cleaning for both training and prediction."""
    mag, err, extinction = [np.array(x, dtype=np.float32, copy=True) for x in (mag, err, extinction)]
    if mag.ndim != 2 or mag.shape[1] != 5 or err.shape != mag.shape or extinction.shape != mag.shape:
        raise ValueError("Photometry must be three arrays with shape (N,5)")
    mag[~np.isfinite(mag) | (mag <= 0) | (mag >= 40)] = np.nan
    err[~np.isfinite(err) | (err <= 0) | (err >= 10)] = np.nan
    extinction[~np.isfinite(extinction) | (extinction < 0) | (extinction >= 10)] = np.nan
    mag0 = mag - extinction
    colors = mag0[:, :-1] - mag0[:, 1:]
    rich = np.concatenate([mag0, colors, err], axis=1)
    keep = np.isfinite(mag0).sum(axis=1) >= 3
    return {"mag": mag0, "rich": rich}, keep


def unique_objects(names, things):
    """One record per connected component sharing name or positive THING_ID."""
    n = len(names)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    seen_names, seen_things = {}, {}
    for i, (name, thing) in enumerate(zip(names, things)):
        for seen, key, usable in [(seen_names, str(name), bool(str(name).strip())),
                                  (seen_things, int(thing), thing > 0)]:
            if usable:
                if key in seen:
                    a, b = root(i), root(seen[key])
                    parent[max(a, b)] = min(a, b)
                else:
                    seen[key] = i
    return np.array([i for i in range(n) if root(i) == i], dtype=np.int64)


def prepare(raw, output, max_samples=200000, seed=42, technical=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; preserve the split or use a new directory")
    validate_fits(raw, official=not technical)
    with fits.open(raw, memmap=True) as h:
        t = h[1].data
        n_catalog = len(t)
        mags, errs, ext = [np.array(t[k], dtype=np.float32) for k in ["PSFMAG", "PSFMAGERR", "EXTINCTION"]]
        y = np.array(t["Z"], dtype=np.float32)
        qso = np.array(t["IS_QSO_FINAL"])
        names = np.char.strip(np.asarray(t["SDSS_NAME"]).astype("U32"))
        things = np.array(t["THING_ID"], dtype=np.int64)
        source = np.asarray(t["SOURCE_Z"]).astype("U24") if "SOURCE_Z" in t.names else np.full(n_catalog, "unknown")
    feat, photo_ok = photometry(mags, errs, ext)
    mask_qso = (qso == 1)
    mask_z = np.isfinite(y) & (y > 0) & (y <= 7)
    mask_name = np.char.str_len(names) > 0
    eligible = np.flatnonzero(mask_qso & mask_z & mask_name & photo_ok)
    n_eligible = len(eligible)
    eligible = eligible[unique_objects(names[eligible], things[eligible])]
    n_unique = len(eligible)
    if n_unique < 100:
        raise ValueError(f"Only {n_unique} valid unique objects; inspect the input")
    if max_samples <= 0:
        raise ValueError("max-samples must be positive")
    rng = np.random.default_rng(seed)
    selected = rng.permutation(eligible)[:max_samples]
    n = len(selected)
    ntrain, nval = int(n * .7), int(n * .15)
    indices = {"train": selected[:ntrain], "val": selected[ntrain:ntrain+nval], "test": selected[ntrain+nval:]}
    output.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    for split, ids in indices.items():
        p = output / f"{split}.npz"
        np.savez_compressed(p, mag=feat["mag"][ids], rich=feat["rich"][ids], y=y[ids],
                            names=names[ids], thing_id=things[ids], row=ids, source_z=source[ids])
        split_hashes[split] = sha256(p)
    counts = {k: int(len(v)) for k, v in indices.items()}
    manifest = {
        "schema_version": 1, "purpose": "TECHNICAL_CHECK_ONLY" if technical else "FORMAL_EXPERIMENT",
        "source": URL, "raw_name": Path(raw).name, "raw_bytes": Path(raw).stat().st_size,
        "raw_sha256": sha256(raw), "seed": seed, "requested_max_samples": max_samples,
        "n_catalog": n_catalog, "n_after_joint_filter": n_eligible, "n_after_dedup": n_unique,
        "n_selected": n, "split_counts": counts, "files": split_hashes, "feature_names": FEATURES,
        "individual_rejection_counts_over_catalog": {
            "not_IS_QSO_FINAL_1": int((~mask_qso).sum()), "invalid_or_outside_0_z_7": int((~mask_z).sum()),
            "blank_name": int((~mask_name).sum()), "fewer_than_3_valid_mags": int((~photo_ok).sum()),
        },
        "missing_features_selected": {
            f: dict(zip(FEATURES[f], [int(v) for v in np.isnan(feat[f][selected]).sum(axis=0)])) for f in FEATURES
        },
        "rules": {
            "label": "Z; 0 < Z <= 7; IS_QSO_FINAL == 1",
            "magnitudes": "PSFMAG - EXTINCTION; before subtraction require 0<mag<40 and 0<=extinction<10",
            "errors": "PSFMAGERR is magnitude error, not inverse variance; require 0<err<10; otherwise NaN",
            "missing": "Retain rows with >=3 finite corrected bands; impute only during training",
            "duplicates": "Connected components by nonblank SDSS_NAME or positive THING_ID; keep first eligible row",
            "split": "Seeded random 70/15/15; fixed before pilot training; no spatial/external holdout",
            "pipeline_warning": "No universal ZWARNING cut: primary Z can be visually corrected",
        },
    }
    manifest["dataset_id"] = digest(manifest)
    save_json(output / "manifest.json", manifest)
    # Actual training objects for a reproducible inference example; no labels.
    examples = []
    for i in indices["train"][:10]:
        r = {"object_id": str(names[i])}
        for prefix, array in [("psfmag", mags), ("psfmagerr", errs), ("extinction", ext)]:
            r.update({f"{prefix}_{b}": float(array[i, j]) for j, b in enumerate(BANDS)})
        examples.append(r)
    save_csv(output / "example_inputs.csv", examples)
    print(f"Prepared {n:,} objects; {counts}; dataset_id={manifest['dataset_id']}", flush=True)
    return manifest
