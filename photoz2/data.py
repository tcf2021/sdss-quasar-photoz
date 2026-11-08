from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import FEATURES, dataset_manifest, digest, load_npz, read_json, save_json, seal, sha256

CATALOG_URL = "https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits"
OFFICIAL_BYTES = 1718484480


def clean_photometry(mag, err, extinction):
    mag, err, extinction = [np.array(x, dtype=np.float32, copy=True) for x in (mag, err, extinction)]
    if mag.ndim != 2 or mag.shape[1] != 5 or mag.shape != err.shape or mag.shape != extinction.shape:
        raise ValueError("Expected three (N,5) photometry arrays")
    mag[~np.isfinite(mag) | (mag <= 0) | (mag >= 40)] = np.nan
    extinction[~np.isfinite(extinction) | (extinction < 0) | (extinction >= 10)] = np.nan
    err[~np.isfinite(err) | (err <= 0) | (err >= 10)] = np.nan
    corrected = mag - extinction
    return corrected, err, np.isfinite(corrected).sum(axis=1) >= 3


def components(names, things):
    """Connected identity components, including links through ineligible rows."""
    parent = np.arange(len(names), dtype=np.int64)

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
    return np.array([root(i) for i in range(len(parent))], dtype=np.int64)


def select_holdout(eligible, component, old_rows, test_size, seed):
    blocked = np.unique(component[old_rows])
    available = eligible[~np.isin(component[eligible], blocked)]
    _, first = np.unique(component[available], return_index=True)
    available = available[np.sort(first)]
    if len(available) < test_size:
        raise ValueError(f"Only {len(available)} genuinely unused objects; need {test_size}")
    chosen = np.random.default_rng(seed).permutation(available)[:test_size]
    assert not np.intersect1d(component[chosen], blocked).size
    return chosen, len(available)


def prepare(v1_dir, raw, output, test_size=30000, seed=20261107, technical=False):
    """V1 train+val form development; ALL V1 objects are excluded from new test."""
    from astropy.io import fits

    v1_dir, raw, output = Path(v1_dir), Path(raw), Path(output)
    old = read_json(v1_dir / "manifest.json")
    if old.get("schema_version") != 1:
        raise ValueError("Expected the original V1 processed-data manifest")
    if not technical and old.get("purpose") != "FORMAL_EXPERIMENT":
        raise ValueError("Formal V2 cannot use a technical V1 dataset")
    if test_size < 20:
        raise ValueError("Need at least 20 new holdout objects")
    if not technical and raw.stat().st_size != OFFICIAL_BYTES:
        raise ValueError("Expected complete official DR16Q_v4.fits")
    raw_hash = sha256(raw)
    if raw_hash != old["raw_sha256"]:
        raise ValueError("Raw catalog differs from the exact snapshot used by V1")
    contract = {"v1_dataset_id": old["dataset_id"], "raw_sha256": raw_hash,
                "test_size": test_size, "seed": seed, "technical": technical}
    if (output / "manifest.json").exists():
        current = dataset_manifest(output, ("dev", "test"))
        if current["prepare_contract"] != contract:
            raise ValueError("Existing V2 data uses different preparation settings; keep it unchanged")
        print(f"Reusing frozen V2 dataset: {current['dataset_id']}", flush=True)
        return current
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty output without a valid manifest: {output}")
    old_rows = []
    old_dev = []
    for split in ["train", "val", "test"]:
        path = v1_dir / f"{split}.npz"
        if sha256(path) != old["files"][split]:
            raise ValueError(f"V1 split changed: {split}")
        with np.load(path, allow_pickle=False) as f:
            ids = f["row"].astype(np.int64)
            old_rows.append(ids)
            if split != "test":
                old_dev.append({k: f[k] for k in ["row", "mag", "rich", "y", "names", "thing_id"]})
    old_rows = np.concatenate(old_rows)
    if len(np.unique(old_rows)) != len(old_rows):
        raise ValueError("V1 contains overlapping catalog rows")
    dev_rows = np.concatenate([x["row"] for x in old_dev]).astype(np.int64)
    with fits.open(raw, memmap=True) as hdus:
        hdus.verify("exception")
        table = hdus[1].data
        if not technical and len(table) != 750414:
            raise ValueError("Unexpected official catalog row count")
        mag, err, photo_ok = clean_photometry(table["PSFMAG"], table["PSFMAGERR"], table["EXTINCTION"])
        y = np.asarray(table["Z"], dtype=np.float32)
        names = np.char.strip(np.asarray(table["SDSS_NAME"]).astype("U32"))
        things = np.asarray(table["THING_ID"], dtype=np.int64)
        qso = np.asarray(table["IS_QSO_FINAL"])
        source = np.asarray(table["SOURCE_Z"]).astype("U24") if "SOURCE_Z" in table.names else np.full(len(y), "unknown")
        warning = np.asarray(table["ZWARNING"], dtype=np.int64) if "ZWARNING" in table.names else np.full(len(y), -1, dtype=np.int64)
    if old_rows.min() < 0 or old_rows.max() >= len(y):
        raise ValueError("V1 row index outside the catalog")
    component = components(names, things)
    if len(np.unique(component[old_rows])) != len(old_rows):
        raise ValueError("Cross-split identity duplicates found in V1; resolve the duplicate groups first")
    for previous in old_dev:
        ids = previous["row"]
        np.testing.assert_array_equal(previous["names"], names[ids])
        np.testing.assert_array_equal(previous["thing_id"], things[ids])
        np.testing.assert_array_equal(previous["y"], y[ids])
        np.testing.assert_allclose(previous["mag"], mag[ids], equal_nan=True, rtol=0, atol=0)
        np.testing.assert_allclose(previous["rich"][:, -5:], err[ids], equal_nan=True, rtol=0, atol=0)
    eligible = np.flatnonzero((qso == 1) & np.isfinite(y) & (y > 0) & (y <= 7)
                              & (np.char.str_len(names) > 0) & photo_ok)
    test_rows, unused_count = select_holdout(eligible, component, old_rows, test_size, seed)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, ids in [("dev", dev_rows), ("test", test_rows)]:
        path = output / f"{split}.npz"
        np.savez_compressed(path, mag=mag[ids], err=err[ids], y=y[ids],
                            names=names[ids], thing_id=things[ids], row=ids,
                            source_z=source[ids], z_warning=warning[ids])
        hashes[split] = sha256(path)
    manifest = seal({
        "schema_version": 2,
        "purpose": "TECHNICAL_CHECK_ONLY" if technical else "FORMAL_EXPERIMENT",
        "prepare_contract": contract, "source": CATALOG_URL,
        "v1_manifest_sha256": sha256(v1_dir / "manifest.json"),
        "v1_all_objects_excluded_from_new_test": int(len(old_rows)),
        "unused_eligible_unique_pool": int(unused_count),
        "counts": {"dev": int(len(dev_rows)), "test": int(len(test_rows))},
        "files": hashes, "feature_names": FEATURES,
        "rules": {
            "development": "Original V1 training + validation only; V1 test remains excluded",
            "new_test": "Seeded sample outside every V1 identity component; frozen before V2 training",
            "identity": "Connected nonblank SDSS_NAME / positive THING_ID over the entire catalog",
            "labels": "Z only, 0<Z<=7 and IS_QSO_FINAL==1",
            "inputs": "PSFMAG-EXTINCTION and PSFMAGERR; at least three valid corrected bands",
            "quality_metadata": "SOURCE_Z and ZWARNING retained for audit, never used as features",
            "generalization": "Same-catalog holdout, not cross-survey validation",
        },
    }, "dataset_id")
    save_json(output / "manifest.json", manifest)
    print(f"Prepared V2: dev={len(dev_rows):,}, NEW test={len(test_rows):,}; no V1 identity overlap", flush=True)
    return manifest
