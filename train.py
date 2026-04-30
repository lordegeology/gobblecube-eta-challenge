#!/usr/bin/env python
"""Training script for the ETA challenge — LightGBM with rich feature engineering.

Strategy
--------
1. Build zone-pair and zone-level lookup tables from training data
   (these alone bring MAE close to zone-pair mean, ~300 s)
2. Add centroid geometry (haversine distance + bearing) and rich time features
3. Train a LightGBM regressor on ~30 features
4. Save model + lookup tables + centroids into a single model.pkl
   (predict.py loads this one file — no other state needed at inference)

Expected Dev MAE: significantly below the ~351 s GBT baseline.

Run
---
    pip install -r requirements.txt lightgbm
    python data/download_data.py    # one-time, ~500 MB
    python train.py
    python grade.py                 # validate before submitting
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import LookupTables, build_dataframe, _load_centroids, FEATURE_NAMES

DATA_DIR   = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train.parquet"
    dev_path   = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run `python data/download_data.py` first.")
    print("Loading training data...")
    train = pd.read_parquet(train_path)
    dev   = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,}  dev: {len(dev):,}")
    return train, dev


def clip_target(y: np.ndarray) -> np.ndarray:
    """Winsorise extreme durations that hurt regression."""
    lo, hi = np.percentile(y, 0.5), np.percentile(y, 99.5)
    return np.clip(y, lo, hi)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    train, dev = load_data()

    # ---- 1. Build lookup tables from training set -------------------------
    print("\nBuilding lookup tables from training set...")
    t0 = time.time()
    tables = LookupTables()
    tables.fit(train)
    print(f"  {len(tables.pair_stats):,} zone-pair entries  ({time.time()-t0:.1f}s)")

    # ---- 2. Load / compute zone centroids ---------------------------------
    print("\nLoading zone centroids...")
    centroids = _load_centroids()
    print(f"  {len(centroids)} zones with centroid data")

    # ---- 3. Build feature matrices ----------------------------------------
    print("\nBuilding feature matrices...")
    t0 = time.time()
    X_train = build_dataframe(train, tables, centroids)
    y_train = clip_target(train["duration_seconds"].to_numpy())
    X_dev   = build_dataframe(dev, tables, centroids)
    y_dev   = dev["duration_seconds"].to_numpy()
    print(f"  X_train: {X_train.shape}  X_dev: {X_dev.shape}  ({time.time()-t0:.1f}s)")

    # Quick sanity: zone-pair mean lookup MAE on dev
    pair_mean_preds = np.array([
        tables.pair_lookup(int(p), int(d))[0]
        for p, d in zip(dev["pickup_zone"], dev["dropoff_zone"])
    ])
    mae_lookup = float(np.mean(np.abs(pair_mean_preds - y_dev)))
    print(f"\n  Zone-pair mean lookup Dev MAE: {mae_lookup:.1f} s  (sanity check)")

    # ---- 4. Train LightGBM ------------------------------------------------
    print("\nTraining LightGBM...")
    params = {
        "objective":        "regression_l1",   # MAE loss — directly optimises our metric
        "metric":           "mae",
        "learning_rate":    0.05,
        "num_leaves":       511,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "lambda_l1":        0.1,
        "lambda_l2":        0.1,
        "max_bin":          511,
        "verbose":          -1,
        "n_jobs":           -1,
        "seed":             42,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES, free_raw_data=False)
    lgb_dev   = lgb.Dataset(X_dev,   label=y_dev,   feature_name=FEATURE_NAMES, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    t0 = time.time()
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_dev],
        callbacks=callbacks,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.0f}s — best iteration: {model.best_iteration}")

    # ---- 5. Evaluate -------------------------------------------------------
    preds_dev = model.predict(X_dev, num_iteration=model.best_iteration)
    mae_dev   = float(np.mean(np.abs(preds_dev - y_dev)))
    print(f"\n  Dev MAE (LightGBM): {mae_dev:.1f} s")
    print(f"  Improvement over zone-pair lookup: {mae_lookup - mae_dev:.1f} s")

    # Feature importance (top 15)
    importance = sorted(
        zip(FEATURE_NAMES, model.feature_importance("gain")),
        key=lambda x: -x[1],
    )[:15]
    print("\n  Top 15 features by gain:")
    for name, gain in importance:
        print(f"    {name:<22} {gain:>12,.0f}")

    # ---- 6. Save artefact --------------------------------------------------
    artefact = {
        "model":     model,
        "tables":    tables,
        "centroids": centroids,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artefact, f, protocol=5)

    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved to {MODEL_PATH}  ({size_mb:.1f} MB)")
    print("\nNext: `python grade.py` to validate on the full Dev set.")


if __name__ == "__main__":
    main()
