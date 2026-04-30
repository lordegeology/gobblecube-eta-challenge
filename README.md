# ETA Challenge Submission

## Approach

The baseline GBT uses 6 sparse features and scores ~367s MAE on eval — worse than a dead-simple zone-pair mean lookup (~301s). The model fails to learn the most obvious signal because it treats zone IDs as raw integers with no spatial or historical context.

Our approach: build the zone-pair lookup table first, hand it directly to the model as features, then let the model learn the residuals using geometry and time.

## Iteration log

### v1 — LightGBM + feature engineering (`258.9s Dev MAE`)

Replaced the baseline XGBoost with LightGBM and 30 engineered features across 6 groups:

| Group | Features | Rationale |
|---|---|---|
| Zone-pair lookup | `pair_mean`, `pair_median`, `pair_p75`, `pair_p90`, `pair_log_count` | Pre-computed from 37M trips. Single biggest unlock — baseline had none of this |
| Zone-level stats | `pu_mean`, `pu_std`, `do_mean`, `do_std` | Marginal zone effects beyond the pair |
| Geometry | `dist_km`, `bearing_sin`, `bearing_cos` | Haversine distance + bearing from TLC zone centroids |
| Cyclical time | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` | Fixes the "hour 23 is far from hour 0" problem of raw integer encoding |
| Boolean flags | `is_weekend`, `is_holiday`, `is_rush_am`, `is_rush_pm`, `is_night`, `is_early_am` | Distinct traffic regimes |
| Raw identifiers | `pickup_zone`, `dropoff_zone`, `hour`, `dow`, `month`, `passenger_count` | Kept alongside cyclical for tree splits |

Model: LightGBM with `objective=regression_l1` (directly optimises MAE, robust to duration outliers). Early stopping on Dev set.

**Result: 258.9s Dev MAE — 30% improvement over the 367s baseline.**

Feature importance (top 5 by gain): `pair_median` >> `pair_mean` > `pair_p75` > `pair_p90` > `hour_cos`. The pair lookup stats dominate, confirming the core hypothesis.

**What went wrong:** Early stopping fired at round 58/2000 — very early. Suggests learning rate (0.05) was too high, causing the model to overshoot and stop before fully converging.

### v2 — Tuned hyperparameters (in progress)

Changes from v1:
- `learning_rate`: 0.05 → 0.02 (slower steps, more trees, better generalisation)
- `num_leaves`: 511 → 255 (less aggressive splits, reduces overfitting)
- `early_stopping` patience: 50 → 100 rounds (gives slower lr room to keep improving)

Expecting the model to train for significantly more rounds and land below 250s.

## Scoreboard

| Approach | Dev MAE |
|---|---|
| Predict global mean | ~580s |
| Zone-pair lookup (10 lines, no ML) | 301s |
| Baseline GBT — their repo | ~367s |
| **v1: LightGBM + feature engineering** | **258.9s** |
| v2: + hyperparameter tuning | TBD |

## How to reproduce

```bash
pip install -r requirements.txt
python data/download_data.py   # one-time, ~500 MB
python train.py                # builds model.pkl
python grade.py                # validate on Dev
python -m pytest tests/
```

## What I'd try next with more time

1. **Weather join** — NOAA hourly data for JFK/LGA. Rain/snow adds 20–40% to trip time. The 2024 eval slice is a winter-holiday period which likely has weather disruption.
2. **OSRM road-network distance** — haversine misses bridges, one-way streets, and East River crossings that dominate Manhattan–Brooklyn/Queens routes.
3. **Zone embeddings + MLP** — a small neural net where each zone gets a learnable embedding, letting the model discover spatial relationships without explicit coordinates.
4. **Hour × zone-pair interaction** — rush hour affects JFK→Midtown very differently from short local trips. An explicit (hour_bucket, pickup_zone, dropoff_zone) lookup could capture this.
