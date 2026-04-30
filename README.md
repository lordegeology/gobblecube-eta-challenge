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

**What went wrong / what we learned:** Early stopping fired at round 58/2000. Suggested learning rate (0.05) might be too high — tried tuning in v2.

---

### v2 — Hyperparameter tuning (`259.9s Dev MAE`)

Changes from v1:
- `learning_rate`: 0.05 → 0.02
- `num_leaves`: 511 → 255
- Early stopping patience: 50 → 100 rounds

Model trained to round 151 this time (vs 58 in v1), but MAE was essentially unchanged — 259.9s vs 258.9s.

**Conclusion: the model is not the bottleneck.** LightGBM is already extracting most of what the current features can offer. Further hyperparameter tuning has diminishing returns. The next lever is data quality — the README hints there is "plenty" of garbage left in the dataset after the basic cleaning in `download_data.py`. Reverting to v1 params (faster training, same score) and moving to data cleaning next.

---

### v3 — Data cleaning (in progress)

The `download_data.py` script applies basic cleaning (drop trips <30s or >3h, invalid zones, fill missing passenger count) but leaves plenty of noise. Planned cleaning pass:

- **Speed-based outlier removal** — compute implied speed as `dist_km / (duration / 3600)`. Drop trips with implied speed > 80 km/h (physically impossible in NYC traffic) or implausibly slow cross-borough trips.
- **Same-zone duration outliers** — trips where pickup == dropoff zone with very long duration are likely meters left running, not real trips.
- **Passenger count capping** — values above 6 are almost certainly data entry errors. Current cleaning fills missing with 1 but doesn't cap absurd values.
- **Duplicate row removal** — exact (pickup_zone, dropoff_zone, requested_at, passenger_count) duplicates are likely logging artifacts.

Cleaning is applied before building the lookup tables so the zone-pair stats are computed on clean data.

---

## Scoreboard

| Approach | Dev MAE |
|---|---|
| Predict global mean | ~580s |
| Baseline GBT — their repo | ~367s |
| Zone-pair lookup (10 lines, no ML) | 301s |
| **v1: LightGBM + feature engineering** | **258.9s** |
| v2: hyperparameter tuning | 259.9s (no improvement) |
| v3: + data cleaning | TBD |

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
3. **Hour × zone-pair interaction** — rush hour affects JFK→Midtown very differently from short local trips. An explicit (hour_bucket, pickup_zone, dropoff_zone) lookup could capture this.
4. **Zone embeddings + MLP** — a small neural net where each zone gets a learnable embedding, letting the model discover spatial relationships without explicit coordinates.
