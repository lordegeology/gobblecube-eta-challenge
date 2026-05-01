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

Model: LightGBM with `objective=regression_l1` (directly optimises MAE, robust to outliers). Early stopping on Dev set.

**Result: 258.9s Dev MAE — 30% improvement over the 367s baseline.**

Feature importance (top 5 by gain): `pair_median` >> `pair_mean` > `pair_p75` > `pair_p90` > `hour_cos`. The pair lookup stats dominate, confirming the core hypothesis.

**What we learned:** Early stopping fired at round 58/2000 — suggested learning rate might be too high.

---

### v2 — Hyperparameter tuning (`259.9s Dev MAE`)

- `learning_rate`: 0.05 → 0.02
- `num_leaves`: 511 → 255  
- Early stopping patience: 50 → 100 rounds

Model trained to round 151 (vs 58 in v1) but MAE was essentially unchanged.

**Conclusion: the model is not the bottleneck.** Reverting to v1 params (faster training, same score) and pivoting to data quality.

---

### v3 — Data cleaning, with duration clip (`261.3s Dev MAE`)

The `download_data.py` script leaves a lot of noise after basic filtering. Added `clean.py` with the following passes applied before building lookup tables:

- **Passenger count in [1, 6]** — removed 557,219 rows. NYC taxis seat max 6; values of 0, 7, 8, 9 are data entry errors.
- **Duplicate rows** — removed 297,907 rows. Same (pickup, dropoff, timestamp, pax) appearing twice is a logging artifact.
- **Implied speed > 80 km/h** — removed 295,359 rows. Physically impossible in NYC traffic; likely bad timestamps or wrong zone IDs.
- **Cross-zone speed < 1 km/h** — removed 6,312 rows. Meter left running after trip ended.
- **Same-zone duration > 45 min** — removed 20,294 rows. Circling within one small zone for 45+ minutes is not a real trip.
- **Duration p0.1–p99.9 clip** — removed 69,543 rows.

Total removed: 1,246,581 / 36,700,289 (3.40%). Remaining: 35,453,708 rows.

**Result: 261.3s — slight regression.** The duration clip was too aggressive — it narrowed the training distribution vs dev and hurt tail calibration.

---

### v4 — Data cleaning, without duration clip (`260.8s Dev MAE`)

Removed the p0.1–p99.9 duration clip. All other filters kept.

Total removed: 1,177,038 / 36,700,289 (3.21%). Remaining: 35,523,251 rows.

**Result: 260.8s — still no meaningful improvement over v1 (258.9s).**

The pattern across v1–v4 is now clear: every run lands within 2s of 259s regardless of cleaning or hyperparameter changes. LightGBM with L1 loss is already robust to the noise we were removing. **The feature set is the bottleneck, not the data quality or model configuration.**

The core problem: `pair_mean` and `pair_median` are single numbers — the average duration for a zone pair across all hours and days. A JFK→Midtown trip at 8am Friday rush is completely different from the same route at 11pm Sunday, but the model only sees one historical average for both. We need time-conditional zone-pair features.

**Next: hour × zone-pair interaction lookup.**

---

## Scoreboard

| Version | Change | Dev MAE |
|---|---|---|
| Baseline GBT (their repo) | — | ~367s |
| Zone-pair lookup (10 lines, no ML) | — | 301s |
| v1 | LightGBM + 30 engineered features | **258.9s** |
| v2 | Hyperparameter tuning | 259.9s |
| v3 | Data cleaning + duration clip | 261.3s |
| v4 | Data cleaning, no duration clip | 260.8s |
| v5 | Hour × zone-pair interaction lookup | TBD |

## How to reproduce

```bash
pip install -r requirements.txt
python data/download_data.py   # one-time, ~500 MB
python train.py                # builds model.pkl
python grade.py                # validate on Dev
python -m pytest tests/
```

## What I'd try next with more time

1. **Hour × zone-pair lookup** — pre-compute mean duration per (pickup_zone, dropoff_zone, hour_bucket) where hour_bucket groups hours into ~6 bands (night, early AM, AM rush, midday, PM rush, evening). Directly addresses the biggest remaining gap.
2. **Weather join** — NOAA hourly data for JFK/LGA. Rain/snow adds 20–40% to trip time. The 2024 eval slice is a winter-holiday period which likely has weather disruption — this could be the single biggest remaining gain.
3. **OSRM road-network distance** — haversine misses bridges, one-way streets, and East River crossings that dominate Manhattan–Brooklyn/Queens routes.
4. **Zone embeddings + MLP** — a small neural net where each zone gets a learnable embedding. Likely marginal given LightGBM already has pair lookup stats, but worth exploring if the above don't move the needle.
