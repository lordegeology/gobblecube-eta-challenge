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

Added `clean.py` with structural filters applied before building lookup tables:

| Filter | Removed | Rationale |
|---|---|---|
| Passenger count in [1, 6] | 557,219 | NYC taxis seat max 6; values >6 are data entry errors |
| Duplicate rows | 297,907 | Same (pickup, dropoff, timestamp, pax) twice = logging artifact |
| Implied speed > 80 km/h | 295,359 | Physically impossible in NYC traffic |
| Cross-zone speed < 1 km/h | 6,312 | Meter left running after trip ended |
| Same-zone duration > 45 min | 20,294 | Circling within one zone for 45+ min is not a real trip |
| Duration p0.1–p99.9 clip | 69,543 | Aggressive tail removal |

Total removed: 1,246,581 / 36,700,289 (3.40%)

**Result: 261.3s — slight regression.** Duration clip narrowed training distribution vs dev, hurting tail calibration.

---

### v4 — Data cleaning, without duration clip (`260.8s Dev MAE`)

Removed the p0.1–p99.9 duration clip. All structural filters kept.

Total removed: 1,177,038 / 36,700,289 (3.21%). Remaining: 35,523,251 rows.

**Result: 260.8s — still no meaningful improvement over v1.** LightGBM with L1 loss was already robust to the noise we were removing. The feature set is the bottleneck, not data quality.

---

### v4.5 — Hyperparameter search on cleaned data (best: `257.7s train / 259.9s grade`)

With clean data locked in, tried several hyperparameter combinations to find the best configuration before moving to new features:

| lr | num_leaves | min_data_in_leaf | Train MAE | Grade MAE |
|---|---|---|---|---|
| 0.05 | 511 | 500 | 258.6s | 260.8s |
| 0.02 | 255 | 500 | 259.1s | 259.9s |
| 0.05 | 511 | 200 | 257.7s | 259.9s |

Best configuration: `lr=0.05`, `num_leaves=511`, `min_data_in_leaf=200`. Marginal differences confirm the model is not the bottleneck — all combinations cluster around 259-261s.

---

### v5 — Hour × zone-pair lookup (`260.3s Dev MAE`)

Added 3 new features to `features.py`:

- **`hour_pair_mean`** — mean duration per (pickup_zone, dropoff_zone, hour_bucket). JFK→Midtown at AM rush gets its own historical average, separate from JFK→Midtown at midnight.
- **`hour_pair_median`** — median for the same grouping, more robust to within-bucket outliers.
- **`hour_pair_ratio`** — `hour_pair_mean / pair_mean`. Captures relative congestion: a ratio of 1.4 means this route runs 40% slower during this time bucket than its daily average. Generalises across routes of different lengths.

6 time buckets: night (22-5), early AM (5-7), AM rush (7-10), midday (10-16), PM rush (16-20), evening (20-22).

**Feature importance shift:** `hour_pair_median` immediately became #1 by a massive margin, displacing `pair_median`. The model finds time-conditional stats far more informative than all-day averages — exactly as predicted.

**Result: 260.3s — MAE unchanged despite better features.** Tried lr=0.03/leaves=255 as well, still ~260s. The model is extracting the right signals but MAE won't budge.

**Conclusion:** LightGBM has hit its ceiling on this feature set at ~259s. The remaining gap is likely a temporal distribution shift — training on 2023 data, evaluating on early 2024. Our lookup tables encode 2023 patterns and can't adapt to January 2024 conditions (different weather, demand patterns, post-holiday effects). No amount of feature engineering on historical averages fixes this without new information.

**Next:** weather join (directly addresses temporal gap) or MLP with zone embeddings (richer representations).

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
| v4.5 | Hyperparameter search on clean data | 259.9s |
| v5 | Hour × zone-pair lookup (33 features) | 260.3s |

## How to reproduce

```bash
pip install -r requirements.txt
python data/download_data.py   # one-time, ~500 MB
python train.py                # builds model.pkl
python grade.py                # validate on Dev
python -m pytest tests/
```

## What I'd try next with more time

1. **Weather join** — NOAA hourly data for JFK/LGA. Rain/snow adds 20–40% to trip time. The 2024 eval slice is a winter-holiday period — systematic weather effects are likely the single biggest remaining source of error.
2. **MLP with zone embeddings** — replace LightGBM with a small neural net where each zone gets a learnable embedding. Lets the model discover spatial relationships beyond what haversine distance captures. Most valuable if combined with the full feature set we've built.
3. **OSRM road-network distance** — haversine misses bridges, one-way streets, and East River crossings that dominate Manhattan–Brooklyn/Queens routes.
4. **Day-of-week × zone-pair lookup** — same idea as hour × zone-pair but for weekday vs weekend. Some routes (e.g. nightlife areas) behave completely differently on weekends.
