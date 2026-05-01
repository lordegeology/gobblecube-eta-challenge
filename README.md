# ETA Challenge Submission

## Approach

The baseline GBT uses 6 sparse features and scores ~367s MAE — worse than a dead-simple zone-pair mean lookup (~301s). It fails because it treats zone IDs as raw integers with no spatial or historical context.

Our approach: pre-compute a zone-pair lookup table from the full training set, hand it directly to LightGBM as features, then let the model learn residuals from geometry, time, and weather. Everything is baked into `model.pkl` at training time — inference is a single dict lookup + LightGBM forward pass, well under the 200ms constraint.

## Iteration log

### v1 — LightGBM + feature engineering (`258.9s Dev MAE`)

Replaced baseline XGBoost with LightGBM and 30 engineered features:

| Group | Features | Rationale |
|---|---|---|
| Zone-pair lookup | `pair_mean`, `pair_median`, `pair_p75`, `pair_p90`, `pair_log_count` | Pre-computed from 37M trips — the single biggest unlock |
| Zone-level stats | `pu_mean`, `pu_std`, `do_mean`, `do_std` | Marginal zone effects beyond the pair |
| Geometry | `dist_km`, `bearing_sin`, `bearing_cos` | Haversine distance + bearing from TLC zone centroids |
| Cyclical time | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` | Fixes discontinuity of raw integer hour encoding |
| Boolean flags | `is_weekend`, `is_holiday`, `is_rush_am`, `is_rush_pm`, `is_night`, `is_early_am` | Distinct NYC traffic regimes |
| Raw identifiers | `pickup_zone`, `dropoff_zone`, `hour`, `dow`, `month`, `passenger_count` | Kept for tree splits |

**Result: 258.9s Dev MAE — 30% improvement over the 367s baseline.**

---

### v2 — Hyperparameter tuning (`259.9s Dev MAE`)

`learning_rate` 0.05→0.02, `num_leaves` 511→255, patience 50→100. Model trained to round 151 vs 58 in v1. MAE unchanged. **Model is not the bottleneck.**

---

### v3 & v4 — Data cleaning (`260.8s Dev MAE`)

Added `clean.py` with structural filters before building lookup tables:

| Filter | Removed | Rationale |
|---|---|---|
| Passenger count in [1, 6] | 557,219 | Data entry errors — NYC taxis seat max 6 |
| Duplicate rows | 297,907 | Same (pickup, dropoff, timestamp, pax) = logging artifact |
| Implied speed > 80 km/h | 295,359 | Physically impossible in NYC traffic |
| Cross-zone speed < 1 km/h | 6,312 | Meter left running |
| Same-zone duration > 45 min | 20,294 | Not a real trip |

Total removed: 1,177,038 / 36,700,289 (3.21%). Also tried a p0.1–p99.9 duration clip (v3) — hurt MAE by narrowing training distribution vs dev. Removed it in v4. **LightGBM with L1 loss is already robust to the noise we were removing.**

---

### v4.5 — Hyperparameter search on clean data (best: `259.9s grade`)

Tried lr ∈ {0.02, 0.05}, num_leaves ∈ {255, 511}, min_data_in_leaf ∈ {200, 500}. All combinations cluster at 259-261s. Definitively confirmed: **the feature set is the bottleneck, not the model.**

---

### v5 — Hour × zone-pair lookup (`260.3s Dev MAE`)

Added 3 features: `hour_pair_mean`, `hour_pair_median`, `hour_pair_ratio` — pre-computed per (pickup_zone, dropoff_zone, hour_bucket) across 6 time bands (night, early AM, AM rush, midday, PM rush, evening).

`hour_pair_median` immediately became #1 feature by gain, displacing `pair_median`. Model finds time-conditional stats far more informative than all-day averages — exactly as predicted. But MAE unchanged. **The model is learning the right things but the overall error floor isn't moving.**

---

### v6 — NOAA weather join (`261.4s Dev MAE`)

Added hourly weather from JFK (USW00094789) and LGA (USW00014732) stations via `data/download_weather.py`, covering 2023-01-01 through 2024-03-31. 8 new features: `temp_c`, `precip_mm`, `wind_kmh`, `snow_depth_mm`, `is_raining`, `is_snowing`, `is_bad_weather`, `wind_strong`. Stored as a `(date, hour)` dict in `model.pkl` — no API calls at inference.

**Result: 261.4s — no improvement.** Weather features didn't appear in top 15 by gain. Likely because `hour_pair_median` already implicitly captures weather effects — the historical average for a route during AM rush already incorporates the rainy days in that bucket. Also encountered a memory error (35M × 41 features in float64 = 10.9 GB) — fixed by pre-allocating a float32 matrix rather than using `np.column_stack` which upcasts before the `astype` cast.

---

### v7 — Log-transform target (`260.7s Dev MAE`)

Trained on `log1p(duration)`, predictions back-transformed with `expm1()`. Model trained to round 169 (vs 57-106 previously) — smoother loss surface. MAE essentially unchanged at 260.7s. With L1 objective already robust to outliers, the log transform adds less than it would with MSE.

---

### On deep learning

Evaluated the DL path and chose not to pursue it. Zone embeddings are the natural contribution — replacing raw zone IDs with learned dense vectors. But our feature importance shows `hour_pair_median` already captures most of the spatial-temporal signal embeddings would provide. The model isn't stuck at 259s because of architecture capacity.

The root cause is temporal: lookup tables encode 2023 patterns, the eval slice is 2024. A neural net trained on the same 2023 data has the identical blind spot — switching architecture doesn't fix distribution shift. The DL approach that would actually help is a sequence model that learns how NYC traffic patterns evolve month-over-month to extrapolate to January 2024 — materially more complex and outside this challenge window.

---

## Scoreboard

| Version | Change | Dev MAE |
|---|---|---|
| Baseline GBT (their repo) | — | ~367s |
| Zone-pair lookup (10 lines) | — | 301s |
| **v1** | **LightGBM + 30 engineered features** | **258.9s** |
| v2 | Hyperparameter tuning | 259.9s |
| v3 | Data cleaning + duration clip | 261.3s |
| v4 | Data cleaning, no clip | 260.8s |
| v4.5 | Hyperparameter search | 259.9s |
| v5 | Hour × zone-pair lookup (33 features) | 260.3s |
| v6 | NOAA weather join (41 features) | 261.4s |
| v7 | Log-transform target | 260.7s |

**Best submission: v1 model.pkl — 258.9s Dev MAE.**

## How to reproduce

```bash
pip install -r requirements.txt
python data/download_data.py       # one-time, ~500 MB
python data/download_weather.py    # one-time, ~2 min
python train.py                    # builds model.pkl (~11 MB)
python grade.py                    # validate: should print ~259s
python -m pytest tests/
```

## What I'd try next

1. **OSRM road-network distance** — haversine underestimates actual driving distance significantly (JFK→Midtown: 18km straight-line vs ~28km via Van Wyck + Queens-Midtown Tunnel). True route distance is a fundamentally better spatial signal regardless of model architecture.
2. **Day-of-week × zone-pair lookup** — Friday PM rush vs Tuesday PM rush are completely different in NYC. A `(pickup_zone, dropoff_zone, dow_bucket)` lookup is the natural extension of the hour × pair approach.
3. **Central Park weather station** — adding USW00094728 would give Manhattan microclimate coverage alongside the two airport stations, potentially making weather features useful for Manhattan-origin trips.
4. **Sequence model for temporal extrapolation** — the only DL approach that targets the actual root cause: learning month-over-month traffic pattern shifts to extrapolate from 2023 training to 2024 eval.
