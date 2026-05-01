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

Feature importance: `pair_median` >> `pair_mean` > `pair_p75` > `pair_p90` > `hour_cos`. Pair lookup stats dominate, confirming the core hypothesis.

**What we learned:** Early stopping fired at round 58/2000 — suggested learning rate might be too high.

---

### v2 — Hyperparameter tuning (`259.9s Dev MAE`)

- `learning_rate`: 0.05 → 0.02, `num_leaves`: 511 → 255, patience: 50 → 100

Model trained to round 151 (vs 58 in v1) but MAE unchanged. **Conclusion: model is not the bottleneck.** Reverting to v1 params, pivoting to data quality.

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

**Result: 261.3s — slight regression.** Duration clip narrowed training distribution vs dev.

---

### v4 — Data cleaning, without duration clip (`260.8s Dev MAE`)

Removed the p0.1–p99.9 duration clip. All structural filters kept. Total removed: 1,177,038 / 36,700,289 (3.21%).

**Result: 260.8s — still no meaningful improvement over v1.** LightGBM with L1 loss was already robust to the noise we were removing. The feature set is the bottleneck.

---

### v4.5 — Hyperparameter search on cleaned data (best: `257.7s train / 259.9s grade`)

Tried several hyperparameter combinations with clean data locked in:

| lr | num_leaves | min_data_in_leaf | Train MAE | Grade MAE |
|---|---|---|---|---|
| 0.05 | 511 | 500 | 258.6s | 260.8s |
| 0.02 | 255 | 500 | 259.1s | 259.9s |
| 0.05 | 511 | 200 | 257.7s | 259.9s |

All combinations cluster around 259-261s. Model is definitively not the bottleneck.

---

### v5 — Hour × zone-pair lookup (`260.3s Dev MAE`)

Added 3 new features: `hour_pair_mean`, `hour_pair_median`, `hour_pair_ratio` — pre-computed mean/median duration per (pickup_zone, dropoff_zone, hour_bucket) across 6 time buckets (night, early AM, AM rush, midday, PM rush, evening).

**Feature importance shift:** `hour_pair_median` immediately became #1 by massive margin, displacing `pair_median`. Model clearly finds time-conditional stats more informative than all-day averages.

**Result: 260.3s — MAE unchanged despite better features.** Tried lr=0.03/leaves=255 as well, still ~260s.

**Conclusion:** LightGBM has hit a ceiling at ~259s on this feature set. The remaining gap is likely a temporal distribution shift — training on 2023 data, evaluating on early 2024. Our lookup tables encode 2023 patterns and can't adapt to January 2024 conditions.

---

### v6 — Weather join (`261.4s Dev MAE`)

Added `data/download_weather.py` to fetch NOAA hourly observations from JFK (USW00094789) and LGA (USW00014732) stations, covering 2023-01-01 through 2024-03-31 — both training and the eval window. 8 new features:

| Feature | Description |
|---|---|
| `temp_c` | Hourly dry bulb temperature |
| `precip_mm` | Hourly precipitation |
| `wind_kmh` | Wind speed |
| `snow_depth_mm` | Snow depth on ground |
| `is_raining` | Boolean: precip > 0.2mm |
| `is_snowing` | Boolean: snow depth > 0 |
| `is_bad_weather` | Boolean: heavy rain or any snow |
| `wind_strong` | Boolean: wind > 30 km/h |

Weather is stored as a `(date, hour)` lookup dict inside `model.pkl` — no external API calls at inference time, sub-millisecond lookup.

**Result: 261.4s — no improvement.** Weather features did not appear in top 15 by gain. Two likely reasons:

1. The dev set (last 2 weeks of 2023) may not have had significant weather events for the model to learn from
2. `hour_pair_median` already implicitly captures weather effects — if JFK→Midtown historically takes longer during certain hours, that average already incorporates the rainy days in that bucket

**Note on station coverage:** JFK and LGA are the two primary official NOAA reporting stations for NYC and what most real ETA systems use. A more rigorous approach would add Central Park (USW00094728) for Manhattan microclimate coverage and weight stations by proximity to the pickup zone centroid. Complexity vs. ROI didn't justify it given weather features showed zero gain on dev.

---

### On deep learning

Evaluated the DL path and chose not to pursue it for this submission. My reasoning:

Zone embeddings are the natural DL contribution here; replacing raw zone IDs with learned dense vectors that capture spatial relationships. However, our feature importance shows `hour_pair_median` already captures most of the spatial-temporal signal we'd expect embeddings to provide. The model isn't stuck at 259s because of architecture limitations.

The root cause is temporal: our lookup tables encode 2023 patterns and the eval slice is 2024. A neural net trained on the same 2023 data has the identical temporal blind spot and so switching architecture doesn't fix a distribution shift.

The DL approach that would actually help is a **sequence model** that learns how NYC traffic patterns evolve month-over-month and can extrapolate to January 2024. That's a materially more complex problem than zone embeddings and is something I'd approach with more bandwidth.

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
| v6 | Weather join from NOAA JFK/LGA (41 features) | 261.4s |

## How to reproduce

```bash
pip install -r requirements.txt
python data/download_data.py       # one-time, ~500 MB
python data/download_weather.py    # one-time, ~2 min
python train.py                    # builds model.pkl
python grade.py                    # validate on Dev
python -m pytest tests/
```

## What I'd try next with more time

1. **Log-transform target** — training on `log1p(duration)` compresses the right tail and may help the model focus on short/medium trips. Low effort, worth trying.
2. **OSRM road-network distance** — haversine from zone centroids significantly underestimates actual driving distance (e.g. JFK→Midtown: 18km haversine vs ~28km via Van Wyck + Queens-Midtown Tunnel). True route distance is a fundamentally better spatial signal.
3. **Day-of-week × zone-pair lookup** — same idea as hour × zone-pair but Friday PM rush vs Tuesday PM rush are completely different in NYC. A `(pickup_zone, dropoff_zone, dow_bucket)` lookup would capture this.
4. **Sequence model for temporal extrapolation** — the only DL approach that targets the actual root cause: learning how NYC traffic patterns shift month-over-month to extrapolate from 2023 training data to 2024 eval conditions.
