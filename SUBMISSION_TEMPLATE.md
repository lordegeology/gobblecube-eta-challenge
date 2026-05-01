# Your Submission: Writeup Template

---

## Your final score

Dev MAE: **258.9 s**

---

## Your approach, in one paragraph

The baseline GBT scores ~367s MAE — worse than a 10-line zone-pair mean lookup (~301s) — because it treats zone IDs as raw integers with no spatial or historical context. The fix is to pre-compute a lookup table of mean/median/p75/p90 trip duration for every (pickup_zone, dropoff_zone) pair from the full 37M-row training set and hand those directly to the model as features, so it learns residuals rather than trying to reconstruct the obvious from scratch. On top of that I added haversine distance and bearing from TLC zone centroids, cyclical time encoding (sin/cos of hour/dow/month to avoid the midnight discontinuity), rush-hour and holiday flags, per-zone marginal stats, time-conditional pair lookups (mean/median per zone pair × hour bucket), and a NOAA weather join from JFK and LGA stations. The model is LightGBM with L1 objective (directly optimises MAE, robust to the heavy-tailed duration distribution). Everything — booster weights, lookup tables, centroids, weather dict — is serialised into a single model.pkl so inference is a dict lookup + forward pass with no network calls, well under the 200ms constraint.

---

## What you tried that didn't work

**Hyperparameter tuning** — lowering learning rate from 0.05 to 0.02 and reducing num_leaves from 511 to 255 made the model train longer (round 151 vs 58) but MAE was identical at ~260s. Tried multiple combinations; all landed within 2s of each other. The model was not the bottleneck.

**Data cleaning** — removed 1.18M rows (3.2%): bad passenger counts, duplicate trips, physically impossible speeds (>80 km/h implied), meter-left-running trips. Also tried a p0.1–p99.9 duration clip which hurt MAE by narrowing the training distribution relative to dev. Without the clip, cleaned data produced the same MAE as uncleaned — LightGBM with L1 loss was already robust to the noise.

**NOAA weather join** — fetched hourly temperature, precipitation, wind, and snow depth from JFK and LGA stations (2023–2024) and added 8 weather features. No improvement. Most likely because `hour_pair_median` already implicitly captures weather effects — the historical average for a zone pair during AM rush already incorporates the rainy days in that bucket. Also tried log-transforming the target (log1p/expm1) — model trained more stably but MAE unchanged with L1 objective.

---

## Where AI tooling sped you up most

Used Claude (claude.ai) throughout — not just for code generation but as a thinking partner for the whole approach.

**Most valuable:** the iterative debugging loop. When the TLC shapefile centroid loader silently returned 0 zones (because `row.get()` doesn't work on pandas Series the way it does on dicts, and the glob pattern `*.shp` missed the file inside a subdirectory), Claude diagnosed both issues from the error output in seconds. Same with the float64 memory blowup when adding weather features and identified the root cause (numpy upcasting in `column_stack` before the `astype(float32)` cast) and rewrote to a pre-allocated float32 matrix. These are the kinds of bugs that cost hours to track down manually.

**Second most valuable:** reasoning about what to try next. After each run, talking through why MAE wasn't improving (model bottleneck vs feature bottleneck vs data bottleneck) helped avoid wasted iterations. The conclusion that the DL path (zone embeddings) doesn't fix the root cause — a temporal distribution shift between 2023 training data and 2024 eval — came out of that kind of conversation.

---

## Next experiments

**OSRM road-network distance** — haversine from zone centroids significantly underestimates actual driving distance (JFK→Midtown: ~18km straight-line vs ~28km via Van Wyck + Queens-Midtown Tunnel). Pre-computing an OSRM distance matrix for all 265×265 zone pairs would give a fundamentally better spatial signal regardless of model architecture, and it's a one-time offline computation.

**Day-of-week × zone-pair lookup** — the natural extension of the hour × zone-pair feature. Friday PM rush vs Tuesday PM rush behave completely differently on routes like Midtown→JFK. A `(pickup_zone, dropoff_zone, dow_bucket)` lookup would capture this without any model changes.

---

## How to reproduce

```bash
pip install -r requirements.txt
python data/download_data.py       # one-time, ~500 MB
python data/download_weather.py    # one-time, ~2 min
python train.py                    # builds model.pkl (~11 MB), takes ~30 min
python grade.py                    # should print ~259s Dev MAE
python -m pytest tests/
```

---

_Total time spent on this challenge: ~32 hours._
