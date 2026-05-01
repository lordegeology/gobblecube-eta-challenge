"""Feature engineering for the ETA challenge.

All feature construction lives here so training (train.py) and inference
(predict.py) use identical logic. Add a feature once; it appears everywhere.

Feature groups
--------------
1. Zone-pair lookup stats        — mean/median/p75/p90/count per (pu, do) pair
2. Hour x zone-pair lookup stats — mean/median per (pu, do, hour_bucket) pair
3. Zone-level stats              — mean duration when this zone is pickup or dropoff
4. Zone centroid geometry        — haversine distance, bearing from TLC shapefile
5. Time features                 — cyclical hour/dow/month, rush-hour flag, weekend, holiday
6. Weather features              — temp, precip, wind, snow from NOAA JFK/LGA stations
7. Raw identifiers               — pickup_zone, dropoff_zone (for tree splits)
"""

from __future__ import annotations

import math
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# NYC federal + state holidays that meaningfully affect taxi demand
# ---------------------------------------------------------------------------
_HOLIDAYS: set[date] = {
    # 2023
    date(2023, 1, 1),   # New Year's Day
    date(2023, 1, 16),  # MLK Day
    date(2023, 2, 20),  # Presidents' Day
    date(2023, 5, 29),  # Memorial Day
    date(2023, 6, 19),  # Juneteenth
    date(2023, 7, 4),   # Independence Day
    date(2023, 9, 4),   # Labor Day
    date(2023, 10, 9),  # Columbus Day
    date(2023, 11, 11), # Veterans Day (observed)
    date(2023, 11, 23), # Thanksgiving
    date(2023, 11, 24), # Black Friday (huge demand spike)
    date(2023, 12, 24), # Christmas Eve
    date(2023, 12, 25), # Christmas
    date(2023, 12, 31), # New Year's Eve
    # 2024 (for inference on the eval slice)
    date(2024, 1, 1),
    date(2024, 1, 15),
    date(2024, 2, 19),
    date(2024, 5, 27),
    date(2024, 6, 19),
    date(2024, 7, 4),
    date(2024, 9, 2),
    date(2024, 10, 14),
    date(2024, 11, 11),
    date(2024, 11, 28),
    date(2024, 11, 29),
    date(2024, 12, 24),
    date(2024, 12, 25),
    date(2024, 12, 31),
}

# ---------------------------------------------------------------------------
# Zone centroid cache — loaded once at module import
# ---------------------------------------------------------------------------
_CENTROID_CACHE: dict[int, tuple[float, float]] | None = None  # zone -> (lat, lon)

def _load_centroids() -> dict[int, tuple[float, float]]:
    """Download TLC zone shapefile once and cache centroids as (lat, lon)."""
    global _CENTROID_CACHE
    if _CENTROID_CACHE is not None:
        return _CENTROID_CACHE

    cache_path = Path(__file__).parent / "data" / "zone_centroids.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        _CENTROID_CACHE = dict(zip(df["zone_id"], zip(df["lat"], df["lon"])))
        return _CENTROID_CACHE

    # Try to compute from shapefile
    try:
        import geopandas as gpd
        import urllib.request, zipfile, io, tempfile, os

        zip_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
        print("Downloading TLC zone shapefile for centroid computation...")
        with urllib.request.urlopen(zip_url) as r:
            zdata = r.read()
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                zf.extractall(tmp)
            shp_files = list(Path(tmp).glob("**/*.shp"))
            gdf = gpd.read_file(shp_files[0])

        gdf = gdf.to_crs("EPSG:3857")  # projected CRS for accurate centroids
        gdf["centroid"] = gdf.geometry.centroid
        gdf = gdf.to_crs("EPSG:4326")  # back to lat/lon for haversine

        # Detect which column holds the zone ID (column names vary by shapefile version)
        id_col = None
        for candidate in ("LocationID", "location_id", "OBJECTID", "objectid", "zone_id"):
            if candidate in gdf.columns:
                id_col = candidate
                break
        if id_col is None:
            raise ValueError(f"No zone ID column found. Columns: {list(gdf.columns)}")
        print(f"  Using column '{id_col}' as zone ID")

        result = {}
        for _, row in gdf.iterrows():
            try:
                zid = int(row[id_col])
            except (ValueError, TypeError):
                continue
            if 1 <= zid <= 265:
                result[zid] = (float(row["centroid"].y), float(row["centroid"].x))
        _CENTROID_CACHE = result
        _save_centroids(result, cache_path)
        print(f"Cached {len(result)} zone centroids to {cache_path}")
        return _CENTROID_CACHE

    except Exception as e:
        print(f"Warning: could not load zone centroids ({e}). Distance features will be 0.")
        _CENTROID_CACHE = {}
        return _CENTROID_CACHE


def _save_centroids(centroids: dict, path: Path) -> None:
    rows = [{"zone_id": k, "lat": v[0], "lon": v[1]} for k, v in centroids.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Weather lookup — loaded once, keyed by (date, hour)
# ---------------------------------------------------------------------------

_WEATHER_CACHE: dict | None = None
_WEATHER_FALLBACK = {
    "temp_c": 10.0, "precip_mm": 0.0, "wind_kmh": 10.0,
    "snow_depth_mm": 0.0, "is_raining": 0, "is_snowing": 0,
    "is_bad_weather": 0, "wind_strong": 0,
}

def load_weather() -> dict:
    """Load weather lookup from CSV. Returns dict keyed by (date, hour)."""
    global _WEATHER_CACHE
    if _WEATHER_CACHE is not None:
        return _WEATHER_CACHE

    weather_path = Path(__file__).parent / "data" / "weather_hourly.csv"
    if not weather_path.exists():
        print("Warning: weather_hourly.csv not found. Run data/download_weather.py first.")
        print("Weather features will be set to neutral fallback values.")
        _WEATHER_CACHE = {}
        return _WEATHER_CACHE

    df = pd.read_csv(weather_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    _WEATHER_CACHE = {}
    weather_cols = ["temp_c", "precip_mm", "wind_kmh", "snow_depth_mm",
                    "is_raining", "is_snowing", "is_bad_weather", "wind_strong"]
    for _, row in df.iterrows():
        key = (row["date"], int(row["hour"]))
        _WEATHER_CACHE[key] = {c: float(row[c]) for c in weather_cols if c in row}
    print(f"  Loaded {len(_WEATHER_CACHE):,} hourly weather records")
    return _WEATHER_CACHE


def get_weather(ts: "datetime") -> dict:
    """Get weather for a given timestamp. Falls back gracefully if missing."""
    cache = load_weather()
    key = (ts.date(), ts.hour)
    return cache.get(key, _WEATHER_FALLBACK)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing in degrees [0, 360)."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dλ = math.radians(lon2 - lon1)
    x = math.sin(dλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Hour bucket helper
# ---------------------------------------------------------------------------

# 6 time-of-day buckets that capture distinct NYC traffic regimes:
#   0 = night       22:00 - 04:59
#   1 = early AM    05:00 - 06:59  (pre-rush)
#   2 = AM rush     07:00 - 09:59
#   3 = midday      10:00 - 15:59
#   4 = PM rush     16:00 - 19:59
#   5 = evening     20:00 - 21:59
_HOUR_BUCKET = [0,0,0,0,0,1,1,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,0,0]  # index = hour

def hour_bucket(h: int) -> int:
    return _HOUR_BUCKET[h]


# ---------------------------------------------------------------------------
# Lookup table helpers
# ---------------------------------------------------------------------------

class LookupTables:
    """Precomputed aggregation tables built from training data.

    Stored as plain dicts so they pickle fast and have zero import deps.
    """

    def __init__(self) -> None:
        # (pickup_zone, dropoff_zone) -> [mean, median, p75, p90, log_count]
        self.pair_stats: dict[tuple[int, int], list[float]] = {}
        # (pickup_zone, dropoff_zone, hour_bucket) -> [mean, median]
        self.hour_pair_stats: dict[tuple[int, int, int], list[float]] = {}
        # pickup_zone -> [mean, std]
        self.pu_stats: dict[int, list[float]] = {}
        # dropoff_zone -> [mean, std]
        self.do_stats: dict[int, list[float]] = {}
        # global fallback
        self.global_mean: float = 900.0
        self.global_std: float = 600.0

    def fit(self, df: pd.DataFrame) -> "LookupTables":
        y = df["duration_seconds"]
        self.global_mean = float(y.mean())
        self.global_std = float(y.std())

        # Zone-pair stats
        grp = df.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        agg = grp.agg(["mean", "median", lambda x: x.quantile(0.75),
                        lambda x: x.quantile(0.90), "count"])
        agg.columns = ["mean", "median", "p75", "p90", "count"]
        for (pu, do), row in agg.iterrows():
            self.pair_stats[(int(pu), int(do))] = [
                float(row["mean"]),
                float(row["median"]),
                float(row["p75"]),
                float(row["p90"]),
                float(math.log1p(row["count"])),
            ]

        # Pickup-zone stats
        grp_pu = df.groupby("pickup_zone")["duration_seconds"].agg(["mean", "std"])
        for zone, row in grp_pu.iterrows():
            self.pu_stats[int(zone)] = [float(row["mean"]), float(row["std"])]

        # Dropoff-zone stats
        grp_do = df.groupby("dropoff_zone")["duration_seconds"].agg(["mean", "std"])
        for zone, row in grp_do.iterrows():
            self.do_stats[int(zone)] = [float(row["mean"]), float(row["std"])]

        # Hour x zone-pair stats
        df_hb = df.copy()
        df_hb["hour_bucket"] = pd.to_datetime(df_hb["requested_at"]).dt.hour.map(lambda h: _HOUR_BUCKET[h])
        grp_hp = df_hb.groupby(["pickup_zone", "dropoff_zone", "hour_bucket"])["duration_seconds"]
        agg_hp = grp_hp.agg(["mean", "median"])
        for (pu, do, hb), row in agg_hp.iterrows():
            self.hour_pair_stats[(int(pu), int(do), int(hb))] = [
                float(row["mean"]),
                float(row["median"]),
            ]

        return self

    def pair_lookup(self, pu: int, do: int) -> list[float]:
        """Return [mean, median, p75, p90, log_count] for zone pair, with fallback."""
        v = self.pair_stats.get((pu, do))
        if v is not None:
            return v
        # Fallback: try pickup-only mean, then global
        pu_m = self.pu_stats.get(pu, [self.global_mean, self.global_std])[0]
        return [pu_m, pu_m, pu_m * 1.15, pu_m * 1.3, 0.0]

    def hour_pair_lookup(self, pu: int, do: int, hb: int) -> list[float]:
        """Return [mean, median] for (zone pair, hour bucket), with fallback to pair stats."""
        v = self.hour_pair_stats.get((pu, do, hb))
        if v is not None:
            return v
        # Fallback to overall pair mean/median
        pair = self.pair_stats.get((pu, do))
        if pair:
            return [pair[0], pair[1]]
        pu_m = self.pu_stats.get(pu, [self.global_mean, self.global_std])[0]
        return [pu_m, pu_m]


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

# Names must stay in sync with build_row() and build_dataframe()
FEATURE_NAMES = [
    # zone identifiers (for tree splits on individual zones)
    "pickup_zone",
    "dropoff_zone",
    # zone-pair lookup
    "pair_mean",
    "pair_median",
    "pair_p75",
    "pair_p90",
    "pair_log_count",
    # hour x zone-pair lookup
    "hour_pair_mean",
    "hour_pair_median",
    "hour_pair_ratio",   # hour_pair_mean / pair_mean — how much this bucket deviates
    # pickup-zone stats
    "pu_mean",
    "pu_std",
    # dropoff-zone stats
    "do_mean",
    "do_std",
    # geometry
    "dist_km",
    "bearing_sin",
    "bearing_cos",
    # time: cyclical
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # time: boolean flags
    "is_weekend",
    "is_holiday",
    "is_rush_am",   # 7-9am weekday
    "is_rush_pm",   # 16-19 weekday
    "is_night",     # 22-5
    "is_early_am",  # 5-7
    # weather
    "temp_c",
    "precip_mm",
    "wind_kmh",
    "snow_depth_mm",
    "is_raining",
    "is_snowing",
    "is_bad_weather",
    "wind_strong",
    # raw (kept for tree splits alongside cyclical)
    "hour",
    "dow",
    "month",
    "passenger_count",
]


def build_row(
    pickup_zone: int,
    dropoff_zone: int,
    requested_at: str | datetime,
    passenger_count: int,
    tables: LookupTables,
    centroids: dict[int, tuple[float, float]],
) -> list[float]:
    """Build one feature vector. Called at inference time (one row)."""

    pu = int(pickup_zone)
    do = int(dropoff_zone)

    if isinstance(requested_at, str):
        ts = datetime.fromisoformat(requested_at)
    else:
        ts = requested_at

    # --- zone-pair lookup ---
    pair = tables.pair_lookup(pu, do)
    pair_mean, pair_median, pair_p75, pair_p90, pair_log_count = pair

    # --- hour x zone-pair lookup ---
    hb = hour_bucket(ts.hour if isinstance(requested_at, datetime) else datetime.fromisoformat(requested_at).hour)
    hp = tables.hour_pair_lookup(pu, do, hb)
    hour_pair_mean, hour_pair_median = hp
    hour_pair_ratio = hour_pair_mean / pair_mean if pair_mean > 0 else 1.0

    # --- zone stats ---
    pu_s = tables.pu_stats.get(pu, [tables.global_mean, tables.global_std])
    do_s = tables.do_stats.get(do, [tables.global_mean, tables.global_std])

    # --- geometry ---
    c_pu = centroids.get(pu)
    c_do = centroids.get(do)
    if c_pu and c_do:
        dist_km = _haversine_km(*c_pu, *c_do)
        bear = _bearing(*c_pu, *c_do)
        bearing_sin = math.sin(math.radians(bear))
        bearing_cos = math.cos(math.radians(bear))
    else:
        dist_km = 0.0
        bearing_sin = 0.0
        bearing_cos = 0.0

    # --- time ---
    h = ts.hour
    dow = ts.weekday()   # 0=Mon, 6=Sun
    month = ts.month

    is_weekend = int(dow >= 5)
    is_holiday = int(ts.date() in _HOLIDAYS)
    is_rush_am = int((not is_weekend) and (7 <= h <= 9))
    is_rush_pm = int((not is_weekend) and (16 <= h <= 19))
    is_night = int(h >= 22 or h <= 4)
    is_early_am = int(5 <= h <= 6)

    # cyclical
    hour_sin  = math.sin(2 * math.pi * h / 24)
    hour_cos  = math.cos(2 * math.pi * h / 24)
    dow_sin   = math.sin(2 * math.pi * dow / 7)
    dow_cos   = math.cos(2 * math.pi * dow / 7)
    month_sin = math.sin(2 * math.pi * (month - 1) / 12)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12)

    # --- weather ---
    w = get_weather(ts)

    return [
        pu, do,
        pair_mean, pair_median, pair_p75, pair_p90, pair_log_count,
        hour_pair_mean, hour_pair_median, hour_pair_ratio,
        pu_s[0], pu_s[1],
        do_s[0], do_s[1],
        dist_km, bearing_sin, bearing_cos,
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        month_sin, month_cos,
        is_weekend, is_holiday, is_rush_am, is_rush_pm, is_night, is_early_am,
        w["temp_c"], w["precip_mm"], w["wind_kmh"], w["snow_depth_mm"],
        w["is_raining"], w["is_snowing"], w["is_bad_weather"], w["wind_strong"],
        h, dow, month,
        int(passenger_count),
    ]


def build_dataframe(df: pd.DataFrame, tables: LookupTables,
                     centroids: dict[int, tuple[float, float]]) -> np.ndarray:
    """Vectorised feature build for training (operates on full DataFrame)."""
    ts = pd.to_datetime(df["requested_at"])

    pu = df["pickup_zone"].astype(int).values
    do = df["dropoff_zone"].astype(int).values

    # Pair lookup (vectorised via list comprehension — fast enough on 37M rows)
    pairs = [tables.pair_lookup(int(p), int(d)) for p, d in zip(pu, do)]
    pairs = np.array(pairs, dtype=np.float32)

    # Hour x pair lookup
    hb_arr = np.array([_HOUR_BUCKET[hh] for hh in ts.dt.hour.values], dtype=np.int8)
    hour_pairs = [tables.hour_pair_lookup(int(p), int(d), int(hb))
                  for p, d, hb in zip(pu, do, hb_arr)]
    hour_pairs = np.array(hour_pairs, dtype=np.float32)
    pair_means = pairs[:, 0]
    hour_pair_ratio = np.where(pair_means > 0, hour_pairs[:, 0] / pair_means, 1.0).astype(np.float32)

    pu_arr = np.array([tables.pu_stats.get(int(z), [tables.global_mean, tables.global_std])
                       for z in pu], dtype=np.float32)
    do_arr = np.array([tables.do_stats.get(int(z), [tables.global_mean, tables.global_std])
                       for z in do], dtype=np.float32)

    # Geometry
    def geo(zones_a, zones_b, fn):
        out = np.zeros(len(zones_a), dtype=np.float32)
        for i, (a, b) in enumerate(zip(zones_a, zones_b)):
            ca, cb = centroids.get(int(a)), centroids.get(int(b))
            if ca and cb:
                out[i] = fn(*ca, *cb)
        return out

    dist_km      = geo(pu, do, _haversine_km)
    bearing_raw  = geo(pu, do, _bearing)
    bearing_sin  = np.sin(np.radians(bearing_raw))
    bearing_cos  = np.cos(np.radians(bearing_raw))

    h     = ts.dt.hour.values.astype(np.int8)
    dow   = ts.dt.dayofweek.values.astype(np.int8)
    month = ts.dt.month.values.astype(np.int8)
    dates = ts.dt.date.values

    is_weekend  = (dow >= 5).astype(np.int8)
    is_holiday  = np.array([int(d in _HOLIDAYS) for d in dates], dtype=np.int8)
    is_rush_am  = ((~is_weekend.astype(bool)) & (h >= 7) & (h <= 9)).astype(np.int8)
    is_rush_pm  = ((~is_weekend.astype(bool)) & (h >= 16) & (h <= 19)).astype(np.int8)
    is_night    = ((h >= 22) | (h <= 4)).astype(np.int8)
    is_early_am = ((h >= 5) & (h <= 6)).astype(np.int8)

    tau_h = 2 * np.pi * h / 24
    tau_d = 2 * np.pi * dow / 7
    tau_m = 2 * np.pi * (month - 1) / 12

    pc = df["passenger_count"].fillna(1).astype(np.int8).values

    # Weather lookup — pre-allocate float32 array to avoid 10GB float64 blowup
    weather_cache = load_weather()
    w_cols = ["temp_c", "precip_mm", "wind_kmh", "snow_depth_mm",
               "is_raining", "is_snowing", "is_bad_weather", "wind_strong"]
    n = len(df)
    weather_arr = np.zeros((n, len(w_cols)), dtype=np.float32)
    fallback_vals = np.array([_WEATHER_FALLBACK[c] for c in w_cols], dtype=np.float32)
    for i, (d, hh) in enumerate(zip(ts.dt.date.values, ts.dt.hour.values)):
        w = weather_cache.get((d, int(hh)))
        if w is not None:
            weather_arr[i] = [w[c] for c in w_cols]
        else:
            weather_arr[i] = fallback_vals

    # Build final matrix entirely in float32 — avoids numpy upcasting to float64
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    col = 0
    def put(*arrs):
        nonlocal col
        for a in arrs:
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 1:
                X[:, col] = a
                col += 1
            else:
                X[:, col:col + a.shape[1]] = a
                col += a.shape[1]

    put(pu, do)
    put(pairs)
    put(hour_pairs, hour_pair_ratio)
    put(pu_arr, do_arr)
    put(dist_km, bearing_sin, bearing_cos)
    put(np.sin(tau_h), np.cos(tau_h))
    put(np.sin(tau_d), np.cos(tau_d))
    put(np.sin(tau_m), np.cos(tau_m))
    put(is_weekend, is_holiday, is_rush_am, is_rush_pm, is_night, is_early_am)
    put(weather_arr)
    put(h, dow, month, pc)
    return X
