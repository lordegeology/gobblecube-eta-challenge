"""Feature engineering for the ETA challenge.

All feature construction lives here so training (train.py) and inference
(predict.py) use identical logic. Add a feature once; it appears everywhere.

Feature groups
--------------
1. Zone-pair lookup stats   — mean/median/p75/p90/count per (pu, do) pair
2. Zone-level stats         — mean duration when this zone is pickup or dropoff
3. Zone centroid geometry   — haversine distance, bearing from TLC shapefile
4. Time features            — cyclical hour/dow/month, rush-hour flag, weekend, holiday
5. Raw identifiers          — pickup_zone, dropoff_zone (for tree splits)
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
# Lookup table helpers
# ---------------------------------------------------------------------------

class LookupTables:
    """Precomputed aggregation tables built from training data.

    Stored as plain dicts so they pickle fast and have zero import deps.
    """

    def __init__(self) -> None:
        # (pickup_zone, dropoff_zone) -> [mean, median, p75, p90, log_count]
        self.pair_stats: dict[tuple[int, int], list[float]] = {}
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

        return self

    def pair_lookup(self, pu: int, do: int) -> list[float]:
        """Return [mean, median, p75, p90, log_count] for zone pair, with fallback."""
        v = self.pair_stats.get((pu, do))
        if v is not None:
            return v
        # Fallback: try pickup-only mean, then global
        pu_m = self.pu_stats.get(pu, [self.global_mean, self.global_std])[0]
        return [pu_m, pu_m, pu_m * 1.15, pu_m * 1.3, 0.0]


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

    return [
        pu, do,
        pair_mean, pair_median, pair_p75, pair_p90, pair_log_count,
        pu_s[0], pu_s[1],
        do_s[0], do_s[1],
        dist_km, bearing_sin, bearing_cos,
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        month_sin, month_cos,
        is_weekend, is_holiday, is_rush_am, is_rush_pm, is_night, is_early_am,
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

    X = np.column_stack([
        pu, do,
        pairs,           # 5 cols
        pu_arr,          # 2 cols
        do_arr,          # 2 cols
        dist_km, bearing_sin, bearing_cos,
        np.sin(tau_h), np.cos(tau_h),
        np.sin(tau_d), np.cos(tau_d),
        np.sin(tau_m), np.cos(tau_m),
        is_weekend, is_holiday, is_rush_am, is_rush_pm, is_night, is_early_am,
        h, dow, month,
        pc,
    ])
    return X.astype(np.float32)
