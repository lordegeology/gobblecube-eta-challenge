#!/usr/bin/env python
"""Download hourly weather data from NOAA for JFK and LGA stations.

Covers 2023-01-01 through 2024-03-31 (training + eval window).
Output: data/weather_hourly.csv — one row per (date, hour) with averaged
readings from both stations.

Run once before training:
    python data/download_weather.py

No API key required for NOAA LCD CSV endpoint.
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Station IDs (WBAN format for LCD CSV endpoint)
# ---------------------------------------------------------------------------
STATIONS = {
    "JFK": "72505394728",   # JFK International Airport
    "LGA": "72503014732",   # LaGuardia Airport
}

# Date range: full 2023 training set + 2024 eval window buffer
START_DATE = "2023-01-01"
END_DATE   = "2024-03-31"

OUTPUT_PATH = Path(__file__).parent / "weather_hourly.csv"

# NOAA LCD CSV API — no key needed, rate limit ~5 req/s
LCD_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1"
    "?dataset=local-climatological-data"
    "&stations={station}"
    "&startDate={start}T00:00:00"
    "&endDate={end}T23:59:59"
    "&dataTypes=HourlyDryBulbTemperature,HourlyPrecipitation,"
    "HourlyWindSpeed,HourlySnowDepth"
    "&format=csv"
    "&units=metric"
)

# ---------------------------------------------------------------------------
# Columns we want and their clean names
# ---------------------------------------------------------------------------
RAW_COLS = {
    "DATE":                       "datetime",
    "HourlyDryBulbTemperature":   "temp_c",
    "HourlyPrecipitation":        "precip_mm",
    "HourlyWindSpeed":            "wind_kmh",
    "HourlySnowDepth":            "snow_depth_mm",
}


def fetch_station(name: str, station_id: str) -> pd.DataFrame:
    url = LCD_URL.format(station=station_id, start=START_DATE, end=END_DATE)
    print(f"  Fetching {name} ({station_id})...")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                raw = r.read().decode("utf-8")
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"    Retry {attempt + 1}/3 after error: {e}")
            time.sleep(5)

    df = pd.read_csv(io.StringIO(raw), low_memory=False)

    # Keep only columns we care about
    keep = [c for c in RAW_COLS if c in df.columns]
    df = df[keep].rename(columns=RAW_COLS)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Keep only the top-of-hour observation (NOAA gives sub-hourly too)
    df["minute"] = df["datetime"].dt.minute
    df = df[df["minute"] == 0].copy()

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["station"] = name

    # Clean numeric cols — NOAA uses "T" for trace precipitation
    for col in ["temp_c", "precip_mm", "wind_kmh", "snow_depth_mm"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("T", "0.001", regex=False)  # trace -> near-zero
                .str.replace("s", "", regex=False)        # "special" flag
                .str.strip()
                .replace("", float("nan"))
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["date", "hour", "station", "temp_c", "precip_mm",
               "wind_kmh", "snow_depth_mm"]].copy()


def build_hourly_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average JFK + LGA readings per (date, hour)."""
    combined = pd.concat(frames, ignore_index=True)

    agg = (
        combined
        .groupby(["date", "hour"])[["temp_c", "precip_mm", "wind_kmh", "snow_depth_mm"]]
        .mean()
        .reset_index()
    )

    # Forward-fill missing hours (station sometimes skips an hour)
    # Build a complete grid first
    all_dates = pd.date_range(START_DATE, END_DATE, freq="D").date
    grid = pd.DataFrame(
        [(d, h) for d in all_dates for h in range(24)],
        columns=["date", "hour"],
    )
    agg = grid.merge(agg, on=["date", "hour"], how="left")
    agg = agg.sort_values(["date", "hour"])

    for col in ["temp_c", "precip_mm", "wind_kmh", "snow_depth_mm"]:
        agg[col] = agg[col].fillna(method="ffill").fillna(0.0)

    # Derived columns used as features
    agg["is_raining"] = (agg["precip_mm"] > 0.2).astype(int)
    agg["is_snowing"] = (agg["snow_depth_mm"] > 0.0).astype(int)
    agg["is_bad_weather"] = ((agg["precip_mm"] > 1.0) | (agg["snow_depth_mm"] > 0.0)).astype(int)
    agg["wind_strong"] = (agg["wind_kmh"] > 30.0).astype(int)

    return agg.reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, sid in STATIONS.items():
        df = fetch_station(name, sid)
        print(f"    {name}: {len(df):,} hourly rows")
        frames.append(df)
        time.sleep(1)  # be polite to NOAA

    print("\nBuilding combined hourly table...")
    weather = build_hourly_table(frames)

    weather.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(weather):,} rows to {OUTPUT_PATH}")
    print(f"\nSample (first 5 rows):")
    print(weather.head().to_string())

    # Quick stats
    print(f"\nRainy hours:      {weather['is_raining'].sum():,}")
    print(f"Snowy hours:      {weather['is_snowing'].sum():,}")
    print(f"Bad weather hrs:  {weather['is_bad_weather'].sum():,}")


if __name__ == "__main__":
    main()
