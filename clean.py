"""Data cleaning for the ETA challenge.

The download_data.py script applies only basic cleaning:
  - duration between 30s and 3 hours
  - zones between 1 and 265
  - year == 2023
  - fills missing passenger_count with 1

This module applies a second pass to remove noise that survives the basic
filter but will hurt model quality. Called from train.py before building
lookup tables — so the zone-pair stats are computed on clean data.

Each filter is logged separately so we know exactly how much each one removes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean(
    df: pd.DataFrame,
    centroids: dict[int, tuple[float, float]],
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply all cleaning passes. Returns a filtered copy of df."""

    original_len = len(df)
    log = []

    def report(label: str, mask: pd.Series) -> pd.Series:
        removed = int((~mask).sum())
        log.append(f"  {label:<45} removed {removed:>8,} rows")
        return mask

    # ------------------------------------------------------------------ #
    # 1. Passenger count — cap at 6 (max legal NYC taxi capacity)
    #    Values of 0, 7, 8, 9 etc. are data entry errors.
    # ------------------------------------------------------------------ #
    mask_pax = report(
        "passenger_count in [1, 6]",
        df["passenger_count"].between(1, 6),
    )

    # ------------------------------------------------------------------ #
    # 2. Duplicate trips
    #    Exact same (pickup_zone, dropoff_zone, requested_at, passenger_count)
    #    appearing more than once is almost certainly a logging artifact.
    # ------------------------------------------------------------------ #
    dup_cols = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count"]
    mask_dup = report(
        "drop duplicate rows",
        ~df.duplicated(subset=dup_cols, keep="first"),
    )

    # Apply filters so far before computing derived columns
    df = df.loc[mask_pax & mask_dup].copy()

    # ------------------------------------------------------------------ #
    # 3. Speed-based outlier removal
    #    Compute implied speed = dist_km / (duration_seconds / 3600).
    #    We need centroids for this; if we don't have them, skip.
    # ------------------------------------------------------------------ #
    if centroids:
        from features import _haversine_km

        pu_lat = np.array([centroids.get(z, (0, 0))[0] for z in df["pickup_zone"]])
        pu_lon = np.array([centroids.get(z, (0, 0))[1] for z in df["pickup_zone"]])
        do_lat = np.array([centroids.get(z, (0, 0))[0] for z in df["dropoff_zone"]])
        do_lon = np.array([centroids.get(z, (0, 0))[1] for z in df["dropoff_zone"]])

        # Vectorised haversine
        R = 6371.0
        phi1 = np.radians(pu_lat)
        phi2 = np.radians(do_lat)
        dphi = np.radians(do_lat - pu_lat)
        dlam = np.radians(do_lon - pu_lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        dist_km = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        hours = df["duration_seconds"].values / 3600.0
        speed_kmh = np.where(hours > 0, dist_km / hours, 0.0)

        # Flag same-zone trips: pickup == dropoff
        same_zone = (df["pickup_zone"].values == df["dropoff_zone"].values)

        # Upper speed limit: 80 km/h average is physically impossible in NYC traffic
        mask_speed_hi = report(
            "implied speed <= 80 km/h",
            pd.Series(speed_kmh <= 80.0, index=df.index),
        )

        # Lower speed limit for cross-zone trips: < 1 km/h means the meter
        # was left running (e.g. driver forgot to end the trip)
        mask_speed_lo = report(
            "cross-zone implied speed >= 1 km/h",
            pd.Series(same_zone | (speed_kmh >= 1.0), index=df.index),
        )

        # Same-zone very long trips: > 45 min for a within-zone trip is suspicious
        mask_same_zone_dur = report(
            "same-zone duration <= 45 min",
            pd.Series(~same_zone | (df["duration_seconds"].values <= 2700), index=df.index),
        )

        df = df.loc[mask_speed_hi & mask_speed_lo & mask_same_zone_dur].copy()

    else:
        print("  Skipping speed-based filters (no centroid data)")

    # # ------------------------------------------------------------------ #
    # # 4. Duration tails — winsorise extreme values that survived basic cleaning
    # #    The download script allows up to 3h; we tighten to p99.9 per zone-pair
    # #    direction (long vs short trip) to remove lingering outliers without
    # #    being too aggressive.
    # #    Simple global percentile clip is fast and effective enough here.
    # # ------------------------------------------------------------------ #
    # p001 = df["duration_seconds"].quantile(0.001)
    # p999 = df["duration_seconds"].quantile(0.999)
    # mask_dur = report(
    #     f"duration in [{p001:.0f}s, {p999:.0f}s] (p0.1–p99.9)",
    #     df["duration_seconds"].between(p001, p999),
    # )
    # df = df.loc[mask_dur].copy()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    final_len = len(df)
    removed_total = original_len - final_len
    pct = 100 * removed_total / original_len

    if verbose:
        print("\n  Cleaning summary:")
        for line in log:
            print(line)
        print(f"\n  Total removed: {removed_total:,} / {original_len:,} ({pct:.2f}%)")
        print(f"  Remaining:     {final_len:,} rows")

    return df.reset_index(drop=True)
