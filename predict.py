"""Submission interface — this is what Gobblecube's grader imports.

The grader calls `predict` once per held-out request. The signature is fixed;
everything else is ours to change.

State loaded at import time (one-time cost):
    model.pkl -> { "model": lgb.Booster, "tables": LookupTables, "centroids": dict }
"""

from __future__ import annotations

import pickle
from pathlib import Path

from features import LookupTables, build_row

_MODEL_PATH = Path(__file__).parent / "model.pkl"

# Load once at import
with open(_MODEL_PATH, "rb") as _f:
    _artefact = pickle.load(_f)

_MODEL:     object              = _artefact["model"]
_TABLES:    LookupTables        = _artefact["tables"]
_CENTROIDS: dict                = _artefact["centroids"]


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    row = build_row(
        pickup_zone     = request["pickup_zone"],
        dropoff_zone    = request["dropoff_zone"],
        requested_at    = request["requested_at"],
        passenger_count = request["passenger_count"],
        tables          = _TABLES,
        centroids       = _CENTROIDS,
    )
    pred = _MODEL.predict([row], num_iteration=_MODEL.best_iteration)
    return float(max(30.0, pred[0]))   # hard floor: 30s (matches data cleaning)
