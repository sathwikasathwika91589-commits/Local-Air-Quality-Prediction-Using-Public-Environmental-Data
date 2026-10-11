"""Validation and canonicalization for observed environmental data."""

from __future__ import annotations

import pandas as pd
import numpy as np

POLLUTANTS = ("pm2_5", "pm10", "no2", "so2", "co", "o3")


def clean_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame:
        raise ValueError("Data must include a timestamp column.")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result = result.dropna(subset=["timestamp"]).sort_values("timestamp")
    result = result.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")
    for column in POLLUTANTS:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
            result.loc[(result[column] < 0) | ~np.isfinite(result[column]), column] = float("nan")
    return result
