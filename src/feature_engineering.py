"""Time and strictly-past pollutant features for chronological forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .data_cleaning import POLLUTANTS

WEATHER_COLUMNS = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                   "wind_direction_10m", "surface_pressure", "precipitation",
                   "cloud_cover", "dew_point_2m")


def make_features(frame: pd.DataFrame, target: str = "pm2_5",
                  timezone: str | None = None) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex is required for time-series features.")
    if target not in frame:
        raise ValueError(f"Target column {target!r} is missing.")
    data = frame.sort_index().copy()
    idx = data.index
    local_index = (idx.tz_convert(timezone) if timezone and idx.tz is not None else idx)
    data["hour"] = local_index.hour
    data["day"] = local_index.day
    data["weekday"] = local_index.dayofweek
    data["weekend"] = (local_index.dayofweek >= 5).astype(int)
    data["week"] = local_index.isocalendar().week.astype(int)
    data["month"] = local_index.month
    data["quarter"] = local_index.quarter
    data["year"] = local_index.year
    data["season"] = ((local_index.month % 12) // 3).astype(int)
    hours = local_index.hour.to_numpy()
    data["rush_hour"] = (((hours >= 7) & (hours <= 10)) |
                         ((hours >= 16) & (hours <= 20))).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * local_index.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * local_index.hour / 24)
    day_of_year = local_index.dayofyear
    data["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    if "wind_direction_10m" in data:
        prior_direction = data["wind_direction_10m"].shift(1)
        radians = np.deg2rad(prior_direction)
        data["wind_direction_sin"] = np.sin(radians)
        data["wind_direction_cos"] = np.cos(radians)
    for column in WEATHER_COLUMNS:
        if column in data:
            prior = data[column].shift(1)
            data[f"{column}_lag_1h"] = prior
            if column in {"temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m"}:
                data[f"{column}_change_1h"] = prior.diff()
    if "precipitation" in data:
        prior_rain = data["precipitation"].shift(1)
        data["precipitation_3h"] = prior_rain.rolling(3, min_periods=3).sum()
        data["precipitation_24h"] = prior_rain.rolling(24, min_periods=24).sum()
    for pollutant in (name for name in POLLUTANTS if name in data):
        past = data[pollutant].shift(1)
        for name, periods in {"1h": 1, "3h": 3, "6h": 6, "12h": 12,
                              "24h": 24, "3d": 72, "7d": 168}.items():
            data[f"{pollutant}_lag_{name}"] = data[pollutant].shift(periods)
        for window in (3, 6, 24, 168):
            roll = past.rolling(window=window, min_periods=window)
            data[f"{pollutant}_rolling_mean_{window}h"] = roll.mean()
            data[f"{pollutant}_rolling_std_{window}h"] = roll.std()
            data[f"{pollutant}_rolling_min_{window}h"] = roll.min()
            data[f"{pollutant}_rolling_max_{window}h"] = roll.max()
        for name, lag in {"1h": 1, "3h": 3, "24h": 24}.items():
            earlier = data[pollutant].shift(lag + 1)
            data[f"{pollutant}_change_{name}"] = past - earlier
            data[f"{pollutant}_pct_change_{name}"] = (past - earlier) / earlier.replace(0, np.nan) * 100
    return data
