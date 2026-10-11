import numpy as np
import pandas as pd

from src.data_cleaning import clean_timeseries
from src.feature_engineering import make_features


def hourly_frame(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"pm2_5": np.arange(n, dtype=float)}, index=idx)


def test_lags_and_rolling_use_past_only():
    result = make_features(hourly_frame())
    assert result["pm2_5_lag_1h"].iloc[24] == 23
    assert result["pm2_5_rolling_mean_3h"].iloc[24] == 22
    assert np.isnan(result["pm2_5_lag_7d"].iloc[100])


def test_time_features_include_cyclic_representation():
    result = make_features(hourly_frame())
    assert {"hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"}.issubset(result)


def test_weather_predictors_use_previous_hour():
    frame = hourly_frame(30)
    frame["temperature_2m"] = np.arange(30, dtype=float)
    frame["wind_direction_10m"] = np.arange(30, dtype=float) * 10
    result = make_features(frame)
    assert result["temperature_2m_lag_1h"].iloc[10] == 9
    assert result["wind_direction_sin"].iloc[10] == np.sin(np.deg2rad(90))


def test_current_row_target_and_weather_do_not_enter_their_predictors():
    frame = hourly_frame(30)
    frame["temperature_2m"] = np.arange(30, dtype=float)
    timestamp = frame.index[15]
    baseline = make_features(frame)
    changed = frame.copy()
    changed.loc[timestamp, "pm2_5"] = 9000
    changed.loc[timestamp, "temperature_2m"] = 9000
    audited = make_features(changed)
    assert baseline.loc[timestamp, "pm2_5_lag_1h"] == audited.loc[timestamp, "pm2_5_lag_1h"]
    assert baseline.loc[timestamp, "temperature_2m_lag_1h"] == audited.loc[timestamp, "temperature_2m_lag_1h"]


def test_temporal_features_use_station_local_timezone():
    index = pd.date_range("2026-01-01 01:00", periods=200, freq="h", tz="UTC")
    frame = pd.DataFrame({"pm2_5": np.arange(200, dtype=float)}, index=index)
    result = make_features(frame, timezone="Asia/Kolkata")
    assert result["hour"].iloc[0] == 6


def test_cleaning_parses_drops_bad_time_and_negative_measurement():
    raw = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z", "bad"], "pm2_5": [-2, 10]})
    result = clean_timeseries(raw)
    assert len(result) == 1
    assert pd.isna(result["pm2_5"].iloc[0])
