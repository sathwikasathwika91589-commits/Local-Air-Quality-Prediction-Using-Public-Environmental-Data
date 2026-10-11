import pandas as pd

from src.spatial_features import attach_spatial_features, distance_km


def test_distance_and_unavailable_neighbor_state():
    assert distance_km(0, 0, 0, 1) == pytest.approx(111.2, abs=0.2)
    history = pd.DataFrame({"pm2_5": [10.0]}, index=pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC"))
    result = attach_spatial_features(history, 10, 0, 0, neighbor_locations=[])
    assert not result.attrs["neighbor_data_available"]
    assert pd.isna(result["neighbor_pm2_5"].iloc[0])
    assert pd.isna(result["neighbor_aqi"].iloc[0])
    assert result["latitude"].iloc[0] == 0


def test_neighbor_values_are_lagged_from_real_prior_hour():
    import pytest
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    history = pd.DataFrame({"pm2_5": [10.0, 11.0]}, index=index)
    locations = [{"id": 10, "coordinates": {"latitude": 0, "longitude": 0}},
                 {"id": 11, "coordinates": {"latitude": 0, "longitude": 1}}]
    observations = pd.DataFrame({"timestamp": [index[0]], "station_id": [11],
                                 "pm2_5": [22.0], "pm10": [40.0], "no2": [1.0],
                                 "so2": [1.0], "co": [0.1], "o3": [10.0]})
    result = attach_spatial_features(history, 10, 0, 0, locations, observations)
    assert pd.isna(result["neighbor_pm2_5"].iloc[0])
    assert result["neighbor_pm2_5"].iloc[1] == 22


def test_neighbor_aqi_uses_complete_sourced_station_series():
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    history = pd.DataFrame({"pm2_5": [1.0]}, index=index[-1:] + pd.Timedelta(hours=1))
    locations = [{"id": 10, "coordinates": {"latitude": 0, "longitude": 0}},
                 {"id": 11, "coordinates": {"latitude": 0, "longitude": 1}}]
    observations = pd.DataFrame({"timestamp": index, "station_id": 11,
                                 "pm2_5": 20.0, "pm10": 40.0, "no2": 20.0,
                                 "so2": 20.0, "co": 0.5, "o3": 30.0})
    result = attach_spatial_features(history, 10, 0, 0, locations, observations)
    assert result["neighbor_aqi"].iloc[0] == 50


import pytest
