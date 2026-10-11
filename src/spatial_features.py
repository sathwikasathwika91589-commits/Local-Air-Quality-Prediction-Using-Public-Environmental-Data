"""Spatial helpers that only use supplied station metadata and observations."""

from __future__ import annotations

from math import radians, sin, cos, asin, sqrt
import pandas as pd
from .data_cleaning import POLLUTANTS
from .aqi_calculator import calculate_aqi_forecast


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    value = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(value))


def attach_spatial_features(history: pd.DataFrame, station_id: int | str,
                            latitude: float, longitude: float,
                            neighbor_locations: list[dict] | None = None,
                            neighbor_observations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add station coordinates and leakage-safe synchronized neighbor observations.

    Neighbor observations must contain real rows with timestamp, station_id, and pollutant
    columns. Only exact observations from the preceding hour are used; missing data stays NA.
    """
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("Station history must use a DatetimeIndex.")
    result = history.copy()
    result["latitude"] = float(latitude)
    result["longitude"] = float(longitude)
    # Station IDs are metadata, not numeric predictors; retain the value for audit/display.
    result.attrs["station_id"] = str(station_id)
    neighbors = [location for location in (neighbor_locations or [])
                 if str(location.get("id")) != str(station_id)
                 and location.get("coordinates", {}).get("latitude") is not None
                 and location.get("coordinates", {}).get("longitude") is not None]
    result.attrs["neighbor_stations"] = [{
        "station_id": str(location["id"]),
        "name": location.get("name"),
        "distance_km": distance_km(latitude, longitude,
                                    location["coordinates"]["latitude"],
                                    location["coordinates"]["longitude"]),
    } for location in neighbors]
    distances = [item["distance_km"] for item in result.attrs["neighbor_stations"]]
    result["nearest_neighbor_distance_km"] = min(distances) if distances else float("nan")
    if neighbor_observations is None or neighbor_observations.empty:
        for pollutant in POLLUTANTS:
            result[f"neighbor_{pollutant}"] = float("nan")
        result["neighbor_particulate_average"] = float("nan")
        result["neighbor_aqi"] = float("nan")
        result["neighbor_station_count"] = float("nan")
        result.attrs["neighbor_data_available"] = False
        return result
    required = {"timestamp", "station_id", *POLLUTANTS}
    missing = required.difference(neighbor_observations.columns)
    if missing:
        raise ValueError(f"Neighbor data missing columns: {', '.join(sorted(missing))}")
    observations = neighbor_observations.copy()
    observations["timestamp"] = pd.to_datetime(observations["timestamp"], utc=True, errors="coerce")
    observations = observations.dropna(subset=["timestamp"])
    observations["station_id"] = observations["station_id"].astype(str)
    neighbor_ids = {str(location["id"]) for location in neighbors}
    observations = observations[observations["station_id"].isin(neighbor_ids)]
    observations = observations.set_index("timestamp").sort_index()
    hourly_by_station = []
    for neighbor_id, station_rows in observations.groupby("station_id"):
        hourly = station_rows[list(POLLUTANTS)].resample("1h").last()
        hourly["station_id"] = neighbor_id
        hourly_by_station.append(hourly.reset_index())
    prior = pd.concat(hourly_by_station, ignore_index=True)
    prior["timestamp"] += pd.Timedelta(hours=1)
    grouped = prior.groupby("timestamp")
    for pollutant in POLLUTANTS:
        result[f"neighbor_{pollutant}"] = grouped[pollutant].mean().reindex(result.index)
    result["neighbor_particulate_average"] = result[["neighbor_pm2_5", "neighbor_pm10"]].mean(axis=1)
    result["neighbor_station_count"] = grouped["station_id"].nunique().reindex(result.index).fillna(0)
    station_aqi = []
    for neighbor_id, station_rows in observations.groupby("station_id"):
        hourly = station_rows.groupby(level=0)[list(POLLUTANTS)].mean().sort_index()
        hourly_aqi = calculate_aqi_forecast(hourly)
        hourly_aqi["timestamp"] = hourly_aqi.index + pd.Timedelta(hours=1)
        hourly_aqi["station_id"] = neighbor_id
        station_aqi.append(hourly_aqi.reset_index(drop=True))
    if station_aqi:
        aqi_rows = pd.concat(station_aqi, ignore_index=True)
        aqi_rows = aqi_rows.dropna(subset=["aqi"]).groupby("timestamp")["aqi"].mean()
        result["neighbor_aqi"] = aqi_rows.reindex(result.index)
    else:
        result["neighbor_aqi"] = float("nan")
    result.attrs["neighbor_data_available"] = bool(result["neighbor_station_count"].max() > 0)
    return result
