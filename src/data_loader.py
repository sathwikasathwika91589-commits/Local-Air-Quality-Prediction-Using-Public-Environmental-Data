"""Public environmental data clients. Missing data is surfaced as an error."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPEN_METEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

POLLUTANT_ALIASES = {"pm25": "pm2_5", "pm2.5": "pm2_5", "pm2_5": "pm2_5",
                     "pm10": "pm10", "no2": "no2", "nitrogen dioxide": "no2",
                     "so2": "so2", "sulfur dioxide": "so2", "sulphur dioxide": "so2",
                     "co": "co", "carbon monoxide": "co", "o3": "o3", "ozone": "o3"}


def normalize_pollutant_unit(pollutant: str, unit: str) -> tuple[float, str] | None:
    """Return scale-to-canonical and canonical unit, or None if unsupported."""
    value = (unit.strip().lower().replace("μ", "u").replace("µ", "u").replace("³", "3")
             .replace(" ", ""))
    if pollutant in {"pm2_5", "pm10", "no2", "so2", "o3"}:
        if value in {"ug/m3", "ug/m^3", "microgram/m3", "micrograms/m3"}:
            return 1.0, "ug/m3"
        return None
    if pollutant == "co":
        if value in {"mg/m3", "mg/m^3"}:
            return 1.0, "mg/m3"
        if value in {"ug/m3", "ug/m^3"}:
            return 0.001, "mg/m3"
    return None


def normalize_latest_measurements(results: list[dict]) -> dict[str, dict]:
    """Normalize most recent pollutant readings while retaining source timestamps."""
    latest: dict[str, dict] = {}
    for row in results:
        parameter = row.get("parameter") or {}
        pollutant = POLLUTANT_ALIASES.get(str(parameter.get("name", "")).lower())
        if pollutant is None or row.get("value") is None:
            continue
        conversion = normalize_pollutant_unit(pollutant, str(parameter.get("units", "")))
        if conversion is None or float(row["value"]) < 0:
            continue
        scale, canonical_unit = conversion
        datetime_info = row.get("datetime") or {}
        stamp = datetime_info.get("utc")
        previous = latest.get(pollutant, {})
        if stamp and previous.get("datetime_utc") and stamp <= previous["datetime_utc"]:
            continue
        latest[pollutant] = {"value": float(row["value"]) * scale,
                             "unit": canonical_unit, "datetime_utc": stamp,
                             "sensor_id": row.get("sensorsId") or row.get("sensorId"),
                             "source_unit": parameter.get("units")}
    return latest


class DataSourceError(RuntimeError):
    """Raised when a public data source cannot provide usable data."""


def find_nearby_locations(latitude: float, longitude: float, api_key: str | None,
                          radius_m: int = 25000) -> list[dict]:
    if not api_key:
        raise DataSourceError("OpenAQ needs an API key. Set OPENAQ_API_KEY in .env.")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180 and 0 < radius_m <= 25000):
        raise ValueError("Coordinates or OpenAQ radius are outside supported ranges.")
    try:
        response = requests.get(
            f"{OPENAQ_BASE_URL}/locations",
            params={"coordinates": f"{latitude},{longitude}", "radius": radius_m,
                    "limit": 100, "monitor": "true"},
            headers={"X-API-Key": api_key}, timeout=20,
        )
        response.raise_for_status()
        return response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        raise DataSourceError(f"OpenAQ location request failed: {exc}") from exc


def fetch_location_latest(location_id: int, api_key: str | None) -> list[dict]:
    if not api_key:
        raise DataSourceError("OpenAQ needs an API key. Set OPENAQ_API_KEY in .env.")
    try:
        response = requests.get(
            f"{OPENAQ_BASE_URL}/locations/{location_id}/latest",
            headers={"X-API-Key": api_key}, timeout=20,
        )
        response.raise_for_status()
        return response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        raise DataSourceError(f"OpenAQ latest measurements request failed: {exc}") from exc


def fetch_location_hourly_history(location_id: int, api_key: str | None,
                                  days: int = 365) -> pd.DataFrame:
    """Load validated hourly pollutant readings, normalizing supported units.

    Unsupported units are skipped instead of silently mixing scales. Gaps remain NaN.
    """
    if not api_key:
        raise DataSourceError("OpenAQ needs an API key. Set OPENAQ_API_KEY in .env.")
    if days < 1 or days > 365:
        raise ValueError("History window must be between 1 and 365 days.")
    headers = {"X-API-Key": api_key}
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    params = {"datetime_from": start.isoformat(),
              "datetime_to": now.isoformat(), "limit": 1000}
    try:
        sensors_response = requests.get(
            f"{OPENAQ_BASE_URL}/locations/{location_id}/sensors",
            headers=headers, timeout=20,
        )
        sensors_response.raise_for_status()
        sensors = sensors_response.json().get("results", [])
        records = []
        unsupported = set()
        for sensor in sensors:
            parameter = sensor.get("parameter", {})
            pollutant = POLLUTANT_ALIASES.get(str(parameter.get("name", "")).lower())
            if not pollutant:
                continue
            unit = str(parameter.get("units", ""))
            conversion = normalize_pollutant_unit(pollutant, unit)
            if conversion is None:
                unsupported.add(f"{pollutant}: {unit or 'unit not provided'}")
                continue
            scale, canonical_unit = conversion
            page = 1
            while True:
                response = requests.get(
                    f"{OPENAQ_BASE_URL}/sensors/{sensor['id']}/hours",
                    headers=headers, params={**params, "page": page}, timeout=20,
                )
                response.raise_for_status()
                result = response.json()
                measurements = result.get("results", [])
                for item in measurements:
                    period = item.get("period", {}).get("datetimeFrom", {}).get("utc")
                    value = item.get("value")
                    if period and value is not None and float(value) >= 0:
                        records.append({"timestamp": period, "pollutant": pollutant,
                                        "value": float(value) * scale,
                                        "sensor_id": sensor["id"], "source_unit": unit,
                                        "canonical_unit": canonical_unit})
                if len(measurements) < 1000:
                    break
                page += 1
        if not records:
            suffix = f" Unsupported units: {', '.join(sorted(unsupported))}." if unsupported else ""
            raise DataSourceError("No hourly observations with supported pollutant units were returned." + suffix)
        long = pd.DataFrame(records)
        # Avoid combining duplicate sensors with unlike measurement units; canonical scaling above
        # aligns supported units, after which station-hour values are averaged.
        wide = long.pivot_table(index="timestamp", columns="pollutant", values="value", aggfunc="mean")
        wide.index = pd.to_datetime(wide.index, utc=True)
        wide.index.name = "timestamp"
        hourly_index = pd.date_range(pd.Timestamp(start).floor("h"),
                                     pd.Timestamp(now).floor("h"), freq="h", tz="UTC")
        wide = wide.reindex(hourly_index).rename_axis("timestamp")
        for pollutant in POLLUTANT_ALIASES.values():
            if pollutant not in wide:
                wide[pollutant] = float("nan")
        wide = wide.loc[:, ["pm2_5", "pm10", "no2", "so2", "co", "o3"]]
        expected = len(wide)
        wide.attrs["source"] = "OpenAQ API v3 hourly sensor observations"
        wide.attrs["coverage"] = {
            pollutant: {"observed_hours": int(wide[pollutant].notna().sum()),
                        "expected_hours": expected,
                        "coverage_percent": (100 * float(wide[pollutant].notna().mean()) if expected else 0.0),
                        "unit": "mg/m3" if pollutant == "co" else "ug/m3"}
            for pollutant in wide.columns
        }
        wide.attrs["unsupported_units"] = sorted(unsupported)
        return wide
    except DataSourceError:
        raise
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DataSourceError(f"OpenAQ hourly history request failed: {exc}") from exc


def fetch_openmeteo(latitude: float, longitude: float, forecast_days: int = 5) -> tuple[dict, dict]:
    """Return real gridded forecasts and recent history when available."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Coordinates are outside valid latitude/longitude ranges.")
    common = {"latitude": latitude, "longitude": longitude, "timezone": "UTC"}
    forecast_days = max(1, min(7, forecast_days))
    air_params = {**common, "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide"}
    air_params.update({"past_days": 5, "forecast_days": forecast_days})
    weather_params = {**common, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation,cloud_cover,dew_point_2m"}
    weather_params.update({"past_days": 5, "forecast_days": forecast_days,
                           "wind_speed_unit": "ms"})
    try:
        air = requests.get(OPEN_METEO_AIR_URL, params=air_params, timeout=20)
        weather = requests.get(OPEN_METEO_WEATHER_URL, params=weather_params, timeout=20)
        air.raise_for_status()
        weather.raise_for_status()
        return air.json(), weather.json()
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        raise DataSourceError(f"Open-Meteo data request failed: {exc}") from exc


def fetch_historical_weather(latitude: float, longitude: float,
                            start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch historical reanalysis and recent gridded model weather when available."""
    variables = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation,cloud_cover,dew_point_2m"
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Coordinates are outside valid latitude/longitude ranges.")
    if start > end:
        raise ValueError("Weather start time must not be after end time.")
    now = datetime.now(timezone.utc)
    archive_cutoff = now.date() - timedelta(days=5)
    archive_end = min(end.date(), archive_cutoff)
    frames = []
    archive_rows = 0
    recent_rows = 0
    recent_status = "not_requested"
    try:
        if start.date() <= archive_end:
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params={
                "latitude": latitude, "longitude": longitude,
                "start_date": start.date().isoformat(), "end_date": archive_end.isoformat(),
                "hourly": variables, "timezone": "UTC", "wind_speed_unit": "ms",
                "models": "era5_land",
            }, timeout=30)
            response.raise_for_status()
            hourly = response.json().get("hourly", {})
            if hourly.get("time"):
                archive = pd.DataFrame(hourly)
                archive["time"] = pd.to_datetime(archive["time"], utc=True)
                archive = archive.set_index("time").rename_axis("timestamp")
                archive_rows = len(archive)
                frames.append(archive)
        recent_start = max(start.date(), archive_end + timedelta(days=1))
        if recent_start <= end.date():
            recent_status = "unavailable"
            try:
                response = requests.get(OPEN_METEO_WEATHER_URL, params={
                    "latitude": latitude, "longitude": longitude, "timezone": "UTC",
                    "hourly": variables, "past_days": 5, "forecast_days": 1,
                    "wind_speed_unit": "ms",
                }, timeout=30)
                response.raise_for_status()
                hourly = response.json().get("hourly", {})
                if hourly.get("time"):
                    recent = pd.DataFrame(hourly)
                    recent["time"] = pd.to_datetime(recent["time"], utc=True)
                    recent = recent.set_index("time").rename_axis("timestamp")
                    recent = recent.loc[(recent.index.date >= recent_start) &
                                        (recent.index.date <= end.date())]
                    recent_rows = len(recent)
                    if recent_rows:
                        frames.append(recent)
                        recent_status = "available"
                    else:
                        recent_status = "no rows returned"
            except (requests.RequestException, ValueError, TypeError, AttributeError):
                # The reanalysis remains usable; recent gaps stay visibly missing.
                recent_status = "request failed"
        if not frames:
            raise DataSourceError("Open-Meteo returned no historical weather rows.")
        frame = pd.concat(frames).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        hourly_index = pd.date_range(pd.Timestamp(start).floor("h"),
                                     pd.Timestamp(end).floor("h"), freq="h", tz="UTC")
        frame = frame.reindex(hourly_index).rename_axis("timestamp")
        expected_rows = len(frame)
        frame.attrs["source"] = "Open-Meteo ERA5-Land reanalysis; recent forecast-model values where available"
        frame.attrs["archive_rows"] = archive_rows
        frame.attrs["recent_model_rows"] = recent_rows
        frame.attrs["recent_model_status"] = recent_status
        frame.attrs["coverage_percent"] = (100 * float(frame["temperature_2m"].notna().mean())
                                            if expected_rows and "temperature_2m" in frame else 0.0)
        return frame
    except DataSourceError:
        raise
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        raise DataSourceError(f"Open-Meteo historical weather request failed: {exc}") from exc
