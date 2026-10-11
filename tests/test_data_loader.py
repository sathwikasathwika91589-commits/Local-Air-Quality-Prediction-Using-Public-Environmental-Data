import pandas as pd
import pytest

from src.data_loader import (DataSourceError, fetch_location_hourly_history,
                             fetch_historical_weather, normalize_latest_measurements,
                             normalize_pollutant_unit)


def test_units_are_converted_or_rejected_explicitly():
    assert normalize_pollutant_unit("co", "µg/m³") == (0.001, "mg/m3")
    assert normalize_pollutant_unit("pm2_5", "ppm") is None
    values = normalize_latest_measurements([{
        "parameter": {"name": "pm25", "units": "µg/m³"}, "value": 12.5,
        "datetime": {"utc": "2026-01-01T00:00:00Z"}, "sensorsId": 8,
    }])
    assert values["pm2_5"]["value"] == 12.5
    assert values["pm2_5"]["unit"] == "ug/m3"


def test_history_loader_reports_real_coverage_and_normalizes_co(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        if url.endswith("/locations/42/sensors"):
            return Response({"results": [
                {"id": 1, "parameter": {"name": "pm25", "units": "µg/m³"}},
                {"id": 2, "parameter": {"name": "co", "units": "µg/m³"}},
                {"id": 3, "parameter": {"name": "pm10", "units": "ppm"}},
            ]})
        if url.endswith("/sensors/1/hours"):
            pollutant_value = 20.0
        elif url.endswith("/sensors/2/hours"):
            pollutant_value = 2500.0
        else:
            raise AssertionError(url)
        stamp = pd.Timestamp(kwargs["params"]["datetime_from"]).floor("h").isoformat().replace("+00:00", "Z")
        return Response({"results": [{"value": pollutant_value,
                                       "period": {"datetimeFrom": {"utc": stamp}}}]})

    monkeypatch.setattr("src.data_loader.requests.get", fake_get)
    data = fetch_location_hourly_history(42, "test-key", days=1)
    first_stamp = data.index[0]
    assert data.loc[first_stamp, "pm2_5"] == 20
    assert data.loc[first_stamp, "co"] == 2.5
    assert data.attrs["coverage"]["pm2_5"]["coverage_percent"] < 100
    assert data.attrs["coverage"]["pm10"]["observed_hours"] == 0
    assert "pm10: ppm" in data.attrs["unsupported_units"]


def test_open_aq_key_is_required_before_network_request():
    with pytest.raises(DataSourceError, match="OPENAQ_API_KEY"):
        fetch_location_hourly_history(42, None)


def test_historical_weather_reports_archive_and_recent_model_coverage(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        if "archive" in url:
            from_date = pd.Timestamp(kwargs["params"]["start_date"], tz="UTC")
            to_date = pd.Timestamp(kwargs["params"]["end_date"], tz="UTC")
            times = pd.date_range(from_date, to_date + pd.Timedelta(hours=23), freq="h")
        else:
            now = pd.Timestamp.now(tz="UTC")
            times = pd.date_range(now.floor("D") - pd.Timedelta(days=5), periods=6 * 24, freq="h")
        return Response({"hourly": {"time": times.strftime("%Y-%m-%dT%H:00").tolist(),
                                     "temperature_2m": [15.0] * len(times),
                                     "relative_humidity_2m": [50.0] * len(times),
                                     "wind_speed_10m": [2.0] * len(times),
                                     "wind_direction_10m": [180.0] * len(times),
                                     "surface_pressure": [1000.0] * len(times),
                                     "precipitation": [0.0] * len(times),
                                     "cloud_cover": [10.0] * len(times),
                                     "dew_point_2m": [5.0] * len(times)}})

    monkeypatch.setattr("src.data_loader.requests.get", fake_get)
    end = pd.Timestamp.now(tz="UTC").to_pydatetime()
    start = end - pd.Timedelta(days=7).to_pytimedelta()
    weather = fetch_historical_weather(28.6, 77.2, start, end)
    assert weather.attrs["archive_rows"] > 0
    assert weather.attrs["recent_model_rows"] > 0
    assert weather.attrs["coverage_percent"] > 95
    assert weather.index.tz is not None
