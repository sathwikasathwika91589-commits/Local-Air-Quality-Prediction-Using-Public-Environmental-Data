import numpy as np
import pandas as pd

from src.aqi_calculator import calculate_aqi, calculate_aqi_forecast


def test_cpcb_requires_three_valid_subindices_including_particulate():
    assert calculate_aqi({"no2": 20, "so2": 20, "co": 0.5})["aqi"] is None
    assert calculate_aqi({"pm2_5": 20, "no2": 20, "so2": 20})["aqi"] is not None


def test_aqi_forecast_uses_complete_pollutant_specific_windows():
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    data = pd.DataFrame({"pm2_5": 20.0, "pm10": 40.0, "no2": 20.0,
                         "so2": 20.0, "co": 0.5, "o3": 30.0}, index=index)
    result = calculate_aqi_forecast(data)
    assert result["aqi"].iloc[:23].isna().all()
    assert pd.notna(result["aqi"].iloc[23])
    data.iloc[-1, data.columns.get_loc("pm2_5")] = np.nan
    data.iloc[-1, data.columns.get_loc("pm10")] = np.nan
    assert pd.isna(calculate_aqi_forecast(data)["aqi"].iloc[-1])
