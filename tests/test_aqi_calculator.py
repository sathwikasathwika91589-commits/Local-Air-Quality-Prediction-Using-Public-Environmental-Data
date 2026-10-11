import pytest

from src.aqi_calculator import calculate_aqi, pollutant_sub_index


def test_known_cpcb_pm25_breakpoints():
    assert pollutant_sub_index("pm2_5", 30) == 50
    assert pollutant_sub_index("pm2_5", 30.5) == 51
    assert pollutant_sub_index("pm2_5", 60) == 100
    assert pollutant_sub_index("pm2_5", -1) is None


def test_overall_index_needs_three_subindices():
    assert calculate_aqi({"pm2_5": 30, "pm10": 50})["aqi"] is None
    result = calculate_aqi({"pm2_5": 30, "pm10": 50, "no2": 40})
    assert result["aqi"] == 50
    assert result["category"] == "Good"


def test_unknown_pollutant_fails_loudly():
    with pytest.raises(ValueError):
        pollutant_sub_index("noise", 1)


def test_co_subindex_uses_cpcb_mg_per_cubic_metre_ranges():
    assert pollutant_sub_index("co", 0.5) == 50
    assert pollutant_sub_index("co", 1.0) == 100
    assert pollutant_sub_index("co", 2.0) == 200
    assert pollutant_sub_index("co", 2.5) == 245
    assert pollutant_sub_index("co", 4.0) > 400
    assert pollutant_sub_index("co", 10.0) == 500
