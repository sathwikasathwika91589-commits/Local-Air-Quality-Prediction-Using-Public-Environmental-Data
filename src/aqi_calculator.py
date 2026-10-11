"""Indian National AQI sub-index calculation using CPCB concentration ranges."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

# CPCB National AQI report: pollutant concentration breakpoints, μg/m³ except CO (mg/m³).
# The 8-hour averaging requirement for CO and O3 is not inferred from a single instant.
BREAKPOINTS = {
    "pm2_5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
             (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "no2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
            (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)],
    "so2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
            (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "co": [(0, 0.5, 0, 50), (0.6, 1.0, 51, 100), (1.1, 2.0, 101, 200),
           (2.1, 3.0, 201, 300), (3.1, 3.4, 301, 400), (3.5, 5.0, 401, 500)],
    "o3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
           (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
}
CATEGORIES = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
              (300, "Poor"), (400, "Very Poor"), (500, "Severe")]
HEALTH_RISK = {
    "Good": "Minimal impact.",
    "Satisfactory": "May cause minor breathing discomfort to sensitive people.",
    "Moderate": "May cause breathing discomfort to people with lung disease, children, older adults, or heart disease.",
    "Poor": "May cause breathing discomfort on prolonged exposure, especially for people with heart disease.",
    "Very Poor": "May cause respiratory illness on prolonged exposure; effects may be stronger for people with lung or heart disease.",
    "Severe": "May cause respiratory effects in healthy people and serious impacts for people with lung or heart disease.",
}


def pollutant_sub_index(pollutant: str, concentration: float | None) -> int | None:
    if pollutant not in BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant: {pollutant}")
    if concentration is None or not math.isfinite(float(concentration)) or float(concentration) < 0:
        return None
    precision = Decimal("0.1") if pollutant == "co" else Decimal("1")
    concentration = float(Decimal(str(concentration)).quantize(precision, rounding=ROUND_HALF_UP))
    for low, high, ilo, ihi in BREAKPOINTS[pollutant]:
        if low <= concentration <= high:
            return round((ihi - ilo) / (high - low) * (concentration - low) + ilo)
    if concentration > BREAKPOINTS[pollutant][-1][1]:
        return 500
    return None


def calculate_aqi(concentrations: dict[str, float | None]) -> dict:
    indices = {name: pollutant_sub_index(name, concentrations.get(name)) for name in BREAKPOINTS}
    valid = [value for value in indices.values() if value is not None]
    particulate_available = any(indices.get(name) is not None for name in ("pm2_5", "pm10"))
    overall = max(valid) if len(valid) >= 3 and particulate_available else None
    category = next((name for threshold, name in CATEGORIES if overall is not None and overall <= threshold), None)
    return {"aqi": overall, "category": category, "sub_indices": indices,
            "health_risk": HEALTH_RISK.get(category),
            "available_pollutants": len(valid), "required_pollutants": 3}


AVERAGING_HOURS = {"pm2_5": 24, "pm10": 24, "no2": 24, "so2": 24, "co": 8, "o3": 8}


def calculate_aqi_forecast(hourly: "pd.DataFrame") -> "pd.DataFrame":
    """Calculate CPCB AQI from trailing pollutant averages in an hourly series.

    Every pollutant window must be complete; no gaps are interpolated. Input pollutant
    units must be canonical: μg/m³, except CO in mg/m³.
    """
    import pandas as pd

    if not isinstance(hourly.index, pd.DatetimeIndex):
        raise ValueError("AQI forecast input must use a DatetimeIndex.")
    data = hourly.sort_index()
    rows = []
    for timestamp in data.index:
        averages = {}
        for pollutant, hours in AVERAGING_HOURS.items():
            if pollutant not in data:
                averages[pollutant] = None
                continue
            window = data.loc[:timestamp, pollutant].tail(hours)
            averages[pollutant] = (float(window.mean())
                                   if len(window) == hours and window.notna().all()
                                   else None)
        result = calculate_aqi(averages)
        dominant = max((key for key, value in result["sub_indices"].items() if value is not None),
                       key=result["sub_indices"].get, default=None)
        rows.append({"timestamp": timestamp, "aqi": result["aqi"],
                     "category": result["category"], "health_risk": result["health_risk"],
                     "available_pollutants": result["available_pollutants"],
                     "dominant_pollutant": dominant,
                     **{f"sub_index_{key}": value for key, value in result["sub_indices"].items()}})
    return pd.DataFrame(rows).set_index("timestamp")
