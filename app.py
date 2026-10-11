"""Streamlit entry point for the local air-quality project."""

import pandas as pd
import streamlit as st

from src.config import Settings
from src.data_loader import (DataSourceError, fetch_location_latest, find_nearby_locations,
                             fetch_openmeteo, fetch_location_hourly_history,
                             fetch_historical_weather, normalize_latest_measurements)
from src.spatial_features import attach_spatial_features

settings = Settings.from_environment()
st.set_page_config(page_title="Local Air Quality", page_icon="🌫️", layout="wide")
st.title("Local Air Quality Prediction")
st.caption("Public monitoring observations, gridded environmental forecasts, and transparent model evaluation.")

with st.sidebar:
    st.header("Location")
    latitude = st.number_input("Latitude", value=28.6139, min_value=-90.0, max_value=90.0, format="%.4f")
    longitude = st.number_input("Longitude", value=77.2090, min_value=-180.0, max_value=180.0, format="%.4f")
    radius = st.slider("Station search radius (km)", 1, 25, 25)
    history_days = st.select_slider("Historical window", options=[30, 90, 180, 365], value=90)

st.subheader("Current air quality")
try:
    locations = find_nearby_locations(latitude, longitude, settings.openaq_api_key, radius * 1000)
    if not locations:
        st.info("OpenAQ returned no monitoring stations in this radius.")
    else:
        labels = {f"{x.get('name', 'Station')} · {x.get('locality') or x.get('country', {}).get('name', '')} · ID {x['id']}": x for x in locations}
        st.session_state["nearby_locations"] = locations
        selected_label = st.selectbox("Monitoring station", list(labels))
        station = labels[selected_label]
        old_metadata = st.session_state.get("station_metadata", {})
        if old_metadata and old_metadata.get("id") != station["id"]:
            for key in ("station_history", "station_metadata", "trained_pollutants",
                        "training_failures", "forecast_pollutants", "forecast_aqi"):
                st.session_state.pop(key, None)
        st.session_state["selected_station"] = station
        results = fetch_location_latest(station["id"], settings.openaq_api_key)
        concentrations = normalize_latest_measurements(results)
        col1, col2, col3 = st.columns(3)
        col1.metric("Indian AQI", "Unavailable")
        col2.metric("Category", "Unavailable")
        col3.metric("Pollutants available", f"{len(concentrations)} / 6")
        st.dataframe(pd.DataFrame([{"Pollutant": k.upper(), "Latest concentration": v["value"],
                                    "Unit": v["unit"], "Measured at (UTC)": v["datetime_utc"]}
                                   for k, v in concentrations.items()]), use_container_width=True, hide_index=True)
        st.warning("OpenAQ latest values are instantaneous observations. CPCB AQI requires pollutant-specific averaging windows, so an official-style current AQI is not calculated from this snapshot.")
        source_times = [v["datetime_utc"] for v in concentrations.values() if v["datetime_utc"]]
        st.caption(f"Source: live OpenAQ station {station.get('name')} (ID {station['id']}). Latest measurement time: {max(source_times) if source_times else 'not supplied'}.")
        if st.button("Load hourly station history", key=f"history-{station['id']}"):
            try:
                history = fetch_location_hourly_history(station["id"], settings.openaq_api_key, history_days)
                coords = station.get("coordinates", {})
                if coords.get("latitude") is None or coords.get("longitude") is None:
                    raise DataSourceError("Station coordinates are unavailable; historical weather cannot be joined.")
                weather = fetch_historical_weather(coords["latitude"], coords["longitude"],
                                                   history.index.min().to_pydatetime(),
                                                   history.index.max().to_pydatetime())
                weather_coverage = weather.attrs.copy()
                coverage = history.attrs.get("coverage", {})
                history = history.join(weather, how="left")
                history = attach_spatial_features(history, station["id"],
                                                  coords["latitude"], coords["longitude"],
                                                  neighbor_locations=locations)
                for key in ("trained_pollutants", "training_failures",
                            "forecast_pollutants", "forecast_aqi"):
                    st.session_state.pop(key, None)
                st.session_state["station_history"] = history
                st.session_state["station_metadata"] = {"id": station["id"], "name": station.get("name"),
                                                         "latitude": coords.get("latitude"),
                                                         "longitude": coords.get("longitude"),
                                                         "timezone": station.get("timezone", "UTC"),
                                                         "coverage": coverage,
                                                         "weather_coverage": weather_coverage,
                                                         "neighbor_stations": history.attrs.get("neighbor_stations", []),
                                                         "neighbor_data_available": history.attrs.get("neighbor_data_available", False)}
                st.success(f"Loaded {len(history):,} OpenAQ hourly timestamps and joined real gridded Open-Meteo weather. Missing values remain missing.")
                st.dataframe(pd.DataFrame(coverage).T,
                             use_container_width=True)
                st.caption(f"Weather source coverage: {weather_coverage.get('coverage_percent', 0):.1f}% temperature hours; ERA5-Land rows={weather_coverage.get('archive_rows', 0)}, recent model rows={weather_coverage.get('recent_model_rows', 0)} ({weather_coverage.get('recent_model_status', 'not reported')}).")
            except DataSourceError as exc:
                st.error(str(exc))
except DataSourceError as exc:
    st.session_state.pop("nearby_locations", None)
    st.session_state.pop("selected_station", None)
    st.warning(str(exc))
    st.info("Live station data is unavailable. No sample values are shown. Add your OpenAQ API key to .env to enable monitoring.")

st.subheader("Regional environmental forecast")
if st.button("Load Open-Meteo forecast"):
    try:
        air, weather = fetch_openmeteo(latitude, longitude)
        air_frame = pd.DataFrame(air.get("hourly", {}))
        weather_frame = pd.DataFrame(weather.get("hourly", {}))
        st.caption("Open-Meteo gridded forecast, not a local monitoring-station observation. Air-quality fields are CAMS model output; attribution: Open-Meteo and Copernicus Atmosphere Monitoring Service (CAMS).")
        st.line_chart(air_frame.set_index("time")[[c for c in ("pm2_5", "pm10") if c in air_frame]])
        st.dataframe(weather_frame.head(24), use_container_width=True, hide_index=True)
    except DataSourceError as exc:
        st.error(str(exc))
