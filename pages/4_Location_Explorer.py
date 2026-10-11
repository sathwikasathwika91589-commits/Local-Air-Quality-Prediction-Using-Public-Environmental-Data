import streamlit as st
import pandas as pd
from src.config import Settings
from src.data_loader import DataSourceError, fetch_location_latest, normalize_latest_measurements
from src.spatial_features import distance_km

st.title("Location Explorer")
locations = st.session_state.get("nearby_locations")
selected = st.session_state.get("selected_station")
if not locations or not selected:
    st.info("Search for real OpenAQ monitoring locations from Home first.")
else:
    api_key = Settings.from_environment().openaq_api_key
    origin = selected.get("coordinates", {})
    station_rows = []
    for location in locations:
        coords = location.get("coordinates", {})
        if coords.get("latitude") is None or coords.get("longitude") is None:
            continue
        station_rows.append({"station_id": location["id"], "station": location.get("name"),
                             "latitude": coords["latitude"], "longitude": coords["longitude"],
                             "distance_km": distance_km(origin["latitude"], origin["longitude"],
                                                         coords["latitude"], coords["longitude"]),
                             "is_selected": location["id"] == selected["id"]})
    station_table = pd.DataFrame(station_rows).sort_values("distance_km")
    st.subheader("Real monitoring locations")
    st.dataframe(station_table, use_container_width=True, hide_index=True)
    st.map(station_table[["latitude", "longitude"]])
    st.caption("Station coordinates and great-circle distances come from OpenAQ metadata.")
    if st.button("Load latest nearby station observations"):
        latest_rows = []
        for station in station_table.loc[~station_table["is_selected"]].head(5).to_dict("records"):
            try:
                raw = fetch_location_latest(int(station["station_id"]), api_key)
                values = normalize_latest_measurements(raw)
                latest_rows.append({"station": station["station"], "station_id": station["station_id"],
                                    "distance_km": station["distance_km"],
                                    "PM2.5 μg/m³": values.get("pm2_5", {}).get("value"),
                                    "PM10 μg/m³": values.get("pm10", {}).get("value"),
                                    "latest UTC": max((v["datetime_utc"] for v in values.values() if v["datetime_utc"]), default=None)})
            except DataSourceError as exc:
                st.warning(f"Could not load station {station['station_id']}: {exc}")
        if latest_rows:
            frame = pd.DataFrame(latest_rows)
            st.subheader("Latest nearby pollutant readings")
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.write("Mean across available nearby latest PM values (timestamps may differ)")
            st.json({column: frame[column].mean() for column in ("PM2.5 μg/m³", "PM10 μg/m³")})
            st.warning("These are latest individual observations, not synchronized historical neighbor features; no neighboring AQI is calculated without CPCB averaging windows.")
    st.info("Neighboring-station time-series features are not used by the trained model until real aligned hourly observations for those stations are acquired. Station ID is retained as metadata; coordinates are included in the selected-station history.")
