import streamlit as st
import pandas as pd
from src.data_cleaning import POLLUTANTS
from src.feature_engineering import WEATHER_COLUMNS

st.title("Historical Analysis")
history = st.session_state.get("station_history")
if history is None:
    st.info("Load real hourly station history from Home to explore observed trends.")
else:
    station = st.session_state.get("station_metadata", {})
    st.caption(f"OpenAQ · {station.get('name', 'station')} · ID {station.get('id', 'unknown')}")
    available = [p for p in POLLUTANTS if p in history]
    if available:
        pollutant = st.selectbox("Pollutant", available)
        st.line_chart(history[[pollutant]])
    st.write("Missing observations by pollutant")
    st.dataframe(history[available].isna().sum().rename("missing_hours").to_frame(), use_container_width=True)
    weather = [column for column in WEATHER_COLUMNS if column in history]
    if weather:
        st.subheader("Historical weather context (gridded, not station observations)")
        weather_feature = st.selectbox("Weather variable", weather)
        st.line_chart(history[[weather_feature]])
    if "pm2_5" in history:
        local_index = history.index.tz_convert(station.get("timezone", "UTC"))
        monthly = history["pm2_5"].groupby(local_index.month).mean().rename("PM2.5 mean")
        st.subheader("Seasonal pattern by local calendar month")
        st.bar_chart(monthly)
    if available:
        st.subheader("Observed AQI history (where averaging windows are complete)")
        from src.aqi_calculator import calculate_aqi_forecast
        observed_aqi = calculate_aqi_forecast(history[available])
        st.line_chart(observed_aqi["aqi"])
    predicted_pollutants = st.session_state.get("forecast_pollutants")
    if predicted_pollutants is not None and available:
        forecast_targets = [p for p in available if p in predicted_pollutants]
        if forecast_targets:
            chart_pollutant = st.selectbox("Historical and predicted pollutant", forecast_targets, key="history_forecast_target")
            joined = pd.concat([history[chart_pollutant].rename("observed"),
                                predicted_pollutants[chart_pollutant].rename("predicted")])
            st.subheader("Observed history and model forecast")
            st.line_chart(joined)
    predicted_aqi = st.session_state.get("forecast_aqi")
    if predicted_aqi is not None:
        st.subheader("Predicted AQI")
        st.line_chart(predicted_aqi["aqi"])
