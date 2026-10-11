import io

import pandas as pd
import streamlit as st

from src.aqi_calculator import calculate_aqi_forecast
from src.data_cleaning import POLLUTANTS
from src.data_loader import DataSourceError, fetch_openmeteo
from src.event_features import build_event_features, merge_event_features, validate_events
from src.feature_engineering import WEATHER_COLUMNS, make_features
from src.model_training import compare_models
from src.prediction import common_forecast_history, forecast_pollutants

st.title("Pollutant and AQI Forecast")
history = st.session_state.get("station_history")
upload = st.file_uploader("Optional sourced event CSV", type="csv",
                          help="Required columns: timestamp,event_type,intensity,duration_hours,source. Only provide events known or scheduled at prediction time.")
st.caption("No automatic event dataset is configured. The CSV must be sourced, cover the forecast period, and include only information known at the prediction origin.")
if upload is not None:
    try:
        events = validate_events(pd.read_csv(io.BytesIO(upload.getvalue())))
        previous = st.session_state.get("event_records")
        if previous is None or not previous.equals(events):
            for key in ("trained_pollutants", "training_failures", "forecast_pollutants", "forecast_aqi"):
                st.session_state.pop(key, None)
        st.session_state["event_records"] = events
        st.success(f"Loaded {len(events)} sourced event record(s) from the uploaded CSV.")
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Event CSV was rejected: {exc}")
elif st.session_state.get("event_records") is not None:
    if st.button("Clear uploaded event data"):
        for key in ("event_records", "trained_pollutants", "training_failures",
                    "forecast_pollutants", "forecast_aqi"):
            st.session_state.pop(key, None)
events = st.session_state.get("event_records")
if events is not None:
    st.dataframe(events, use_container_width=True, hide_index=True)

if history is None:
    st.info("Load live OpenAQ station history from Home. The model needs real pollutant history joined to historical weather.")
else:
    station = st.session_state.get("station_metadata", {})
    st.caption(f"Live OpenAQ station: {station.get('name')} · ID {station.get('id')} · {len(history):,} hourly rows")
    targets = [pollutant for pollutant in POLLUTANTS if pollutant in history and history[pollutant].notna().sum()]
    missing_weather = [column for column in WEATHER_COLUMNS if column not in history]
    if not targets:
        st.warning("No supported, unit-normalized pollutant history is available for training.")
    elif missing_weather:
        st.warning(f"Weather inputs are missing ({', '.join(missing_weather)}). Reload historical weather before model training.")
    else:
        st.write("Available pollutant targets: " + ", ".join(p.upper() for p in targets))
        if st.button("Compare and train available pollutant models", type="primary"):
            trained, failures = {}, {}
            for target in targets:
                engineered = make_features(history, target, timezone=station.get("timezone", "UTC"))
                if events is not None:
                    engineered = merge_event_features(engineered, events)
                features = engineered.drop(columns=[p for p in POLLUTANTS if p in engineered], errors="ignore")
                try:
                    trained[target] = compare_models(features, engineered[target])
                except ValueError as exc:
                    failures[target] = str(exc)
            st.session_state["trained_pollutants"] = trained
            st.session_state["training_failures"] = failures
            st.session_state["training_event_count"] = 0 if events is None else len(events)
        trained = st.session_state.get("trained_pollutants", {})
        failures = st.session_state.get("training_failures", {})
        if failures:
            st.warning("Some targets could not be trained: " + "; ".join(f"{p.upper()}: {reason}" for p, reason in failures.items()))
        if trained:
            comparison = pd.concat({p: result["validation_metrics"].set_index("model")
                                    for p, result in trained.items()}, names=["pollutant", "model"])
            st.subheader("Chronological validation model comparison")
            st.dataframe(comparison.reset_index(), use_container_width=True, hide_index=True)
            horizon = st.select_slider("Forecast horizon (hours)", options=[1, 3, 6, 12, 24, 72, 168], value=24)
            if st.button("Forecast pollutants and calculate AQI"):
                try:
                    forecast_history, origin = common_forecast_history(history, list(trained))
                    _, weather = fetch_openmeteo(station["latitude"], station["longitude"],
                                                 forecast_days=min(7, max(1, (horizon + 23) // 24)))
                    covariates = pd.DataFrame(weather["hourly"])
                    covariates["time"] = pd.to_datetime(covariates["time"], utc=True)
                    covariates = covariates.set_index("time")
                    if events is not None:
                        covariates = covariates.join(build_event_features(covariates.index, events))
                    forecasts = forecast_pollutants(forecast_history, trained, horizon, covariates,
                                                    timezone=station.get("timezone", "UTC"))
                    combined = pd.concat([forecast_history[list(POLLUTANTS)], forecasts]).sort_index()
                    aqi = calculate_aqi_forecast(combined).loc[forecasts.index]
                    st.session_state["forecast_pollutants"] = forecasts
                    st.session_state["forecast_aqi"] = aqi
                    st.caption(f"Forecast origin: {origin.isoformat()} (latest common pollutant observation)")
                    st.caption("Meteorological forecast covariates: Open-Meteo gridded model data, aligned in UTC.")
                    st.subheader("Predicted pollutant concentrations")
                    st.line_chart(forecasts)
                    st.subheader("Predicted Indian AQI")
                    if aqi["aqi"].notna().any():
                        st.line_chart(aqi["aqi"])
                        st.dataframe(aqi.reset_index(), use_container_width=True, hide_index=True)
                    else:
                        st.warning("AQI unavailable: each timestamp needs complete CPCB pollutant averaging windows, at least three valid pollutant sub-indices, and at least one particulate pollutant prediction.")
                    st.caption("Forecast AQI uses strict complete trailing 24-hour PM/NO₂/SO₂ and 8-hour CO/O₃ means. No missing hourly values are interpolated.")
                    st.subheader("Validation-based uncertainty reference")
                    rows = [{"Pollutant": p.upper(), "Validation MAE (one step)": r["validation_error_mae"],
                             "90th percentile absolute error (one step)": r["validation_error_p90"]}
                            for p, r in trained.items()]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption("These are chronological validation errors for the next-hour model. They are descriptive error references, not confidence intervals, and do not quantify 24-hour or multi-day uncertainty.")
                except (DataSourceError, ValueError, KeyError) as exc:
                    st.error(f"Forecast unavailable: {exc}")
