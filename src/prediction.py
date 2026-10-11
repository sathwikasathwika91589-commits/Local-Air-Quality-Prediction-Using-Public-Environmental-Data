"""One-step recursive predictions using the fitted feature pipeline."""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone
from .feature_engineering import make_features
from .feature_engineering import WEATHER_COLUMNS


def forecast(model, history: pd.DataFrame, feature_columns: list[str],
             target: str = "pm2_5", horizon_hours: int = 1,
             future_weather: pd.DataFrame | None = None,
             timezone: str | None = None) -> pd.Series:
    if horizon_hours < 1:
        raise ValueError("Forecast horizon must be positive.")
    values = history.sort_index().copy()
    if not isinstance(values.index, pd.DatetimeIndex) or values.empty:
        raise ValueError("History must be a non-empty time-indexed DataFrame.")
    weather_inputs = [column for column in WEATHER_COLUMNS
                      if any(feature.startswith(f"{column}_") for feature in feature_columns)]
    if weather_inputs and future_weather is None:
        raise ValueError("A real weather forecast is required for this model's meteorological features.")
    output = {}
    for _ in range(horizon_hours):
        next_time = values.index[-1] + pd.Timedelta(hours=1)
        row = {target: float("nan")}
        if future_weather is not None and next_time in future_weather.index:
            row.update(future_weather.loc[next_time].to_dict())
        if weather_inputs and any(pd.isna(row.get(column)) for column in weather_inputs):
            raise ValueError(f"Weather forecast is missing required covariates at {next_time}.")
        seed = pd.concat([values, pd.DataFrame(row, index=[next_time])])
        features = make_features(seed, target, timezone=timezone).iloc[[-1]][feature_columns]
        prediction = float(model.predict(features)[0])
        prediction = max(0.0, prediction)
        output[next_time] = prediction
        values.loc[next_time, target] = prediction
    return pd.Series(output, name=f"predicted_{target}")


def forecast_pollutants(history: pd.DataFrame, trained: dict,
                        horizon_hours: int,
                        future_covariates: pd.DataFrame | None = None,
                        timezone: str | None = None) -> pd.DataFrame:
    """Jointly roll forward pollutant models so cross-pollutant lags stay populated."""
    if horizon_hours < 1 or not trained:
        raise ValueError("At least one trained pollutant and a positive horizon are required.")
    values = history.sort_index().copy()
    if not isinstance(values.index, pd.DatetimeIndex) or values.empty:
        raise ValueError("History must be a non-empty time-indexed DataFrame.")
    weather_inputs = {column for result in trained.values()
                      for column in WEATHER_COLUMNS
                      if any(feature.startswith(f"{column}_") for feature in result["feature_columns"])}
    if any(feature in {"wind_direction_sin", "wind_direction_cos"}
           for result in trained.values() for feature in result["feature_columns"]):
        weather_inputs.add("wind_direction_10m")
    if weather_inputs and future_covariates is None:
        raise ValueError("A real weather forecast is required for the trained models.")
    output = []
    for _ in range(horizon_hours):
        next_time = values.index[-1] + pd.Timedelta(hours=1)
        future_row = {pollutant: float("nan") for pollutant in trained}
        for static_column in ("latitude", "longitude", "nearest_neighbor_distance_km"):
            if static_column in values:
                future_row[static_column] = values[static_column].iloc[-1]
        if future_covariates is not None and next_time in future_covariates.index:
            future_row.update(future_covariates.loc[next_time].to_dict())
        if weather_inputs and any(pd.isna(future_row.get(column)) for column in weather_inputs):
            raise ValueError(f"Weather forecast is missing required covariates at {next_time}.")
        seed = pd.concat([values, pd.DataFrame(future_row, index=[next_time])])
        predicted = {}
        for pollutant, result in trained.items():
            row_features = make_features(seed, pollutant, timezone=timezone).iloc[[-1]][result["feature_columns"]]
            estimator = result["models"][result["selected_model"]]
            predicted[pollutant] = max(0.0, float(estimator.predict(row_features)[0]))
        output.append({"timestamp": next_time, **predicted})
        for pollutant, value in predicted.items():
            values.loc[next_time, pollutant] = value
        for column, value in future_row.items():
            if column not in trained:
                values.loc[next_time, column] = value
    return pd.DataFrame(output).set_index("timestamp")


def common_forecast_history(history: pd.DataFrame, targets: list[str],
                           max_age_hours: float = 24) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Trim to the freshest common target origin and reject stale live observations."""
    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive.")
    latest = []
    for target in targets:
        if target not in history or not history[target].notna().any():
            raise ValueError(f"No observed history is available for {target}.")
        latest.append(history[target].last_valid_index())
    origin = min(latest)
    age_hours = (pd.Timestamp(datetime.now(timezone.utc)) - origin).total_seconds() / 3600
    if age_hours > max_age_hours:
        raise ValueError(f"Latest common pollutant observations are {age_hours:.1f} hours old; refresh station data before forecasting.")
    return history.loc[:origin].copy(), origin
