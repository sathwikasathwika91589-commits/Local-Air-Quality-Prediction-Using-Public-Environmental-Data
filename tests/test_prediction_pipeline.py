import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from src.feature_engineering import make_features
from src.model_training import chronological_splits
from src.model_training import compare_models
from src.prediction import forecast
from src.prediction import forecast_pollutants, common_forecast_history


def test_chronological_split_preserves_order():
    frame = pd.DataFrame({"value": range(100)}, index=pd.date_range("2024-01-01", periods=100, freq="h"))
    train, valid, test = chronological_splits(frame)
    assert train.index.max() < valid.index.min() < test.index.min()


def test_forecast_pipeline_returns_nonnegative_future_values():
    idx = pd.date_range("2024-01-01", periods=220, freq="h", tz="UTC")
    history = pd.DataFrame({"pm2_5": np.linspace(10, 30, len(idx))}, index=idx)
    engineered = make_features(history).dropna()
    columns = ["hour_sin", "hour_cos", "pm2_5_lag_1h", "pm2_5_lag_3h"]
    model = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    model.fit(engineered[columns], engineered["pm2_5"])
    prediction = forecast(model, history, columns, horizon_hours=3)
    assert len(prediction) == 3
    assert (prediction >= 0).all()


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def test_joint_pollutant_forecast_keeps_all_predictions_on_shared_timestamps():
    idx = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
    history = pd.DataFrame({"pm2_5": 10.0, "pm10": 20.0,
                            "temperature_2m": 15.0}, index=idx)
    future_index = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=2, freq="h")
    covariates = pd.DataFrame({"temperature_2m": [15.0, 15.0]}, index=future_index)
    models = {}
    for pollutant, value in (("pm2_5", 11), ("pm10", 21)):
        models[pollutant] = {"models": {"constant": ConstantModel(value)},
                             "selected_model": "constant",
                             "feature_columns": ["pm2_5_lag_1h", "temperature_2m_lag_1h"]}
    result = forecast_pollutants(history, models, 2, covariates)
    assert result["pm2_5"].tolist() == [11, 11]
    assert result["pm10"].tolist() == [21, 21]
    assert result.index[0] == idx[-1] + pd.Timedelta(hours=1)


def test_models_use_chronological_metrics_and_drop_features_unavailable_in_train():
    idx = pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC")
    target = pd.Series(np.sin(np.arange(400) / 12) + 10, index=idx, name="pm2_5")
    features = pd.DataFrame({"lag": target.shift(1),
                             "future_only": [np.nan] * 300 + list(range(100))}, index=idx)
    result = compare_models(features, target)
    assert set(result["validation_metrics"]["model"]) == {
        "Linear Regression", "Random Forest", "Gradient Boosting"}
    assert "future_only" not in result["feature_columns"]
    expected_mae = np.mean(np.abs(result["test_actual"] - result["test_predictions"]))
    assert result["test_metrics"]["mae"] == pytest.approx(expected_mae)
    assert result["validation_error_p90"] >= 0


def test_common_forecast_origin_rejects_stale_pollutant_history():
    idx = pd.date_range(pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4), periods=4, freq="h")
    history = pd.DataFrame({"pm2_5": [1, 2, 3, 4], "pm10": [1, 2, 3, 4]}, index=idx)
    aligned, origin = common_forecast_history(history, ["pm2_5", "pm10"], max_age_hours=24)
    assert aligned.index[-1] == origin
    with pytest.raises(ValueError, match="hours old"):
        common_forecast_history(history, ["pm2_5", "pm10"], max_age_hours=0.5)
