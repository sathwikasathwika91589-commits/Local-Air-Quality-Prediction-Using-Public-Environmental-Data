"""Leakage-conscious chronological model comparison and fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from .data_cleaning import POLLUTANTS
from .feature_engineering import WEATHER_COLUMNS


def chronological_splits(frame: pd.DataFrame, train_fraction: float = 0.70,
                         validation_fraction: float = 0.15):
    ordered = frame.sort_index()
    first = int(len(ordered) * train_fraction)
    second = int(len(ordered) * (train_fraction + validation_fraction))
    if first < 1 or second <= first or second >= len(ordered):
        raise ValueError("Need enough chronological observations for train/validation/test splits.")
    return ordered.iloc[:first], ordered.iloc[first:second], ordered.iloc[second:]


def compare_models(features: pd.DataFrame, target: pd.Series) -> dict:
    aligned = features.join(target.rename("__target__"), how="inner").dropna(subset=["__target__"])
    if len(aligned) < 240:
        raise ValueError("At least 240 hourly target observations are required for a minimally useful chronological comparison.")
    train, validation, test = chronological_splits(aligned)
    raw_inputs = set(POLLUTANTS) | set(WEATHER_COLUMNS)
    xcols = [c for c in features.columns if c in aligned.columns and c not in raw_inputs
             and features.loc[train.index, c].notna().any()]
    if not xcols:
        raise ValueError("No usable historical, temporal, weather, spatial, or event features are available.")
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, random_state=42),
    }
    fitted, metrics = {}, []
    for name, estimator in candidates.items():
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), estimator)
        model.fit(train[xcols], train["__target__"])
        prediction = model.predict(validation[xcols])
        metrics.append({"model": name,
                        "mae": mean_absolute_error(validation["__target__"], prediction),
                        "rmse": np.sqrt(mean_squared_error(validation["__target__"], prediction)),
                        "r2": r2_score(validation["__target__"], prediction)})
        fitted[name] = model
    chosen = min(metrics, key=lambda item: item["mae"])["model"]
    selected_template = candidates[chosen]
    selection_model = fitted[chosen]
    validation_prediction = selection_model.predict(validation[xcols])
    validation_error = np.abs(validation["__target__"].to_numpy() - validation_prediction)
    test_model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                               clone(selected_template))
    development = pd.concat([train, validation]).sort_index()
    test_model.fit(development[xcols], development["__target__"])
    test_prediction = test_model.predict(test[xcols])
    test_metrics = {"mae": mean_absolute_error(test["__target__"], test_prediction),
                    "rmse": np.sqrt(mean_squared_error(test["__target__"], test_prediction)),
                    "r2": r2_score(test["__target__"], test_prediction)}
    # After the untouched test score is recorded, refit the selected deployment estimator
    # on all available history. The reported test metrics remain from the preceding fit.
    deployment_model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                     clone(selected_template))
    deployment_model.fit(aligned[xcols], aligned["__target__"])
    fitted[chosen] = deployment_model
    return {"models": fitted, "selected_model": chosen, "validation_metrics": pd.DataFrame(metrics),
            "test_metrics": test_metrics, "feature_columns": xcols,
            "validation_error_mae": float(np.mean(validation_error)),
            "validation_error_p90": float(np.quantile(validation_error, 0.90)),
            "test_predictions": pd.Series(test_prediction, index=test.index),
            "test_actual": test["__target__"]}
