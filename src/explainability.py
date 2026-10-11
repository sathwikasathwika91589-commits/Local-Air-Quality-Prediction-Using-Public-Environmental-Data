"""Feature contribution summaries supported by scikit-learn estimators."""


def feature_importance(pipeline, feature_columns):
    estimator = pipeline.steps[-1][1]
    if not hasattr(estimator, "feature_importances_"):
        return None
    import pandas as pd
    return pd.Series(estimator.feature_importances_, index=feature_columns).sort_values(ascending=False)
