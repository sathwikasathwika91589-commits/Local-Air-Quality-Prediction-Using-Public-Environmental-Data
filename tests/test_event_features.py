import pandas as pd
import pytest

from src.event_features import build_event_features, validate_events


def test_sourced_event_expands_over_declared_duration_only():
    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    events = pd.DataFrame([{"timestamp": index[1], "event_type": "traffic",
                            "intensity": 0.8, "duration_hours": 2,
                            "source": "municipal traffic bulletin"}])
    features = build_event_features(index, events)
    assert features["event_traffic"].tolist() == [0, 1, 1]
    assert features["event_intensity_traffic"].tolist() == [0, 0.8, 0.8]


def test_event_records_require_provenance_and_valid_types():
    invalid = pd.DataFrame([{"timestamp": "2026-01-01", "event_type": "festival",
                             "intensity": 0.5, "duration_hours": 2, "source": ""}])
    with pytest.raises(ValueError, match="source"):
        validate_events(invalid)
