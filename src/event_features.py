"""Event feature framework for sourced, timestamped user-supplied records."""

from __future__ import annotations

import pandas as pd

EVENT_TYPES = ("public_holiday", "festival", "traffic", "construction_dust", "fire", "industrial")
EVENT_COLUMNS = {"timestamp", "event_type", "intensity", "duration_hours", "source"}


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = EVENT_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"Event data missing required columns: {', '.join(sorted(missing))}")
    result = events.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("Every event needs a valid timestamp.")
    result["event_type"] = result["event_type"].astype(str).str.strip().str.lower()
    invalid = sorted(set(result["event_type"]) - set(EVENT_TYPES))
    if invalid:
        raise ValueError(f"Unsupported event type(s): {', '.join(invalid)}")
    result["source"] = result["source"].astype(str).str.strip()
    if result["source"].eq("").any() or result["source"].str.lower().eq("nan").any():
        raise ValueError("Every event record must include a source citation or dataset identifier.")
    result["intensity"] = pd.to_numeric(result["intensity"], errors="coerce")
    result["duration_hours"] = pd.to_numeric(result["duration_hours"], errors="coerce")
    if result["intensity"].isna().any() or (~result["intensity"].between(0, 1)).any():
        raise ValueError("Event intensity must be numeric from 0 (low) to 1 (high).")
    if result["duration_hours"].isna().any() or (result["duration_hours"] <= 0).any():
        raise ValueError("Event duration_hours must be positive.")
    return result


def build_event_features(index: pd.DatetimeIndex,
                         events: pd.DataFrame | None) -> pd.DataFrame:
    """Expand sourced events over their declared duration at requested timestamps."""
    output = pd.DataFrame(0.0, index=index, columns=[
        *(f"event_{kind}" for kind in EVENT_TYPES),
        *(f"event_intensity_{kind}" for kind in EVENT_TYPES),
        "event_count",
    ])
    if events is None or events.empty:
        return output
    context = validate_events(events)
    utc_index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    for event in context.itertuples(index=False):
        end = event.timestamp + pd.Timedelta(hours=float(event.duration_hours))
        active = (utc_index >= event.timestamp) & (utc_index < end)
        feature = f"event_{event.event_type}"
        intensity = f"event_intensity_{event.event_type}"
        output.loc[active, feature] = 1.0
        output.loc[active, intensity] = output.loc[active, intensity].combine(
            pd.Series(float(event.intensity), index=output.index[active]), max)
        output.loc[active, "event_count"] += 1
    return output


def merge_event_features(frame: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    return frame.join(build_event_features(frame.index, events))
