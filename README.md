# Local Air Quality Prediction Using Public Environmental Data

A Python 3.11 and Streamlit project for monitoring public air-quality measurements, training pollutant forecasts from real station history, and calculating predicted Indian AQI from predicted pollutant concentrations.

## Problem and objectives

Air-quality coverage and sensor availability vary by location. The app keeps observed station readings distinct from gridded reanalysis/forecast model products, reports source coverage and supported units, and refuses to fill missing measurements with synthetic values. It supports a complete path from public inputs through chronological evaluation to pollutant and AQI forecasts when data is adequate.

## Architecture

`app.py` handles station selection and current observations. `src/data_loader.py` owns public API access and unit normalization. Cleaning, five-group feature engineering, CPCB AQI, spatial/event context, chronological model comparison, recursive prediction, and explainability are separate modules. Streamlit pages expose current conditions, forecast, history, station explorer, model insights, and methodology.

## Public data sources

- [OpenAQ API v3](https://docs.openaq.org/): location/sensor metadata, latest values, and hourly observations. An API key is required.
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api): ERA5-Land hourly gridded reanalysis used for the archive period. Recent weather-model rows are used to bridge the reanalysis delay where available; both products are grid model data, not station observations.
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs): hourly gridded weather forecast covariates.
- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api): CAMS-based grid forecasts for context on Home. These values are not used as station targets or substituted for local measurements.
- [CPCB National AQI report](https://airquality.cpcb.gov.in/ccr_docs/FINAL-REPORT_AQI_.pdf): Indian AQI categories, concentration breakpoints, and averaging rules.

No sample observations or fake forecast outputs are bundled. Without a valid OpenAQ key, the app shows that live data is unavailable. This workspace has no `.env` file and `OPENAQ_API_KEY` is not configured, so live OpenAQ requests and actual station coverage could not be verified in this run. API behavior and parsing are covered with mocked response tests. Open-Meteo requests are made only when the user asks to load data.

## Feature inventory

| Feature name | Group | Source | Calculation | Available | Used by model |
|---|---|---|---|---|---|
| `temperature_2m_lag_1h`, `relative_humidity_2m_lag_1h`, `wind_speed_10m_lag_1h`, `surface_pressure_lag_1h`, `cloud_cover_lag_1h`, `dew_point_2m_lag_1h`, `precipitation_lag_1h` | Meteorological | Open-Meteo historical weather; future Open-Meteo forecast | Previous-hour gridded value aligned to UTC; wind speed is m/s, temperature °C, pressure hPa, rainfall mm | When both APIs return usable rows | Yes |
| `wind_direction_sin`, `wind_direction_cos` | Meteorological | Open-Meteo wind direction | Sine/cosine encoding of prior-hour degrees | When wind direction is available | Yes |
| `temperature_2m_change_1h`, `relative_humidity_2m_change_1h`, `surface_pressure_change_1h`, `wind_speed_10m_change_1h` | Meteorological | Open-Meteo historical/forecast weather | Difference between two prior hourly observations | When preceding weather rows exist | Yes |
| `precipitation_3h`, `precipitation_24h` | Meteorological | Open-Meteo precipitation | Sum of the previous 3/24 hourly precipitation values | When complete prior windows exist | Yes |
| `{pollutant}_lag_{1h,3h,6h,12h,24h,3d,7d}` | Historical pollutants | OpenAQ hourly sensor data | Shifted pollutant concentration at the named previous-hour offset | Per pollutant sensor coverage | Yes |
| `{pollutant}_rolling_{mean,std,min,max}_{3,6,24,168}h` | Historical pollutants | OpenAQ hourly sensor data | Rolling statistic over prior values only; current row shifted out | Per pollutant sensor coverage and complete window | Yes |
| `{pollutant}_change_{1h,3h,24h}`, `{pollutant}_pct_change_*` | Historical pollutants | OpenAQ hourly sensor data | Difference/percentage change between prior-hour lag values | Per pollutant sensor coverage; percentage unavailable for zero baseline | Yes |
| `hour`, `day`, `weekday`, `weekend`, `week`, `month`, `quarter`, `year`, `season`, `rush_hour` | Temporal | Selected station timezone from OpenAQ | Local calendar fields and weekday commute-hour indicator | Always when timezone is valid; UTC fallback otherwise | Yes |
| `hour_sin`, `hour_cos`, `day_of_year_sin`, `day_of_year_cos` | Temporal | Station-local timestamp | Cyclic encoding of local hour and day of year | Always when timezone is valid; UTC fallback otherwise | Yes |
| public holiday/festival calendar | Temporal/event | No automatic reliable source configured; may be supplied by CSV | Explicit event interval and source | Only when provided in event CSV | Yes, when provided |
| `latitude`, `longitude`, `nearest_neighbor_distance_km` | Geographic/spatial | OpenAQ station metadata | Station coordinate and great-circle distance to nearest returned station | When coordinates/stations exist | Yes (distance omitted when unavailable) |
| `neighbor_{pollutant}`, `neighbor_particulate_average`, `neighbor_aqi`, `neighbor_station_count` | Geographic/spatial | Real hourly histories for nearby OpenAQ stations | Prior-hour synchronized neighbor mean; neighbor AQI calculated from each neighbor's complete CPCB windows | Infrastructure implemented; selected-station history is not automatically downloaded for neighboring stations | Only when real aligned neighbor rows are supplied; otherwise unavailable and excluded |
| station name and `station_id` | Geographic/spatial metadata | OpenAQ location record | Retained as station identity metadata | When station selected | Not encoded as numeric model input; current model is per station |
| `event_{type}`, `event_intensity_{type}`, `event_count` | External events | User-supplied, sourced event CSV | Active interval from `timestamp` through `duration_hours`, declared intensity from 0 to 1, and active event count | Only when uploaded | Yes, when uploaded for training and prediction |
| visibility | Meteorological | Not currently returned by configured historical weather request | Not calculated | Unavailable | No |

The event CSV schema is `timestamp,event_type,intensity,duration_hours,source`. `event_type` must be one of `public_holiday`, `festival`, `traffic`, `construction_dust`, `fire`, or `industrial`. `source` must identify a citation or dataset. Upload data only when its time coverage is adequate; an absent CSV row is interpreted as no event in that supplied event feed. Do not upload unsourced or incomplete feeds as if they were comprehensive. Scheduled/known future events can be included for forecasting; events discovered only after the forecast origin are not valid future features.

## Cleaning, units, and coverage

Hourly OpenAQ data is requested for the selected date window and aligned to a full hourly UTC index. Negative and non-finite values are discarded. PM2.5, PM10, NO2, SO2, and O3 are accepted only in μg/m³. CO is normalized to mg/m³ from mg/m³ or μg/m³. Unknown units are skipped and reported; they are never guessed. Coverage is shown as observed hours divided by all requested hours for each pollutant. Weather coverage is shown separately, with archive and recent model row counts. A low percentage or sparse series may make that target ineligible for training.

## Leakage prevention and machine learning

Every pollutant lag and rolling feature uses values strictly before its target timestamp. Weather predictors are also shifted one hour; future predictions receive Open-Meteo weather forecast covariates, never same-hour historical reanalysis. Event inputs must be known/scheduled at the forecast origin. Features unavailable in the training segment are discarded based only on the chronological training data. Missing feature values are median-imputed inside each model pipeline; missing target rows are dropped. This imputation is a model operation, not a displayed observation.

At least 240 usable hourly target observations are required. Rows are sorted chronologically into 70% train, 15% validation, and 15% test. Linear Regression, Random Forest, and HistGradientBoostingRegressor are compared; minimum validation MAE selects the model. MAE, RMSE, and R² are computed for validation and the held-out test segment. Test data is not used for selection. XGBoost is omitted to keep installation and platform dependencies stable; no XGBoost result is implied.

The app displays validation MAE and the 90th percentile absolute validation residual as a next-hour descriptive error reference. This is not represented as a confidence interval or guaranteed coverage, and is not applied as a multi-hour prediction interval. Recursive forecast errors can grow with horizon; archived weather forecasts for fixed lead-time backtesting and horizon-specific calibration remain future improvements.

## CPCB/Indian AQI

Pollutant concentrations are forecast first. The AQI calculator then uses CPCB concentration breakpoints, rounds concentrations to integer μg/m³ (CO to 0.1 mg/m³), interpolates pollutant sub-indices, and takes their maximum when at least three sub-indices exist and at least one is PM2.5 or PM10. Forecast AQI uses complete trailing hourly windows: 24 hours for PM2.5, PM10, NO2, and SO2; 8 hours for CO and O3. Missing observations/predictions are not interpolated. Concentrations are normalized to μg/m³, except CO in mg/m³. AQI/category remain unavailable until window coverage and pollutant count requirements are met. Current instant readings are not mislabeled as CPCB AQI.

## Geographic and event limitations

The Location Explorer shows real nearby station coordinates/distances and can fetch latest PM2.5/PM10 readings for up to five neighbors. Its displayed average is across latest available values whose timestamps can differ; it is not a synchronized spatial predictor or AQI. Neighboring hourly-history/AQI feature generation is implemented in `src/spatial_features.py`, but no neighbor historical dataset is automatically retrieved or used in training yet. The UI states this unavailable condition explicitly. There is no automatic festival, traffic, construction, fire, or industrial event feed configured; the documented sourced CSV route is the supported input.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your OpenAQ key to `.env` as `OPENAQ_API_KEY=...`. Keep `.env private`; it is ignored by Git. Open-Meteo's configured public endpoints do not require a key.

Run the app:

```powershell
streamlit run app.py
```

Run tests:

```powershell
python -m pytest -q
```

## Known limitations

- Live API success depends on a valid OpenAQ key, upstream availability, station location, pollutant sensors, coverage, and supported units.
- The app does not make up missing readings. Low coverage can disable training or AQI.
- Training is per selected station. Station identity is not pooled across stations; latitude/longitude are constant within a station-specific fit.
- Neighbor history must be separately acquired and supplied before the spatial neighbor time-series features can enter a model.
- Event features depend on user-supplied source quality and completeness; no reliable global/local historical event source is configured.
- Multi-day uncertainty is not calibrated. Validation error shown is next-hour only.
- Open-Meteo reanalysis is gridded retrospective weather. It is not a local weather station and actual reanalysis in validation can differ from weather forecasts available at the time.

## GitHub readiness

`.gitignore` excludes `.env`, virtual environments, caches, logs, generated models, and raw/processed data. Public source code and documentation can be committed; review upstream dataset/API attribution and terms before distributing acquired data.
