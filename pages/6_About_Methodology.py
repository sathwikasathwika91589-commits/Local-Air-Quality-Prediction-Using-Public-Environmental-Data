import streamlit as st
st.title("About / Methodology")
st.markdown("""
### Sources and scope
- **OpenAQ API v3**: live station metadata and hourly historical measurements; an API key is required. Unsupported units are omitted and coverage by pollutant is reported.
- **Open-Meteo ERA5-Land**: gridded retrospective weather, joined in UTC. These are reanalysis products, not local station observations.
- **Open-Meteo Forecast**: gridded forecast weather covariates used by the recursive pollutant model.
- **CPCB National AQI**: forecast pollutant concentrations are averaged over complete pollutant-specific windows, mapped to sub-indices, then aggregated. At least three pollutant sub-indices including PM2.5 or PM10 are required.

### Leakage controls
Pollutant and weather predictors are shifted so each row only uses information from before its target hour. The multi-pollutant forecast rolls all available trained targets forward together. Final evaluation uses chronological train, validation, and test partitions. Model selection uses validation MAE; the test partition is held back for reporting. Validation errors are displayed as descriptive one-hour error references, not confidence intervals.

### Limits
Station availability and pollutant coverage vary by location. Current instant measurements do not have a CPCB AQI label. Neighbor history is not automatically downloaded; neighbor features are enabled only when real aligned data is supplied. There is no automatic holiday, festival, traffic, fire, construction, or industrial feed; the Forecast page accepts source-cited event CSV rows. Missing values are not fabricated.
""")
