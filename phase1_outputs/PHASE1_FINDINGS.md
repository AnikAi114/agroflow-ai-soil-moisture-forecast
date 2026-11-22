# AgroFlow AI - Phase 1: Dataset Exploration and Quality Audit

## Purpose
Explore the published five-zone preprocessed agriculture dataset. The goal is to
understand sampling, spatial heterogeneity, missing time intervals, and forecast
target validity before treating a model result as trustworthy.

**Dataset landing page:** https://data.mendeley.com/datasets/c837v6p8ph/2
**Data used:** The five `dataset_zone_*_preprocessed.csv` files, unmodified.
**Times:** Naive Europe/Rome local timestamps, as described in DATA_DICTIONARY.md.

## Key numbers

- Total processed observations: **47,068** across five agricultural zones.
- Exact 24-hour forecast windows: **17,614** (**37.4%** of rows).
- The original published targets are 144 **rows** ahead, not necessarily 24 **hours** ahead.
- For this audit, a valid window must be 144 observations ahead, exactly 24 hours apart,
  with the published target matching the observed soil moisture at that time.
- Consecutive regular 10-minute pairs across all zones: **42,322**;
  time gaps longer than 10 minutes: **4,741**.
- Rows with pH outside the illustrative 3-9 screening range: **45**;
  rows with EC outside the illustrative 0-1000 range: **45**.
  These are flagged observations, **not** proof that every flag is a sensor malfunction.

## Zone-level summary

| Zone | Processed rows | Exact-24h windows | Valid share | Median moisture | Regular 10-min pairs |
|---|---:|---:|---:|---:|---:|
| 1 | 9,531 | 4,533 | 47.56% | 23.00% | 91.54% |
| 2 | 11,929 | 7,077 | 59.33% | 19.00% | 91.24% |
| 3 | 1,118 | 0 | 0.00% | 79.00% | 72.25% |
| 4 | 12,676 | 3,774 | 29.77% | 80.00% | 92.22% |
| 5 | 11,814 | 2,230 | 18.88% | 66.00% | 86.51% |

## Figures

1. `01_zone_coverage.png`: Processed rows versus verified 24-hour windows.
2. `02_soil_moisture_distributions.png`: Different zone distributions.
3. `03_monthly_moisture_trends.png`: Monthly medians from observed processed readings.
4. `04_data_quality.png`: Timestamp regularity and screened pH/EC values.
5. `05_zone4_activity_window.png`: Historical zone 4 activity, 2025-06-15 to 2025-06-21.
6. `06_verified_windows_by_month.png`: Exact-24h windows by month.

## Important methodological limitations

- These are *preprocessed* rather than raw sensor values. The publisher cleaned,
  clipped, despiked, and interpolated soil-moisture readings; chart trends can
  reflect processing decisions. See `DATA_DICTIONARY.md` and the source README.
- The `weather_*` variables originate from Open-Meteo ERA5 historical/reanalysis
  data; they are not necessarily co-located live weather-station measurements.
- A nonzero `irrigation_duration_minutes` is a recorded interval with valve
  activity; do not equate number of active intervals with number of irrigation events.
- Temporal gaps are substantial. When 144 successive 10-minute intervals are
  not available, the publisher's 144-row future target is not exactly 24 hours.
- Zone 3 has no eligible exact-24h windows in this dataset. It remains in EDA
  but is **not** valid for the current exact-24h forecasting benchmark.
- The 24-hour windows in this analysis overlap heavily, so adjacent samples
  are not statistically independent.
- `water_vol_to_24h` and `real_moisture_delta` contain future outcomes; never
  use them as predictors in ordinary forecasting experiments.
- This report does not establish irrigation causality or field-ready control.

## Suggested use in later phases

- Phase 2: Compare time-aware machine-learning models against persistence;
  do not tune against an already-inspected historical test period.
- Phase 3: Use feature importance / SHAP carefully as a model explanation,
  not an explanation of physical causality.
- Phase 4: Display the verified-coverage and limitations alongside forecasts.

## Reproduce

From the project root: `python agroflow_phase1_eda.py`.
Artifacts are written to `phase1_outputs/` without modifying the input CSVs.
