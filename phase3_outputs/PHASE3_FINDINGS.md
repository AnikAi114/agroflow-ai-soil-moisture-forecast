# AgroFlow AI | Phase 3 Explainable AI Findings

## Research setup

- Verified full dataset: 17,614 24-hour rows.
- Chronological validation rows: 2,213; exploratory feature explanation.
- Permutation subsample: at most 800 time-spread rows, 4 shuffles per group.
- SHAP subsample: at most 300 time-spread rows.
- ML models are fitted Phase 2 models trained on the earlier **training** period only; no refitting here.
- Phase 2 validation-selected candidate: **Regularized Gradient Boosting (unshrunk)**.
- All walk-forward CV blend weights were **0.0**. Those blended methods equal persistence,
  so explanations here concern the actual **unshrunk ML models**, not blended persistence.
- Historical-test rows are read only to reproduce the established chronological split;
  no historical-test rows are used to calculate feature explanations.

## Exploratory validation permutation sensitivity

The metric is the change in 24-hour forecast MAE after shuffling one input
feature group on a validation subsample, keeping the rest of the inputs unchanged.
A positive score indicates sensitivity on these observations, not causal influence.
Group-level shuffling across dates/zones can form unrealistic combinations.

### Top 5 positive sensitivities: Gradient Boosting

- `agricultural_zone`: MAE change +1.2702 percentage points
- `soil_moisture`: MAE change +0.4174 percentage points
- `weather_temp`: MAE change +0.0000 percentage points
- `weather_rain`: MAE change +0.0000 percentage points
- `weather_humidity`: MAE change +0.0000 percentage points

### Top 5 positive sensitivities: Random Forest

- `agricultural_zone`: MAE change +1.6082 percentage points
- `soil_moisture`: MAE change +0.5400 percentage points
- `weather_radiation`: MAE change +0.0059 percentage points
- `weather_wind_speed`: MAE change +0.0023 percentage points
- `weather_rain`: MAE change -0.0000 percentage points

## SHAP explanation of the Gradient Boosting residual

SHAP decomposes the model's **predicted 24-hour soil-moisture CHANGE**.
The forecast itself adds that learned change to current moisture (with clipping).
SHAP here does not explain persistence, future irrigation need, or causal effects.
Correlated weather inputs and one-hot zone indicators can share attribution.

### Top 5 by mean absolute SHAP magnitude

- `agricultural_zone`: 1.8738 percentage points
- `soil_moisture`: 0.7670 percentage points
- `soil_temperature_7-18cm`: 0.1416 percentage points
- `weather_pressure`: 0.0024 percentage points
- `weather_temp`: 0.0011 percentage points

### Illustrative single forecast (from validation)

- Timestamp: `2025-09-04 05:40:00`; zone: `5`.
- Current moisture: 22.00%.
- Model predicted delta (before clipping): -3.92 pp.
- Predicted 24-hour moisture: 18.08%.
- Historical actual 24-hour moisture: 19.00%.

## Limitations and appropriate claims

1. This is model interpretability, **not a validation of predictive skill**.
2. Phase 2 unshrunk Gradient Boosting had lower validation MAE than persistence,
   but persistence had lower MAE on the previously viewed historical test.
3. The existing historical test was already reviewed in past phases; it is not pristine.
4. Sensor readings and 10-minute windows are strongly dependent; the row count
   overstates the number of independent environmental situations.
5. SHAP and permutation sensitivity can be unstable with correlated features.
6. Future rainfall/irrigation are unknown and a field-ready controller has not been established.
7. For real generalization claims, collect a truly new season, site, or future-period dataset.

## Output files

- `01_permutation_sensitivity.png`, `02_shap_global_importance.png`, `03_shap_local_example.png`.
- `permutation_sensitivity.csv`, `shap_global_importance.csv`, `shap_local_example.csv`,
  `shap_local_example.json`, and `phase3_summary.json`.

**Dataset source:** https://data.mendeley.com/datasets/c837v6p8ph/2

**Phase 3 complete.** Next: Phase 4 portfolio dashboard and GitHub packaging.
