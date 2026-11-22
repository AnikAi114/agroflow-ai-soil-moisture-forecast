# Phase 2 - Research Findings

**Objective:** observed soil moisture at exactly t + 24 hours.

**Verified dataset:** 17,614 exact-24h windows across four usable zones; 47,068 processed source records across five zones.

## Experiment design

- Chronological training, validation, and historical test periods with a 24-hour purge at training boundaries.
- Three expanding walk-forward folds *inside training data only* for shrinkage-weight calibration.
- Baseline: current soil moisture stays unchanged over the next 24 hours.
- ML regressors learn the change relative to current moisture. All predictions are clipped to 0-100%.
- Model parameters are fixed exploratory choices. No fresh independent external farm was used.
- Both unshrunk ML forecasts and CV-blended forecasts are reported; a zero CV weight equals persistence.

## Comparison (MAE in soil-moisture percentage points)

| Model | Train | Validation | Previously inspected historical test | Walk-forward weight |
|---|---:|---:|---:|---:|
| Persistence | 1.029 | 1.076 | 0.421 | 0.00 |
| Ridge (unshrunk) | 1.004 | 1.924 | 1.288 | 1.00 |
| Ridge (CV blended) | 1.029 | 1.076 | 0.421 | 0.00 |
| Random Forest (unshrunk) | 0.668 | 1.209 | 1.008 | 1.00 |
| Random Forest (CV blended) | 1.029 | 1.076 | 0.421 | 0.00 |
| Regularized Gradient Boosting (unshrunk) | 0.826 | 0.855 | 0.652 | 1.00 |
| Regularized Gradient Boosting (CV blended) | 1.029 | 1.076 | 0.421 | 0.00 |

**Validation-selected model:** Regularized Gradient Boosting (unshrunk) (validation MAE 0.855).
**Historical test:** selected model MAE 0.652; persistence MAE 0.421.

## Interpretation and integrity limits

The earlier historical test period had already influenced project development. Therefore any model differences on that period are **exploratory** and not independent proof of out-of-sample improvement.
Adjacent 10-minute observations have overlapping 24-hour targets, reducing the effective number of independent examples. Data coverage and response differ by zone. A different season or farm is needed for stronger evidence.

The error-regime table groups cases by the actual future 24-hour change, which is known **only after the fact**. It is for diagnostics, not a deployment rule.

No forecasting model here should automatically switch agricultural pumps or valves.

## Output files

`model_comparison.csv`, `walk_forward_fold_scores.csv`, `walk_forward_weight_selection.csv`, `error_analysis.csv`, `historical_test_predictions.csv`, `phase2_results.json`, `phase2_selected_model.joblib`, and three PNG charts.

**Dataset credit:** https://data.mendeley.com/datasets/c837v6p8ph/2
