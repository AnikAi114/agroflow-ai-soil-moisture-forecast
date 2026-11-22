# 🌱 AgroFlow AI — 24-Hour Soil Moisture Forecast Research

**An end-to-end, explainable machine-learning research prototype for precision-agriculture sensor data.**

AgroFlow AI audits five zones of historical Italian agricultural IoT sensor readings, verifies whether the published targets are truly 24 **clock-hours** ahead, compares interpretable baseline forecasts against several learning methods, and explains the resulting models with SHAP and permutation sensitivity. A Streamlit dashboard brings the evidence together.

> **Research-only.** This is **not** a field-validated irrigation recommendation system or an automatic irrigation controller.

## At a glance

| Item | Value |
|---|---:|
| Preprocessed sensor observations | 47,068 |
| Timestamp-verified 24-hour forecasting windows | 17,614 |
| Agricultural zones | 5 (4 eligible for the exact-24h model) |
| Original period | 2025, Italy |
| Phase 2 train / validation / previously inspected historical test | 11,813 / 2,213 / 2,643 |
| Explanations | SHAP, validation permutation sensitivity |
| Interface | Streamlit research dashboard |

## Research question

**Can historic soil and weather measurements predict soil moisture exactly 24 hours later more reliably than assuming moisture stays unchanged?**

The original preprocessed dataset contains targets defined **144 rows ahead** on an intended 10-minute grid. When there are observation gaps, 144 rows need not correspond to 24 elapsed hours. Our target audit retains only records where the timestamps and target values confirm a true 24-hour horizon. We exclude future-looking columns such as `water_vol_to_24h` and `real_moisture_delta` from ordinary predictive features.

## Four research phases

| Phase | Evidence and scripts |
|---|---|
| 1. Data exploration | `agroflow_phase1_eda.py` → six charts and coverage/quality report |
| 2. ML benchmark | `agroflow_phase2_benchmark.py` → persistence vs Ridge, Random Forest and Gradient Boosting, expanding walk-forward checks and error analysis |
| 3. Explainability | `agroflow_phase3_explain.py` → validation permutation sensitivity, global/local tree SHAP |
| 4. Portfolio interface | `agroflow_dashboard_final.py` → unified Streamlit dashboard and research limitations |

The earlier V1 and V2 training experiments are retained as separately labeled iterations (`agroflow_train.py`, `agroflow_train_v2.py`). The interactive forecast lab uses those models; the Phase 2 benchmark is presented as a separate set of experiments.

## Key evaluation evidence

**Phase 2 unshrunk models, MAE in soil-moisture percentage points (lower is better):**

| Model | Chronological validation | Previously inspected historical test |
|---|---:|---:|
| Persistence (no-change) | 1.076 | **0.421** |
| Ridge Regression | 1.924 | 1.288 |
| Random Forest | 1.209 | 1.008 |
| Regularized Gradient Boosting | **0.855** | 0.652 |

- Train-only walk-forward tuning selected a **zero learned-change weight** for each CV-blended ML model. A zero weight is **pure persistence**, not an ML performance win.
- Gradient Boosting improves over persistence on the original validation period, but **not** on the historical test period.
- The prior V2 experiment achieved validation MAE **0.563** and historical-test MAE **0.474**, also above the persistence historical-test MAE **0.421**.
- **Scientific honesty:** This historical test had already been examined during project development. The comparison is exploratory and **cannot** be advertised as a pristine, independently held-out result.

## Explainability findings

On 300 sampled validation records, global SHAP magnitude for the unshrunk Phase 2 Gradient Boosting model's **predicted change** was dominated by **agricultural zone (~1.874 pp)** and **current soil moisture (~0.767 pp)**, then soil temperature at 7–18 cm (~0.142 pp). These are learned model attributions, **not** evidence of causal relationships or irrigation necessity.

## Quick start (Windows / Python 3.13)

1. Clone/download this code repository.
2. Download the dataset from the **original publisher**: [Mendeley Data: Soil Moisture, Irrigation Actuator and Weather Dataset](https://data.mendeley.com/datasets/c837v6p8ph/2).
3. Place the five published preprocessed files in `02_processed_data/preprocessed/`:

```text
02_processed_data/
  preprocessed/
    dataset_zone_1_preprocessed.csv
    dataset_zone_2_preprocessed.csv
    dataset_zone_3_preprocessed.csv
    dataset_zone_4_preprocessed.csv
    dataset_zone_5_preprocessed.csv
```

4. Create a Python virtual environment and run the experiments **from the repository root**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r agroflow_project_requirements.txt
python agroflow_phase1_eda.py
python agroflow_train.py
python agroflow_train_v2.py
python agroflow_phase2_benchmark.py
python agroflow_phase3_explain.py
python agroflow_phase4_check.py
python -m streamlit run agroflow_dashboard_final.py
```

The experiment outputs are written to `agroflow_artifacts/`, `phase1_outputs/`, `phase2_outputs/`, `phase3_outputs/`, and `phase4_outputs/`.

**Important:** Only open `joblib` model artifacts produced by your own trusted local scripts. Do not load unknown model binaries from the web.

## Suggested repository structure

```text
agroflow-ai-soil-moisture-forecast/
├── agroflow_train.py
├── agroflow_train_v2.py
├── agroflow_phase1_eda.py
├── agroflow_phase2_benchmark.py
├── agroflow_phase3_explain.py
├── agroflow_phase4_check.py
├── agroflow_dashboard_final.py
├── agroflow_project_requirements.txt
├── phase1_outputs/            # curated charts / summary / findings
├── phase2_outputs/            # curated charts / tables / findings
├── phase3_outputs/            # curated charts / tables / findings
└── README.md
```

The publisher's complete raw/processed dataset and binary fitted models are **not** included in this repository. See `.gitignore`.

## Limitations

1. Source data have gaps and preprocessing, including interpolation and cleaning.
2. Adjacent 10-minute records share overlapping 24-hour outcomes; 17,614 windows are **not** 17,614 independent trials.
3. Zone 3 has zero verified exact-24h windows and is excluded from model training.
4. Historical weather is partly ERA5 reanalysis, not a demonstrated live forecast service.
5. Future precipitation and farmer irrigation decisions are unknown at inference time.
6. The previous historical test period was already inspected in V1/V2 work. No newly collected season, independent farm or field trial has validated this model.
7. Permutation/SHAP importance describes model sensitivity and association, not agronomic causality.
8. No automatic pump or valve operation is implemented or recommended.

## Data provenance & attribution

Data: **Soil Moisture, Irrigation Actuator and Weather Dataset**, Mendeley Data, available at [https://data.mendeley.com/datasets/c837v6p8ph/2](https://data.mendeley.com/datasets/c837v6p8ph/2). Respect the publisher's license, citation requirements, source README and data dictionary when using the original files. The code in this repository is a separate experimental analysis, not a claim of ownership of the sensor measurements.

## Appropriate portfolio summary

> Developed a reproducible precision-agriculture research prototype that audited five-zone soil and weather data, verified 24-hour forecast targets, compared time-aware ML regressors against persistence, and added SHAP-based explanations and a Streamlit dashboard. Reported the finding that a simple baseline was stronger in the previously inspected historical test period.

---

**Status:** Research prototype / portfolio demonstration. **Not field-ready.**
