# AgroFlow AI — Phase 4 Final Dashboard & Portfolio Packaging

## 1. Where to place the files

Copy these **three** files into your existing PyCharm **project root**, next to `agroflow_phase3_explain.py` and `agroflow_train_v2.py`:

- `agroflow_dashboard_final.py` — the unified Streamlit app.
- `agroflow_phase4_check.py` — non-destructive readiness check.
- `PHASE4_GUIDE.md` — this guide.

Optional GitHub materials:

- `PROJECT_README.md` — a proposed public-facing README. Do **not** replace the dataset publisher's original `README.md` blindly.
- `AGROFLOW_GITIGNORE.txt` — review and merge with your `.gitignore` before renaming.
- `GITHUB_PUBLISH_GUIDE.md` — safe, selective publishing instructions.
- `agroflow_project_requirements.txt` — all libraries used by all phases.

**Do not move or delete** the existing `01_raw_data`, `02_processed_data`, `03_code`, V1, V2, or Phase 1–3 files.

## 2. Run locally (Windows / PyCharm)

From the project root (example: `D:\AgroFlow AI — Smart Irrigation Assistant`):

```powershell
python -m pip install -r agroflow_project_requirements.txt
python agroflow_phase4_check.py
python -m streamlit run agroflow_dashboard_final.py
```

If Phases 1–3 and V2 have been run already, no retraining is needed. Your existing `phase1_outputs`, `phase2_outputs`, `phase3_outputs` and `agroflow_artifacts/models_v2.joblib` are reused.

**Important:** The local V2 model file is a joblib/pickle artifact. Load only the files your own trusted training script generated; never load a model binary from an unknown source.

## 3. What you'll see

- **Overview:** the study's key numbers and research caveats.
- **Data exploration:** verified 24h data coverage, descriptive visualizations and exportable data-quality table.
- **Model evidence:** an honest comparison of persistence, Ridge, Random Forest and Gradient Boosting, including MAE differences over time.
- **Explainability:** SHAP and permutation-feature-sensitivity charts of fixed Phase 2 models.
- **Forecast lab:** the prior V1/V2 interactive demonstration versus persistence, clearly labeled as exploratory.
- **Methods & limitations:** data provenance, leakage safeguards, reproducibility commands and source reports.

## 4. What Phase 4 does NOT do

- Does not train a new model or claim to improve the historical test.
- Does not turn on irrigation hardware, use live farm sensor feeds or calculate real-world watering instructions.
- Does not claim that SHAP values are causal crop-water relationships.
- Does not convert an already-inspected historical test into an untouched holdout.
- Does not include original large datasets in the public GitHub package.

## 5. GitHub publication

You can publish a **source-code research portfolio**, but do not upload the entire working folder (raw source datasets and generated binaries). See `GITHUB_PUBLISH_GUIDE.md` and `PROJECT_README.md` for the exact curated file list and source attribution.

A GitHub-hosted code repository is **not automatically a deployed Streamlit website**. The full interactive forecast requires regenerating model artifacts locally after downloading the source dataset.

Dataset citation: https://data.mendeley.com/datasets/c837v6p8ph/2
