# AgroFlow AI | Publishing to GitHub safely

## Repo name (suggested)

`agroflow-ai-soil-moisture-forecast`

Description: `Time-aware soil-moisture forecasting research with baseline benchmarking, SHAP explanations and a Streamlit dashboard.`

## Easiest safe approach

1. Open a NEW empty public repository on GitHub. Do not upload your entire PyCharm folder.
2. Review `PROJECT_README.md`. Copy it to the new repository as `README.md`. Keep the original dataset publisher's README locally; do not overwrite it accidentally.
3. Review `AGROFLOW_GITIGNORE.txt` and copy its content into `.gitignore` if needed.
4. Upload the reproducible **Python code** and report files, not personal data, generated models, or the raw/preprocessed datasets.
5. Add one screenshot of the final running app, if you wish, to `docs/dashboard.png` and refer to it from the README.
6. Ensure the README links to the original dataset rather than claiming it is your own.

## Useful source files to publish

- `agroflow_train.py`, `agroflow_train_v2.py`
- `agroflow_phase1_eda.py`, `agroflow_phase2_benchmark.py`, `agroflow_phase3_explain.py`
- `agroflow_dashboard_final.py`, `agroflow_phase4_check.py`
- `agroflow_project_requirements.txt` (you may copy to `requirements.txt` in the published repository)
- `README.md` (copied from `PROJECT_README.md`), `.gitignore`
- `PHASE1_GUIDE.md`, `PHASE2_GUIDE.md`, `PHASE3_GUIDE.md`, `PHASE4_GUIDE.md` (optional)
- Selected **small** charts and documented CSV summaries from `phase1_outputs/`, `phase2_outputs/`, `phase3_outputs/`.

## Exclude

- `01_raw_data/`, `02_processed_data/` (can be obtained from the publisher).
- `.venv/`, `.idea/`, Python caches.
- `agroflow_artifacts/` and `phase2_outputs/*.joblib` (fitted binaries).
- Large prediction-level files such as `phase2_outputs/historical_test_predictions.csv`.
- Source dataset author code (`03_code/`) until you verify redistribution terms separately.
- Passwords, access tokens or private identifying information.

## Important caveats

- Dataset attribution: https://data.mendeley.com/datasets/c837v6p8ph/2
- GitHub is a code portfolio. Deploying to Streamlit Community Cloud would need a supported setup for downloading source data and generating model files; this package does not configure that service.
- Include the full caveat: the previously explored historical test is **not an untouched holdout**.
- Avoid claims of 100% accuracy, causality or automatic irrigation.

## Example Git commands (optional)

```powershell
git init
git add .
git status
# Inspect the staged list carefully before committing.
git commit -m "Add reproducible AgroFlow AI research portfolio"
```

The `git add .` example is appropriate only **inside a curated repository directory** with a verified `.gitignore`; do not run it blindly over the original full project.
