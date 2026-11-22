"""AgroFlow AI | Phase 3: careful explanation of trained ML regressors.

Run from the project root: python agroflow_phase3_explain.py

This script loads Phase 2's EXISTING fitted models. It does NOT retrain them,
inspect the historical test set, or modify earlier phase outputs. Interpretations
are exploratory: SHAP values describe learned residual associations, not causal
irrigation effects or field reliability.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from agroflow_train import TARGET, chronological_split, load_verified_samples
from agroflow_phase2_benchmark import FEATURE_COLUMNS, NUMERIC_FEATURES, features

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "phase3_outputs"
BUNDLE_FILE = ROOT / "phase2_outputs" / "phase2_selected_model.joblib"
SEED = 42
MAX_PERMUTATION_ROWS = 800
MAX_SHAP_ROWS = 300
REPEATS = 4
HGB_NAME = "Regularized Gradient Boosting"
RF_NAME = "Random Forest"


def evaluation_subset(validation: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Take a reproducible, time-spread subset, not independently sampled trials."""
    if len(validation) <= cap:
        return validation.copy()
    indices = np.linspace(0, len(validation) - 1, cap, dtype=int)
    return validation.iloc[indices].copy()


def load_training_artifacts():
    if not BUNDLE_FILE.is_file():
        raise FileNotFoundError(
            "Phase 2 fitted model not found: " + str(BUNDLE_FILE)
            + "\nRun python agroflow_phase2_benchmark.py from the project root."
        )
    # joblib uses pickle. Only load model files produced locally by your own
    # Phase 2 script; never load an untrusted downloaded .joblib file.
    bundle = joblib.load(BUNDLE_FILE)
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Phase 2 model features differ from Phase 3 expectations.")
    models = bundle["all_fitted_nonbaseline_models"]
    for name in [RF_NAME, HGB_NAME]:
        if name not in models:
            raise ValueError(f"Missing Phase 2 model: {name}")
    return bundle, models


def forecast(model, x: pd.DataFrame, current: np.ndarray) -> np.ndarray:
    """Phase 2 models predict change in moisture, not absolute moisture."""
    return np.clip(current + model.predict(x), 0, 100)


def permutation_groups() -> dict[str, list[str]]:
    groups = {name: [name] for name in NUMERIC_FEATURES}
    groups["agricultural_zone"] = [f"zone_{zone}" for zone in range(1, 6)]
    return groups


def permutation_sensitivity(models, validation: pd.DataFrame) -> pd.DataFrame:
    sample = evaluation_subset(validation, MAX_PERMUTATION_ROWS)
    x = features(sample)
    current = sample["soil_moisture"].to_numpy(dtype=float)
    y = sample[TARGET].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    rows = []
    for model_name in [RF_NAME, HGB_NAME]:
        model = models[model_name]
        reference_mae = float(mean_absolute_error(y, forecast(model, x, current)))
        for group_name, cols in permutation_groups().items():
            changes = []
            for _ in range(REPEATS):
                order = rng.permutation(len(x))
                xp = x.copy()
                xp.loc[:, cols] = x.iloc[order][cols].to_numpy()
                permuted_mae = float(mean_absolute_error(y, forecast(model, xp, current)))
                changes.append(permuted_mae - reference_mae)
            rows.append({
                "model": model_name,
                "feature_group": group_name,
                "validation_rows": int(len(sample)),
                "unpermuted_mae": round(reference_mae, 6),
                "mae_increase_mean_pp": round(float(np.mean(changes)), 6),
                "mae_increase_std_pp": round(float(np.std(changes)), 6),
                "repeats": REPEATS,
            })
    return pd.DataFrame(rows)


def save_permutation_chart(table: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6.8), sharey=False)
    for ax, name in zip(axes, [RF_NAME, HGB_NAME]):
        sub = table[table["model"] == name].sort_values("mae_increase_mean_pp")
        ax.barh(sub["feature_group"], sub["mae_increase_mean_pp"])
        ax.axvline(0, linewidth=1, color="black")
        ax.set_title(name)
        ax.set_xlabel("MAE increase after shuffling (percentage points)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Validation permutation sensitivity | Unshrunk ML models")
    fig.text(0.5, 0.005,
             "Negative bars: shuffling improved MAE on the sample; these are NOT causal effects.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT / "01_permutation_sensitivity.png", dpi=165)
    plt.close(fig)


def group_shap(values: np.ndarray) -> dict[str, np.ndarray]:
    grouped = {name: values[:, idx] for idx, name in enumerate(NUMERIC_FEATURES)}
    group_indices = [FEATURE_COLUMNS.index(f"zone_{z}") for z in range(1, 6)]
    grouped["agricultural_zone"] = values[:, group_indices].sum(axis=1)
    return grouped


def shap_analysis(model, validation: pd.DataFrame):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is not installed. Run: python -m pip install shap"
        ) from exc

    sample = evaluation_subset(validation, MAX_SHAP_ROWS)
    x = features(sample)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x), dtype=float)
    if values.shape != x.shape:
        raise ValueError(f"Unexpected SHAP result shape: {values.shape}; expected {x.shape}")

    # SHAP explains predicted 24-hour DELTA, not the soil moisture baseline.
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    reconstructed = base + values.sum(axis=1)
    actual_delta_prediction = np.asarray(model.predict(x), dtype=float)
    discrepancy = float(np.max(np.abs(reconstructed - actual_delta_prediction)))
    if discrepancy > 1e-3:
        raise ValueError(f"SHAP additive consistency check failed: {discrepancy}")

    grouped = group_shap(values)
    global_table = pd.DataFrame({
        "feature_group": list(grouped),
        "mean_abs_shap_on_predicted_change_pp": [
            float(np.mean(np.abs(attributions))) for attributions in grouped.values()
        ],
        "mean_signed_shap_on_predicted_change_pp": [
            float(np.mean(attributions)) for attributions in grouped.values()
        ],
        "n": len(x),
    }).sort_values("mean_abs_shap_on_predicted_change_pp", ascending=False)
    global_table.to_csv(OUTPUT / "shap_global_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7.0))
    top = global_table.head(12).iloc[::-1]
    ax.barh(top["feature_group"], top["mean_abs_shap_on_predicted_change_pp"])
    ax.set_xlabel("Mean |SHAP| on predicted 24h moisture change (percentage points)")
    ax.set_title("Gradient Boosting: global SHAP magnitude (validation sample)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_shap_global_importance.png", dpi=165)
    plt.close(fig)

    # A local case with a relatively large estimated change illustrates that
    # SHAP is a decomposition of an ML forecast, not a guarantee of truth.
    local_i = int(np.argmax(np.abs(actual_delta_prediction)))
    local_values = {name: float(v[local_i]) for name, v in grouped.items()}
    local_table = pd.DataFrame({
        "feature_group": list(local_values),
        "shap_contribution_to_change_pp": list(local_values.values()),
    }).sort_values("shap_contribution_to_change_pp")
    local_table.to_csv(OUTPUT / "shap_local_example.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.barh(local_table["feature_group"], local_table["shap_contribution_to_change_pp"])
    ax.axvline(0, linewidth=1, color="black")
    ax.set_xlabel("Contribution to predicted moisture CHANGE (percentage points)")
    ax.set_title("One illustrative validation forecast: grouped SHAP contributions")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_shap_local_example.png", dpi=165)
    plt.close(fig)

    row = sample.iloc[local_i]
    example = {
        "timestamp": str(row["ts"]),
        "zone": int(row["zone"]),
        "current_soil_moisture": round(float(row["soil_moisture"]), 4),
        "actual_24h_soil_moisture": round(float(row[TARGET]), 4),
        "unclipped_model_predicted_delta": round(float(actual_delta_prediction[local_i]), 4),
        "forecast_soil_moisture": round(float(np.clip(
            row["soil_moisture"] + actual_delta_prediction[local_i], 0, 100
        )), 4),
        "shap_base_delta": round(base, 4),
        "grouped_shap_sum": round(float(sum(local_values.values())), 4),
        "max_shap_reconstruction_difference": discrepancy,
    }
    (OUTPUT / "shap_local_example.json").write_text(
        json.dumps(example, indent=2), encoding="utf-8"
    )
    return global_table, example


def save_report(bundle, validation, permutation, shap_global, example, full_count):
    gb_perm = permutation[permutation["model"] == HGB_NAME].sort_values(
        "mae_increase_mean_pp", ascending=False
    )
    rf_perm = permutation[permutation["model"] == RF_NAME].sort_values(
        "mae_increase_mean_pp", ascending=False
    )
    lines = [
        "# AgroFlow AI | Phase 3 Explainable AI Findings", "",
        "## Research setup", "",
        f"- Verified full dataset: {full_count:,} 24-hour rows.",
        f"- Chronological validation rows: {len(validation):,}; exploratory feature explanation.",
        f"- Permutation subsample: at most {MAX_PERMUTATION_ROWS:,} time-spread rows, {REPEATS} shuffles per group.",
        f"- SHAP subsample: at most {MAX_SHAP_ROWS:,} time-spread rows.",
        "- ML models are fitted Phase 2 models trained on the earlier **training** period only; no refitting here.",
        f"- Phase 2 validation-selected candidate: **{bundle['selected_model_name']}**.",
        "- All walk-forward CV blend weights were **0.0**. Those blended methods equal persistence,",
        "  so explanations here concern the actual **unshrunk ML models**, not blended persistence.",
        "- Historical-test rows are read only to reproduce the established chronological split;",
        "  no historical-test rows are used to calculate feature explanations.", "",
        "## Exploratory validation permutation sensitivity", "",
        "The metric is the change in 24-hour forecast MAE after shuffling one input",
        "feature group on a validation subsample, keeping the rest of the inputs unchanged.",
        "A positive score indicates sensitivity on these observations, not causal influence.",
        "Group-level shuffling across dates/zones can form unrealistic combinations.", "",
        "### Top 5 positive sensitivities: Gradient Boosting", "",
    ]
    for row in gb_perm.head(5).itertuples():
        lines.append(f"- `{row.feature_group}`: MAE change {row.mae_increase_mean_pp:+.4f} percentage points")
    lines.extend(["", "### Top 5 positive sensitivities: Random Forest", ""])
    for row in rf_perm.head(5).itertuples():
        lines.append(f"- `{row.feature_group}`: MAE change {row.mae_increase_mean_pp:+.4f} percentage points")
    lines.extend(["", "## SHAP explanation of the Gradient Boosting residual", "",
                  "SHAP decomposes the model's **predicted 24-hour soil-moisture CHANGE**.",
                  "The forecast itself adds that learned change to current moisture (with clipping).",
                  "SHAP here does not explain persistence, future irrigation need, or causal effects.",
                  "Correlated weather inputs and one-hot zone indicators can share attribution.", "",
                  "### Top 5 by mean absolute SHAP magnitude", ""])
    for row in shap_global.head(5).itertuples():
        lines.append(f"- `{row.feature_group}`: {row.mean_abs_shap_on_predicted_change_pp:.4f} percentage points")
    lines.extend(["", "### Illustrative single forecast (from validation)", "",
                  f"- Timestamp: `{example['timestamp']}`; zone: `{example['zone']}`.",
                  f"- Current moisture: {example['current_soil_moisture']:.2f}%.",
                  f"- Model predicted delta (before clipping): {example['unclipped_model_predicted_delta']:+.2f} pp.",
                  f"- Predicted 24-hour moisture: {example['forecast_soil_moisture']:.2f}%.",
                  f"- Historical actual 24-hour moisture: {example['actual_24h_soil_moisture']:.2f}%.",
                  "", "## Limitations and appropriate claims", "",
                  "1. This is model interpretability, **not a validation of predictive skill**.",
                  "2. Phase 2 unshrunk Gradient Boosting had lower validation MAE than persistence,",
                  "   but persistence had lower MAE on the previously viewed historical test.",
                  "3. The existing historical test was already reviewed in past phases; it is not pristine.",
                  "4. Sensor readings and 10-minute windows are strongly dependent; the row count",
                  "   overstates the number of independent environmental situations.",
                  "5. SHAP and permutation sensitivity can be unstable with correlated features.",
                  "6. Future rainfall/irrigation are unknown and a field-ready controller has not been established.",
                  "7. For real generalization claims, collect a truly new season, site, or future-period dataset.",
                  "", "## Output files", "",
                  "- `01_permutation_sensitivity.png`, `02_shap_global_importance.png`, `03_shap_local_example.png`.",
                  "- `permutation_sensitivity.csv`, `shap_global_importance.csv`, `shap_local_example.csv`,",
                  "  `shap_local_example.json`, and `phase3_summary.json`.",
                  "", "**Dataset source:** https://data.mendeley.com/datasets/c837v6p8ph/2", "",
                  "**Phase 3 complete.** Next: Phase 4 portfolio dashboard and GitHub packaging.", ""])
    (OUTPUT / "PHASE3_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bundle, models = load_training_artifacts()
    all_rows, audit = load_verified_samples()
    train, validation, _historical_test, *_ = chronological_split(all_rows)
    # Do not use the historical test. Discard its reference immediately.
    del _historical_test

    perm = permutation_sensitivity(models, validation)
    perm.to_csv(OUTPUT / "permutation_sensitivity.csv", index=False)
    save_permutation_chart(perm)
    shap_global, example = shap_analysis(models[HGB_NAME], validation)
    save_report(bundle, validation, perm, shap_global, example, len(all_rows))

    summary = {
        "phase": 3,
        "purpose": "Exploratory explanation of existing Phase 2 unshrunk models",
        "selected_phase2_candidate": bundle["selected_model_name"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "permutation_rows": min(len(validation), MAX_PERMUTATION_ROWS),
        "shap_rows": min(len(validation), MAX_SHAP_ROWS),
        "verified_24h_rows": len(all_rows),
        "dataset_source": "https://data.mendeley.com/datasets/c837v6p8ph/2",
        "important_note": "No model retraining; no new independent generalization claim; not for irrigation control",
        "example": example,
    }
    (OUTPUT / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAGROFLOW AI - PHASE 3 EXPLAINABLE AI COMPLETED")
    print("=" * 61)
    print(f"Verified 24-hour records: {len(all_rows):,}")
    print(f"Train records (models were already fitted in Phase 2): {len(train):,}")
    print(f"Validation records: {len(validation):,}")
    print(f"Permutation sample rows: {min(len(validation), MAX_PERMUTATION_ROWS):,}")
    print(f"SHAP sample rows: {min(len(validation), MAX_SHAP_ROWS):,}")
    print(f"Selected Phase 2 model: {bundle['selected_model_name']}")
    print("\nTop 5 Gradient Boosting SHAP magnitude features:")
    for row in shap_global.head(5).itertuples():
        print(f"  {row.feature_group:<30} {row.mean_abs_shap_on_predicted_change_pp:.4f} pp")
    print("\nSaved 3 charts, CSV tables, and PHASE3_FINDINGS.md to phase3_outputs/.")
    print("No earlier models, scripts, data, or dashboards were modified.")
    print("Reminder: importance does not imply causality or field readiness.")


if __name__ == "__main__":
    main()
