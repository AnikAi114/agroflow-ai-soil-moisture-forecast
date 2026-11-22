"""AgroFlow AI - Phase 2: honest, time-aware model benchmarking.

Put this file in the project root next to agroflow_train.py and run:
    python agroflow_phase2_benchmark.py

Reads the already audited source through agroflow_train.py. Saves ONLY to
phase2_outputs/. Does not overwrite datasets, V1/V2 models, or dashboards.
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agroflow_train import HORIZON, TARGET, chronological_split, load_verified_samples

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "phase2_outputs"
SEED = 42
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

# All values describe conditions available at prediction time. In particular,
# future irrigation amounts, target deltas, and target means are excluded.
NUMERIC_FEATURES = [
    "soil_moisture", "weather_temp", "weather_humidity", "weather_rain",
    "weather_pressure", "weather_wind_speed", "weather_radiation",
    "soil_temperature_0-7cm", "soil_temperature_7-18cm",
    "irrigation_duration_minutes", "month", "hour",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + [f"zone_{z}" for z in range(1, 6)]
MODEL_NAMES = ("Ridge", "Random Forest", "Regularized Gradient Boosting")


def features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[NUMERIC_FEATURES].copy()
    for zone in range(1, 6):
        result[f"zone_{zone}"] = (frame["zone"] == zone).astype(float)
    return result[FEATURE_COLUMNS]


def make_model(name: str):
    # Parameters are fixed for a transparent *exploratory* comparison. They are
    # not claimed to have been selected without exposure to earlier results.
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=7, min_samples_leaf=50,
            max_features=0.9, n_jobs=-1, random_state=SEED,
        )
    if name == "Regularized Gradient Boosting":
        return HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=6, min_samples_leaf=100,
            l2_regularization=200, learning_rate=0.02,
            random_state=SEED,
        )
    raise ValueError(f"Unknown model: {name}")


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 5),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 5),
        "r2": round(float(r2_score(actual, predicted)), 5),
    }


def predict(frame: pd.DataFrame, model=None, weight: float = 0.0) -> np.ndarray:
    current = frame["soil_moisture"].to_numpy(dtype=float)
    if model is None or weight == 0:
        return current.copy()
    learned_change = model.predict(features(frame))
    return np.clip(current + weight * learned_change, 0.0, 100.0)


def safe_relative_gain(baseline_mae: float, candidate_mae: float):
    if baseline_mae <= 1e-12:
        return None
    return round(100.0 * (baseline_mae - candidate_mae) / baseline_mae, 2)


def training_walk_forward_folds(train: pd.DataFrame) -> list[dict]:
    """Three expanding-history validations using ONLY the original train subset.

    The last label of each fold's training period must occur *before* the
    fold validation starts: t + 24h < validation_start. Fold validation blocks
    may contain overlapping forecast targets: these are not independent rows.
    """
    unique_times = pd.Index(train["ts"].drop_duplicates().sort_values())
    windows = ((0.45, 0.60), (0.60, 0.78), (0.78, 1.0))
    result = []
    for i, (start_fraction, end_fraction) in enumerate(windows, start=1):
        first = pd.Timestamp(unique_times[int((len(unique_times) - 1) * start_fraction)])
        last = pd.Timestamp(unique_times[int((len(unique_times) - 1) * end_fraction)])
        train_fold = train.loc[(train["ts"] + HORIZON) < first].copy()
        val_fold = train.loc[(train["ts"] >= first) & (train["ts"] <= last)].copy()
        if len(train_fold) < 200 or len(val_fold) < 100:
            raise ValueError(f"Insufficient data in walk-forward fold {i}")
        if (train_fold["ts"] + HORIZON).max() >= val_fold["ts"].min():
            raise AssertionError("Target crosses the walk-forward validation boundary")
        result.append({
            "fold": i, "train": train_fold, "validation": val_fold,
            "train_rows": int(len(train_fold)), "validation_rows": int(len(val_fold)),
            "validation_start": str(first), "validation_end": str(last),
        })
    return result


def cross_validate(train: pd.DataFrame):
    folds = training_walk_forward_folds(train)
    detailed = []
    for fold in folds:
        tr = fold["train"]
        va = fold["validation"]
        y_delta = (tr[TARGET] - tr["soil_moisture"]).to_numpy(dtype=float)
        y_val = va[TARGET].to_numpy(dtype=float)
        current_val = va["soil_moisture"].to_numpy(dtype=float)
        for name in MODEL_NAMES:
            model = make_model(name)
            model.fit(features(tr), y_delta)
            raw_delta = model.predict(features(va))
            for weight in BLEND_WEIGHTS:
                prediction = np.clip(current_val + weight * raw_delta, 0.0, 100.0)
                detailed.append({
                    "fold": fold["fold"], "model": name, "weight": weight,
                    "n": int(len(va)), "abs_error_sum": float(np.abs(y_val - prediction).sum()),
                    "mae": float(mean_absolute_error(y_val, prediction)),
                })

    details = pd.DataFrame(detailed)
    grouped = details.groupby(["model", "weight"], as_index=False).agg(
        n=("n", "sum"), total_absolute_error=("abs_error_sum", "sum"),
    )
    grouped["pooled_walkforward_mae"] = grouped["total_absolute_error"] / grouped["n"]

    weights = {}
    for name in MODEL_NAMES:
        sub = grouped[grouped["model"] == name].sort_values(
            ["pooled_walkforward_mae", "weight"], kind="stable"
        )
        weights[name] = float(sub.iloc[0]["weight"])
    return folds, details, grouped, weights


def evaluation_table(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                     models: dict, weights: dict):
    """Report both genuine ML forecasts and CV-calibrated forecasts.

    If a CV weight is zero, that candidate has effectively rejected its own
    learned change. This MUST NOT be hidden or described as ML improvement.
    """
    sets = {"train": train, "validation": val, "historical_test": test}
    rows, forecasts = [], []
    for split_name, subset in sets.items():
        actual = subset[TARGET].to_numpy(dtype=float)
        methods = {"Persistence": (predict(subset), 0.0)}
        for name in MODEL_NAMES:
            methods[f"{name} (unshrunk)"] = (predict(subset, models[name], 1.0), 1.0)
            methods[f"{name} (CV blended)"] = (
                predict(subset, models[name], weights[name]), weights[name]
            )
        for name, (values, weight) in methods.items():
            metrics = score(actual, values)
            rows.append({
                "split": split_name, "model": name,
                "blend_weight": float(weight), "n": int(len(subset)), **metrics,
            })
        if split_name == "historical_test":
            record = subset[["ts", "zone", "soil_moisture", TARGET]].copy()
            record = record.rename(columns={
                "soil_moisture": "current_moisture",
                TARGET: "actual_24h_moisture",
            })
            for name, (values, _) in methods.items():
                col_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_prediction"
                record[col_name] = values
            record["actual_change_24h"] = (
                record["actual_24h_moisture"] - record["current_moisture"]
            )
            forecasts.append(record)
    return pd.DataFrame(rows), pd.concat(forecasts, ignore_index=True)


def make_error_table(forecast: pd.DataFrame) -> pd.DataFrame:
    display_names = ["Persistence"] + [
        f"{name} ({variant})" for name in MODEL_NAMES
        for variant in ("unshrunk", "CV blended")
    ]
    columns = {
        name: name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_prediction"
        for name in display_names
    }
    subset_masks = {
        "all": np.ones(len(forecast), dtype=bool),
        "little_change_lt_2pp": (forecast["actual_change_24h"].abs() < 2).to_numpy(),
        "large_change_ge_2pp": (forecast["actual_change_24h"].abs() >= 2).to_numpy(),
    }
    rows = []
    for zone in ["all"] + sorted(forecast["zone"].unique().tolist()):
        zone_mask = np.ones(len(forecast), dtype=bool) if zone == "all" else (forecast["zone"] == zone).to_numpy()
        for subgroup, subgroup_mask in subset_masks.items():
            block = forecast.loc[zone_mask & subgroup_mask]
            if block.empty:
                continue
            for name, col in columns.items():
                actual = block["actual_24h_moisture"].to_numpy(dtype=float)
                predicted = block[col].to_numpy(dtype=float)
                rows.append({
                    "zone": zone, "regime": subgroup, "model": name,
                    "n": int(len(block)), **score(actual, predicted),
                })
    return pd.DataFrame(rows)


def chart_error_comparison(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 6.8))
    values = metrics.pivot(index="model", columns="split", values="mae")
    models = ["Persistence"] + [
        f"{name} ({variant})" for name in MODEL_NAMES
        for variant in ("unshrunk", "CV blended")
    ]
    x = np.arange(len(models))
    width = 0.36
    ax.bar(x - width / 2, values.loc[models, "validation"], width, label="Validation")
    ax.bar(x + width / 2, values.loc[models, "historical_test"], width,
           label="Historical test (previously inspected)")
    ax.set_xticks(x, models, rotation=24, ha="right", fontsize=9)
    ax.set_ylabel("MAE (soil moisture percentage points)")
    ax.set_title("Raw ML predictions versus train-CV shrinkage and persistence")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_model_comparison.png", dpi=160)
    plt.close(fig)


def chart_zone_errors(err: pd.DataFrame) -> None:
    rows = err[(err["zone"] != "all") & (err["regime"] == "all")]
    zones = sorted(rows["zone"].unique().tolist())
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    x = np.arange(len(zones))
    width = 0.19
    for i, name in enumerate(["Persistence"] + [f"{n} (unshrunk)" for n in MODEL_NAMES]):
        chunk = rows.loc[rows["model"] == name].set_index("zone")
        ax.bar(x + (i - 1.5) * width, chunk.loc[zones, "mae"], width, label=name)
    ax.set_xticks(x, [f"Zone {zone}" for zone in zones])
    ax.set_ylabel("Historical test MAE (percentage points)")
    ax.set_title("Unshrunk ML error varies across agricultural zones")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_zone_error_comparison.png", dpi=160)
    plt.close(fig)


def chart_change_regimes(err: pd.DataFrame) -> None:
    rows = err[(err["zone"] == "all") & (err["regime"] != "all")]
    fig, ax = plt.subplots(figsize=(11.1, 5.5))
    regimes = ["little_change_lt_2pp", "large_change_ge_2pp"]
    x = np.arange(2)
    width = 0.19
    for i, name in enumerate(["Persistence"] + [f"{n} (unshrunk)" for n in MODEL_NAMES]):
        chunk = rows.loc[rows["model"] == name].set_index("regime")
        ax.bar(x + (i - 1.5) * width, chunk.loc[regimes, "mae"], width,
               label=name)
    ax.set_xticks(x, ["Actual change < 2 pp", "Actual change >= 2 pp"])
    ax.set_ylabel("Historical test MAE (percentage points)")
    ax.set_title("Post-hoc error analysis: stable vs changing moisture")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_change_regime_errors.png", dpi=160)
    plt.close(fig)


def save_report(df, audit, train, val, test, cut1, cut2, folds, cv_details,
                cv_summary, weights, metrics, test_forecasts, errors, selected):
    baseline_val = metrics.loc[(metrics["split"] == "validation") & (
        metrics["model"] == "Persistence"), "mae"].iloc[0]
    baseline_test = metrics.loc[(metrics["split"] == "historical_test") & (
        metrics["model"] == "Persistence"), "mae"].iloc[0]
    selected_val = metrics.loc[(metrics["split"] == "validation") & (
        metrics["model"] == selected), "mae"].iloc[0]
    selected_test = metrics.loc[(metrics["split"] == "historical_test") & (
        metrics["model"] == selected), "mae"].iloc[0]
    fold_summary = [
        {k: fold[k] for k in ("fold", "train_rows", "validation_rows",
                              "validation_start", "validation_end")}
        for fold in folds
    ]
    report = {
        "title": "AgroFlow AI Phase 2: exploratory time-aware benchmarking",
        "source": "https://data.mendeley.com/datasets/c837v6p8ph/2",
        "total_source_rows": int(sum(x["source_rows"] for x in audit)),
        "verified_exact_24h_windows": int(len(df)),
        "per_zone_audit": audit,
        "chronological_boundaries": {"train_val": str(cut1), "val_test": str(cut2)},
        "split_rows": {"train": len(train), "validation": len(val), "historical_test": len(test)},
        "walk_forward_folds_within_train_only": fold_summary,
        "model_features": FEATURE_COLUMNS,
        "selected_weights_from_train_only_walk_forward": weights,
        "model_selected_on_original_validation": selected,
        "metrics": metrics.to_dict(orient="records"),
        "validation_selected_percent_improvement_vs_persistence": safe_relative_gain(baseline_val, selected_val),
        "historical_test_selected_percent_improvement_vs_persistence": safe_relative_gain(baseline_test, selected_test),
        "important_limitations": [
            "This research forecast is not a field-tested irrigation controller.",
            "The historical test period was repeatedly viewed in earlier V1/V2 development; it is NOT an untouched independent holdout.",
            "A zero walk-forward blend weight means the model's learned correction was rejected; it is exactly persistence, not ML success.",
            "V2-style gradient boosting regularization was informed by earlier exploratory work; not a blind comparison.",
            "Only targets exactly 24 clock-hours after the input reading are included; publisher's raw 144-row target is not always 24 hours.",
            "For a strictly causal prediction, features include only contemporaneous or past information, not future rain/water/targets.",
            "An expanding walk-forward with a 24h training-label purge is used inside the train partition; adjacent test targets still overlap heavily.",
            "The same sites, one summer season, sensor gaps, and temporal shift limit generalization.",
            "Zone 3 provides no exact 24-hour forecasting windows. Later historical test includes only zones 2, 4, 5.",
            "Post-hoc error grouping by observed future change must never be used as an input at inference.",
            "The best validation model is chosen without optimizing against the historical test results in this script; however earlier development viewed the test.",
        ],
    }
    (OUTPUT / "phase2_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics.to_csv(OUTPUT / "model_comparison.csv", index=False)
    cv_details.to_csv(OUTPUT / "walk_forward_fold_scores.csv", index=False)
    cv_summary.drop(columns=["total_absolute_error"]).to_csv(
        OUTPUT / "walk_forward_weight_selection.csv", index=False
    )
    errors.to_csv(OUTPUT / "error_analysis.csv", index=False)
    test_forecasts.to_csv(OUTPUT / "historical_test_predictions.csv", index=False)
    chart_error_comparison(metrics)
    chart_zone_errors(errors)
    chart_change_regimes(errors)

    compact = metrics.pivot(index="model", columns="split", values="mae")
    lines = [
        "# Phase 2 - Research Findings",
        "",
        "**Objective:** observed soil moisture at exactly t + 24 hours.",
        "",
        f"**Verified dataset:** {len(df):,} exact-24h windows across four usable zones; "
        f"{sum(x['source_rows'] for x in audit):,} processed source records across five zones.",
        "",
        "## Experiment design",
        "",
        "- Chronological training, validation, and historical test periods with a 24-hour purge at training boundaries.",
        "- Three expanding walk-forward folds *inside training data only* for shrinkage-weight calibration.",
        "- Baseline: current soil moisture stays unchanged over the next 24 hours.",
        "- ML regressors learn the change relative to current moisture. All predictions are clipped to 0-100%.",
        "- Model parameters are fixed exploratory choices. No fresh independent external farm was used.",
        "- Both unshrunk ML forecasts and CV-blended forecasts are reported; a zero CV weight equals persistence.",
        "",
        "## Comparison (MAE in soil-moisture percentage points)",
        "",
        "| Model | Train | Validation | Previously inspected historical test | Walk-forward weight |",
        "|---|---:|---:|---:|---:|",
    ]
    display_names = ["Persistence"] + [
        f"{name} ({variant})" for name in MODEL_NAMES
        for variant in ("unshrunk", "CV blended")
    ]
    for name in display_names:
        line = compact.loc[name]
        if name == "Persistence":
            weight = 0.0
        elif name.endswith("(unshrunk)"):
            weight = 1.0
        else:
            base_name = name.replace(" (CV blended)", "")
            weight = weights[base_name]
        lines.append(
            f"| {name} | {line['train']:.3f} | {line['validation']:.3f} | "
            f"{line['historical_test']:.3f} | {weight:.2f} |"
        )
    lines.extend([
        "",
        f"**Validation-selected model:** {selected} (validation MAE {selected_val:.3f}).",
        f"**Historical test:** selected model MAE {selected_test:.3f}; persistence MAE {baseline_test:.3f}.",
        "",
        "## Interpretation and integrity limits",
        "",
        "The earlier historical test period had already influenced project development. Therefore any model differences on that period are **exploratory** and not independent proof of out-of-sample improvement.",
        "Adjacent 10-minute observations have overlapping 24-hour targets, reducing the effective number of independent examples. Data coverage and response differ by zone. A different season or farm is needed for stronger evidence.",
        "",
        "The error-regime table groups cases by the actual future 24-hour change, which is known **only after the fact**. It is for diagnostics, not a deployment rule.",
        "",
        "No forecasting model here should automatically switch agricultural pumps or valves.",
        "",
        "## Output files",
        "",
        "`model_comparison.csv`, `walk_forward_fold_scores.csv`, `walk_forward_weight_selection.csv`, `error_analysis.csv`, `historical_test_predictions.csv`, `phase2_results.json`, `phase2_selected_model.joblib`, and three PNG charts.",
        "",
        "**Dataset credit:** https://data.mendeley.com/datasets/c837v6p8ph/2",
        "",
    ])
    (OUTPUT / "PHASE2_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df, audit = load_verified_samples()
    train, val, test, cut1, cut2 = chronological_split(df)

    folds, cv_details, cv_summary, weights = cross_validate(train)
    models = {}
    y_train_delta = (train[TARGET] - train["soil_moisture"]).to_numpy(dtype=float)
    for name in MODEL_NAMES:
        models[name] = make_model(name)
        models[name].fit(features(train), y_train_delta)

    metrics, test_forecasts = evaluation_table(train, val, test, models, weights)
    errors = make_error_table(test_forecasts)
    validation_rank = metrics[metrics["split"] == "validation"].sort_values(
        ["mae", "model"], kind="stable"
    )
    # Break exact ties in favor of the simpler persistence baseline.
    validation_rank = validation_rank.copy()
    validation_rank["complexity"] = (validation_rank["model"] != "Persistence").astype(int)
    validation_rank = validation_rank.sort_values(
        ["mae", "complexity", "model"], kind="stable"
    )
    selected = str(validation_rank.iloc[0]["model"])

    # Only the original training partition is used to fit stored candidate models.
    # The stored selection is from the chronological validation, not the test.
    if selected == "Persistence":
        selected_base_model, selected_weight = None, 0.0
    elif selected.endswith(" (unshrunk)"):
        base_name = selected.removesuffix(" (unshrunk)")
        selected_base_model, selected_weight = models[base_name], 1.0
    else:
        base_name = selected.removesuffix(" (CV blended)")
        selected_base_model, selected_weight = models[base_name], weights[base_name]
    model_record = {
        "selected_model_name": selected,
        "selected_model": selected_base_model,
        "selected_blend_weight": float(selected_weight),
        "all_fitted_nonbaseline_models": models,
        "cross_validation_blend_weights": weights,
        "candidate_model_names": list(MODEL_NAMES),
        "numeric_features": NUMERIC_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "purpose": "Exploratory 24-hour soil-moisture forecast; not automatic irrigation control",
    }
    joblib.dump(model_record, OUTPUT / "phase2_selected_model.joblib")
    report = save_report(df, audit, train, val, test, cut1, cut2, folds,
                         cv_details, cv_summary, weights, metrics,
                         test_forecasts, errors, selected)

    print("\nAGROFLOW AI - PHASE 2 MODEL BENCHMARK COMPLETED")
    print("=" * 61)
    print(f"Source records: {report['total_source_rows']:,}")
    print(f"Verified exact-24h forecasting windows: {len(df):,}")
    print(f"Train: {len(train):,} | Validation: {len(val):,} | Historical test: {len(test):,}")
    print("\nShrinkage weights chosen on TRAIN-only walk-forward folds:")
    for name, weight in weights.items():
        print(f"  {name:<31} {weight:.2f}")
    print("\nA zero CV weight MEANS pure persistence, not an ML win.")
    print("\nMAE comparison (lower = better):")
    print(metrics.pivot(index="model", columns="split", values="mae").round(3).to_string())
    print(f"\nSelected by chronological validation: {selected}")
    print("The historical test was previously inspected; results are exploratory, NOT independent confirmation.")
    print("Saved results, predictions, model, and 3 charts to phase2_outputs/.")
    print("All Phase 1 / V1 / V2 files and source datasets remain unchanged.")


if __name__ == "__main__":
    main()
