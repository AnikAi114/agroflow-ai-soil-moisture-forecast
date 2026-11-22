"""AgroFlow AI V2: validation-calibrated, regularized 24-hour forecast.

Place beside existing agroflow_train.py in the project root and run:
    python agroflow_train_v2.py

The V1 files are NOT changed. Dataset files remain in 02_processed_data/preprocessed.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Reuse the already-audited exact-24h target verification and the chronological
# split from the original project. This avoids reimplementing critical checks.
from agroflow_train import (
    DATA_DIR,
    FEATURES,
    HORIZON,
    OUTPUT_DIR,
    TARGET,
    chronological_split,
    load_verified_samples,
    score,
)

OLD_PARAMS = {
    "max_iter": 200,
    "max_leaf_nodes": 10,
    "l2_regularization": 8,
    "learning_rate": 0.05,
    "random_state": 42,
}

# Stronger regularization to reduce fitting to large, rare moisture changes.
V2_PARAMS = {
    "max_iter": 100,
    "max_leaf_nodes": 6,
    "min_samples_leaf": 100,
    "l2_regularization": 200,
    "learning_rate": 0.02,
    "random_state": 42,
}

BLEND_WEIGHTS = np.round(np.arange(0.0, 1.001, 0.1), 2)


def predict(model, data: pd.DataFrame, weight: float) -> np.ndarray:
    """Persistence + weight * predicted 24h moisture change."""
    current = data["soil_moisture"].to_numpy(dtype=float)
    delta = model.predict(data[FEATURES])
    return np.clip(current + weight * delta, 0, 100)


def evaluate(data: pd.DataFrame, old_model, new_model, weight: float) -> dict:
    actual = data[TARGET].to_numpy(dtype=float)
    persistence = data["soil_moisture"].to_numpy(dtype=float)
    return {
        "n": int(len(data)),
        "first_timestamp": str(data["ts"].min()),
        "last_timestamp": str(data["ts"].max()),
        "persistence": score(actual, persistence),
        "original_ml": score(actual, predict(old_model, data, 1.0)),
        "regularized_ml": score(actual, predict(new_model, data, weight)),
    }


def main() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            "Data directory missing: " + str(DATA_DIR) + "\n"
            "Place this script next to agroflow_train.py in the main project folder."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, audit = load_verified_samples()
    train, validation, test, train_cut, val_cut = chronological_split(df)
    y_train_delta = train[TARGET] - train["soil_moisture"]

    old_model = HistGradientBoostingRegressor(**OLD_PARAMS)
    old_model.fit(train[FEATURES], y_train_delta)

    v2_model = HistGradientBoostingRegressor(**V2_PARAMS)
    v2_model.fit(train[FEATURES], y_train_delta)

    # Select ONLY using validation. Do not search weights on the test set.
    tuning = []
    for weight in BLEND_WEIGHTS:
        predictions = predict(v2_model, validation, float(weight))
        metrics = score(validation[TARGET], predictions)
        tuning.append({"weight": float(weight), "validation_mae": metrics["mae"]})
    best_weight = min(tuning, key=lambda row: (row["validation_mae"], row["weight"]))["weight"]

    # Keep the training-only model so the validation metrics remain comparable.
    # An optional train+validation refit would change this evaluation and must
    # be reported as a separate experiment instead of silently replacing it.
    results = {
        name: evaluate(subset, old_model, v2_model, best_weight)
        for name, subset in [("train", train), ("validation", validation), ("test", test)]
    }

    truth = test[TARGET].to_numpy(dtype=float)
    current = test["soil_moisture"].to_numpy(dtype=float)
    forecasts = pd.DataFrame({
        "ts": test["ts"].to_numpy(),
        "zone": test["zone"].to_numpy(),
        "actual_24h_moisture": truth,
        "persistence_prediction": current,
        "original_ml_prediction": predict(old_model, test, 1.0),
        "regularized_ml_prediction": predict(v2_model, test, best_weight),
    })

    by_zone = {}
    for zone, subset in forecasts.groupby("zone"):
        actual = subset["actual_24h_moisture"]
        by_zone[str(int(zone))] = {
            "n": int(len(subset)),
            "persistence": score(actual, subset["persistence_prediction"]),
            "original_ml": score(actual, subset["original_ml_prediction"]),
            "regularized_ml": score(actual, subset["regularized_ml_prediction"]),
        }

    report = {
        "project": "AgroFlow AI - V2 regularized forecast research benchmark",
        "dataset_source": "https://data.mendeley.com/datasets/c837v6p8ph/2",
        "goal": "Forecast observed soil moisture at exactly t + 24 hours",
        "source_rows": int(sum(row["source_rows"] for row in audit)),
        "verified_windows": int(len(df)),
        "audit": audit,
        "time_cutoffs": {
            "train_validation": str(train_cut),
            "validation_test": str(val_cut),
        },
        "features": FEATURES,
        "target": TARGET,
        "original_params": OLD_PARAMS,
        "v2_params": V2_PARAMS,
        "best_blend_weight": best_weight,
        "weight_selection": "Minimum validation MAE over 0.0, 0.1, ..., 1.0; zero means pure persistence.",
        "validation_tuning": tuning,
        "evaluation": results,
        "test_by_zone": by_zone,
        "limitations": [
            "This is retrospective research, not a deployable irrigation controller.",
            "The earlier test period was inspected in V1 and during exploratory development. V2 test figures are NOT independent confirmation.",
            "Adjacent 10-minute forecasting windows overlap heavily.",
            "Published targets are 144 rows ahead; only rows with exactly 24 elapsed hours are retained and the target is cross-checked.",
            "Zone 3 has no valid exact-24h windows and is excluded; later test dates include only zones 2, 4 and 5.",
            "No future water volume, future rainfall, future irrigation plan or future-target-derived features are used.",
            "The observed current 10-minute irrigation duration is historical, not a prediction of future watering.",
            "Sensor quality and missingness vary by zone and time; no independent external-farm data were tested.",
            "Future ML gains are not guaranteed; compare against persistence before interpreting forecasts.",
        ],
    }

    joblib.dump(
        {
            "original_model": old_model,
            "regularized_model": v2_model,
            "features": FEATURES,
            "blend_weight": best_weight,
        },
        OUTPUT_DIR / "models_v2.joblib",
    )
    forecasts.to_csv(OUTPUT_DIR / "test_forecasts_v2.csv", index=False)
    (OUTPUT_DIR / "evaluation_v2.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nAGROFLOW AI - V2 MODEL IMPROVEMENT")
    print("=" * 57)
    print(f"Source records: {report['source_rows']:,} | Verified 24h windows: {len(df):,}")
    print(f"Train: {len(train):,} | Validation: {len(validation):,} | Test: {len(test):,}")
    print(f"Selected V2 weight (validation only): {best_weight:.1f}")
    print("\nSplit        Persistence MAE     Original ML MAE     V2 ML MAE")
    for split_name in ("train", "validation", "test"):
        row = results[split_name]
        print(
            f"{split_name:<13}"
            f"{row['persistence']['mae']:<20.4f}"
            f"{row['original_ml']['mae']:<20.4f}"
            f"{row['regularized_ml']['mae']:.4f}"
        )
    test_vals = results["test"]
    if test_vals["regularized_ml"]["mae"] >= test_vals["persistence"]["mae"]:
        print("\nNote: The simple persistence baseline still wins on this historical test.")
    else:
        print("\nNote: V2 wins on this historical test, but it is NOT an independent confirmation.")
    print("Test results were explored previously; do not claim an untouched holdout.")
    print("Saved V2-only files to agroflow_artifacts/; V1 files were preserved.")


if __name__ == "__main__":
    main()
