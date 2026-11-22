"""Reproducible benchmark for actual 24-hour soil moisture forecasting.

Run from the AgroFlow project root:
    python agroflow_train.py

The five source CSV files must remain in 02_processed_data/preprocessed/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "02_processed_data" / "preprocessed"
OUTPUT_DIR = ROOT / "agroflow_artifacts"
HORIZON = pd.Timedelta(hours=24)
HORIZON_ROWS = 144  # Original files use 10-minute readings.

# Only present/observed information is allowed here. NO future water volume,
# future moisture delta, or forward-looking target-derived columns.
FEATURES = [
    "soil_moisture",
    "weather_temp",
    "weather_humidity",
    "weather_rain",
    "weather_pressure",
    "weather_wind_speed",
    "weather_radiation",
    "soil_temperature_0-7cm",
    "soil_temperature_7-18cm",
    "irrigation_duration_minutes",
    "zone",
    "month",
    "hour",
]
TARGET = "target_point_24h"


def load_verified_samples() -> tuple[pd.DataFrame, list[dict]]:
    """Keep only t where the 144th next sample occurs at t + 24h."""
    frames: list[pd.DataFrame] = []
    audit: list[dict] = []

    for zone in range(1, 6):
        csv_file = DATA_DIR / f"dataset_zone_{zone}_preprocessed.csv"
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_file}\n"
                "Keep the source's 02_processed_data folder in the project root."
            )
        source = pd.read_csv(csv_file, parse_dates=["ts"])
        required = set(FEATURES).difference({"zone"}) | {"ts", TARGET}
        missing_columns = required.difference(source.columns)
        if missing_columns:
            raise ValueError(f"Zone {zone} is missing: {sorted(missing_columns)}")

        source = source.sort_values("ts").reset_index(drop=True)
        if source["ts"].duplicated().any():
            raise ValueError(f"Zone {zone} contains duplicate timestamps")

        exact_horizon = (source["ts"].shift(-HORIZON_ROWS) - source["ts"]) == HORIZON
        verified = source.loc[exact_horizon].copy()
        verified["zone"] = zone
        verified = verified.dropna(subset=FEATURES + [TARGET])

        # Detect unintended target misalignment before model training.
        if not verified.empty:
            true_tplus24 = source["soil_moisture"].shift(-HORIZON_ROWS)
            if not np.allclose(
                verified[TARGET].to_numpy(),
                true_tplus24.loc[verified.index].to_numpy(),
                rtol=0,
                atol=1e-6,
            ):
                raise ValueError(f"Zone {zone}: published target does not match t+24h")

        audit.append({
            "zone": zone,
            "source_rows": int(len(source)),
            "verified_24h_rows": int(len(verified)),
            "excluded_rows": int(len(source) - len(verified)),
            "first_valid_timestamp": str(verified["ts"].min()) if len(verified) else None,
            "last_valid_timestamp": str(verified["ts"].max()) if len(verified) else None,
        })
        if not verified.empty:
            frames.append(verified)

    if not frames:
        raise ValueError("No verified 24-hour samples found")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ts", "zone"]).reset_index(drop=True)
    return combined, audit


def chronological_split(df: pd.DataFrame):
    """Global-time split; remove labels crossing train/validation boundaries."""
    train_cut = df["ts"].quantile(0.70)
    validation_cut = df["ts"].quantile(0.85)
    outcome_time = df["ts"] + HORIZON

    train = df.loc[outcome_time < train_cut].copy()
    validation = df.loc[(df["ts"] >= train_cut) & (outcome_time < validation_cut)].copy()
    test = df.loc[df["ts"] >= validation_cut].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split returned an empty subset")
    return train, validation, test, train_cut, validation_cut


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 5),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 5),
        "r2": round(float(r2_score(actual, predicted)), 5),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df, audit = load_verified_samples()
    train, validation, test, train_cut, validation_cut = chronological_split(df)

    # Residual forecasting: learn how moisture changes relative to its present value.
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=10,
        l2_regularization=8,
        learning_rate=0.05,
        random_state=42,
    )
    delta = train[TARGET] - train["soil_moisture"]
    model.fit(train[FEATURES], delta)

    result = {}
    for split_name, split in [("train", train), ("validation", validation), ("test", test)]:
        baseline = split["soil_moisture"].to_numpy(dtype=float)
        ml = np.clip(baseline + model.predict(split[FEATURES]), 0, 100)
        truth = split[TARGET].to_numpy(dtype=float)
        result[split_name] = {
            "n": int(len(split)),
            "time_start": str(split["ts"].min()),
            "time_end": str(split["ts"].max()),
            "persistence": score(truth, baseline),
            "experimental_ml": score(truth, ml),
        }
        if split_name == "test":
            forecast = pd.DataFrame({
                "ts": split["ts"].to_numpy(),
                "zone": split["zone"].to_numpy(),
                "actual_24h_moisture": truth,
                "current_moisture": baseline,
                "persistence_prediction": baseline,
                "experimental_ml_prediction": ml,
            })
            forecast.to_csv(OUTPUT_DIR / "test_forecasts.csv", index=False)

            result["test"]["by_zone"] = {}
            for zone, part in forecast.groupby("zone"):
                result["test"]["by_zone"][str(zone)] = {
                    "n": int(len(part)),
                    "persistence_mae": score(part["actual_24h_moisture"], part["persistence_prediction"])["mae"],
                    "experimental_ml_mae": score(part["actual_24h_moisture"], part["experimental_ml_prediction"])["mae"],
                }

    summary = {
        "project": "AgroFlow AI - 24h Soil Moisture Forecast Lab",
        "objective": "Observed soil moisture exactly 24 hours later",
        "source": "https://data.mendeley.com/datasets/c837v6p8ph/2",
        "verified_windows": int(len(df)),
        "source_rows": int(sum(x["source_rows"] for x in audit)),
        "audit": audit,
        "cutoffs": {
            "train_to_validation": str(train_cut),
            "validation_to_test": str(validation_cut),
        },
        "features": FEATURES,
        "target": TARGET,
        "evaluation": result,
        "limitations": [
            "The original forward target counts rows, not necessarily elapsed hours.",
            "Only continuous 144-step windows whose actual span is 24h are included.",
            "Adjacent 10-minute windows are strongly correlated, not independent samples.",
            "Zone 3 has no qualifying 24-hour windows.",
            "The future irrigation schedule is not provided to the forecasting model.",
            "Future water volume and future moisture delta are excluded to prevent leakage.",
            "Original pH/EC readings and sector-5-derived water volumes have quality/calibration caveats and are excluded.",
            "Forecast errors can vary greatly between dates and zones; results are not field-validated.",
            "The test period was inspected during exploratory development, so it should not be presented as a pristine untouched holdout.",
        ],
    }

    joblib.dump({"model": model, "features": FEATURES}, OUTPUT_DIR / "residual_model.joblib")
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAGROFLOW AI - REPRODUCIBLE FORECAST BENCHMARK")
    print("=" * 52)
    print(f"Source records: {summary['source_rows']:,}")
    print(f"Verified 24-hour windows: {len(df):,}")
    for a in audit:
        print(f"Zone {a['zone']}: {a['verified_24h_rows']:,} / {a['source_rows']:,} verified")
    print(f"Time cutoffs: {train_cut} | {validation_cut}")
    for split_name in ("train", "validation", "test"):
        e = result[split_name]
        print(
            f"{split_name:<11} n={e['n']:<6} "
            f"Persistence MAE={e['persistence']['mae']:.3f}  "
            f"ML MAE={e['experimental_ml']['mae']:.3f}"
        )
    print("\nLower MAE is better. ML is exploratory and may be worse than persistence.")
    print("Saved evaluation and forecasts to agroflow_artifacts/.")


if __name__ == "__main__":
    main()
