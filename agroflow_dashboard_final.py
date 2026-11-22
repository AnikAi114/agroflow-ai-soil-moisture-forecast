"""AgroFlow AI | Phase 4 research dashboard.

This app presents completed Phase 1-3 results alongside the existing V2
interactive demonstration. It does not retrain models or operate hardware.
Run: python -m streamlit run agroflow_dashboard_final.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
P1 = ROOT / "phase1_outputs"
P2 = ROOT / "phase2_outputs"
P3 = ROOT / "phase3_outputs"
ART = ROOT / "agroflow_artifacts"
SOURCE = "https://data.mendeley.com/datasets/c837v6p8ph/2"

st.set_page_config(
    page_title="AgroFlow AI | Soil Moisture Research",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {max-width: 1380px; padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(115deg, #153b31, #1a5f4d 58%, #173c3b);
       padding: 2rem 2.2rem; border-radius: 18px; color: #fff;
       margin-bottom: 1.2rem; border: 1px solid #3e786b;}
.hero h1 {font-size: clamp(2rem, 4vw, 3.2rem); margin: 0 0 .35rem; color: #fff;}
.hero p {font-size: 1rem; opacity: .87; margin: 0; color: #f3f9f5;}
.eyebrow {text-transform: uppercase; letter-spacing: .17em; font-weight: 700;
          font-size: .72rem; color: #a7e1ca; margin-bottom: .7rem;}
.note {background: rgba(40,130,105,.09); border-left: 3px solid #49a587;
       padding: .8rem 1rem; border-radius: .3rem; margin: 1rem 0;}
[data-testid="stMetric"] {border: 1px solid rgba(120,140,130,.25);
    border-radius: 11px; padding: .85rem 1rem; background: rgba(105,135,115,.035);}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def json_file(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

@st.cache_resource(show_spinner=False)
def load_v2_models(path: str) -> dict:
    # Load only artifacts that YOU generated locally, never untrusted joblib files.
    return joblib.load(path)


def chart(path: Path, label: str) -> None:
    if path.is_file():
        st.image(str(path), width="stretch")
        st.caption(label)
    else:
        st.info(f"Chart not generated yet: {path.name}. Run the corresponding phase script.")


def header(label: str, description: str | None = None) -> None:
    st.subheader(label)
    if description:
        st.caption(description)


def safe_float(frame: pd.DataFrame, split: str, model: str, metric: str = "mae") -> float | None:
    found = frame.loc[(frame["split"] == split) & (frame["model"] == model)]
    return float(found.iloc[0][metric]) if not found.empty else None


st.markdown(
    '<div class="hero"><div class="eyebrow">Precision agriculture · Research prototype</div>'
    '<h1>🌱 AgroFlow AI</h1>'
    '<p>24-hour soil moisture forecasting · Five agricultural zones · '
    'Interpretable model comparison</p></div>',
    unsafe_allow_html=True,
)

st.caption(
    "Historical agricultural data from Italy (2025)  |  "
    "Exploratory research only  |  NOT for automatic irrigation control"
)

p1_ok = (P1 / "zone_quality_summary.csv").exists()
p2_ok = (P2 / "model_comparison.csv").exists()
p3_ok = (P3 / "shap_global_importance.csv").exists()
v2_ok = (ART / "models_v2.joblib").exists()

if not all((p1_ok, p2_ok, p3_ok)):
    st.warning(
        "Some research outputs are missing. Run Phase 1, Phase 2, and Phase 3 "
        "scripts from your project root to enable all dashboard sections."
    )

zones = csv_file(str(P1 / "zone_quality_summary.csv")) if p1_ok else pd.DataFrame()
comparison = csv_file(str(P2 / "model_comparison.csv")) if p2_ok else pd.DataFrame()

ov, eda, eval_tab, explain, predict, methods = st.tabs([
    "Overview", "Data exploration", "Model evidence", "Explainability",
    "Forecast lab", "Methods & limitations",
])

with ov:
    header("Research at a glance", "Data coverage, forecast validity, and benchmark evidence.")
    a, b, c, d = st.columns(4)
    a.metric("Processed observations", "47,068")
    b.metric("Exact 24h windows", "17,614")
    c.metric("Agricultural zones", "5")
    d.metric("Trainable zones", "4 / 5")

    st.markdown(
        '<div class="note"><strong>Main finding.</strong> A simple persistence forecast '
        'is stronger on the previously inspected historical test segment than all '
        'Phase 2 unshrunk ML candidates. Model rankings change across periods.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        chart(P1 / "01_zone_coverage.png", "Phase 1 · All five zones; Zone 3 has no exact-24h windows.")
    with right:
        chart(P2 / "01_model_comparison.png", "Phase 2 · Model error across time segments (lower is better).")
    if p2_ok:
        baseline = safe_float(comparison, "historical_test", "Persistence")
        gradient = safe_float(comparison, "historical_test", "Regularized Gradient Boosting (unshrunk)")
        x, y = st.columns(2)
        x.metric("Historical test · Persistence MAE", f"{baseline:.3f} pp" if baseline is not None else "–")
        y.metric("Historical test · Phase 2 GB MAE", f"{gradient:.3f} pp" if gradient is not None else "–")
    st.info(
        "The previous historical test was viewed during earlier project development. "
        "Do not call it a pristine holdout or claim external generalization."
    )

with eda:
    header("Explore the sensor dataset", "Published preprocessed measurements and a verified 24-clock-hour target audit.")
    if p1_ok:
        view = zones[["zone", "records", "verified_exact_24h_windows", "verified_24h_percent",
                      "moisture_median", "gaps_over_10min", "irrigation_active_intervals"]].copy()
        view.columns = ["Zone", "Rows", "Valid 24h windows", "Valid (%)",
                        "Median moisture (%)", "Gaps >10min", "Valve-active intervals"]
        st.dataframe(view, hide_index=True, width="stretch")
        st.caption("Valve-active intervals are not the same as irrigation events.")
        l, r = st.columns(2)
        with l:
            chart(P1 / "02_soil_moisture_distributions.png", "Different zones have different moisture distributions.")
            chart(P1 / "04_data_quality.png", "Timestamp gaps and illustrative pH/EC screening flags.")
        with r:
            chart(P1 / "03_monthly_moisture_trends.png", "Monthly medians of available processed readings.")
            chart(P1 / "06_verified_windows_by_month.png", "Valid exact-24h examples by month.")
        with st.expander("Historical irrigation activity example"):
            chart(P1 / "05_zone4_activity_window.png", "One example period from Zone 4, not causal evidence.")
        st.download_button(
            "Download zone coverage CSV",
            zones.to_csv(index=False).encode("utf-8"),
            file_name="agroflow_zone_quality_summary.csv", mime="text/csv",
        )

with eval_tab:
    header("Model evidence", "Metrics for actual 24-hour soil moisture, expressed in percentage points.")
    if p2_ok:
        choices = ["Persistence", "Ridge (unshrunk)", "Random Forest (unshrunk)",
                   "Regularized Gradient Boosting (unshrunk)"]
        selected = comparison.loc[comparison["model"].isin(choices),
                                  ["model", "split", "n", "mae", "rmse"]].copy()
        pivot = selected.pivot(index="model", columns="split", values="mae")
        ordering = [c for c in ("train", "validation", "historical_test") if c in pivot]
        pivot = pivot.reindex(choices)[ordering]
        st.dataframe(pivot.rename(columns={"historical_test": "Historical test"}).style.format("{:.3f}"),
                     width="stretch")
        st.caption("Lower MAE is better. Historical test results were inspected previously and are exploratory.")
        st.bar_chart(pivot[[c for c in ("validation", "historical_test") if c in pivot]],
                     y_label="MAE (percentage points)")
        st.warning(
            "Train-only walk-forward selection assigned a 0.00 correction weight to Ridge, "
            "Random Forest and Gradient Boosting. Their CV-blended forecasts are therefore "
            "exactly persistence—not independent machine-learning gains."
        )
        l, r = st.columns(2)
        with l:
            chart(P2 / "02_zone_error_comparison.png", "Historical error differs by zone.")
        with r:
            chart(P2 / "03_change_regime_errors.png", "Error grouped by actual FUTURE change (post-hoc diagnostic).")
        forecast_csv = P2 / "historical_test_predictions.csv"
        if forecast_csv.is_file():
            st.markdown("#### Example: overlapping historical forecast windows")
            forecast = csv_file(str(forecast_csv)).copy()
            forecast["ts"] = pd.to_datetime(forecast["ts"])
            selected_zone = st.selectbox("Agricultural zone for chart", sorted(forecast["zone"].dropna().unique().tolist()))
            subset = forecast[forecast["zone"] == selected_zone].sort_values("ts").head(144)
            st.line_chart(subset.set_index("ts")[[
                "actual_24h_moisture", "persistence_prediction",
                "regularized_gradient_boosting_unshrunk_prediction",
            ]])
            st.caption("The first 144 overlapping windows in this zone. This is not a live forecast timeline.")
        st.download_button(
            "Download full model comparison",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="agroflow_model_comparison.csv", mime="text/csv",
        )

with explain:
    header("Explainability & model behavior", "Attributions explain model predictions, NOT agronomic causality.")
    if p3_ok:
        importance = csv_file(str(P3 / "shap_global_importance.csv")).sort_values(
            "mean_abs_shap_on_predicted_change_pp", ascending=False
        )
        top = importance.head(8)
        st.markdown("#### Global SHAP importance — predicted 24h change")
        st.bar_chart(top.set_index("feature_group")[["mean_abs_shap_on_predicted_change_pp"]],
                     horizontal=True, x_label="Mean |SHAP| (percentage points)")
        st.caption("Agricultural zone and current soil moisture dominate the explored model's change prediction.")
        l, r = st.columns(2)
        with l:
            chart(P3 / "02_shap_global_importance.png", "Global tree SHAP, 300 sampled validation rows.")
        with r:
            chart(P3 / "01_permutation_sensitivity.png", "Validation sensitivity; correlated inputs affect interpretation.")
        with st.expander("One local SHAP explanation"):
            chart(P3 / "03_shap_local_example.png", "Contribution to learned change for one historical validation example.")
            one = P3 / "shap_local_example.json"
            if one.exists():
                e = json_file(str(one))
                st.json(e, expanded=False)
        st.info(
            "All Phase 2 CV-blended weights were zero. SHAP therefore explains the "
            "UNSHRUNK Gradient Boosting model, not the persistence-equivalent blends."
        )
        st.download_button("Download global SHAP CSV", importance.to_csv(index=False).encode("utf-8"),
                           file_name="agroflow_shap_importance.csv", mime="text/csv")

with predict:
    header("Interactive 24-hour forecast laboratory", "Compare existing V1/V2 models and persistence for a hypothetical input.")
    st.warning("Educational demonstration only. Not a pump command or irrigation recommendation.")
    if not v2_ok:
        st.error("Missing V2 model: run `python agroflow_train_v2.py` to enable this section.")
    else:
        models = load_v2_models(str(ART / "models_v2.joblib"))
        with st.form("prediction_inputs"):
            a, b, c = st.columns(3)
            with a:
                zone = st.selectbox("Agricultural zone", [1, 2, 4, 5], index=1)
                soil_moisture = st.slider("Current soil moisture (%)", 0.0, 100.0, 24.0, 0.5)
                weather_temp = st.number_input("Air temperature (°C)", -10.0, 60.0, 25.0, step=0.5)
                weather_humidity = st.slider("Air humidity (% RH)", 0.0, 100.0, 60.0)
                weather_rain = st.number_input("Observed rain (last interval, mm)", 0.0, 100.0, 0.0, step=0.1)
            with b:
                weather_pressure = st.number_input("Surface pressure (hPa)", 900.0, 1100.0, 1012.0, step=0.5)
                weather_wind_speed = st.number_input("Wind speed (km/h)", 0.0, 120.0, 8.0, step=0.5)
                weather_radiation = st.number_input("Solar radiation (W/m²)", 0.0, 1400.0, 250.0, step=10.0)
                irrigation_duration_minutes = st.slider("Valve-open minutes (current 10 min)", 0.0, 10.0, 0.0, 0.5)
            with c:
                soil_temp_0 = st.number_input("Soil temperature 0–7cm (°C)", -10.0, 60.0, 24.0, step=0.5)
                soil_temp_7 = st.number_input("Soil temperature 7–18cm (°C)", -10.0, 60.0, 22.0, step=0.5)
                month = st.selectbox("Observation month", list(range(1, 13)), index=7)
                hour = st.slider("Observation hour", 0, 23, 12)
            pressed = st.form_submit_button("Compare three forecast methods", type="primary")
        if pressed:
            row = pd.DataFrame([{
                "soil_moisture": soil_moisture,
                "weather_temp": weather_temp,
                "weather_humidity": weather_humidity,
                "weather_rain": weather_rain,
                "weather_pressure": weather_pressure,
                "weather_wind_speed": weather_wind_speed,
                "weather_radiation": weather_radiation,
                "soil_temperature_0-7cm": soil_temp_0,
                "soil_temperature_7-18cm": soil_temp_7,
                "irrigation_duration_minutes": irrigation_duration_minutes,
                "zone": zone,
                "month": month,
                "hour": hour,
            }])[models["features"]]
            old_delta = float(models["original_model"].predict(row)[0])
            new_delta = float(models["regularized_model"].predict(row)[0])
            w = float(models["blend_weight"])
            v1 = float(np.clip(soil_moisture + old_delta, 0, 100))
            v2 = float(np.clip(soil_moisture + w * new_delta, 0, 100))
            left, center, right = st.columns(3)
            left.metric("Persistence (%)", f"{soil_moisture:.1f}")
            center.metric("Original ML V1 (%)", f"{v1:.1f}", f"{v1-soil_moisture:+.1f} pp")
            right.metric("Regularized V2 (%)", f"{v2:.1f}", f"{v2-soil_moisture:+.1f} pp")
            st.info(
                f"V2 blends {w:.0%} of the regularized model's predicted change with persistence. "
                "The inputs include observed current-interval rain and valve duration, not "
                "unknown future irrigation or weather. These are illustrative predictions."
            )
        st.caption("Only zones 1, 2, 4 and 5 have eligible exact-24h training windows. Zone 3 is excluded.")

with methods:
    header("Research protocol & limitations", "What the project can and cannot establish.")
    st.markdown("""
**Study design**

- **Source:** historical five-zone Italian agricultural sensor/weather dataset.
- **Target:** soil moisture 24 *clock-hours* after an observation, not merely 144 rows later.
- **Audit:** only keep verified targets with exactly 144 consecutive 10-minute readings.
- **Features:** current/past soil moisture, historical weather, soil temperature, current irrigation minutes,
  calendar markers and zone identity. Future watering and target-derived features are excluded.
- **Splits:** chronological, with a 24-hour label-boundary purge in Phase 2 train-only
  walk-forward folds. A previous test period had already been inspected before Phase 2.
- **Benchmarks:** persistence, Ridge, Random Forest and regularized Gradient Boosting;
  V1/V2 remain separate earlier experiments.
- **Explanations:** permutation sensitivity and tree SHAP for fitted, *unshrunk* ML models.

**Scientific limitations**

1. Observation windows overlap and are **not** independent farm trials.
2. Soil moisture is preprocessed, despiked and sometimes interpolated.
3. Weather inputs include historical reanalysis; live operational performance is unknown.
4. Data gaps remove most rows from exact 24-hour forecasting; Zone 3 has no valid windows.
5. Future rainfall and irrigation events cannot be known from these historical instantaneous inputs.
6. The earlier historical test was inspected and **is not an untouched holdout**.
7. Model attributions identify learned associations, not cause-and-effect or crop-water demand.
8. No independent new farm/season validation or safe actuation trial has been conducted.

**Appropriate portfolio claim:** An end-to-end reproducible research prototype for
sensor-data auditing, honest baseline benchmarking and interpretability—not a deployed
irrigation controller.
""")
    st.markdown(f"**Dataset source:** [Mendeley Data — Soil Moisture, Irrigation Actuator and Weather Dataset]({SOURCE})")
    st.markdown("**Reproduce locally (run in project root):**")
    st.code(
        "python agroflow_phase1_eda.py\n"
        "python agroflow_phase2_benchmark.py\n"
        "python agroflow_phase3_explain.py\n"
        "python agroflow_train_v2.py\n"
        "python -m streamlit run agroflow_dashboard_final.py",
        language="powershell",
    )
    for folder, file in [(P1, "PHASE1_FINDINGS.md"), (P2, "PHASE2_FINDINGS.md"), (P3, "PHASE3_FINDINGS.md")]:
        path = folder / file
        if path.exists():
            with st.expander(file.replace("_", " ")):
                st.markdown(path.read_text(encoding="utf-8"))
