"""AgroFlow AI - Phase 1: Exploratory Data Analysis and Data Quality Audit.

Run from a project with 02_processed_data/preprocessed/ in the project root:

    python agroflow_phase1_eda.py

Uses only the five provided preprocessed sector CSV files. This script does NOT
modify datasets, train models, or change the existing V1 / V2 project files.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "02_processed_data" / "preprocessed"
OUTPUT_DIR = ROOT / "phase1_outputs"
HORIZON_ROWS = 144
TIME_STEP = pd.Timedelta(minutes=10)
HORIZON = pd.Timedelta(hours=24)
SOURCE_URL = "https://data.mendeley.com/datasets/c837v6p8ph/2"

REQUIRED_COLUMNS = {
    "ts", "soil_moisture", "weather_temp", "weather_humidity", "weather_rain",
    "irrigation_duration_minutes", "ph", "ec", "target_point_24h",
}


def read_zone(zone: int) -> pd.DataFrame:
    csv_path = DATA_DIR / f"dataset_zone_{zone}_preprocessed.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing data file: {csv_path}\n"
            "Keep the original 02_processed_data/preprocessed folder in the project root."
        )
    data = pd.read_csv(csv_path, parse_dates=["ts"])
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Zone {zone}: missing columns: {sorted(missing)}")
    if data["ts"].isna().any():
        raise ValueError(f"Zone {zone}: invalid timestamps found.")
    data = data.sort_values("ts").reset_index(drop=True)
    if data["ts"].duplicated().any():
        raise ValueError(f"Zone {zone}: duplicate timestamps found.")
    data["zone"] = zone
    return data


def summarize_zone(zone: int, data: pd.DataFrame) -> dict:
    diffs = data["ts"].diff().iloc[1:]
    valid_horizon = (
        data["ts"].shift(-HORIZON_ROWS) - data["ts"] == HORIZON
    )
    expected_targets = data["soil_moisture"].shift(-HORIZON_ROWS)
    values_match = np.isclose(
        data["target_point_24h"].to_numpy(dtype=float),
        expected_targets.to_numpy(dtype=float),
        rtol=0.0, atol=1e-6, equal_nan=False,
    )
    verified = valid_horizon.to_numpy() & values_match
    mismatches = int((valid_horizon.to_numpy() & ~values_match).sum())
    if mismatches:
        raise ValueError(f"Zone {zone}: {mismatches} exact-24h target mismatches")

    moisture = data["soil_moisture"]
    ph = data["ph"]
    ec = data["ec"]
    intervals = max(len(data) - 1, 1)
    return {
        "zone": zone,
        "records": len(data),
        "first_timestamp": str(data["ts"].min()),
        "last_timestamp": str(data["ts"].max()),
        "regular_10min_intervals": int(diffs.eq(TIME_STEP).sum()),
        "gaps_over_10min": int(diffs.gt(TIME_STEP).sum()),
        "short_intervals_below_10min": int(diffs.lt(TIME_STEP).sum()),
        "regular_interval_percent": round(float(100 * diffs.eq(TIME_STEP).sum() / intervals), 2),
        "verified_exact_24h_windows": int(verified.sum()),
        "verified_24h_percent": round(float(100 * verified.sum() / len(data)), 2),
        "moisture_mean": round(float(moisture.mean()), 2),
        "moisture_median": round(float(moisture.median()), 2),
        "moisture_min": round(float(moisture.min()), 2),
        "moisture_max": round(float(moisture.max()), 2),
        "moisture_at_0_percent": round(float(100 * moisture.eq(0).mean()), 2),
        "moisture_at_100_percent": round(float(100 * moisture.eq(100).mean()), 2),
        "irrigation_active_intervals": int(data["irrigation_duration_minutes"].gt(0).sum()),
        "observed_valve_open_minutes": round(float(data["irrigation_duration_minutes"].sum()), 1),
        "rain_positive_intervals": int(data["weather_rain"].gt(0).sum()),
        "ph_outside_3_to_9": int((ph.notna() & ~ph.between(3, 9)).sum()),
        "ec_outside_0_to_1000": int((ec.notna() & ~ec.between(0, 1000)).sum()),
        "ph_missing": int(ph.isna().sum()),
        "ec_missing": int(ec.isna().sum()),
    }


def save_chart(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_coverage(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    y = np.arange(len(summary))
    ax.barh(y + 0.19, summary["records"], height=0.36, label="Source rows")
    ax.barh(y - 0.19, summary["verified_exact_24h_windows"], height=0.36,
            label="Verified exact-24h windows")
    ax.set_yticks(y, [f"Zone {z}" for z in summary["zone"]])
    ax.invert_yaxis()
    ax.set_xlabel("Records")
    ax.set_title("Observation coverage versus valid 24-hour forecasting windows", pad=18)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for i, row in enumerate(summary.itertuples(index=False)):
        ax.annotate(f"{row.verified_24h_percent:.1f}% valid",
                    (row.verified_exact_24h_windows, i - 0.19),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    save_chart(fig, "01_zone_coverage.png")


def chart_moisture_distribution(frames: dict[int, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6))
    samples = [frames[zone]["soil_moisture"].dropna().to_numpy() for zone in range(1, 6)]
    ax.boxplot(samples, tick_labels=[f"Zone {z}" for z in range(1, 6)],
               showfliers=False, patch_artist=False)
    ax.set_title("Processed soil-moisture distributions across five zones", pad=18)
    ax.set_ylabel("Soil moisture (relative probe %, 0-100)")
    ax.set_ylim(-4, 104)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.1, 0.015, "Boxes: interquartile range and median. Outliers hidden for legibility; values are preprocessed.",
             fontsize=9)
    save_chart(fig, "02_soil_moisture_distributions.png")


def chart_monthly_trend(frames: dict[int, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for zone, data in frames.items():
        grouped = data.set_index("ts")["soil_moisture"].resample("MS").median()
        ax.plot(grouped.index, grouped.values, marker="o", markersize=5,
                linewidth=1.8, label=f"Zone {zone}")
    ax.set_title("Monthly median soil moisture (available processed readings only)", pad=18)
    ax.set_ylabel("Median soil moisture (%)")
    ax.set_xlabel("Month in 2025")
    ax.set_ylim(-3, 103)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(alpha=0.22)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.autofmt_xdate()
    save_chart(fig, "03_monthly_moisture_trends.png")


def chart_quality(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(5)
    regular = summary["regular_interval_percent"].to_numpy(dtype=float)
    axes[0].bar(x, regular)
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x, [f"Zone {z}" for z in summary["zone"]])
    axes[0].set_ylabel("Percent of consecutive intervals")
    axes[0].set_title("Consecutive readings exactly 10 minutes apart")
    for i, val in enumerate(regular):
        axes[0].text(i, val + 2, f"{val:.1f}%", ha="center", fontsize=9)
    ph_share = summary["ph_outside_3_to_9"] / summary["records"] * 100
    ec_share = summary["ec_outside_0_to_1000"] / summary["records"] * 100
    axes[1].bar(x - 0.18, ph_share, width=0.36, label="pH outside 3-9")
    axes[1].bar(x + 0.18, ec_share, width=0.36, label="EC outside 0-1000")
    axes[1].set_xticks(x, [f"Zone {z}" for z in summary["zone"]])
    axes[1].set_ylabel("Share of rows (%)")
    axes[1].set_title("Flagged sensor values (screening ranges)")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle("Data-quality checks on published preprocessed records", fontsize=15, y=1.03)
    fig.tight_layout()
    save_chart(fig, "04_data_quality.png")


def chart_example_activity(frames: dict[int, pd.DataFrame]) -> tuple[int, str, str]:
    # Pick a readable real-data sample: a 7-day window in Zone 4 with at least
    # 300 readings and the most recorded valve-open minutes. Not a causal study.
    zone = 4
    data = frames[zone].set_index("ts").sort_index()
    daily = data.resample("D").agg(
        count=("soil_moisture", "count"),
        valve_minutes=("irrigation_duration_minutes", "sum"),
        rainfall_mm=("weather_rain", "sum"),
    )
    rolling_count = daily["count"].rolling(7, min_periods=7).sum()
    rolling_valve = daily["valve_minutes"].rolling(7, min_periods=7).sum()
    rolling_rain = daily["rainfall_mm"].rolling(7, min_periods=7).sum()
    # Prefer a seven-day interval containing both rain and recorded valve activity.
    eligible = rolling_valve[(rolling_count >= 300) & (rolling_rain > 0) & (rolling_valve > 0)]
    if eligible.empty:
        eligible = rolling_valve[rolling_count >= 300]
    if eligible.empty:
        eligible = rolling_valve.dropna()
    window_end = eligible.idxmax().normalize()
    window_start = window_end - pd.Timedelta(days=6)
    window_stop = window_end + pd.Timedelta(days=1)
    sample = data.loc[(data.index >= window_start) & (data.index < window_stop)].copy()
    # Hourly aggregates preserve missing hours as NaN. Do not connect across gaps.
    hourly = sample.resample("h").agg({
        "soil_moisture": "mean",
        "weather_rain": "sum",
        "irrigation_duration_minutes": "sum",
    })
    hourly_counts = sample["soil_moisture"].resample("h").count()
    hourly.loc[hourly_counts.eq(0), ["weather_rain", "irrigation_duration_minutes"]] = np.nan
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})
    axes[0].plot(hourly.index, hourly["soil_moisture"], linewidth=1.7)
    axes[0].set_ylabel("Soil moisture (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title(f"Zone {zone}: illustrative 7-day observation window ({window_start:%d %b} - {window_end:%d %b %Y})", pad=12)
    axes[1].bar(hourly.index, hourly["irrigation_duration_minutes"], width=0.03)
    axes[1].set_ylabel("Valve-open\nminutes/hour")
    axes[2].bar(hourly.index, hourly["weather_rain"], width=0.03)
    axes[2].set_ylabel("Rainfall\nmm/hour")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].set_xlabel("Local time (Europe/Rome as provided)")
    fig.text(0.08, 0.008, "Historical co-occurrence only. Gaps remain gaps; this plot does not show irrigation causing moisture changes.",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_chart(fig, "05_zone4_activity_window.png")
    return zone, str(window_start.date()), str(window_end.date())


def chart_monthly_horizon_coverage(frames: dict[int, pd.DataFrame]) -> None:
    # The 144-row targets are not always 24-hour targets: audit by time elapsed.
    month_labels = pd.date_range("2025-02-01", "2025-09-01", freq="MS")
    matrix = np.zeros((5, len(month_labels)), dtype=int)
    for zone, data in frames.items():
        is_exact = ((data["ts"].shift(-HORIZON_ROWS) - data["ts"]) == HORIZON)
        counts = data.loc[is_exact].groupby(data.loc[is_exact, "ts"].dt.to_period("M")).size()
        for j, month in enumerate(month_labels):
            matrix[zone - 1, j] = int(counts.get(month.to_period("M"), 0))
    fig, ax = plt.subplots(figsize=(12.5, 4.7))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_yticks(range(5), [f"Zone {zone}" for zone in range(1, 6)])
    ax.set_xticks(range(len(month_labels)), [x.strftime("%b") for x in month_labels])
    ax.set_title("Exact 24-hour forecast windows by zone and month (2025)", pad=15)
    for i in range(5):
        for j in range(len(month_labels)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=8,
                    color="white" if value > matrix.max() * 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Verified forecast windows", fraction=0.023, pad=0.02)
    fig.tight_layout()
    save_chart(fig, "06_verified_windows_by_month.png")


def create_report(summary: pd.DataFrame, activity: tuple[int, str, str]) -> None:
    total_rows = int(summary["records"].sum())
    valid_rows = int(summary["verified_exact_24h_windows"].sum())
    count_irregular = int(summary["gaps_over_10min"].sum())
    count_regular = int(summary["regular_10min_intervals"].sum())
    invalid_ph = int(summary["ph_outside_3_to_9"].sum())
    invalid_ec = int(summary["ec_outside_0_to_1000"].sum())
    share = 100.0 * valid_rows / total_rows
    rows = []
    for s in summary.itertuples(index=False):
        rows.append(
            f"| {s.zone} | {s.records:,} | {s.verified_exact_24h_windows:,} | "
            f"{s.verified_24h_percent:.2f}% | {s.moisture_median:.2f}% | "
            f"{s.regular_interval_percent:.2f}% |"
        )
    lines = [
        "# AgroFlow AI - Phase 1: Dataset Exploration and Quality Audit",
        "",
        "## Purpose",
        "Explore the published five-zone preprocessed agriculture dataset. The goal is to",
        "understand sampling, spatial heterogeneity, missing time intervals, and forecast",
        "target validity before treating a model result as trustworthy.",
        "",
        f"**Dataset landing page:** {SOURCE_URL}",
        "**Data used:** The five `dataset_zone_*_preprocessed.csv` files, unmodified.",
        "**Times:** Naive Europe/Rome local timestamps, as described in DATA_DICTIONARY.md.",
        "",
        "## Key numbers",
        "",
        f"- Total processed observations: **{total_rows:,}** across five agricultural zones.",
        f"- Exact 24-hour forecast windows: **{valid_rows:,}** (**{share:.1f}%** of rows).",
        "- The original published targets are 144 **rows** ahead, not necessarily 24 **hours** ahead.",
        "- For this audit, a valid window must be 144 observations ahead, exactly 24 hours apart,",
        "  with the published target matching the observed soil moisture at that time.",
        f"- Consecutive regular 10-minute pairs across all zones: **{count_regular:,}**;",
        f"  time gaps longer than 10 minutes: **{count_irregular:,}**.",
        f"- Rows with pH outside the illustrative 3-9 screening range: **{invalid_ph:,}**;",
        f"  rows with EC outside the illustrative 0-1000 range: **{invalid_ec:,}**.",
        "  These are flagged observations, **not** proof that every flag is a sensor malfunction.",
        "",
        "## Zone-level summary",
        "",
        "| Zone | Processed rows | Exact-24h windows | Valid share | Median moisture | Regular 10-min pairs |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Figures",
        "",
        "1. `01_zone_coverage.png`: Processed rows versus verified 24-hour windows.",
        "2. `02_soil_moisture_distributions.png`: Different zone distributions.",
        "3. `03_monthly_moisture_trends.png`: Monthly medians from observed processed readings.",
        "4. `04_data_quality.png`: Timestamp regularity and screened pH/EC values.",
        f"5. `05_zone4_activity_window.png`: Historical zone {activity[0]} activity, {activity[1]} to {activity[2]}.",
        "6. `06_verified_windows_by_month.png`: Exact-24h windows by month.",
        "",
        "## Important methodological limitations",
        "",
        "- These are *preprocessed* rather than raw sensor values. The publisher cleaned,",
        "  clipped, despiked, and interpolated soil-moisture readings; chart trends can",
        "  reflect processing decisions. See `DATA_DICTIONARY.md` and the source README.",
        "- The `weather_*` variables originate from Open-Meteo ERA5 historical/reanalysis",
        "  data; they are not necessarily co-located live weather-station measurements.",
        "- A nonzero `irrigation_duration_minutes` is a recorded interval with valve",
        "  activity; do not equate number of active intervals with number of irrigation events.",
        "- Temporal gaps are substantial. When 144 successive 10-minute intervals are",
        "  not available, the publisher's 144-row future target is not exactly 24 hours.",
        "- Zone 3 has no eligible exact-24h windows in this dataset. It remains in EDA",
        "  but is **not** valid for the current exact-24h forecasting benchmark.",
        "- The 24-hour windows in this analysis overlap heavily, so adjacent samples",
        "  are not statistically independent.",
        "- `water_vol_to_24h` and `real_moisture_delta` contain future outcomes; never",
        "  use them as predictors in ordinary forecasting experiments.",
        "- This report does not establish irrigation causality or field-ready control.",
        "",
        "## Suggested use in later phases",
        "",
        "- Phase 2: Compare time-aware machine-learning models against persistence;",
        "  do not tune against an already-inspected historical test period.",
        "- Phase 3: Use feature importance / SHAP carefully as a model explanation,",
        "  not an explanation of physical causality.",
        "- Phase 4: Display the verified-coverage and limitations alongside forecasts.",
        "",
        "## Reproduce",
        "",
        "From the project root: `python agroflow_phase1_eda.py`.",
        "Artifacts are written to `phase1_outputs/` without modifying the input CSVs.",
        "",
    ]
    (OUTPUT_DIR / "PHASE1_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {zone: read_zone(zone) for zone in range(1, 6)}
    summary = pd.DataFrame([
        summarize_zone(zone, data) for zone, data in frames.items()
    ])
    summary.to_csv(OUTPUT_DIR / "zone_quality_summary.csv", index=False)

    chart_coverage(summary)
    chart_moisture_distribution(frames)
    chart_monthly_trend(frames)
    chart_quality(summary)
    activity = chart_example_activity(frames)
    chart_monthly_horizon_coverage(frames)
    create_report(summary, activity)

    print("=" * 62)
    print("AGROFLOW AI - PHASE 1 DATA EXPLORATION COMPLETED")
    print("=" * 62)
    print(f"Processed records: {summary['records'].sum():,}")
    print(f"Verified exact-24h windows: {summary['verified_exact_24h_windows'].sum():,}")
    print("\nPer-zone processed rows / verified 24-hour windows:")
    for row in summary.itertuples(index=False):
        print(f"Zone {row.zone}: {row.records:,} / {row.verified_exact_24h_windows:,}")
    print("\nSaved six charts, zone_quality_summary.csv, and PHASE1_FINDINGS.md")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Original datasets and V1/V2 files were NOT changed.")


if __name__ == "__main__":
    main()
