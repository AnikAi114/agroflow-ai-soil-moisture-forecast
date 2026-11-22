"""Non-destructive Phase 4 local portfolio readiness check.

Run in the AgroFlow project root: python agroflow_phase4_check.py
Reports what is present, what must be generated, and reproducible metrics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "phase4_outputs"
SOURCE_URL = "https://data.mendeley.com/datasets/c837v6p8ph/2"

REQUIRED_FILES = {
    "V1 training code": ROOT / "agroflow_train.py",
    "V2 training code": ROOT / "agroflow_train_v2.py",
    "Phase 1 EDA": ROOT / "agroflow_phase1_eda.py",
    "Phase 2 benchmarks": ROOT / "agroflow_phase2_benchmark.py",
    "Phase 3 explanations": ROOT / "agroflow_phase3_explain.py",
    "Final dashboard": ROOT / "agroflow_dashboard_final.py",
    "EDA report": ROOT / "phase1_outputs/PHASE1_FINDINGS.md",
    "Model comparison": ROOT / "phase2_outputs/model_comparison.csv",
    "Explainability report": ROOT / "phase3_outputs/PHASE3_FINDINGS.md",
    "V2 locally fitted models": ROOT / "agroflow_artifacts/models_v2.joblib",
}
REQUIRED_FILES.update({
    f"Dataset Zone {zone}": ROOT / f"02_processed_data/preprocessed/dataset_zone_{zone}_preprocessed.csv"
    for zone in range(1, 6)
})


def read_mae(split: str, model: str) -> float | None:
    table = ROOT / "phase2_outputs/model_comparison.csv"
    if not table.exists():
        return None
    with table.open(encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["split"] == split and row["model"] == model:
                return round(float(row["mae"]), 4)
    return None


def main() -> None:
    print("=" * 64)
    print("AGROFLOW AI - PHASE 4 PORTFOLIO READINESS CHECK")
    print("=" * 64)
    statuses = {name: path.exists() for name, path in REQUIRED_FILES.items()}
    for name, ok in statuses.items():
        print(f"{'OK     ' if ok else 'MISSING'}  {name}")

    baseline = read_mae("historical_test", "Persistence")
    candidate = read_mae("historical_test", "Regularized Gradient Boosting (unshrunk)")
    validations = {
        "historical_test_persistence_mae": baseline,
        "historical_test_phase2_gradient_boosting_mae": candidate,
        "source_url": SOURCE_URL,
        "note": "Historical test was previously inspected; no independent claim.",
    }

    OUTPUT.mkdir(exist_ok=True)
    report = {
        "files": statuses,
        "ready_for_full_local_dashboard": all(statuses.values()),
        "metrics": validations,
        "github_note": "Do not upload raw data, preprocessed source CSVs, or fitted model binaries.",
        "safety_note": "Research prototype; not for automatic irrigation control.",
    }
    (OUTPUT / "portfolio_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# AgroFlow AI | Phase 4 Release Check", "",
        "This is a local project audit, not a model-validation certificate.", "",
        "## File readiness", "",
    ]
    lines.extend(f"- [{'x' if ok else ' '}] {name}" for name, ok in statuses.items())
    lines.extend([
        "", "## Main findings", "",
        f"- Previously inspected historical-test persistence MAE: {baseline if baseline is not None else 'not available'} pp.",
        f"- Previously inspected historical-test unshrunk Phase 2 Gradient Boosting MAE: {candidate if candidate is not None else 'not available'} pp.",
        "- None of the numbers establishes field-ready performance.",
        "- Final research dashboard is suitable for a **local** demonstration when all files exist.",
        "- GitHub publication requires excluding original data and generated model binaries.",
        "", f"Dataset source: {SOURCE_URL}", "",
    ])
    (OUTPUT / "PHASE4_RELEASE_CHECK.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nHistorical-test MAE (previously inspected; lower is better):")
    print(f"  Persistence:          {baseline if baseline is not None else 'N/A'}")
    print(f"  Phase 2 GradientBoost: {candidate if candidate is not None else 'N/A'}")
    print(f"\nReady for full local demo: {'YES' if all(statuses.values()) else 'NO - review missing items'}")
    print("Saved phase4_outputs/portfolio_check.json and PHASE4_RELEASE_CHECK.md")
    print("No existing model, dataset, or earlier output was modified.")


if __name__ == "__main__":
    main()
