from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "europe_external_risk_input_panel.csv"

OUTPUT_DIAGNOSTICS = PROJECT_ROOT / "data" / "processed" / "europe_external_risk_diagnostics.csv"
OUTPUT_ANNUAL = PROJECT_ROOT / "data" / "processed" / "europe_external_risk_annual_dynamics.csv"
OUTPUT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "europe_external_risk_diagnostics_report.txt"

INDICATORS = {
    "BCA_NGDPD": "current_account_balance_pct_gdp",
    "TX_RPCH": "exports_volume_growth_pct",
    "TM_RPCH": "imports_volume_growth_pct",
}
PERIODS = {
    "long_run_1980_2025": (1980, 2025),
    "modern_2000_2019": (2000, 2019),
    "pre_pandemic_2015_2019": (2015, 2019),
    "recent_2020_2024": (2020, 2024),
}
EXPECTED_COUNTRIES = 20


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if series.notna().any() else np.nan


def safe_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if series.notna().sum() >= 2 else np.nan


def safe_median(series: pd.Series) -> float:
    return float(series.median()) if series.notna().any() else np.nan


def safe_mad(series: pd.Series) -> float:
    values = series.dropna()
    return float((values - values.median()).abs().median()) if len(values) else np.nan


def safe_iqr(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan


def longest_run(mask: pd.Series, years: pd.Series) -> int:
    pairs = pd.DataFrame({"flag": mask.fillna(False).astype(bool), "year": years})
    pairs = pairs.sort_values("year")
    best = current = 0
    previous_year = None
    for row in pairs.itertuples(index=False):
        if row.flag and (previous_year is None or row.year == previous_year + 1):
            current += 1
        elif row.flag:
            current = 1
        else:
            current = 0
        best = max(best, current)
        previous_year = row.year
    return best


def build_metrics(group: pd.DataFrame, start: int, end: int) -> dict:
    group = group.loc[group["year"].between(start, end)].sort_values("year").copy()
    values = group["value"].dropna()
    n = len(values)
    years_observed = group.loc[group["value"].notna(), "year"].nunique()

    result = {
        "records": len(group),
        "numeric_observations": n,
        "missing_numeric_observations": int(len(group) - n),
        "first_numeric_year": int(group.loc[group["value"].notna(), "year"].min()) if n else np.nan,
        "last_numeric_year": int(group.loc[group["value"].notna(), "year"].max()) if n else np.nan,
        "mean": safe_mean(group["value"]),
        "median": safe_median(group["value"]),
        "std": safe_std(group["value"]),
        "mad": safe_mad(group["value"]),
        "iqr": safe_iqr(group["value"]),
        "minimum": float(values.min()) if n else np.nan,
        "maximum": float(values.max()) if n else np.nan,
        "numeric_year_coverage": years_observed / (end - start + 1),
    }

    if group["indicator_code"].iloc[0] == "BCA_NGDPD":
        deficit = group["value"] < 0
        surplus = group["value"] > 0
        result.update({
            "negative_balance_frequency": float(deficit.sum() / n) if n else np.nan,
            "positive_balance_frequency": float(surplus.sum() / n) if n else np.nan,
            "largest_consecutive_deficit_run_years": longest_run(deficit, group["year"]),
            "absolute_mean_balance": float(group["value"].abs().mean()) if n else np.nan,
        })
    else:
        contraction = group["value"] < 0
        result.update({
            "negative_growth_frequency": float(contraction.sum() / n) if n else np.nan,
            "positive_growth_frequency": float((group["value"] > 0).sum() / n) if n else np.nan,
            "largest_consecutive_negative_growth_run_years": longest_run(contraction, group["year"]),
            "mean_absolute_growth": float(group["value"].abs().mean()) if n else np.nan,
        })
    return result


def main() -> int:
    print("=" * 76)
    print("EUROPEAN EXTERNAL RISK DIAGNOSTICS")
    print("=" * 76)
    require(INPUT_FILE.exists(), f"Input file not found: {INPUT_FILE}")

    panel = pd.read_csv(INPUT_FILE, low_memory=False)
    required = {"iso", "year", "indicator_code", "indicator_name", "value"}
    missing_columns = required - set(panel.columns)
    require(not missing_columns, f"Missing required columns: {sorted(missing_columns)}")

    panel["iso"] = panel["iso"].astype(str).str.strip().str.upper()
    panel["indicator_code"] = panel["indicator_code"].astype(str).str.strip()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    require(panel["year"].notna().all(), "Non-numeric or missing years detected.")
    panel["year"] = panel["year"].astype(int)

    panel = panel.loc[panel["indicator_code"].isin(INDICATORS)].copy()
    require(set(panel["indicator_code"].unique()) == set(INDICATORS),
            "One or more expected indicators are absent.")
    require(panel["iso"].nunique() == EXPECTED_COUNTRIES,
            f"Expected {EXPECTED_COUNTRIES} countries; found {panel['iso'].nunique()}.")
    require(not panel.duplicated(["iso", "year", "indicator_code"]).any(),
            "Duplicate country-year-indicator keys detected.")

    # Balanced recent panel is the anchor for cross-country comparisons.
    recent = panel.loc[panel["year"].between(2020, 2024)]
    for code in INDICATORS:
        subset = recent.loc[recent["indicator_code"].eq(code)]
        require(
            len(subset) == EXPECTED_COUNTRIES * 5
            and subset["iso"].nunique() == EXPECTED_COUNTRIES
            and subset["value"].notna().all()
            and subset.groupby("iso")["year"].nunique().eq(5).all(),
            f"2020-2024 panel is incomplete for {code}.",
        )

    rows = []
    for (iso, code), group in panel.groupby(["iso", "indicator_code"], sort=True):
        country = group["country"].iloc[0] if "country" in group.columns else iso
        for period_name, (start, end) in PERIODS.items():
            metrics = build_metrics(group, start, end)
            rows.append({
                "iso": iso,
                "country": country,
                "indicator_code": code,
                "indicator_name": INDICATORS[code],
                "period": period_name,
                "start_year": start,
                "end_year": end,
                **metrics,
            })
    diagnostics = pd.DataFrame(rows)

    # Annual changes in the current-account ratio and in export/import growth.
    # These are descriptive changes, not causal contributions.
    annual_rows = []
    for (iso, code), group in panel.groupby(["iso", "indicator_code"], sort=True):
        group = group.sort_values("year").copy()
        group["annual_change"] = group["value"].diff()
        group["previous_year"] = group["year"].shift(1)
        group.loc[group["year"] - group["previous_year"] != 1, "annual_change"] = np.nan
        country = group["country"].iloc[0] if "country" in group.columns else iso
        for row in group.itertuples(index=False):
            annual_rows.append({
                "iso": iso,
                "country": country,
                "year": int(row.year),
                "indicator_code": code,
                "indicator_name": INDICATORS[code],
                "value": row.value,
                "annual_change": row.annual_change,
            })
    annual = pd.DataFrame(annual_rows)

    require(len(diagnostics) == EXPECTED_COUNTRIES * len(INDICATORS) * len(PERIODS),
            "Unexpected diagnostics output dimensions.")
    require(not diagnostics.duplicated(["iso", "indicator_code", "period"]).any(),
            "Duplicate country-indicator-period diagnostics.")
    require(not annual.duplicated(["iso", "year", "indicator_code"]).any(),
            "Duplicate annual dynamics records.")

    OUTPUT_DIAGNOSTICS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(OUTPUT_DIAGNOSTICS, index=False)
    annual.to_csv(OUTPUT_ANNUAL, index=False)

    recent_diag = diagnostics.loc[diagnostics["period"].eq("recent_2020_2024")]
    report_lines = [
        "=" * 76,
        "EUROPEAN EXTERNAL RISK DIAGNOSTICS REPORT",
        "=" * 76,
        "",
        "INPUT",
        "-" * 76,
        str(INPUT_FILE),
        f"Panel records: {len(panel):,}",
        f"Countries: {panel['iso'].nunique()}",
        f"Indicators: {panel['indicator_code'].nunique()}",
        f"Diagnostics records: {len(diagnostics)}",
        f"Annual dynamics records: {len(annual)}",
        "",
        "RECENT PERIOD: 2020-2024",
        "-" * 76,
        recent_diag[[
            "iso", "indicator_code", "numeric_observations", "mean", "median",
            "std", "mad", "iqr", "minimum", "maximum"
        ]].to_string(index=False),
        "",
        "METHODOLOGICAL NOTES",
        "-" * 76,
        "Current-account deficits are not automatically external crises.",
        "A current-account balance ratio does not reveal financing composition or reserve adequacy.",
        "Export/import indicators are annual volume growth rates, not trade values or trade balances.",
        "Negative export/import volume growth indicates contraction in volume, not necessarily a worsening external position.",
        "Historical averages may be influenced by transition episodes and structural changes.",
        "Annual changes are descriptive and do not establish causality.",
        "This stage does not estimate external default probability or financing stress.",
        "",
        "VALIDATION",
        "-" * 76,
        "PASS: Required input columns present.",
        "PASS: All three expected indicators present.",
        "PASS: 20-country coverage.",
        "PASS: No duplicate country-year-indicator records.",
        "PASS: Balanced and numeric 2020-2024 panel for all indicators.",
        "PASS: Expected 240 country-indicator-period diagnostics records.",
        "PASS: No duplicate diagnostics or annual dynamics keys.",
        "",
        "OUTPUTS",
        "-" * 76,
        str(OUTPUT_DIAGNOSTICS),
        str(OUTPUT_ANNUAL),
        str(OUTPUT_REPORT),
    ]
    OUTPUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print("\nEUROPEAN EXTERNAL RISK DIAGNOSTICS VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nVALIDATION FAILED: {exc}", file=sys.stderr)
        raise
