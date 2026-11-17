from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "europe_weo_classified.csv"

OUTPUT_PANEL = PROJECT_ROOT / "data" / "processed" / "europe_external_risk_input_panel.csv"
OUTPUT_COVERAGE = PROJECT_ROOT / "outputs" / "reports" / "europe_external_risk_country_coverage.csv"
OUTPUT_MISSINGNESS = PROJECT_ROOT / "outputs" / "reports" / "europe_external_risk_missingness.csv"
OUTPUT_COMMON = PROJECT_ROOT / "outputs" / "reports" / "europe_external_risk_common_period_coverage.csv"
OUTPUT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "europe_external_risk_input_report.txt"

INDICATORS = {
    "BCA_NGDPD": "current_account_balance_pct_gdp",
    "TX_RPCH": "exports_volume_growth_pct",
    "TM_RPCH": "imports_volume_growth_pct",
}
EXPECTED_COUNTRIES = 20
HISTORICAL_STATUS = "historical_actual"
COMMON_START = 2020
COMMON_END = 2024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_column(columns: list[str], candidates: tuple[str, ...], required: bool = True) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    if required:
        raise ValueError(f"Required source field not found. Tried: {candidates}")
    return None


def main() -> int:
    print("=" * 76)
    print("EUROPEAN EXTERNAL RISK INPUT PANEL BUILDER")
    print("=" * 76)

    require(INPUT_FILE.exists(), f"Input file not found: {INPUT_FILE}")
    source = pd.read_csv(INPUT_FILE, low_memory=False)

    if "ISO" in source.columns and "iso" in source.columns:
        source = source.drop(columns=["ISO"])

    cols = source.columns.tolist()

    iso_col = resolve_column(cols, ("iso", "ISO"))
    year_col = resolve_column(cols, ("year", "Year"))
    code_col = resolve_column(cols, ("weo_code", "WEO Subject Code", "subject_code"))
    value_col = resolve_column(cols, ("value", "Value"))
    status_col = resolve_column(cols, ("observation_status", "status"))
    country_col = resolve_column(cols, ("country", "Country", "country_name"), False)
    unit_col = resolve_column(cols, ("unit", "Units", "weo_unit"), False)
    scale_col = resolve_column(cols, ("scale", "Scale"), False)
    descriptor_col = resolve_column(cols, ("subject_descriptor", "Subject Descriptor"), False)

    work = source.copy()
    work[iso_col] = work[iso_col].astype(str).str.strip().str.upper()
    work[code_col] = work[code_col].astype(str).str.strip()
    work[status_col] = work[status_col].astype(str).str.strip().str.lower()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    selected = work.loc[
        work[code_col].isin(INDICATORS)
        & work[status_col].eq(HISTORICAL_STATUS)
    ].copy()
    require(not selected.empty, "No historical actual observations found for target indicators.")

    rename = {
        iso_col: "iso",
        year_col: "year",
        code_col: "indicator_code",
        value_col: "value",
        status_col: "observation_status",
    }
    for source_name, target_name in (
        (country_col, "country"),
        (unit_col, "unit"),
        (scale_col, "scale"),
        (descriptor_col, "source_subject_descriptor"),
    ):
        if source_name:
            rename[source_name] = target_name

    panel = selected.rename(columns=rename)
    keep = [
        "iso", "country", "year", "indicator_code", "value", "unit",
        "scale", "source_subject_descriptor", "observation_status",
    ]
    panel = panel[[c for c in keep if c in panel.columns]].copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    require(panel["year"].notna().all(), "Missing or non-numeric years detected.")
    panel["year"] = panel["year"].astype(int)
    panel["indicator_name"] = panel["indicator_code"].map(INDICATORS)

    dupes = panel.duplicated(["iso", "year", "indicator_code"], keep=False)
    require(not dupes.any(), "Duplicate country-year-indicator keys detected:\n" +
            panel.loc[dupes, ["iso", "year", "indicator_code"]].head(12).to_string(index=False))

    country_sets = {
        code: set(panel.loc[panel["indicator_code"].eq(code), "iso"].unique())
        for code in INDICATORS
    }
    all_countries = set.union(*country_sets.values())
    require(len(all_countries) == EXPECTED_COUNTRIES,
            f"Expected {EXPECTED_COUNTRIES} countries; found {len(all_countries)}.")
    require(all(cset == all_countries for cset in country_sets.values()),
            "The three indicators do not have identical country coverage.")

    coverage_rows = []
    missing_rows = []
    for code, label in INDICATORS.items():
        subset = panel.loc[panel["indicator_code"].eq(code)]
        for iso in sorted(all_countries):
            country_data = subset.loc[subset["iso"].eq(iso)]
            coverage_rows.append({
                "iso": iso,
                "indicator_code": code,
                "indicator_name": label,
                "records": len(country_data),
                "numeric_observations": int(country_data["value"].notna().sum()),
                "missing_numeric_observations": int(country_data["value"].isna().sum()),
                "first_year": int(country_data["year"].min()) if len(country_data) else np.nan,
                "last_year": int(country_data["year"].max()) if len(country_data) else np.nan,
            })
        missing_rows.append({
            "indicator_code": code,
            "indicator_name": label,
            "records": len(subset),
            "numeric_observations": int(subset["value"].notna().sum()),
            "missing_numeric_observations": int(subset["value"].isna().sum()),
            "missing_pct": round(float(subset["value"].isna().mean() * 100), 4),
        })

    coverage = pd.DataFrame(coverage_rows)
    missingness = pd.DataFrame(missing_rows)

    common = panel.loc[panel["year"].between(COMMON_START, COMMON_END)]
    expected = EXPECTED_COUNTRIES * (COMMON_END - COMMON_START + 1)
    common_rows = []
    for code, label in INDICATORS.items():
        subset = common.loc[common["indicator_code"].eq(code)]
        n_numeric = int(subset["value"].notna().sum())
        common_rows.append({
            "indicator_code": code,
            "indicator_name": label,
            "expected_country_years": expected,
            "records_found": len(subset),
            "countries_found": subset["iso"].nunique(),
            "numeric_observations": n_numeric,
            "missing_numeric_observations": int(len(subset) - n_numeric),
            "balanced_20_country_2020_2024": (
                len(subset) == expected
                and subset["iso"].nunique() == EXPECTED_COUNTRIES
                and n_numeric == expected
                and subset.groupby("iso")["year"].nunique().eq(5).all()
            ),
        })
    common_coverage = pd.DataFrame(common_rows)
    require(common_coverage["balanced_20_country_2020_2024"].all(),
            "The 2020-2024 common panel is not balanced and numeric for all indicators.")

    OUTPUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    panel.sort_values(["iso", "indicator_code", "year"]).to_csv(OUTPUT_PANEL, index=False)
    coverage.to_csv(OUTPUT_COVERAGE, index=False)
    missingness.to_csv(OUTPUT_MISSINGNESS, index=False)
    common_coverage.to_csv(OUTPUT_COMMON, index=False)

    total_numeric = int(panel["value"].notna().sum())
    total_missing = int(panel["value"].isna().sum())
    report = [
        "=" * 76,
        "EUROPEAN EXTERNAL RISK INPUT PANEL REPORT",
        "=" * 76,
        "",
        "SOURCE",
        "-" * 76,
        str(INPUT_FILE),
        f"Source rows: {len(source):,}",
        f"Historical target records: {len(panel):,}",
        f"Countries: {panel['iso'].nunique()}",
        f"Historical period: {panel['year'].min()}-{panel['year'].max()}",
        f"Numeric observations: {total_numeric:,}",
        f"Missing numeric observations: {total_missing:,}",
        "",
        "INDICATOR COVERAGE",
        "-" * 76,
        missingness.to_string(index=False),
        "",
        "COMMON PERIOD: 2020-2024",
        "-" * 76,
        common_coverage.to_string(index=False),
        "",
        "METHODOLOGICAL NOTES",
        "-" * 76,
        "Only observations classified as historical_actual are included.",
        "Missing numeric observations are retained; they are not imputed as zero.",
        "Current-account balance is not, by itself, a measure of financing vulnerability.",
        "Export and import indicators are volume growth rates, not trade levels or trade balance.",
        "This panel does not include reserves, external debt, exchange rates or financing structure.",
        "",
        "VALIDATION",
        "-" * 76,
        "PASS: Required source fields resolved.",
        "PASS: Three target indicators present.",
        "PASS: Exactly 20 countries represented consistently.",
        "PASS: No duplicate country-year-indicator keys.",
        "PASS: 2020-2024 panels balanced and numeric for all indicators.",
        "PASS: Missing numeric values preserved without imputation.",
        "",
        "OUTPUTS",
        "-" * 76,
        str(OUTPUT_PANEL),
        str(OUTPUT_COVERAGE),
        str(OUTPUT_MISSINGNESS),
        str(OUTPUT_COMMON),
        str(OUTPUT_REPORT),
    ]
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print("\nEUROPEAN EXTERNAL RISK INPUT PANEL VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nVALIDATION FAILED: {exc}", file=sys.stderr)
        raise
