from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "europe_weo_classified.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

TARGET_CODES = {
    "BCA_NGDPD": "current_account_balance_pct_gdp",
    "TX_RPCH": "exports_volume_growth_pct",
    "TM_RPCH": "imports_volume_growth_pct",
}
EXPECTED_COUNTRIES = 20
HISTORICAL_START = 1980
HISTORICAL_END = 2025
COMMON_PERIOD_START = 2020
COMMON_PERIOD_END = 2024

ALIASES = {
    "iso": ["iso", "ISO", "country_iso", "country_code"],
    "year": ["year", "Year", "TIME_PERIOD", "time_period"],
    "code": ["weo_code", "Subject Code", "subject_code", "WEO Subject Code", "indicator_code"],
    "value": ["value", "Value", "numeric_value", "value_numeric", "OBS_VALUE"],
    "status": ["observation_status", "status", "data_status", "classification"],
}


def resolve_column(df: pd.DataFrame, logical_name: str, required: bool = True):
    for candidate in ALIASES[logical_name]:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(
            f"Could not identify required '{logical_name}' column. "
            f"Tried {ALIASES[logical_name]}. Available columns: {list(df.columns)}"
        )
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({
            "": pd.NA, "--": pd.NA, "...": pd.NA, "…": pd.NA,
            "n/a": pd.NA, "N/A": pd.NA, "NA": pd.NA, "nan": pd.NA,
        })
    )
    return pd.to_numeric(cleaned, errors="coerce")


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    print("=" * 76)
    print("EUROPEAN EXTERNAL RISK INPUT INSPECTION")
    print("=" * 76)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    if df.empty:
        raise ValueError("Input dataset is empty.")

    iso_col = resolve_column(df, "iso")
    year_col = resolve_column(df, "year")
    code_col = resolve_column(df, "code")
    value_col = resolve_column(df, "value")
    status_col = resolve_column(df, "status", required=False)

    print(f"\nInput: {INPUT_FILE}")
    print(f"Input rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nRESOLVED SOURCE FIELDS")
    print("-" * 76)
    print(f"Country code: {iso_col}")
    print(f"Year:         {year_col}")
    print(f"Indicator:    {code_col}")
    print(f"Value:        {value_col}")
    print(f"Status:       {status_col or 'not found'}")

    work = df.copy()
    work["_iso"] = work[iso_col].astype("string").str.strip().str.upper()
    work["_year"] = pd.to_numeric(work[year_col], errors="coerce")
    work["_code"] = work[code_col].astype("string").str.strip().str.upper()
    work["_value"] = to_numeric(work[value_col])

    relevant = work[work["_code"].isin(TARGET_CODES)].copy()
    if relevant.empty:
        raise ValueError(
            f"None of the target WEO codes {list(TARGET_CODES)} were found "
            f"in column '{code_col}'."
        )

    # Keep all statuses for status coverage, but use historical actuals for
    # historical risk-input diagnostics.
    if status_col:
        relevant["_status"] = (
            relevant[status_col].astype("string").str.strip().str.lower()
        )
    else:
        relevant["_status"] = "unavailable"

    print("\nTARGET INDICATOR PRESENCE")
    print("-" * 76)
    for code, name in TARGET_CODES.items():
        subset = relevant[relevant["_code"] == code]
        print(f"{code} ({name}): {len(subset):,} rows")

    # Do not silently infer actual/estimate status if the source lacks it.
    if status_col:
        historical = relevant[
            relevant["_status"].str.replace(" ", "_", regex=False).isin(
                ["historical_actual", "historical"]
            )
        ].copy()
        if historical.empty:
            # Report the actual labels rather than guessing an alternative.
            labels = sorted(relevant["_status"].dropna().unique().tolist())
            raise ValueError(
                "No historical-actual rows matched the recognised status labels. "
                f"Observed labels: {labels}"
            )
    else:
        raise ValueError(
            "The classified dataset has no recognisable observation-status column. "
            "Cannot safely distinguish historical actuals from estimates/projections."
        )

    historical = historical[
        historical["_year"].between(HISTORICAL_START, HISTORICAL_END)
    ].copy()
    historical["_year"] = historical["_year"].astype(int)

    # Duplicate keys are an inspection failure: do not aggregate them away.
    duplicate_mask = historical.duplicated(
        subset=["_iso", "_code", "_year"], keep=False
    )
    duplicates = historical.loc[
        duplicate_mask, ["_iso", "_code", "_year", "_value", "_status"]
    ].sort_values(["_code", "_iso", "_year"])

    if not duplicates.empty:
        duplicate_path = REPORT_DIR / "europe_external_risk_duplicate_records.csv"
        write_csv(duplicates, duplicate_path)
        raise ValueError(
            f"Duplicate historical country-indicator-year keys found: "
            f"{len(duplicates)} rows. See {duplicate_path}"
        )

    expected_iso = set(historical["_iso"].dropna().unique())
    country_count = len(expected_iso)
    print(f"\nHistorical countries found: {country_count}")
    print(f"Historical period inspected: {HISTORICAL_START}-{HISTORICAL_END}")

    # Build country-indicator coverage and distribution diagnostics.
    profile_rows = []
    missing_rows = []
    extreme_rows = []

    for code, indicator_name in TARGET_CODES.items():
        subset = historical[historical["_code"] == code].copy()
        for iso, country_df in subset.groupby("_iso", dropna=False):
            country_df = country_df.sort_values("_year")
            numeric = country_df["_value"].dropna()
            years = country_df.loc[country_df["_value"].notna(), "_year"]
            expected_years = set(range(HISTORICAL_START, HISTORICAL_END + 1))
            observed_years = set(country_df["_year"].dropna().astype(int))
            numeric_years = set(years.astype(int))
            missing_years = sorted(expected_years - numeric_years)

            row = {
                "iso": iso,
                "indicator_code": code,
                "indicator_name": indicator_name,
                "records": len(country_df),
                "numeric_observations": len(numeric),
                "missing_numeric_observations": len(country_df) - len(numeric),
                "first_numeric_year": min(numeric_years) if numeric_years else np.nan,
                "last_numeric_year": max(numeric_years) if numeric_years else np.nan,
                "numeric_years_covered": len(numeric_years),
                "expected_years_1980_2025": len(expected_years),
                "missing_years": ";".join(map(str, missing_years)),
                "mean": numeric.mean() if len(numeric) else np.nan,
                "median": numeric.median() if len(numeric) else np.nan,
                "standard_deviation": numeric.std() if len(numeric) > 1 else np.nan,
                "minimum": numeric.min() if len(numeric) else np.nan,
                "maximum": numeric.max() if len(numeric) else np.nan,
                "2020_2024_numeric_observations": int(
                    country_df.loc[
                        country_df["_year"].between(COMMON_PERIOD_START, COMMON_PERIOD_END),
                        "_value",
                    ].notna().sum()
                ),
            }
            profile_rows.append(row)

            missing_rows.append({
                "iso": iso,
                "indicator_code": code,
                "indicator_name": indicator_name,
                "missing_year_count": len(missing_years),
                "missing_years": ";".join(map(str, missing_years)),
            })

            if len(numeric):
                min_idx = country_df["_value"].idxmin()
                max_idx = country_df["_value"].idxmax()
                for label, idx in [("minimum", min_idx), ("maximum", max_idx)]:
                    record = country_df.loc[idx]
                    extreme_rows.append({
                        "iso": iso,
                        "indicator_code": code,
                        "indicator_name": indicator_name,
                        "extreme_type": label,
                        "year": int(record["_year"]),
                        "value": float(record["_value"]),
                    })

    profile = pd.DataFrame(profile_rows)
    missingness = pd.DataFrame(missing_rows)
    extremes = pd.DataFrame(extreme_rows)

    # Common-period balanced panel diagnostics.
    common = historical[
        historical["_year"].between(COMMON_PERIOD_START, COMMON_PERIOD_END)
    ].copy()
    common_rows = []
    for code, indicator_name in TARGET_CODES.items():
        sub = common[common["_code"] == code]
        numeric_count = int(sub["_value"].notna().sum())
        unique_countries = int(sub["_iso"].nunique())
        common_rows.append({
            "indicator_code": code,
            "indicator_name": indicator_name,
            "expected_country_years": EXPECTED_COUNTRIES * 5,
            "records_found": len(sub),
            "countries_found": unique_countries,
            "numeric_observations": numeric_count,
            "missing_numeric_observations": len(sub) - numeric_count,
            "balanced_20_country_2020_2024": (
                unique_countries == EXPECTED_COUNTRIES
                and len(sub) == EXPECTED_COUNTRIES * 5
                and numeric_count == EXPECTED_COUNTRIES * 5
            ),
        })
    common_summary = pd.DataFrame(common_rows)

    # Status coverage across the complete target-code extract.
    status_summary = (
        relevant.groupby(["_code", "_status"], dropna=False)
        .agg(records=("_value", "size"), numeric_observations=("_value", "count"))
        .reset_index()
        .rename(columns={"_code": "indicator_code", "_status": "observation_status"})
    )

    # Metadata fields are reported when available; do not infer units.
    metadata_candidates = [
        "Subject Descriptor", "subject_descriptor", "weo_subject",
        "Units", "units", "Scale", "scale", "Country", "country",
    ]
    metadata_cols = [c for c in metadata_candidates if c in relevant.columns]
    metadata_rows = []
    for code in TARGET_CODES:
        sub = relevant[relevant["_code"] == code]
        item = {"indicator_code": code, "records": len(sub)}
        for col in metadata_cols:
            vals = sub[col].dropna().astype(str).unique().tolist()
            item[col] = " | ".join(vals[:10])
        metadata_rows.append(item)
    metadata = pd.DataFrame(metadata_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    profile_path = PROCESSED_DIR / "europe_external_risk_input_profile.csv"
    missing_path = REPORT_DIR / "europe_external_risk_missingness.csv"
    extremes_path = REPORT_DIR / "europe_external_risk_extremes.csv"
    common_path = REPORT_DIR / "europe_external_risk_common_period_coverage.csv"
    status_path = REPORT_DIR / "europe_external_risk_observation_status.csv"
    metadata_path = REPORT_DIR / "europe_external_risk_source_metadata.csv"
    report_path = REPORT_DIR / "europe_external_risk_input_inspection.txt"

    write_csv(profile, profile_path)
    write_csv(missingness, missing_path)
    write_csv(extremes, extremes_path)
    write_csv(common_summary, common_path)
    write_csv(status_summary, status_path)
    write_csv(metadata, metadata_path)

    total_rows = len(historical)
    numeric_total = int(historical["_value"].notna().sum())
    missing_total = total_rows - numeric_total
    duplicate_count = len(duplicates)

    lines = [
        "=" * 76,
        "EUROPEAN EXTERNAL RISK INPUT INSPECTION REPORT",
        "=" * 76,
        "",
        f"Input: {INPUT_FILE}",
        f"Inspection timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Historical rows: {total_rows}",
        f"Countries: {country_count}",
        f"Historical period: {HISTORICAL_START}-{HISTORICAL_END}",
        f"Numeric observations: {numeric_total}",
        f"Missing numeric observations: {missing_total}",
        f"Duplicate historical country-indicator-year keys: {duplicate_count}",
        "",
        "INDICATOR COVERAGE",
        "-" * 76,
        profile.to_string(index=False) if not profile.empty else "No profiles generated.",
        "",
        "COMMON PERIOD COVERAGE: 2020-2024",
        "-" * 76,
        common_summary.to_string(index=False),
        "",
        "OBSERVATION STATUS COVERAGE",
        "-" * 76,
        status_summary.to_string(index=False),
        "",
        "SOURCE METADATA",
        "-" * 76,
        metadata.to_string(index=False),
        "",
        "INTERPRETATION LIMITATIONS",
        "-" * 76,
        "Current-account balance is not a standalone measure of external vulnerability.",
        "A deficit must be assessed alongside financing, reserves, external debt and flows.",
        "Export/import volume growth rates are not export/import levels or trade balances.",
        "WEO historical observations do not provide a complete external financing profile.",
        "Extreme historical values require source and country-context review before treatment.",
        "Missing observations are not zero and have not been imputed.",
        "Historical actuals are separated from estimates and projections.",
        "",
        "OUTPUTS CREATED",
        "-" * 76,
        str(profile_path),
        str(missing_path),
        str(extremes_path),
        str(common_path),
        str(status_path),
        str(metadata_path),
        str(report_path),
        "",
        "=" * 76,
        "EUROPEAN EXTERNAL RISK INPUT INSPECTION COMPLETED",
        "=" * 76,
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nSUMMARY")
    print("-" * 76)
    print(f"Historical records: {total_rows:,}")
    print(f"Countries: {country_count}")
    print(f"Numeric observations: {numeric_total:,}")
    print(f"Missing numeric observations: {missing_total:,}")
    print("\nCOMMON PERIOD: 2020-2024")
    print(common_summary.to_string(index=False))
    print("\nOUTPUTS CREATED")
    print("-" * 76)
    for path in [profile_path, missing_path, extremes_path, common_path,
                 status_path, metadata_path, report_path]:
        print(path)
    print("\nInspection completed. Review coverage and metadata before building risk indicators.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
