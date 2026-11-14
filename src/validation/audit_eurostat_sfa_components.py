from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SFA_FILE = (
    ROOT / "data/processed/europe_eurostat_sfa_reconciliation.csv"
)

GDP_FILE = (
    ROOT / "data/processed/europe_eurostat_edp_gdp_2019_2024.csv"
)

COVERAGE_FILE = (
    ROOT / "outputs/reports/europe_sfa_component_coverage.csv"
)

COMPARISON_FILE = (
    ROOT / "data/processed/europe_sfa_component_unit_comparison.csv"
)

SUMMARY_FILE = (
    ROOT / "outputs/reports/europe_sfa_component_unit_summary.csv"
)

REPORT_FILE = (
    ROOT / "outputs/reports/europe_sfa_component_audit_report.txt"
)

EXPECTED_COUNTRIES = 17
EXPECTED_YEARS = 5
EXPECTED_PAIRS = EXPECTED_COUNTRIES * EXPECTED_YEARS

EXPECTED_UNITS = {"MIO_NAC", "PC_GDP"}

UNIT_TOLERANCE_PP = 0.05

IDENTIFIER_COLUMNS = {
    "country",
    "year",
    "unit",
    "sfa_available",
    "reconciliation_status",
}


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    if not SFA_FILE.exists():
        raise FileNotFoundError(
            f"SFA input not found: {SFA_FILE}"
        )

    if not GDP_FILE.exists():
        raise FileNotFoundError(
            f"GDP input not found: {GDP_FILE}"
        )

    sfa = pd.read_csv(SFA_FILE)
    gdp = pd.read_csv(GDP_FILE)

    required_sfa = {
        "country",
        "year",
        "unit",
        "GD_CH",
        "B9_T3",
    }

    required_gdp = {
        "country_code",
        "year",
        "nominal_gdp",
    }

    missing_sfa = required_sfa - set(sfa.columns)
    missing_gdp = required_gdp - set(gdp.columns)

    if missing_sfa:
        raise ValueError(
            f"Missing SFA columns: {sorted(missing_sfa)}"
        )

    if missing_gdp:
        raise ValueError(
            f"Missing GDP columns: {sorted(missing_gdp)}"
        )

    sfa["country"] = (
        sfa["country"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    gdp["country_code"] = (
        gdp["country_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    sfa["unit"] = (
        sfa["unit"]
        .astype(str)
        .str.strip()
    )

    sfa["year"] = pd.to_numeric(
        sfa["year"],
        errors="raise"
    ).astype(int)

    gdp["year"] = pd.to_numeric(
        gdp["year"],
        errors="raise"
    ).astype(int)

    gdp = gdp[
        gdp["year"].between(2020, 2024)
    ].copy()

    gdp["nominal_gdp"] = pd.to_numeric(
        gdp["nominal_gdp"],
        errors="coerce"
    )

    if sfa.duplicated(
        ["country", "year", "unit"]
    ).any():
        raise ValueError(
            "Duplicate SFA country-year-unit observations detected."
        )

    if gdp.duplicated(
        ["country_code", "year"]
    ).any():
        raise ValueError(
            "Duplicate GDP country-year observations detected."
        )

    return sfa, gdp


# ============================================================
# IDENTIFY NUMERIC INDICATORS
# ============================================================

def identify_numeric_indicators(sfa):

    indicators = []

    for column in sfa.columns:

        if column in IDENTIFIER_COLUMNS:
            continue

        numeric = pd.to_numeric(
            sfa[column],
            errors="coerce"
        )

        if numeric.notna().any():
            sfa[column] = numeric
            indicators.append(column)

    if not indicators:
        raise ValueError(
            "No numeric SFA indicators found."
        )

    return indicators


# ============================================================
# COVERAGE AUDIT
# ============================================================

def build_coverage(sfa, indicators):

    rows = []

    for indicator in indicators:

        for unit in sorted(EXPECTED_UNITS):

            subset = sfa[
                sfa["unit"].eq(unit)
            ]

            available = subset[indicator].notna()

            rows.append({

                "indicator": indicator,

                "unit": unit,

                "expected_country_years": EXPECTED_PAIRS,

                "records": len(subset),

                "available_observations": int(
                    available.sum()
                ),

                "missing_observations": int(
                    (~available).sum()
                ),

                "coverage_pct": (
                    available.mean() * 100
                    if len(subset)
                    else np.nan
                ),

                "countries_covered": (
                    subset.loc[
                        available, "country"
                    ].nunique()
                ),

                "years_covered": (
                    subset.loc[
                        available, "year"
                    ].nunique()
                ),

            })

    return pd.DataFrame(rows)


# ============================================================
# MONETARY VS PERCENTAGE COMPARISON
# ============================================================

def build_unit_comparison(sfa, gdp, indicators):

    monetary = sfa[
        sfa["unit"].eq("MIO_NAC")
    ].copy()

    percentage = sfa[
        sfa["unit"].eq("PC_GDP")
    ].copy()

    monetary = monetary.drop(
        columns=["unit"]
    )

    percentage = percentage.drop(
        columns=["unit"]
    )

    monetary = monetary.rename(
        columns={
            indicator: f"{indicator}_mio"
            for indicator in indicators
        }
    )

    percentage = percentage.rename(
        columns={
            indicator: f"{indicator}_pct"
            for indicator in indicators
        }
    )

    comparison = monetary.merge(
        percentage,
        on=["country", "year"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    comparison = comparison.merge(
        gdp.rename(
            columns={
                "country_code": "country",
                "nominal_gdp": "current_gdp",
            }
        )[
            ["country", "year", "current_gdp"]
        ],
        on=["country", "year"],
        how="left",
        validate="one_to_one",
    )

    comparison["unit_pair_matched"] = (
        comparison["_merge"].eq("both")
    )

    comparison.drop(
        columns="_merge",
        inplace=True
    )

    comparison["valid_gdp"] = (
        comparison["current_gdp"].notna()
        & comparison["current_gdp"].gt(0)
    )

    result_rows = []

    for indicator in indicators:

        mio_col = f"{indicator}_mio"
        pct_col = f"{indicator}_pct"

        valid = (
            comparison["unit_pair_matched"]
            & comparison["valid_gdp"]
            & comparison[mio_col].notna()
            & comparison[pct_col].notna()
        )

        subset = comparison.loc[
            valid,
            [
                "country",
                "year",
                "current_gdp",
                mio_col,
                pct_col,
            ],
        ].copy()

        subset["indicator"] = indicator

        subset["mio_converted_pct"] = (
            subset[mio_col]
            / subset["current_gdp"]
            * 100
        )

        subset["direct_pct_gdp"] = subset[pct_col]

        subset["difference_pp"] = (
            subset["mio_converted_pct"]
            - subset["direct_pct_gdp"]
        )

        subset["absolute_difference_pp"] = (
            subset["difference_pp"].abs()
        )

        tolerance = (
            0.10 if indicator == "sfa_calculated"
            else UNIT_TOLERANCE_PP
        )

        subset["within_tolerance"] = (
            subset["absolute_difference_pp"]
            <= tolerance
        )

        result_rows.append(subset)

    if not result_rows:
        return pd.DataFrame(
            columns=[
                "country",
                "year",
                "current_gdp",
                "indicator",
                "mio_converted_pct",
                "direct_pct_gdp",
                "difference_pp",
                "absolute_difference_pp",
                "within_tolerance",
            ]
        ), comparison

    return pd.concat(
        result_rows,
        ignore_index=True
    ), comparison


# ============================================================
# UNIT COMPARISON SUMMARY
# ============================================================

def build_comparison_summary(comparison, indicators):

    rows = []

    for indicator in indicators:

        subset = comparison[
            comparison["indicator"].eq(indicator)
        ]

        rows.append({

            "indicator": indicator,

            "paired_observations": len(subset),

            "within_tolerance": int(
                subset["within_tolerance"].sum()
            ),

            "outside_tolerance": int(
                (~subset["within_tolerance"]).sum()
            ),

            "mean_difference_pp": (
                subset["difference_pp"].mean()
                if len(subset)
                else np.nan
            ),

            "mean_absolute_difference_pp": (
                subset["absolute_difference_pp"].mean()
                if len(subset)
                else np.nan
            ),

            "maximum_absolute_difference_pp": (
                subset["absolute_difference_pp"].max()
                if len(subset)
                else np.nan
            ),

            "tolerance_pp": (
                0.10 if indicator == "sfa_calculated"
                else UNIT_TOLERANCE_PP
            ),

        })

    return pd.DataFrame(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    sfa,
    gdp,
    indicators,
    coverage,
    comparison,
    summary,
    paired,
):

    checks = {}

    checks["Expected country count"] = (
        sfa["country"].nunique()
        == EXPECTED_COUNTRIES
    )

    checks["Expected SFA year range"] = (
        sfa["year"].min() == 2020
        and sfa["year"].max() == 2024
    )

    checks["Expected GDP coverage"] = (
        len(gdp) == EXPECTED_PAIRS
        and gdp["nominal_gdp"].notna().all()
        and gdp["nominal_gdp"].gt(0).all()
    )

    checks["Expected SFA units"] = (
        set(sfa["unit"].unique())
        == EXPECTED_UNITS
    )

    checks["Expected numeric indicators"] = (
        len(indicators) > 0
    )

    checks["No duplicate SFA observations"] = (
        not sfa.duplicated(
            ["country", "year", "unit"]
        ).any()
    )

    checks["No duplicate GDP observations"] = (
        not gdp.duplicated(
            ["country_code", "year"]
        ).any()
    )

    checks["Complete unit-pair coverage"] = (
        paired["unit_pair_matched"].all()
        and len(paired) > 0
    )

    checks["Positive GDP for unit comparisons"] = (
        paired["valid_gdp"].all()
    )

    checks["Both unit comparison values available"] = (
        len(comparison) > 0
        and comparison["mio_converted_pct"].notna().all()
        and comparison["direct_pct_gdp"].notna().all()
    )

    checks["Both unit values reconcile within tolerance"] = (
        len(comparison) > 0
        and comparison["within_tolerance"].all()
    )

    checks["All indicators represented in summary"] = (
        set(summary["indicator"]) == set(indicators)
    )

    return checks


# ============================================================
# REPORT
# ============================================================

def write_report(
    sfa,
    gdp,
    indicators,
    coverage,
    comparison,
    summary,
    checks,
):

    lines = []

    def add(value=""):
        lines.append(str(value))

    add("EUROSTAT SFA COMPONENT AUDIT")
    add("=" * 75)
    add()

    add("PERIOD: 2020-2024")
    add(f"EXPECTED COUNTRY-YEAR PAIRS: {EXPECTED_PAIRS}")
    add(f"SFA SOURCE RECORDS: {len(sfa)}")
    add(f"GDP SOURCE RECORDS: {len(gdp)}")
    add(f"NUMERIC INDICATORS AUDITED: {len(indicators)}")
    add()

    add("INDICATORS")
    add("-" * 75)

    for indicator in indicators:
        add(indicator)

    add()

    add("UNIT COVERAGE BY INDICATOR")
    add("-" * 75)

    add(
        coverage.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    add()

    add("MONETARY VS PERCENTAGE UNIT COMPARISON")
    add("-" * 75)

    add(f"Paired country-year observations: {len(comparison)}")
    add(f"Comparison tolerance: +/- {UNIT_TOLERANCE_PP:.2f} pp")
    add()

    add(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    add()

    add("UNIT COMPARISON EXCEPTIONS")
    add("-" * 75)

    if comparison.empty:

        add("No comparable observations.")

    else:

        exceptions = comparison[
            ~comparison["within_tolerance"]
        ]

        if exceptions.empty:

            add("No exceptions.")

        else:

            columns = [
                "country",
                "year",
                "indicator",
                "mio_converted_pct",
                "direct_pct_gdp",
                "difference_pp",
                "absolute_difference_pp",
            ]

            add(
                exceptions[columns].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}"
                )
            )

    add()

    add("METHODOLOGICAL NOTES")
    add("-" * 75)

    add(
        "1. Monetary values are converted using current-year GDP."
    )

    add(
        "2. PC_GDP observations are compared against converted MIO_NAC values."
    )

    add(
        "3. Unit consistency does not establish component additivity."
    )

    add(
        "4. Missing observations are not treated as zero."
    )

    add(
        "5. Aggregate and component indicators remain separate until "
        "their accounting relationships are established."
    )

    add(
        "6. This audit does not sum component indicators or attribute "
        "causal contributions."
    )

    add()

    add("VALIDATION RESULTS")
    add("-" * 75)

    for name, passed in checks.items():

        add(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    add()

    overall_pass = all(checks.values())

    add(
        "OVERALL RESULT: "
        f"{'PASS' if overall_pass else 'REVIEW REQUIRED'}"
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    sfa, gdp = load_inputs()

    indicators = identify_numeric_indicators(sfa)

    coverage = build_coverage(
        sfa,
        indicators
    )

    comparison, paired = build_unit_comparison(
        sfa,
        gdp,
        indicators
    )

    summary = build_comparison_summary(
        comparison,
        indicators
    )

    checks = validate_results(
        sfa,
        gdp,
        indicators,
        coverage,
        comparison,
        summary,
        paired
    )

    COVERAGE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    COMPARISON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    write_report(
        sfa,
        gdp,
        indicators,
        coverage,
        comparison,
        summary,
        checks
    )

    print("EUROSTAT SFA COMPONENT AUDIT")
    print("=" * 65)

    print(f"Countries: {sfa['country'].nunique()}")
    print(f"Years: {sfa['year'].min()}-{sfa['year'].max()}")
    print(f"Indicators: {len(indicators)}")
    print(f"Unit comparisons: {len(comparison)}")

    print()
    print("UNIT COMPARISON SUMMARY")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print("VALIDATION RESULTS")

    for name, passed in checks.items():

        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    print()

    print(
        "OVERALL RESULT:",
        "PASS" if all(checks.values()) else "REVIEW REQUIRED"
    )

    print()
    print(f"Coverage: {COVERAGE_FILE}")
    print(f"Comparison: {COMPARISON_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()