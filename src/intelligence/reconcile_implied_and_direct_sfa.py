from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SFA_FILE = (
    ROOT / "data/processed/europe_eurostat_sfa_reconciliation.csv"
)

IGC_FILE = (
    ROOT / "data/processed/europe_interest_growth_contribution.csv"
)

OUTPUT_FILE = (
    ROOT / "data/processed/europe_sfa_independent_reconciliation.csv"
)

SUMMARY_FILE = (
    ROOT / "data/processed/europe_sfa_independent_reconciliation_summary.csv"
)

REPORT_FILE = (
    ROOT / "outputs/reports/europe_sfa_independent_reconciliation_report.txt"
)

EXPECTED_COUNTRIES = 17
EXPECTED_YEARS = 5
EXPECTED_OBSERVATIONS = EXPECTED_COUNTRIES * EXPECTED_YEARS

ROUNDING_TOLERANCE_BY_UNIT = {
    "MIO_NAC": 0.05,
    "PC_GDP": 0.10,
}

EXPECTED_UNITS = set(ROUNDING_TOLERANCE_BY_UNIT)


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():

    if not SFA_FILE.exists():
        raise FileNotFoundError(SFA_FILE)

    if not IGC_FILE.exists():
        raise FileNotFoundError(IGC_FILE)

    sfa = pd.read_csv(SFA_FILE)
    igc = pd.read_csv(IGC_FILE)

    required_sfa = {
        "country",
        "year",
        "unit",
        "sfa_calculated",
    }

    required_igc = {
        "country_code",
        "country",
        "year",
        "closing_gdp",
        "implied_sfa_pp",
    }

    missing_sfa = required_sfa - set(sfa.columns)
    missing_igc = required_igc - set(igc.columns)

    if missing_sfa:
        raise ValueError(
            f"Missing SFA columns: {sorted(missing_sfa)}"
        )

    if missing_igc:
        raise ValueError(
            f"Missing interest-growth columns: {sorted(missing_igc)}"
        )

    sfa["year"] = pd.to_numeric(
        sfa["year"], errors="raise"
    ).astype(int)

    igc["year"] = pd.to_numeric(
        igc["year"], errors="raise"
    ).astype(int)

    sfa["sfa_calculated"] = pd.to_numeric(
        sfa["sfa_calculated"], errors="coerce"
    )

    igc["closing_gdp"] = pd.to_numeric(
        igc["closing_gdp"], errors="coerce"
    )

    igc["implied_sfa_pp"] = pd.to_numeric(
        igc["implied_sfa_pp"], errors="coerce"
    )

    sfa["country"] = sfa["country"].astype(str).str.strip().str.upper()
    sfa["unit"] = sfa["unit"].astype(str).str.strip()

    igc["country_code"] = (
        igc["country_code"].astype(str).str.strip().str.upper()
    )

    return sfa, igc


# ============================================================
# BUILD DIRECT VS IMPLIED SFA COMPARISON
# ============================================================

def build_comparison(sfa, igc):

    direct = sfa[
        sfa["unit"].isin(EXPECTED_UNITS)
        & sfa["sfa_calculated"].notna()
    ].copy()

    direct = direct.rename(
        columns={"country": "country_code"}
    )

    duplicate_direct = direct.duplicated(
        ["country_code", "year", "unit"]
    ).sum()

    if duplicate_direct:
        raise ValueError(
            f"Duplicate direct SFA records: {duplicate_direct}"
        )

    duplicate_igc = igc.duplicated(
        ["country_code", "year"]
    ).sum()

    if duplicate_igc:
        raise ValueError(
            f"Duplicate interest-growth records: {duplicate_igc}"
        )

    comparison = direct.merge(
        igc[
            [
                "country_code",
                "year",
                "country",
                "closing_gdp",
                "implied_sfa_pp",
            ]
        ],
        on=["country_code", "year"],
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_sfa", "_igc"),
    )

    comparison["matched_igc"] = (
        comparison["_merge"].eq("both")
    )

    comparison.drop(columns="_merge", inplace=True)

    # --------------------------------------------------------
    # Convert direct SFA into percentage points of GDP.
    # --------------------------------------------------------

    comparison["direct_sfa_pp"] = np.nan

    monetary = comparison["unit"].eq("MIO_NAC")
    percentage = comparison["unit"].eq("PC_GDP")

    valid_gdp = (
        comparison["closing_gdp"].notna()
        & comparison["closing_gdp"].gt(0)
    )

    comparison.loc[
        monetary & valid_gdp,
        "direct_sfa_pp"
    ] = (
        comparison.loc[
            monetary & valid_gdp,
            "sfa_calculated"
        ]
        / comparison.loc[
            monetary & valid_gdp,
            "closing_gdp"
        ]
        * 100
    )

    comparison.loc[
        percentage,
        "direct_sfa_pp"
    ] = comparison.loc[
        percentage,
        "sfa_calculated"
    ]

    # --------------------------------------------------------
    # Calculate differences.
    # --------------------------------------------------------

    comparison["difference_pp"] = (
        comparison["implied_sfa_pp"]
        - comparison["direct_sfa_pp"]
    )

    comparison["absolute_difference_pp"] = (
        comparison["difference_pp"].abs()
    )

    comparison["rounding_tolerance_pp"] = (
        comparison["unit"].map(
            ROUNDING_TOLERANCE_BY_UNIT
        )
    )

    comparison["within_rounding_tolerance"] = (
        comparison["absolute_difference_pp"]
        <= comparison["rounding_tolerance_pp"]
    )

    # --------------------------------------------------------
    # Classify each comparison.
    # --------------------------------------------------------

    comparison["comparison_status"] = np.select(
        [
            ~comparison["matched_igc"],

            comparison["direct_sfa_pp"].isna(),

            comparison["implied_sfa_pp"].isna(),

            comparison["within_rounding_tolerance"],
        ],
        [
            "unmatched_interest_growth",

            "missing_direct_sfa_or_gdp",

            "missing_implied_sfa",

            "within_rounding_tolerance",
        ],
        default="difference_exceeds_tolerance",
    )

    comparison = comparison.sort_values(
        ["country_code", "year", "unit"]
    ).reset_index(drop=True)

    return comparison


# ============================================================
# BUILD UNIT-LEVEL SUMMARY
# ============================================================

def build_summary(comparison):

    rows = []

    for unit in sorted(EXPECTED_UNITS):

        group = comparison[
            comparison["unit"].eq(unit)
        ].copy()

        valid = group.dropna(
            subset=[
                "implied_sfa_pp",
                "direct_sfa_pp",
                "difference_pp",
            ]
        )

        within = valid[
            valid["within_rounding_tolerance"]
        ]

        outside = valid[
            ~valid["within_rounding_tolerance"]
        ]

        rows.append({

            "unit": unit,

            "direct_observations": len(group),

            "matched_observations": int(
                group["matched_igc"].sum()
            ),

            "comparable_observations": len(valid),

            "within_tolerance": len(within),

            "outside_tolerance": len(outside),

            "mean_difference_pp": (
                valid["difference_pp"].mean()
                if len(valid) else np.nan
            ),

            "mean_absolute_difference_pp": (
                valid["absolute_difference_pp"].mean()
                if len(valid) else np.nan
            ),

            "maximum_absolute_difference_pp": (
                valid["absolute_difference_pp"].max()
                if len(valid) else np.nan
            ),

            "rounding_tolerance_pp": (
                ROUNDING_TOLERANCE_BY_UNIT[unit]
            ),

        })

    return pd.DataFrame(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_results(sfa, igc, comparison, summary):

    checks = {}

    checks["Expected country count"] = (
        igc["country_code"].nunique()
        == EXPECTED_COUNTRIES
    )

    checks["Expected interest-growth observations"] = (
        len(igc) == EXPECTED_OBSERVATIONS
    )

    checks["Expected year range"] = (
        igc["year"].min() == 2020
        and igc["year"].max() == 2024
    )

    checks["Expected SFA units"] = (
        set(comparison["unit"].unique())
        == EXPECTED_UNITS
    )

    checks["No duplicate interest-growth records"] = (
        not igc.duplicated(
            ["country_code", "year"]
        ).any()
    )

    checks["No duplicate direct SFA records"] = (
        not comparison.duplicated(
            ["country_code", "year", "unit"]
        ).any()
    )

    checks["All direct SFA records matched"] = (
        comparison["matched_igc"].all()
    )

    checks["No missing direct SFA or GDP values"] = (
        comparison["direct_sfa_pp"].notna().all()
    )

    checks["No missing implied SFA values"] = (
        comparison["implied_sfa_pp"].notna().all()
    )

    checks["All comparisons within unit tolerance"] = (
        comparison["within_rounding_tolerance"].all()
    )

    checks["Both unit summaries present"] = (
        set(summary["unit"]) == EXPECTED_UNITS
    )

    checks["All direct observations included in summaries"] = (
        (
            summary["direct_observations"]
            == summary["matched_observations"]
        ).all()
    )

    return checks


# ============================================================
# REPORT
# ============================================================

def write_report(
    sfa,
    igc,
    comparison,
    summary,
    checks,
):

    lines = []

    def add(value=""):
        lines.append(str(value))

    add("EUROSTAT IMPLIED VS DIRECT SFA RECONCILIATION")
    add("=" * 75)
    add()

    add("PERIOD: 2020-2024")
    add(f"EXPECTED COUNTRY-YEAR OBSERVATIONS: {EXPECTED_OBSERVATIONS}")
    add(f"INTEREST-GROWTH OBSERVATIONS: {len(igc)}")
    add()

    add("COMPARISON METHOD")
    add("-" * 75)
    add("MIO_NAC direct SFA = (GD_CH - B9_T3) / current-year GDP * 100")
    add("PC_GDP direct SFA = GD_CH - B9_T3")
    add()
    add("Implied SFA = observed debt-ratio change")
    add("              - interest-growth contribution")
    add("              + primary balance")
    add()
    add("Difference = implied SFA - direct SFA")
    add()

    add("UNIT-SPECIFIC ROUNDING TOLERANCES")
    add("-" * 75)

    for unit, tolerance in ROUNDING_TOLERANCE_BY_UNIT.items():
        add(f"{unit}: +/- {tolerance:.2f} percentage points")

    add()
    add("SOURCE COVERAGE")
    add("-" * 75)
    add(f"Raw SFA records: {len(sfa)}")
    add(f"Interest-growth records: {len(igc)}")

    add("Direct SFA calculated records by unit:")

    for unit in sorted(EXPECTED_UNITS):

        count = len(
            sfa[
                sfa["unit"].eq(unit)
                & sfa["sfa_calculated"].notna()
            ]
        )

        add(f"  {unit}: {count}")

    add(
        "Matched country-year-unit records: "
        f"{int(comparison['matched_igc'].sum())}"
    )

    add(
        "Missing interest-growth matches: "
        f"{int((~comparison['matched_igc']).sum())}"
    )

    add()

    add("RECONCILIATION SUMMARY")
    add("-" * 75)

    add(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    add()

    add("COUNTRY-YEAR EXCEPTIONS")
    add("-" * 75)

    exceptions = comparison[
        comparison["comparison_status"].ne(
            "within_rounding_tolerance"
        )
    ]

    if exceptions.empty:

        add("No exceptions.")

    else:

        columns = [
            "country_code",
            "year",
            "unit",
            "sfa_calculated",
            "closing_gdp",
            "implied_sfa_pp",
            "direct_sfa_pp",
            "difference_pp",
            "rounding_tolerance_pp",
            "comparison_status",
        ]

        add(
            exceptions[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    add()

    add("VALIDATION RESULTS")
    add("-" * 75)

    for name, passed in checks.items():

        add(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    add()

    add("METHODOLOGICAL NOTES")
    add("-" * 75)

    add(
        "1. MIO_NAC monetary SFA is converted using current-year GDP."
    )

    add(
        "2. PC_GDP is independently calculated from rounded percentage "
        "series and therefore has a wider rounding tolerance."
    )

    add(
        "3. MIO_NAC is the primary reconciliation measure."
    )

    add(
        "4. Implied SFA is an accounting residual, not an independent "
        "measurement."
    )

    add(
        "5. Agreement does not validate the attribution of individual "
        "stock-flow adjustment components."
    )

    add(
        "6. Tolerance is a measurement-precision allowance, not an "
        "economic materiality threshold."
    )

    add()

    overall_pass = all(checks.values())

    add(
        f"OVERALL RESULT: "
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

    sfa, igc = load_inputs()

    comparison = build_comparison(
        sfa,
        igc
    )

    summary = build_summary(
        comparison
    )

    checks = validate_results(
        sfa,
        igc,
        comparison,
        summary
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    write_report(
        sfa,
        igc,
        comparison,
        summary,
        checks
    )

    print("EUROSTAT SFA INDEPENDENT RECONCILIATION")
    print("=" * 65)

    print(
        f"Comparison records: {len(comparison)}"
    )

    print(
        f"Countries: {comparison['country_code'].nunique()}"
    )

    print(
        f"Years: {comparison['year'].min()}-"
        f"{comparison['year'].max()}"
    )

    print()

    print("UNIT SUMMARY")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print("VALIDATION")

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
    print(f"Detailed output: {OUTPUT_FILE}")
    print(f"Summary output:  {SUMMARY_FILE}")
    print(f"Report output:   {REPORT_FILE}")


if __name__ == "__main__":
    main()
