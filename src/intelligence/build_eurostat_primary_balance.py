from pathlib import Path

import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

B9_FILE = ROOT / "data/processed/europe_eurostat_b9.csv"
GDP_FILE = ROOT / "data/processed/europe_eurostat_edp_gdp_2019_2024.csv"
INTEREST_FILE = ROOT / "data/processed/europe_effective_interest_rate_panel.csv"

OUTPUT_FILE = ROOT / "data/processed/europe_eurostat_primary_balance.csv"
REPORT_FILE = ROOT / "outputs/reports/europe_eurostat_primary_balance_report.txt"

COUNTRIES = [
    "BE", "CZ", "DK", "DE", "IE", "EL", "ES", "FR", "IT",
    "HU", "NL", "AT", "PL", "PT", "RO", "FI", "SE"
]

YEARS = list(range(2020, 2025))

EXPECTED = len(COUNTRIES) * len(YEARS)

GDP_MATCH_TOLERANCE = 0.01
INTEREST_GDP_TOLERANCE_PP = 0.02
IDENTITY_TOLERANCE = 1e-10


# ============================================================
# MAIN PROCESS
# ============================================================

def main():

    print("=" * 72)
    print("EUROSTAT PRIMARY BALANCE CALCULATION")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. CHECK INPUT FILES
    # --------------------------------------------------------

    input_files = [B9_FILE, GDP_FILE, INTEREST_FILE]

    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    # --------------------------------------------------------
    # 2. LOAD INPUT DATA
    # --------------------------------------------------------

    b9 = pd.read_csv(B9_FILE)
    gdp = pd.read_csv(GDP_FILE)
    interest = pd.read_csv(INTEREST_FILE)

    print("\nINPUT DATA SHAPES")
    print("B9:", b9.shape)
    print("GDP:", gdp.shape)
    print("Interest:", interest.shape)

    # --------------------------------------------------------
    # 3. VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = {
        "B9": {
            "country_code",
            "year",
            "b9_net_lending",
        },
        "GDP": {
            "country_code",
            "year",
            "nominal_gdp",
        },
        "Interest": {
            "country_code",
            "country",
            "year",
            "interest_expenditure",
            "nominal_gdp",
            "interest_to_gdp_pct",
            "opening_debt",
            "closing_debt",
            "effective_interest_rate_pct",
        },
    }

    datasets = {
        "B9": b9,
        "GDP": gdp,
        "Interest": interest,
    }

    for name, required in required_columns.items():

        missing = required - set(datasets[name].columns)

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {sorted(missing)}"
            )

    # --------------------------------------------------------
    # 4. SELECT COMMON PERIOD AND REQUIRED FIELDS
    # --------------------------------------------------------

    b9 = b9[
        b9["year"].isin(YEARS)
    ][
        ["country_code", "year", "b9_net_lending"]
    ].copy()

    gdp = gdp[
        gdp["year"].isin(YEARS)
    ][
        ["country_code", "year", "nominal_gdp"]
    ].copy()

    interest = interest[
        interest["year"].isin(YEARS)
    ][
        [
            "country_code",
            "country",
            "year",
            "interest_expenditure",
            "nominal_gdp",
            "interest_to_gdp_pct",
            "opening_debt",
            "closing_debt",
            "effective_interest_rate_pct",
        ]
    ].copy()

    # Preserve the GDP value from the interest panel.
    interest = interest.rename(
        columns={
            "nominal_gdp": "interest_panel_gdp"
        }
    )

    keys = ["country_code", "year"]

    # --------------------------------------------------------
    # 5. VALIDATE INPUT UNIQUENESS
    # --------------------------------------------------------

    duplicate_b9 = int(b9.duplicated(keys).sum())
    duplicate_gdp = int(gdp.duplicated(keys).sum())
    duplicate_interest = int(interest.duplicated(keys).sum())

    if any([
        duplicate_b9,
        duplicate_gdp,
        duplicate_interest,
    ]):
        raise ValueError(
            "Duplicate country-year records detected. "
            f"B9={duplicate_b9}, "
            f"GDP={duplicate_gdp}, "
            f"Interest={duplicate_interest}"
        )

    # --------------------------------------------------------
    # 6. MERGE THE THREE DATASETS
    # --------------------------------------------------------

    panel = b9.merge(
        interest,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator="b9_interest_merge",
    )

    panel = panel.merge(
        gdp,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator="gdp_merge",
    )

    # --------------------------------------------------------
    # 7. VALIDATE COUNTRY-YEAR COVERAGE
    # --------------------------------------------------------

    expected_pairs = {
        (country, year)
        for country in COUNTRIES
        for year in YEARS
    }

    actual_pairs = set(
        zip(panel["country_code"], panel["year"])
    )

    missing_pairs = sorted(
        expected_pairs - actual_pairs
    )

    unexpected_pairs = sorted(
        actual_pairs - expected_pairs
    )

    unmatched_b9_interest = int(
        (panel["b9_interest_merge"] != "both").sum()
    )

    unmatched_gdp = int(
        (panel["gdp_merge"] != "both").sum()
    )

    # --------------------------------------------------------
    # 8. VALIDATE REQUIRED VALUES
    # --------------------------------------------------------

    required_values = [
        "b9_net_lending",
        "interest_expenditure",
        "interest_panel_gdp",
        "nominal_gdp",
        "interest_to_gdp_pct",
        "opening_debt",
        "closing_debt",
        "effective_interest_rate_pct",
    ]

    missing_values_by_column = (
        panel[required_values].isna().sum()
    )

    total_missing_values = int(
        missing_values_by_column.sum()
    )

    # --------------------------------------------------------
    # 9. INDEPENDENT GDP RECONCILIATION
    # --------------------------------------------------------

    panel["gdp_source_difference"] = (
        panel["nominal_gdp"]
        - panel["interest_panel_gdp"]
    ).abs()

    max_gdp_difference = float(
        panel["gdp_source_difference"].max()
    )

    # --------------------------------------------------------
    # 10. CALCULATE INTEREST-TO-GDP RATIO
    # --------------------------------------------------------

    panel["calculated_interest_to_gdp_pct"] = (
        panel["interest_expenditure"]
        / panel["nominal_gdp"]
        * 100
    )

    panel["interest_gdp_difference_pp"] = (
        panel["calculated_interest_to_gdp_pct"]
        - panel["interest_to_gdp_pct"]
    )

    max_interest_ratio_difference = float(
        panel["interest_gdp_difference_pp"].abs().max()
    )

    # --------------------------------------------------------
    # 11. CALCULATE PRIMARY BALANCE
    # --------------------------------------------------------

    # Eurostat B9:
    # Positive = net lending.
    # Negative = net borrowing.
    #
    # D41PAY:
    # Interest expenditure, expected positive.
    #
    # Primary balance = B9 + interest expenditure.

    panel["primary_balance"] = (
        panel["b9_net_lending"]
        + panel["interest_expenditure"]
    )

    # --------------------------------------------------------
    # 12. EXPRESS BALANCES AS PERCENTAGE OF GDP
    # --------------------------------------------------------

    panel["overall_balance_pct_gdp"] = (
        panel["b9_net_lending"]
        / panel["nominal_gdp"]
        * 100
    )

    panel["primary_balance_pct_gdp"] = (
        panel["primary_balance"]
        / panel["nominal_gdp"]
        * 100
    )

    # --------------------------------------------------------
    # 13. ACCOUNTING IDENTITY RECONCILIATION
    # --------------------------------------------------------

    panel["primary_balance_identity_difference"] = (
        panel["primary_balance_pct_gdp"]
        - (
            panel["overall_balance_pct_gdp"]
            + panel["calculated_interest_to_gdp_pct"]
        )
    )

    max_identity_error = float(
        panel["primary_balance_identity_difference"].abs().max()
    )

    # --------------------------------------------------------
    # 14. ECONOMIC AND SIGN CHECKS
    # --------------------------------------------------------

    negative_interest = int(
        (panel["interest_expenditure"] < 0).sum()
    )

    non_positive_gdp = int(
        (panel["nominal_gdp"] <= 0).sum()
    )

    non_positive_debt = int(
        (
            (panel["opening_debt"] <= 0)
            | (panel["closing_debt"] <= 0)
        ).sum()
    )

    negative_effective_rates = int(
        (panel["effective_interest_rate_pct"] < 0).sum()
    )

    # --------------------------------------------------------
    # 15. VALIDATION CHECKS
    # --------------------------------------------------------

    checks = {
        "Expected observation count": (
            len(panel) == EXPECTED
        ),

        "Complete country-year coverage": (
            len(missing_pairs) == 0
        ),

        "No unexpected country-years": (
            len(unexpected_pairs) == 0
        ),

        "No unmatched B9-interest records": (
            unmatched_b9_interest == 0
        ),

        "No unmatched GDP records": (
            unmatched_gdp == 0
        ),

        "No missing required values": (
            total_missing_values == 0
        ),

        "GDP matches independent extraction": (
            max_gdp_difference <= GDP_MATCH_TOLERANCE
        ),

        "Interest expenditure non-negative": (
            negative_interest == 0
        ),

        "GDP positive": (
            non_positive_gdp == 0
        ),

        "Debt stocks positive": (
            non_positive_debt == 0
        ),

        "Effective interest rates non-negative": (
            negative_effective_rates == 0
        ),

        "Interest/GDP ratio reconciles": (
            max_interest_ratio_difference
            <= INTEREST_GDP_TOLERANCE_PP
        ),

        "Primary balance identity reconciles": (
            max_identity_error < IDENTITY_TOLERANCE
        ),
    }

    # --------------------------------------------------------
    # 16. BUILD OUTPUT DATASET
    # --------------------------------------------------------

    output_columns = [
        "country_code",
        "country",
        "year",
        "b9_net_lending",
        "interest_expenditure",
        "primary_balance",
        "nominal_gdp",
        "interest_panel_gdp",
        "gdp_source_difference",
        "overall_balance_pct_gdp",
        "interest_to_gdp_pct",
        "calculated_interest_to_gdp_pct",
        "interest_gdp_difference_pp",
        "primary_balance_pct_gdp",
        "opening_debt",
        "closing_debt",
        "effective_interest_rate_pct",
        "primary_balance_identity_difference",
    ]

    panel = panel.sort_values(keys)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8f",
    )

    # --------------------------------------------------------
    # 17. COUNTRY SUMMARY
    # --------------------------------------------------------

    summary = (
        panel.groupby(
            ["country_code", "country"],
            as_index=False,
        )
        .agg(
            mean_primary_balance_pct_gdp=(
                "primary_balance_pct_gdp",
                "mean",
            ),
            mean_overall_balance_pct_gdp=(
                "overall_balance_pct_gdp",
                "mean",
            ),
            mean_interest_to_gdp_pct=(
                "calculated_interest_to_gdp_pct",
                "mean",
            ),
            observations=(
                "year",
                "count",
            ),
        )
        .sort_values("mean_primary_balance_pct_gdp")
    )

    # --------------------------------------------------------
    # 18. GENERATE REPORT
    # --------------------------------------------------------

    report = [
        "EUROSTAT PRIMARY BALANCE CALCULATION REPORT",
        "=" * 75,
        "",
        "SOURCE DATASETS",
        "B9: gov_10dd_edpt1",
        "Interest: gov_10a_main, D41PAY",
        "GDP: gov_10dd_edpt1, B1GQ, S1",
        "",
        "CALCULATION",
        "Primary balance = B9 + D41PAY",
        "Period: 2020-2024",
        "Countries: 17",
        "",
        "DATA COVERAGE",
        f"Expected observations: {EXPECTED}",
        f"Retrieved observations: {len(panel)}",
        f"Missing country-years: {len(missing_pairs)}",
        f"Unexpected country-years: {len(unexpected_pairs)}",
        f"Unmatched B9-interest records: {unmatched_b9_interest}",
        f"Unmatched GDP records: {unmatched_gdp}",
        f"Missing required values: {total_missing_values}",
        "",
        "DATA RECONCILIATION",
        f"Maximum GDP source difference: {max_gdp_difference:.10f}",
        f"Maximum interest/GDP difference (pp): "
        f"{max_interest_ratio_difference:.10f}",
        f"Maximum primary balance identity error: "
        f"{max_identity_error:.12f}",
        "",
        "ECONOMIC VALIDATION",
        f"Negative interest observations: {negative_interest}",
        f"Non-positive GDP observations: {non_positive_gdp}",
        f"Non-positive debt observations: {non_positive_debt}",
        f"Negative effective interest rates: {negative_effective_rates}",
        "",
        "MISSING COUNTRY-YEARS",
        str(missing_pairs if missing_pairs else "None"),
        "",
        "VALIDATION RESULTS",
    ]

    for name, passed in checks.items():
        report.append(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    overall_pass = all(checks.values())

    report.extend([
        "",
        "OVERALL RESULT: "
        + ("PASS" if overall_pass else "INVESTIGATION REQUIRED"),
        "",
        "COUNTRY SUMMARY",
        "Mean primary balance as percentage of GDP, 2020-2024",
        "",
        summary.to_string(index=False),
        "",
        "INTERPRETATION",
        "Positive primary balance = primary surplus.",
        "Negative primary balance = primary deficit.",
        "This is an observed, non-cyclically-adjusted balance.",
        "It is not a structural primary balance.",
        "",
        f"Processed dataset: {OUTPUT_FILE}",
    ])

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 19. DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n".join(report))

    print("\nOUTPUT FILES")
    print("Processed dataset:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)

    if not overall_pass:
        raise SystemExit(
            "\nPrimary balance validation failed. "
            "Inspect the report before proceeding."
        )


if __name__ == "__main__":
    main()
