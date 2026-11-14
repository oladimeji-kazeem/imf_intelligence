
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GDP_FILE = ROOT / "data/processed/europe_eurostat_edp_gdp_2019_2024.csv"
INTEREST_FILE = ROOT / "data/processed/europe_effective_interest_rate_panel.csv"
PRIMARY_FILE = ROOT / "data/processed/europe_eurostat_primary_balance.csv"

OUTPUT_FILE = ROOT / "data/processed/europe_interest_growth_contribution.csv"
REPORT_FILE = ROOT / "outputs/reports/europe_interest_growth_contribution_report.txt"

COUNTRIES = [
    "BE", "CZ", "DK", "DE", "IE", "EL", "ES", "FR", "IT",
    "HU", "NL", "AT", "PL", "PT", "RO", "FI", "SE"
]

YEARS = list(range(2020, 2025))
EXPECTED = len(COUNTRIES) * len(YEARS)

TOLERANCE = 1e-8


def main():

    print("=" * 75)
    print("EUROSTAT INTEREST-GROWTH CONTRIBUTION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD INPUTS
    # --------------------------------------------------------

    for path in [GDP_FILE, INTEREST_FILE, PRIMARY_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    gdp = pd.read_csv(GDP_FILE)
    interest = pd.read_csv(INTEREST_FILE)
    primary = pd.read_csv(PRIMARY_FILE)

    # --------------------------------------------------------
    # 2. SELECT REQUIRED FIELDS
    # --------------------------------------------------------

    gdp = gdp[
        ["country_code", "year", "nominal_gdp"]
    ].copy()

    interest = interest[
        [
            "country_code",
            "country",
            "year",
            "interest_expenditure",
            "opening_debt",
            "closing_debt",
            "effective_interest_rate_pct",
        ]
    ].copy()

    primary = primary[
        [
            "country_code",
            "year",
            "primary_balance",
            "primary_balance_pct_gdp",
        ]
    ].copy()

    # --------------------------------------------------------
    # 3. CHECK UNIQUE COUNTRY-YEAR RECORDS
    # --------------------------------------------------------

    keys = ["country_code", "year"]

    duplicate_counts = {
        "GDP": int(gdp.duplicated(keys).sum()),
        "Interest": int(interest.duplicated(keys).sum()),
        "Primary balance": int(primary.duplicated(keys).sum()),
    }

    if any(duplicate_counts.values()):
        raise ValueError(
            f"Duplicate country-year records: {duplicate_counts}"
        )

    # --------------------------------------------------------
    # 4. BUILD GDP LAG
    # --------------------------------------------------------

    gdp = gdp.sort_values(keys)

    gdp["opening_gdp"] = (
        gdp.groupby("country_code")["nominal_gdp"].shift(1)
    )

    gdp = gdp.rename(
        columns={"nominal_gdp": "closing_gdp"}
    )

    # Retain only calculation years.
    gdp = gdp[gdp["year"].isin(YEARS)].copy()

    # --------------------------------------------------------
    # 5. MERGE DATASETS
    # --------------------------------------------------------

    panel = interest.merge(
        gdp,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator="gdp_merge",
    )

    panel = panel.merge(
        primary,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator="primary_merge",
    )

    # --------------------------------------------------------
    # 6. CHECK COVERAGE
    # --------------------------------------------------------

    expected_pairs = {
        (country, year)
        for country in COUNTRIES
        for year in YEARS
    }

    actual_pairs = set(
        zip(panel["country_code"], panel["year"])
    )

    missing_pairs = sorted(expected_pairs - actual_pairs)
    unexpected_pairs = sorted(actual_pairs - expected_pairs)

    unmatched_gdp = int((panel["gdp_merge"] != "both").sum())
    unmatched_primary = int(
        (panel["primary_merge"] != "both").sum()
    )

    required = [
        "interest_expenditure",
        "opening_debt",
        "closing_debt",
        "opening_gdp",
        "closing_gdp",
        "primary_balance",
        "primary_balance_pct_gdp",
    ]

    missing_values = int(panel[required].isna().sum().sum())

    # --------------------------------------------------------
    # 7. CALCULATE NOMINAL GDP GROWTH
    # --------------------------------------------------------

    panel["nominal_gdp_growth"] = (
        panel["closing_gdp"] / panel["opening_gdp"]
    ) - 1

    panel["nominal_gdp_growth_pct"] = (
        panel["nominal_gdp_growth"] * 100
    )

    # --------------------------------------------------------
    # 8. CALCULATE OPENING-DEBT-BASED EFFECTIVE RATE
    # --------------------------------------------------------

    panel["opening_debt_effective_rate"] = (
        panel["interest_expenditure"] / panel["opening_debt"]
    )

    panel["opening_debt_effective_rate_pct"] = (
        panel["opening_debt_effective_rate"] * 100
    )

    # --------------------------------------------------------
    # 9. CALCULATE OPENING DEBT RATIO
    # --------------------------------------------------------

    panel["opening_debt_to_gdp"] = (
        panel["opening_debt"] / panel["opening_gdp"]
    )

    panel["opening_debt_to_gdp_pct"] = (
        panel["opening_debt_to_gdp"] * 100
    )

    # --------------------------------------------------------
    # 10. CALCULATE CLOSING DEBT RATIO
    # --------------------------------------------------------

    panel["closing_debt_to_gdp"] = (
        panel["closing_debt"] / panel["closing_gdp"]
    )

    panel["closing_debt_to_gdp_pct"] = (
        panel["closing_debt_to_gdp"] * 100
    )

    # Observed debt ratio change in percentage points.
    panel["observed_debt_change_pp"] = (
        panel["closing_debt_to_gdp_pct"]
        - panel["opening_debt_to_gdp_pct"]
    )

    # --------------------------------------------------------
    # 11. CALCULATE INTEREST-GROWTH CONTRIBUTION
    # --------------------------------------------------------

    # IGC = ((r - g) / (1 + g)) * opening debt ratio.
    #
    # r and g are decimal rates.
    # Debt ratio is a decimal.
    # Result is converted to percentage points of GDP.

    panel["interest_growth_contribution"] = (
        (
            panel["opening_debt_effective_rate"]
            - panel["nominal_gdp_growth"]
        )
        / (1 + panel["nominal_gdp_growth"])
    ) * panel["opening_debt_to_gdp"]

    panel["interest_growth_contribution_pp"] = (
        panel["interest_growth_contribution"] * 100
    )

    # --------------------------------------------------------
    # 12. CALCULATE IMPLIED STOCK-FLOW ADJUSTMENT
    # --------------------------------------------------------

    # Debt change = IGC - primary balance + SFA
    #
    # Therefore:
    # Implied SFA = debt change - IGC + primary balance.

    panel["implied_sfa_pp"] = (
        panel["observed_debt_change_pp"]
        - panel["interest_growth_contribution_pp"]
        + panel["primary_balance_pct_gdp"]
    )

    # --------------------------------------------------------
    # 13. RECONSTRUCT DEBT CHANGE
    # --------------------------------------------------------

    panel["reconstructed_debt_change_pp"] = (
        panel["interest_growth_contribution_pp"]
        - panel["primary_balance_pct_gdp"]
        + panel["implied_sfa_pp"]
    )

    panel["debt_dynamics_reconciliation_error_pp"] = (
        panel["observed_debt_change_pp"]
        - panel["reconstructed_debt_change_pp"]
    )

    max_reconciliation_error = float(
        panel["debt_dynamics_reconciliation_error_pp"]
        .abs()
        .max()
    )

    # --------------------------------------------------------
    # 14. ECONOMIC VALIDATION
    # --------------------------------------------------------

    invalid_opening_debt = int(
        (panel["opening_debt"] <= 0).sum()
    )

    invalid_closing_debt = int(
        (panel["closing_debt"] <= 0).sum()
    )

    invalid_opening_gdp = int(
        (panel["opening_gdp"] <= 0).sum()
    )

    invalid_closing_gdp = int(
        (panel["closing_gdp"] <= 0).sum()
    )

    negative_interest = int(
        (panel["interest_expenditure"] < 0).sum()
    )

    invalid_growth_denominator = int(
        ((1 + panel["nominal_gdp_growth"]) <= 0).sum()
    )

    # --------------------------------------------------------
    # 15. VALIDATION CHECKS
    # --------------------------------------------------------

    checks = {
        "Expected observation count": len(panel) == EXPECTED,
        "Complete country-year coverage": len(missing_pairs) == 0,
        "No unexpected country-years": len(unexpected_pairs) == 0,
        "No unmatched GDP records": unmatched_gdp == 0,
        "No unmatched primary-balance records": unmatched_primary == 0,
        "No missing required values": missing_values == 0,
        "Positive opening debt": invalid_opening_debt == 0,
        "Positive closing debt": invalid_closing_debt == 0,
        "Positive opening GDP": invalid_opening_gdp == 0,
        "Positive closing GDP": invalid_closing_gdp == 0,
        "Interest expenditure non-negative": negative_interest == 0,
        "Valid GDP growth denominator": invalid_growth_denominator == 0,
        "Debt dynamics identity reconciles": (
            max_reconciliation_error <= TOLERANCE
        ),
    }

    # --------------------------------------------------------
    # 16. OUTPUT DATASET
    # --------------------------------------------------------

    output_columns = [
        "country_code",
        "country",
        "year",
        "interest_expenditure",
        "opening_debt",
        "closing_debt",
        "opening_gdp",
        "closing_gdp",
        "nominal_gdp_growth_pct",
        "opening_debt_effective_rate_pct",
        "effective_interest_rate_pct",
        "opening_debt_to_gdp_pct",
        "closing_debt_to_gdp_pct",
        "observed_debt_change_pp",
        "primary_balance",
        "primary_balance_pct_gdp",
        "interest_growth_contribution_pp",
        "implied_sfa_pp",
        "reconstructed_debt_change_pp",
        "debt_dynamics_reconciliation_error_pp",
    ]

    panel = panel.sort_values(keys)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

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
            cumulative_debt_change_pp=(
                "observed_debt_change_pp",
                "sum",
            ),
            cumulative_interest_growth_pp=(
                "interest_growth_contribution_pp",
                "sum",
            ),
            cumulative_primary_contribution_pp=(
                "primary_balance_pct_gdp",
                lambda x: -x.sum(),
            ),
            cumulative_implied_sfa_pp=(
                "implied_sfa_pp",
                "sum",
            ),
            mean_nominal_gdp_growth_pct=(
                "nominal_gdp_growth_pct",
                "mean",
            ),
            mean_opening_debt_rate_pct=(
                "opening_debt_effective_rate_pct",
                "mean",
            ),
            observations=("year", "count"),
        )
        .sort_values("cumulative_debt_change_pp", ascending=False)
    )

    # --------------------------------------------------------
    # 18. GENERATE REPORT
    # --------------------------------------------------------

    report = [
        "EUROSTAT INTEREST-GROWTH CONTRIBUTION REPORT",
        "=" * 75,
        "",
        "PERIOD: 2020-2024",
        "COUNTRIES: 17",
        f"EXPECTED OBSERVATIONS: {EXPECTED}",
        f"RETRIEVED OBSERVATIONS: {len(panel)}",
        "",
        "FORMULA",
        "IGC = ((r - g) / (1 + g)) * opening debt ratio",
        "r = interest expenditure / opening debt",
        "g = nominal GDP growth",
        "",
        "DEBT DYNAMICS",
        "Debt change = IGC - primary balance + implied SFA",
        "",
        "COVERAGE",
        f"Missing country-years: {len(missing_pairs)}",
        f"Unexpected country-years: {len(unexpected_pairs)}",
        f"Unmatched GDP records: {unmatched_gdp}",
        f"Unmatched primary-balance records: {unmatched_primary}",
        f"Missing required values: {missing_values}",
        "",
        "ECONOMIC VALIDATION",
        f"Invalid opening debt: {invalid_opening_debt}",
        f"Invalid closing debt: {invalid_closing_debt}",
        f"Invalid opening GDP: {invalid_opening_gdp}",
        f"Invalid closing GDP: {invalid_closing_gdp}",
        f"Negative interest observations: {negative_interest}",
        f"Invalid growth denominators: {invalid_growth_denominator}",
        f"Maximum reconciliation error: {max_reconciliation_error:.12f}",
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
        summary.to_string(index=False),
        "",
        "METHODOLOGICAL NOTES",
        "1. Interest rate uses opening debt, not average debt.",
        "2. Nominal GDP growth uses the Eurostat 2019-2024 GDP series.",
        "3. Primary balance is the Eurostat-derived observed balance.",
        "4. Implied SFA is an accounting residual, not a causal estimate.",
        "5. Direct SFA reconciliation is a separate validation step.",
        "6. Annual contributions are in percentage points of GDP.",
        "",
        f"OUTPUT FILE: {OUTPUT_FILE}",
    ])

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n".join(report))
    print("\nProcessed dataset:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)

    if not overall_pass:
        raise SystemExit(
            "Interest-growth contribution validation failed."
        )


if __name__ == "__main__":
    main()
