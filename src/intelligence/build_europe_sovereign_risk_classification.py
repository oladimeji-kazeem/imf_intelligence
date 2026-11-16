
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BRIDGE_FILE = (
    ROOT / "data" / "processed"
    / "europe_sovereign_debt_bridge.csv"
)

DEBT_FILE = (
    ROOT / "data" / "processed"
    / "europe_effective_interest_rate_panel.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_PROFILES = (
    OUTPUT_DIR / "europe_sovereign_risk_classifications.csv"
)

OUTPUT_DETAILS = (
    OUTPUT_DIR / "europe_sovereign_risk_dimension_detail.csv"
)

OUTPUT_REPORT = (
    REPORT_DIR / "europe_sovereign_risk_classification_report.txt"
)

REFERENCE_YEAR = 2024
START_YEAR = 2020
END_YEAR = 2024

EXPECTED_YEARS = list(range(START_YEAR, END_YEAR + 1))
EXPECTED_PERIODS = len(EXPECTED_YEARS)

MINIMUM_PEER_GROUP = 5


# ============================================================
# REQUIRED INPUT SCHEMAS
# ============================================================

BRIDGE_REQUIRED = [
    "country_code",
    "country",
    "year",
    "interest_growth_contribution_pp",
    "observed_debt_change_pp",
    "primary_balance_pct_gdp",
    "F_ASS_pp",
    "ORADJ_pp",
    "YA3_pp",
    "excluded_liability_adjustments_pp",
    "debt_valuation_adjustments_pp",
    "foreign_currency_valuation_pp",
    "other_volume_changes_pp",
    "top_level_sfa_complete",
    "all_oradj_categories_complete",
]

DEBT_REQUIRED = [
    "country_code",
    "year",
    "debt_to_gdp_pct",
]


# ============================================================
# CLASSIFICATION DIMENSIONS
# ============================================================

DIMENSIONS = [
    {
        "name": "debt_burden",
        "source": "debt_to_gdp_pct",
        "unit": "% GDP",
        "period": "2024",
        "higher": True,
        "description": "Higher gross debt-to-GDP represents greater exposure.",
    },
    {
        "name": "primary_balance_pressure_2024",
        "source": "primary_balance_pct_gdp",
        "unit": "% GDP",
        "period": "2024",
        "higher": False,
        "description": "A lower primary balance represents greater fiscal pressure.",
    },
    {
        "name": "debt_trajectory_2020_2024",
        "source": "observed_debt_change_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A larger cumulative increase in debt ratio represents greater exposure.",
    },
    {
        "name": "interest_growth_pressure_2020_2024",
        "source": "interest_growth_contribution_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A larger positive interest-growth contribution increases debt-ratio pressure.",
    },
    {
        "name": "financial_asset_acquisition_2020_2024",
        "source": "F_ASS_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A positive contribution increases gross debt through financial asset acquisition.",
    },
    {
        "name": "oradj_exposure_2020_2024",
        "source": "ORADJ_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A positive ORADJ contribution increases the debt ratio.",
    },
    {
        "name": "statistical_discrepancy_2020_2024",
        "source": "YA3_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A positive statistical discrepancy contributes to higher debt.",
    },
    {
        "name": "excluded_liability_adjustments_2020_2024",
        "source": "excluded_liability_adjustments_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A positive contribution increases the debt ratio.",
    },
    {
        "name": "debt_valuation_adjustments_2020_2024",
        "source": "debt_valuation_adjustments_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A positive valuation contribution increases the debt ratio.",
    },
    {
        "name": "foreign_currency_valuation_2020_2024",
        "source": "foreign_currency_valuation_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A positive foreign-currency valuation contribution increases the debt ratio.",
    },
    {
        "name": "other_volume_changes_2020_2024",
        "source": "other_volume_changes_pp",
        "unit": "Percentage points",
        "period": "2020-2024",
        "higher": True,
        "description": "A positive other-volume contribution increases the debt ratio.",
    },
]


# ============================================================
# GENERAL VALIDATION
# ============================================================

def validate_columns(df, required, label):

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def validate_country_year_keys(df, label):

    duplicates = df.duplicated(
        subset=["country_code", "year"],
        keep=False,
    )

    if duplicates.any():

        sample = (
            df.loc[duplicates, ["country_code", "year"]]
            .sort_values(["country_code", "year"])
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            f"{label} contains duplicate country-year keys.\n"
            f"{sample}"
        )


def load_inputs():

    if not BRIDGE_FILE.exists():
        raise FileNotFoundError(BRIDGE_FILE)

    if not DEBT_FILE.exists():
        raise FileNotFoundError(DEBT_FILE)

    bridge = pd.read_csv(BRIDGE_FILE)
    debt = pd.read_csv(DEBT_FILE)

    validate_columns(
        bridge,
        BRIDGE_REQUIRED,
        "Sovereign debt bridge",
    )

    validate_columns(
        debt,
        DEBT_REQUIRED,
        "Effective interest rate panel",
    )

    for df in (bridge, debt):

        df["country_code"] = (
            df["country_code"].astype(str).str.strip()
        )

        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce",
        )

        if df["year"].isna().any():
            raise ValueError("Invalid year values found.")

        df["year"] = df["year"].astype(int)

    validate_country_year_keys(
        bridge,
        "Sovereign debt bridge",
    )

    validate_country_year_keys(
        debt,
        "Effective interest rate panel",
    )

    bridge = bridge[
        bridge["year"].between(START_YEAR, END_YEAR)
    ].copy()

    debt = debt[
        debt["year"] == REFERENCE_YEAR
    ].copy()

    if bridge.empty:
        raise ValueError("No observations found in the bridge period.")

    if debt.empty:
        raise ValueError("No 2024 debt observations found.")

    return bridge, debt


# ============================================================
# PERIOD AGGREGATION
# ============================================================

def complete_period_sum(group, column):

    period = group[
        group["year"].isin(EXPECTED_YEARS)
    ].copy()

    period = period.sort_values("year")

    values = period[column]

    if len(period) != EXPECTED_PERIODS:
        return np.nan

    if period["year"].nunique() != EXPECTED_PERIODS:
        return np.nan

    if values.isna().any():
        return np.nan

    return float(values.sum())


def count_complete_period_years(group, flag_column):

    period = group[
        group["year"].isin(EXPECTED_YEARS)
    ]

    if flag_column not in period.columns:
        return 0

    return int(
        period[flag_column].fillna(False).astype(bool).sum()
    )


# ============================================================
# BUILD COUNTRY-LEVEL INPUT PROFILES
# ============================================================

def build_country_profiles(bridge, debt):

    country_names = (
        bridge[["country_code", "country"]]
        .drop_duplicates(subset=["country_code"])
    )

    if country_names["country_code"].duplicated().any():
        raise ValueError("Country names are not unique by country code.")

    debt_reference = debt[
        ["country_code", "debt_to_gdp_pct"]
    ].copy()

    if debt_reference["country_code"].duplicated().any():
        raise ValueError("Duplicate 2024 debt observations.")

    profiles = country_names.merge(
        debt_reference,
        on="country_code",
        how="left",
        validate="one_to_one",
    )

    bridge_reference = bridge[
        bridge["year"] == REFERENCE_YEAR
    ][
        ["country_code", "primary_balance_pct_gdp"]
    ].copy()

    if bridge_reference["country_code"].duplicated().any():
        raise ValueError("Duplicate 2024 primary balance observations.")

    profiles = profiles.merge(
        bridge_reference,
        on="country_code",
        how="left",
        validate="one_to_one",
    )

    period_records = []

    for country_code, group in bridge.groupby("country_code"):

        record = {
            "country_code": country_code,
            "bridge_years_available": int(
                group["year"].nunique()
            ),
            "top_level_sfa_years_available":
                count_complete_period_years(
                    group,
                    "top_level_sfa_complete",
                ),
            "detailed_oradj_years_available":
                count_complete_period_years(
                    group,
                    "all_oradj_categories_complete",
                ),
        }

        for column in [
            "observed_debt_change_pp",
            "interest_growth_contribution_pp",
            "F_ASS_pp",
            "ORADJ_pp",
            "YA3_pp",
            "excluded_liability_adjustments_pp",
            "debt_valuation_adjustments_pp",
            "foreign_currency_valuation_pp",
            "other_volume_changes_pp",
        ]:

            record[column] = complete_period_sum(
                group,
                column,
            )

        period_records.append(record)

    period_df = pd.DataFrame(period_records)

    profiles = profiles.merge(
        period_df,
        on="country_code",
        how="left",
        validate="one_to_one",
    )

    if profiles["country_code"].duplicated().any():
        raise ValueError("Duplicate country profiles after merging.")

    return profiles


# ============================================================
# CLASSIFICATION LOGIC
# ============================================================

def classify_exposure(percentile):

    if pd.isna(percentile):
        return "insufficient_data"

    if percentile <= (1 / 3):
        return "lower_relative_exposure"

    if percentile > (2 / 3):
        return "higher_relative_exposure"

    return "moderate_relative_exposure"


def add_classification(
    records,
    country,
    dimension,
    value,
    peer_group,
    higher_value_means_higher_exposure=True,
    unit="",
    reference_period="",
):

    valid = pd.Series(peer_group).dropna()

    if pd.isna(value) or len(valid) < MINIMUM_PEER_GROUP:
        records.append({
            "country_code": country,
            "dimension": dimension,
            "value": value,
            "unit": unit,
            "reference_period": reference_period,
            "peer_count": len(valid),
            "exposure_percentile": np.nan,
            "classification": "insufficient_data",
        })
        return

    # Country must occur exactly once in the peer population.
    if valid.index.duplicated().any():
        raise ValueError(
            f"Duplicate country index in peer group: {dimension}"
        )

    if country not in valid.index:
        raise ValueError(
            f"Target country {country} missing from peer group "
            f"for {dimension}."
        )

    exposure_values = (
        valid
        if higher_value_means_higher_exposure
        else -valid
    )

    percentile = exposure_values.rank(
        pct=True,
        method="average",
    ).loc[country]

    records.append({
        "country_code": country,
        "dimension": dimension,
        "value": float(value),
        "unit": unit,
        "reference_period": reference_period,
        "peer_count": len(valid),
        "exposure_percentile": float(percentile),
        "classification": classify_exposure(percentile),
    })


def build_classifications(profiles):

    records = []

    indexed = profiles.set_index("country_code")

    if indexed.index.duplicated().any():
        raise ValueError(
            "Duplicate country codes in classification profiles."
        )

    for dimension in DIMENSIONS:

        dimension_name = dimension["name"]
        source = dimension["source"]

        if source not in indexed.columns:
            raise ValueError(
                f"Missing classification input: {source}"
            )

        peer_values = indexed[source].dropna()

        for country in indexed.index:

            value = indexed.loc[country, source]

            add_classification(
                records=records,
                country=country,
                dimension=dimension_name,
                value=value,
                peer_group=peer_values,
                higher_value_means_higher_exposure=dimension["higher"],
                unit=dimension["unit"],
                reference_period=dimension["period"],
            )

    details = pd.DataFrame(records)

    expected_rows = len(profiles) * len(DIMENSIONS)

    if len(details) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} dimension observations; "
            f"received {len(details)}."
        )

    if details.duplicated(
        subset=["country_code", "dimension"]
    ).any():
        raise ValueError(
            "Duplicate country-dimension classification records."
        )

    return details


# ============================================================
# ACCOUNTING COVERAGE GRADES
# ============================================================

def build_data_confidence(profiles):

    results = []

    grade_order = {
        "A_complete": 1,
        "B_high": 2,
        "C_partial": 3,
        "D_limited": 4,
        "D_unavailable": 5,
    }

    def coverage_grade(years):

        if years == 5:
            return "A_complete"

        if years == 4:
            return "B_high"

        if years == 3:
            return "C_partial"

        if years > 0:
            return "D_limited"

        return "D_unavailable"

    for _, row in profiles.iterrows():

        top_years = int(
            row["top_level_sfa_years_available"]
        )

        detail_years = int(
            row["detailed_oradj_years_available"]
        )

        top_grade = coverage_grade(top_years)
        detail_grade = coverage_grade(detail_years)

        overall_grade = max(
            [top_grade, detail_grade],
            key=lambda grade: grade_order[grade],
        )

        results.append({
            "country_code": row["country_code"],
            "debt_stock_available_2024":
                pd.notna(row["debt_to_gdp_pct"]),
            "bridge_years_available":
                int(row["bridge_years_available"]),
            "top_level_sfa_years_available": top_years,
            "detailed_oradj_years_available": detail_years,
            "top_level_sfa_coverage_grade": top_grade,
            "detailed_oradj_coverage_grade": detail_grade,
            "overall_accounting_coverage_grade": overall_grade,
        })

    return pd.DataFrame(results)


# ============================================================
# BUILD FINAL COUNTRY CLASSIFICATION TABLE
# ============================================================

def build_final_profiles(profiles, details, coverage):

    final = profiles.copy()

    for dimension in DIMENSIONS:

        dimension_name = dimension["name"]

        subset = details[
            details["dimension"] == dimension_name
        ][
            [
                "country_code",
                "value",
                "exposure_percentile",
                "classification",
                "peer_count",
            ]
        ].copy()

        subset = subset.rename(columns={
            "value": f"{dimension_name}_value",
            "exposure_percentile":
                f"{dimension_name}_percentile",
            "classification":
                f"{dimension_name}_classification",
            "peer_count":
                f"{dimension_name}_peer_count",
        })

        if subset["country_code"].duplicated().any():
            raise ValueError(
                f"Duplicate countries in {dimension_name}."
            )

        final = final.merge(
            subset,
            on="country_code",
            how="left",
            validate="one_to_one",
        )

    final = final.merge(
        coverage,
        on="country_code",
        how="left",
        validate="one_to_one",
    )

    if final["country_code"].duplicated().any():
        raise ValueError(
            "Duplicate country profiles in final output."
        )

    return final


# ============================================================
# REPORT GENERATION
# ============================================================

def format_value(value, decimals=2):

    if pd.isna(value):
        return "N/A"

    return f"{value:+.{decimals}f}"


def generate_report(profiles, details, coverage):

    lines = []

    def add(text=""):
        lines.append(str(text))

    add("=" * 78)
    add("EUROPEAN SOVEREIGN RISK CLASSIFICATION REPORT")
    add("=" * 78)

    add(f"Countries: {len(profiles)}")
    add(f"Reference year: {REFERENCE_YEAR}")
    add(f"Historical period: {START_YEAR}-{END_YEAR}")

    add()
    add("CLASSIFICATION METHODOLOGY")
    add("-" * 78)
    add(
        "Classifications are peer-relative terciles among available countries."
    )
    add(
        "They are not absolute risk thresholds, credit ratings "
        "or default probabilities."
    )
    add(
        "Higher debt, debt increases and positive debt-increasing "
        "contributions represent higher exposure."
    )
    add(
        "For primary balance, a lower value represents greater "
        "fiscal pressure."
    )
    add(
        "Accounting adjustments are exposures, not causal estimates."
    )
    add(
        "Five-year cumulative accounting components are classified "
        "only when all five annual observations are available."
    )

    add()
    add("COUNTRY DEBT BURDEN")
    add("-" * 78)

    debt_sorted = profiles.sort_values(
        "debt_to_gdp_pct",
        ascending=False,
    )

    for _, row in debt_sorted.iterrows():

        add(
            f"{row['country_code']}: "
            f"debt/GDP={row['debt_to_gdp_pct']:.2f}%"
        )

    add()
    add("COUNTRY DEBT TRAJECTORY")
    add("-" * 78)

    trajectory = profiles.sort_values(
        "observed_debt_change_pp",
        ascending=False,
    )

    for _, row in trajectory.iterrows():

        add(
            f"{row['country_code']}: "
            f"debt change="
            f"{format_value(row['observed_debt_change_pp'], 4)} pp"
        )

    add()
    add("PRIMARY BALANCE PRESSURE — 2024")
    add("-" * 78)

    primary = profiles.sort_values(
        "primary_balance_pct_gdp",
        ascending=True,
    )

    for _, row in primary.iterrows():

        value = row["primary_balance_pct_gdp"]

        if pd.isna(value):
            display = "N/A"
        else:
            display = f"{value:+.4f}% GDP"

        add(f"{row['country_code']}: primary balance={display}")

    add()
    add("ACCOUNTING COVERAGE")
    add("-" * 78)

    coverage_sorted = coverage.sort_values(
        ["overall_accounting_coverage_grade", "country_code"]
    )

    for _, row in coverage_sorted.iterrows():

        add(
            f"{row['country_code']}: "
            f"{row['overall_accounting_coverage_grade']}; "
            f"top-level SFA="
            f"{row['top_level_sfa_years_available']}/5 "
            f"({row['top_level_sfa_coverage_grade']}); "
            f"detailed ORADJ="
            f"{row['detailed_oradj_years_available']}/5 "
            f"({row['detailed_oradj_coverage_grade']})"
        )

    add()
    add("CLASSIFICATION COUNTS")
    add("-" * 78)

    for dimension in DIMENSIONS:

        name = dimension["name"]

        counts = (
            details.loc[
                details["dimension"] == name,
                "classification",
            ]
            .value_counts()
        )

        add(f"{name}:")

        for classification, count in counts.items():

            add(f"  {classification}: {count}")

    add()
    add("VALIDATION")
    add("-" * 78)

    add(f"Country profiles: {len(profiles)}")
    add(
        "Dimension observations: "
        f"{len(details)}"
    )
    add(
        "Unique countries in profiles: "
        f"{profiles['country_code'].nunique()}"
    )
    add(
        "Duplicate country profiles: "
        f"{profiles['country_code'].duplicated().sum()}"
    )
    add(
        "Missing 2024 debt ratios: "
        f"{profiles['debt_to_gdp_pct'].isna().sum()}"
    )
    add(
        "Missing country classifications: "
        f"{details['classification'].isna().sum()}"
    )
    add(
        "Duplicate country-dimension records: "
        f"{details.duplicated(['country_code', 'dimension']).sum()}"
    )

    add()
    add("PEER GROUP VALIDATION")
    add("-" * 78)

    for dimension in DIMENSIONS:

        name = dimension["name"]

        subset = details[
            details["dimension"] == name
        ]

        peer_counts = sorted(
            subset["peer_count"].dropna().unique().tolist()
        )

        add(f"{name}: peer counts={peer_counts}")

    add()
    add("INTERPRETATION LIMITATIONS")
    add("-" * 78)

    limitations = [
        "Peer-relative terciles do not establish absolute risk.",
        "The peer group contains 17 European countries, not all European sovereigns.",
        "Debt burden uses the 2024 debt-to-GDP observation.",
        "Debt trajectory uses five annual debt changes from 2020-2024.",
        "Primary balance is not cyclically adjusted.",
        "Interest-growth contribution uses the opening-debt-implied interest-rate proxy.",
        "Financial-asset acquisition can increase gross debt while adding assets.",
        "SFA and ORADJ components are accounting adjustments, not causal estimates.",
        "Incomplete five-year component histories are not treated as complete cumulative totals.",
        "Accounting coverage grades measure coverage, not source accuracy.",
        "A missing component is not interpreted as zero.",
        "No composite sovereign risk score or default probability is calculated.",
    ]

    for i, item in enumerate(limitations, start=1):
        add(f"{i}. {item}")

    return "\n".join(lines)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EUROPEAN SOVEREIGN RISK CLASSIFICATION ENGINE")
    print("=" * 78)

    bridge, debt = load_inputs()

    print("Input validation passed.")

    print(f"Bridge observations: {len(bridge)}")
    print(f"Bridge countries: {bridge['country_code'].nunique()}")
    print(
        f"Bridge period: "
        f"{bridge['year'].min()}-{bridge['year'].max()}"
    )

    profiles = build_country_profiles(
        bridge,
        debt,
    )

    print(
        f"Country input profiles built: {len(profiles)}"
    )

    details = build_classifications(profiles)

    print(
        f"Dimension classifications built: {len(details)}"
    )

    coverage = build_data_confidence(profiles)

    final_profiles = build_final_profiles(
        profiles,
        details,
        coverage,
    )

    # --------------------------------------------------------
    # Additional integrity checks
    # --------------------------------------------------------

    if final_profiles["country_code"].duplicated().any():
        raise ValueError("Final country profiles contain duplicates.")

    if details.duplicated(
        ["country_code", "dimension"]
    ).any():
        raise ValueError("Duplicate country-dimension classifications.")

    expected_dimensions = len(DIMENSIONS)

    dimension_counts = details.groupby(
        "country_code"
    )["dimension"].nunique()

    if not (dimension_counts == expected_dimensions).all():
        raise ValueError(
            "One or more countries have missing classification dimensions."
        )

    # The broad dimensions should use the full available peer group.
    broad_dimensions = [
        "debt_burden",
        "primary_balance_pressure_2024",
        "debt_trajectory_2020_2024",
        "interest_growth_pressure_2020_2024",
    ]

    for dimension in broad_dimensions:

        subset = details[
            details["dimension"] == dimension
        ]

        if len(subset) != len(final_profiles):
            raise ValueError(
                f"Unexpected classification record count: {dimension}"
            )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    final_profiles.to_csv(
        OUTPUT_PROFILES,
        index=False,
    )

    details.to_csv(
        OUTPUT_DETAILS,
        index=False,
    )

    report = generate_report(
        profiles,
        details,
        coverage,
    )

    OUTPUT_REPORT.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("OUTPUTS CREATED")
    print("-" * 78)
    print(OUTPUT_PROFILES)
    print(OUTPUT_DETAILS)
    print(OUTPUT_REPORT)

    print()
    print("VALIDATION PASSED")
    print(f"Country profiles: {len(final_profiles)}")
    print(f"Dimension observations: {len(details)}")
    print(
        "Duplicate country profiles: "
        f"{final_profiles['country_code'].duplicated().sum()}"
    )
    print(
        "Duplicate country-dimension records: "
        f"{details.duplicated(['country_code', 'dimension']).sum()}"
    )

    print()
    print("ACCOUNTING COVERAGE CHECK")
    print("-" * 78)

    for code in ["NL", "RO"]:

        row = coverage[
            coverage["country_code"] == code
        ]

        if not row.empty:

            item = row.iloc[0]

            print(
                f"{code}: overall="
                f"{item['overall_accounting_coverage_grade']}; "
                f"top-level="
                f"{item['top_level_sfa_years_available']}/5; "
                f"detailed="
                f"{item['detailed_oradj_years_available']}/5"
            )

    print()
    print("CLASSIFICATION REPORT")
    print("=" * 78)
    print(report)


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("ERROR: SOVEREIGN RISK CLASSIFICATION FAILED")
        print("=" * 78)

        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

        sys.exit(1)
