from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "europe_eurostat_sfa_reconciliation.csv"
)

HIERARCHY_FILE = (
    ROOT / "outputs" / "reports"
    / "europe_sfa_accounting_hierarchy.csv"
)

COVERAGE_FILE = (
    ROOT / "outputs" / "reports"
    / "europe_sfa_accounting_hierarchy_coverage.csv"
)

REPORT_FILE = (
    ROOT / "outputs" / "reports"
    / "europe_sfa_accounting_hierarchy_audit.txt"
)

EXPECTED_COUNTRIES = 17
EXPECTED_YEARS = list(range(2020, 2025))
EXPECTED_UNITS = {"MIO_NAC", "PC_GDP"}

# These are source indicators, not the derived sfa_calculated field.
HIERARCHY = {
    "GD_CH": {
        "role": "debt_transition_anchor",
        "class": "outcome_measure",
        "additivity": "Do not add",
        "explanation": (
            "Observed change in consolidated gross debt. "
            "This is the debt-change outcome being explained."
        ),
    },
    "B9_T3": {
        "role": "deficit_surplus_bridge",
        "class": "primary_transition_measure",
        "additivity": "Do not add independently",
        "explanation": (
            "Reversed-sign net lending/net borrowing measure used "
            "with GD_CH to calculate the SFA bridge."
        ),
    },
    "F_ASS": {
        "role": "financial_asset_transactions",
        "class": "broad_component_aggregate",
        "additivity": "Do not add to its own subcomponents",
        "explanation": (
            "Net acquisition of financial assets across multiple "
            "instrument categories. Treat as an aggregate unless "
            "the detailed instrument breakdown is used instead."
        ),
    },
    "F71_ADJ_LIAB": {
        "role": "financial_derivative_liabilities",
        "class": "candidate_adjustment_component",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Adjustment for net incurrence of liabilities in "
            "financial derivatives."
        ),
    },
    "F8_ADJ_LIAB": {
        "role": "other_accounts_payable",
        "class": "candidate_adjustment_component",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Adjustment for net incurrence of other accounts payable."
        ),
    },
    "FV_ADJ_LIAB": {
        "role": "other_liabilities",
        "class": "candidate_adjustment_component",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Adjustment relating to other liabilities. "
            "Check its detailed instrument coverage before combining."
        ),
    },
    "ORINV": {
        "role": "issuance_price_adjustment",
        "class": "candidate_valuation_adjustment",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Adjustment for debt issuance above or below nominal value."
        ),
    },
    "ORD41A_ADJ": {
        "role": "interest_accrual_cash_timing",
        "class": "candidate_timing_adjustment",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Difference between D.41 interest accrued and interest paid. "
            "Do not confuse this with total interest expenditure."
        ),
    },
    "ORFCD": {
        "role": "foreign_currency_valuation",
        "class": "candidate_valuation_adjustment",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Adjustment for appreciation or depreciation of "
            "foreign-currency debt."
        ),
    },
    "ORRNV": {
        "role": "redemption_price_adjustment",
        "class": "candidate_valuation_adjustment",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Adjustment for debt redemptions or repurchases above "
            "or below nominal value."
        ),
    },
    "KX": {
        "role": "other_volume_changes",
        "class": "candidate_volume_adjustment",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Other volume changes in financial liabilities. "
            "Check the exact coverage of the selected series."
        ),
    },
    "K61": {
        "role": "sector_reclassification",
        "class": "candidate_reclassification_adjustment",
        "additivity": "Potentially additive; verify identity and sign",
        "explanation": (
            "Changes in sector classification and institutional structure."
        ),
    },
    "B9FX9": {
        "role": "capital_financial_account_discrepancy",
        "class": "statistical_discrepancy",
        "additivity": "Keep separate pending identity verification",
        "explanation": (
            "Discrepancy between financial and non-financial "
            "net lending/net borrowing measures."
        ),
    },
    "YA3": {
        "role": "total_statistical_discrepancies",
        "class": "broad_discrepancy_aggregate",
        "additivity": "Do not add to its subcomponents",
        "explanation": (
            "Total statistical discrepancies. Treat as an aggregate "
            "until its detailed composition is established."
        ),
    },
    "YA3O": {
        "role": "other_statistical_discrepancies",
        "class": "statistical_discrepancy",
        "additivity": "Do not add to YA3 without identity verification",
        "explanation": (
            "Other statistical discrepancies. May overlap with a "
            "broader discrepancy aggregate."
        ),
    },
    "ORADJ": {
        "role": "total_adjustments",
        "class": "broad_adjustment_aggregate",
        "additivity": "Do not add to its underlying components",
        "explanation": (
            "Broad total-adjustments measure. Its precise relationship "
            "to individual adjustment series must be verified before use."
        ),
    },
}

DERIVED_MEASURES = {
    "sfa_calculated": {
        "role": "derived_sfa_control",
        "class": "derived_reconciliation_measure",
        "additivity": "Never add as a separate component",
        "explanation": (
            "Derived from GD_CH and B9_T3. It is a reconciliation "
            "measure, not an independent source component."
        ),
    }
}


def load_input():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"country", "year", "unit"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["unit"] = df["unit"].astype(str).str.strip()

    return df


def build_hierarchy(df):
    rows = []

    for indicator, metadata in HIERARCHY.items():
        rows.append({
            "indicator": indicator,
            **metadata,
            "source_or_derived": "source",
        })

    for indicator, metadata in DERIVED_MEASURES.items():
        rows.append({
            "indicator": indicator,
            **metadata,
            "source_or_derived": "derived",
        })

    hierarchy = pd.DataFrame(rows)

    hierarchy["present_in_input"] = (
        hierarchy["indicator"].isin(df.columns)
    )

    return hierarchy


def build_coverage(df, hierarchy):
    countries = sorted(df["country"].dropna().unique())
    years = EXPECTED_YEARS
    expected_pairs = len(countries) * len(years)

    rows = []

    for unit in sorted(EXPECTED_UNITS):
        unit_df = df[
            (df["unit"] == unit)
            & (df["year"].isin(years))
        ].copy()

        for indicator in hierarchy["indicator"]:
            if indicator not in df.columns:
                available = 0
                country_count = 0
                year_count = 0
            else:
                values = pd.to_numeric(
                    unit_df[indicator], errors="coerce"
                )

                valid = unit_df.loc[values.notna(), ["country", "year"]]

                available = int(values.notna().sum())
                country_count = int(valid["country"].nunique())
                year_count = int(valid["year"].nunique())

            rows.append({
                "unit": unit,
                "indicator": indicator,
                "source_or_derived": (
                    "derived"
                    if indicator in DERIVED_MEASURES
                    else "source"
                ),
                "expected_country_year_pairs": expected_pairs,
                "available_observations": available,
                "missing_observations": expected_pairs - available,
                "coverage_pct": (
                    available / expected_pairs * 100
                    if expected_pairs else 0
                ),
                "countries_covered": country_count,
                "years_covered": year_count,
            })

    return pd.DataFrame(rows)


def validate(df, hierarchy, coverage):
    checks = []

    def add(name, passed, detail):
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    countries = df["country"].nunique()
    years = sorted(
        df.loc[df["year"].isin(EXPECTED_YEARS), "year"]
        .dropna().astype(int).unique().tolist()
    )
    units = set(df["unit"].dropna().unique())

    add(
        "Expected country coverage",
        countries == EXPECTED_COUNTRIES,
        f"Found {countries}; expected {EXPECTED_COUNTRIES}",
    )

    add(
        "Expected reference years",
        years == EXPECTED_YEARS,
        f"Found {years}; expected {EXPECTED_YEARS}",
    )

    add(
        "Expected units",
        units == EXPECTED_UNITS,
        f"Found {sorted(units)}; expected {sorted(EXPECTED_UNITS)}",
    )

    duplicate_count = int(
        df.duplicated(["country", "year", "unit"]).sum()
    )

    add(
        "No duplicate country-year-unit records",
        duplicate_count == 0,
        f"Duplicate records: {duplicate_count}",
    )

    missing_indicators = hierarchy.loc[
        ~hierarchy["present_in_input"], "indicator"
    ].tolist()

    add(
        "All expected indicators present",
        len(missing_indicators) == 0,
        (
            "All expected indicators found"
            if not missing_indicators
            else f"Missing: {missing_indicators}"
        ),
    )

    coverage_valid = (
        coverage["available_observations"].ge(0).all()
        & coverage["available_observations"].le(
            coverage["expected_country_year_pairs"]
        ).all()
    )

    add(
        "Coverage counts are valid",
        bool(coverage_valid),
        "Missing component observations are reported, not treated as zero",
    )

    return pd.DataFrame(checks)


def write_report(df, hierarchy, coverage, checks):
    lines = [
        "EUROSTAT SFA ACCOUNTING HIERARCHY AUDIT",
        "=" * 72,
        "",
        f"Input: {INPUT_FILE}",
        f"Countries: {df['country'].nunique()}",
        (
            "Years: "
            + ", ".join(
                map(
                    str,
                    sorted(
                        df.loc[
                            df["year"].isin(EXPECTED_YEARS), "year"
                        ].dropna().astype(int).unique()
                    ),
                )
            )
        ),
        f"Units: {', '.join(sorted(df['unit'].dropna().unique()))}",
        "",
        "HIERARCHY CLASSIFICATION",
        "-" * 72,
    ]

    for _, row in hierarchy.iterrows():
        lines.extend([
            f"{row['indicator']}: {row['class']}",
            f"  Role: {row['role']}",
            f"  Treatment: {row['additivity']}",
            f"  Explanation: {row['explanation']}",
            f"  Present in input: {row['present_in_input']}",
            "",
        ])

    lines.extend([
        "COVERAGE SUMMARY",
        "-" * 72,
    ])

    for unit in sorted(EXPECTED_UNITS):
        unit_cov = coverage[coverage["unit"] == unit]

        lines.append(f"\nUnit: {unit}")

        for _, row in unit_cov.iterrows():
            lines.append(
                f"  {row['indicator']}: "
                f"{row['available_observations']}/"
                f"{row['expected_country_year_pairs']} "
                f"({row['coverage_pct']:.1f}%)"
            )

    lines.extend([
        "",
        "VALIDATION CHECKS",
        "-" * 72,
    ])

    for _, row in checks.iterrows():
        lines.append(
            f"{row['status']}: {row['check']} — {row['detail']}"
        )

    overall = (
        "PASS"
        if checks["status"].eq("PASS").all()
        else "FAIL"
    )

    lines.extend([
        "",
        f"OVERALL STRUCTURAL AUDIT: {overall}",
        "",
        "INTERPRETATION LIMITATIONS",
        "-" * 72,
        "1. This is a classification and coverage audit, not an additive identity test.",
        "2. Broad aggregates must not be added to their underlying components.",
        "3. Statistical discrepancies are kept separate from economic drivers.",
        "4. Candidate adjustment components require official sign and identity verification.",
        "5. Missing observations are not zero and are not imputed.",
        "6. Cross-unit agreement does not prove economic additivity.",
        "7. The derived sfa_calculated measure is not an independent source series.",
        "",
        "Official metadata reference:",
        "https://ec.europa.eu/eurostat/cache/metadata/en/gov_10dd_esms.htm",
    ])

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    return overall


def main():
    HIERARCHY_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = load_input()
    hierarchy = build_hierarchy(df)
    coverage = build_coverage(df, hierarchy)
    checks = validate(df, hierarchy, coverage)

    hierarchy.to_csv(HIERARCHY_FILE, index=False)
    coverage.to_csv(COVERAGE_FILE, index=False)

    overall = write_report(df, hierarchy, coverage, checks)

    print("\nEUROSTAT SFA ACCOUNTING HIERARCHY AUDIT")
    print("=" * 60)
    print(f"Input records: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Hierarchy entries: {len(hierarchy)}")
    print(f"Source indicators: {len(HIERARCHY)}")
    print(f"Derived measures: {len(DERIVED_MEASURES)}")
    print(f"Coverage rows: {len(coverage)}")
    print(f"Validation checks: {len(checks)}")
    print(f"Failed checks: {(checks['status'] == 'FAIL').sum()}")
    print(f"Overall structural audit: {overall}")

    print("\nARTIFACTS")
    print(HIERARCHY_FILE)
    print(COVERAGE_FILE)
    print(REPORT_FILE)

    if overall != "PASS":
        raise SystemExit(
            "Hierarchy audit failed. Inspect the report before proceeding."
        )


if __name__ == "__main__":
    main()
