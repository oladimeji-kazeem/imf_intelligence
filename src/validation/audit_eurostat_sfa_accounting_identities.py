from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "europe_eurostat_sfa_reconciliation.csv"
)

DETAIL_FILE = (
    ROOT / "data" / "processed"
    / "europe_sfa_accounting_identity_tests.csv"
)

SUMMARY_FILE = (
    ROOT / "outputs" / "reports"
    / "europe_sfa_accounting_identity_summary.csv"
)

REPORT_FILE = (
    ROOT / "outputs" / "reports"
    / "europe_sfa_accounting_identity_report.txt"
)

EXPECTED_COUNTRIES = 17
EXPECTED_YEARS = list(range(2020, 2025))
EXPECTED_UNITS = {"MIO_NAC", "PC_GDP"}

# Lower-level adjustment series.
# Their sum is a diagnostic candidate, not an assumed identity.
ADJUSTMENT_DETAIL = [
    "F71_ADJ_LIAB",
    "F8_ADJ_LIAB",
    "FV_ADJ_LIAB",
    "ORINV",
    "ORD41A_ADJ",
    "ORFCD",
    "ORRNV",
    "KX",
    "K61",
]

IDENTITY_CANDIDATES = [
    "TOP_LEVEL_SFA_PLUS",
    "STAT_DISCREPANCY_PLUS_B9FX9",
    "STAT_DISCREPANCY_MINUS_B9FX9",
    "ADJUSTMENT_DETAIL_SUM",
]


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "country",
        "year",
        "unit",
        "sfa_calculated",
        "F_ASS",
        "ORADJ",
        "YA3",
        "B9FX9",
        "YA3O",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["unit"] = df["unit"].astype(str).str.strip()

    numeric_columns = (
        list(required - {"country", "year", "unit"})
        + ADJUSTMENT_DETAIL
    )

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    df = df[
        df["year"].isin(EXPECTED_YEARS)
        & df["unit"].isin(EXPECTED_UNITS)
    ].copy()

    return df


def add_comparison(
    results,
    row,
    test,
    observed,
    expected,
    component_count,
    formula_description,
):
    if pd.isna(observed) or pd.isna(expected):
        return

    residual = observed - expected

    results.append({
        "test": test,
        "country": row["country"],
        "year": int(row["year"]),
        "unit": row["unit"],
        "observed_value": observed,
        "candidate_value": expected,
        "residual": residual,
        "absolute_residual": abs(residual),
        "component_count": component_count,
        "formula": formula_description,
    })


def build_comparisons(df):
    results = []

    for _, row in df.iterrows():

        # --------------------------------------------------
        # A. Top-level SFA bridge candidate
        # --------------------------------------------------
        top_fields = ["sfa_calculated", "F_ASS", "ORADJ", "YA3"]

        if all(pd.notna(row[field]) for field in top_fields):
            candidate = row["F_ASS"] + row["ORADJ"] + row["YA3"]

            add_comparison(
                results,
                row,
                "TOP_LEVEL_SFA_PLUS",
                row["sfa_calculated"],
                candidate,
                3,
                "sfa_calculated versus F_ASS + ORADJ + YA3",
            )

        # --------------------------------------------------
        # B. Statistical discrepancy: test both signs
        # --------------------------------------------------
        if (
            pd.notna(row["YA3"])
            and pd.notna(row["B9FX9"])
            and pd.notna(row["YA3O"])
        ):
            plus_candidate = row["B9FX9"] + row["YA3O"]

            add_comparison(
                results,
                row,
                "STAT_DISCREPANCY_PLUS_B9FX9",
                row["YA3"],
                plus_candidate,
                2,
                "YA3 versus B9FX9 + YA3O",
            )

            minus_candidate = -row["B9FX9"] + row["YA3O"]

            add_comparison(
                results,
                row,
                "STAT_DISCREPANCY_MINUS_B9FX9",
                row["YA3"],
                minus_candidate,
                2,
                "YA3 versus -B9FX9 + YA3O",
            )

        # --------------------------------------------------
        # C. Lower-level adjustment diagnostic
        # --------------------------------------------------
        if pd.notna(row["ORADJ"]):
            available = [
                row[field]
                for field in ADJUSTMENT_DETAIL
                if field in df.columns and pd.notna(row[field])
            ]

            # Only calculate the complete selected-detail sum.
            if len(available) == len(ADJUSTMENT_DETAIL):
                detail_sum = sum(available)

                add_comparison(
                    results,
                    row,
                    "ADJUSTMENT_DETAIL_SUM",
                    row["ORADJ"],
                    detail_sum,
                    len(available),
                    "ORADJ versus sum of selected adjustment details",
                )

    return pd.DataFrame(results)


def build_summary(comparisons):
    if comparisons.empty:
        return pd.DataFrame()

    return (
        comparisons.groupby(
            ["test", "unit", "formula"],
            dropna=False,
        )
        .agg(
            comparisons=("residual", "count"),
            mean_residual=("residual", "mean"),
            mean_absolute_residual=("absolute_residual", "mean"),
            median_absolute_residual=("absolute_residual", "median"),
            max_absolute_residual=("absolute_residual", "max"),
            min_absolute_residual=("absolute_residual", "min"),
        )
        .reset_index()
    )


def build_coverage(df):
    rows = []

    expected = EXPECTED_COUNTRIES * len(EXPECTED_YEARS)

    for unit in sorted(EXPECTED_UNITS):
        subset = df[df["unit"] == unit]

        for indicator in ADJUSTMENT_DETAIL:
            if indicator not in subset.columns:
                available = 0
            else:
                available = int(subset[indicator].notna().sum())

            rows.append({
                "unit": unit,
                "indicator": indicator,
                "available_observations": available,
                "expected_observations": expected,
                "coverage_pct": available / expected * 100,
            })

    return pd.DataFrame(rows)


def validate_structure(df, comparisons):
    checks = []

    def add(name, passed, detail):
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    countries = df["country"].nunique()

    years = sorted(
        df["year"].dropna().astype(int).unique().tolist()
    )

    units = set(df["unit"].dropna().unique())

    duplicates = int(
        df.duplicated(["country", "year", "unit"]).sum()
    )

    add(
        "Expected country coverage",
        countries == EXPECTED_COUNTRIES,
        f"Found {countries}; expected {EXPECTED_COUNTRIES}",
    )

    add(
        "Expected years",
        years == EXPECTED_YEARS,
        f"Found {years}; expected {EXPECTED_YEARS}",
    )

    add(
        "Expected units",
        units == EXPECTED_UNITS,
        f"Found {sorted(units)}; expected {sorted(EXPECTED_UNITS)}",
    )

    add(
        "No duplicate country-year-unit records",
        duplicates == 0,
        f"Duplicate records: {duplicates}",
    )

    add(
        "Comparison output generated",
        not comparisons.empty,
        f"Comparison records: {len(comparisons)}",
    )

    return pd.DataFrame(checks)


def write_report(df, comparisons, summary, coverage, checks):
    lines = [
        "EUROSTAT SFA ACCOUNTING RELATIONSHIP DIAGNOSTIC",
        "=" * 76,
        "",
        f"Input: {INPUT_FILE}",
        f"Countries: {df['country'].nunique()}",
        f"Years: {', '.join(map(str, EXPECTED_YEARS))}",
        f"Units: {', '.join(sorted(df['unit'].unique()))}",
        "",
        "IMPORTANT METHODOLOGICAL NOTE",
        "-" * 76,
        "The following are candidate accounting relationships.",
        "They are not automatically treated as verified identities.",
        "Residuals are reported without imposing arbitrary pass/fail",
        "rounding thresholds.",
        "",
        "TEST A — TOP-LEVEL SFA BRIDGE CANDIDATE",
        "-" * 76,
        "Compare sfa_calculated with F_ASS + ORADJ + YA3.",
        "This tests whether the selected aggregate series reconcile.",
        "",
        "TEST B — STATISTICAL DISCREPANCY SIGN DIAGNOSTIC",
        "-" * 76,
        "Compare YA3 with both:",
        "  B9FX9 + YA3O",
        "  -B9FX9 + YA3O",
        "B9FX9 is defined as B9F-B9, while the EDP transition",
        "description refers to B9-B9F. The sign must be verified.",
        "",
        "TEST C — ADJUSTMENT DETAIL DIAGNOSTIC",
        "-" * 76,
        "Compare ORADJ with the complete-case sum of:",
        ", ".join(ADJUSTMENT_DETAIL),
        "This is not assumed to be exhaustive or additive.",
        "Only observations with all selected detail values are included.",
        "",
        "SUMMARY RESULTS",
        "-" * 76,
    ]

    if summary.empty:
        lines.append("No comparisons available.")
    else:
        for _, row in summary.iterrows():
            lines.extend([
                "",
                f"{row['test']} | {row['unit']}",
                f"  Formula: {row['formula']}",
                f"  Comparisons: {int(row['comparisons'])}",
                f"  Mean residual: {row['mean_residual']:.6f}",
                (
                    "  Mean absolute residual: "
                    f"{row['mean_absolute_residual']:.6f}"
                ),
                (
                    "  Median absolute residual: "
                    f"{row['median_absolute_residual']:.6f}"
                ),
                (
                    "  Maximum absolute residual: "
                    f"{row['max_absolute_residual']:.6f}"
                ),
            ])

    lines.extend([
        "",
        "ADJUSTMENT DETAIL COVERAGE",
        "-" * 76,
    ])

    for _, row in coverage.iterrows():
        lines.append(
            f"{row['unit']} | {row['indicator']}: "
            f"{row['available_observations']}/"
            f"{row['expected_observations']} "
            f"({row['coverage_pct']:.1f}%)"
        )

    lines.extend([
        "",
        "VALIDATION CHECKS",
        "-" * 76,
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
        f"OVERALL STRUCTURAL CHECK: {overall}",
        "",
        "INTERPRETATION LIMITATIONS",
        "-" * 76,
        "1. Structural PASS does not mean accounting identities passed.",
        "2. Residuals must be evaluated against source precision and",
        "   official indicator definitions.",
        "3. The two B9FX9 sign candidates are diagnostic alternatives.",
        "4. ORADJ may contain adjustment categories not included in",
        "   the selected detail series.",
        "5. Missing values are not zero and are not imputed.",
        "6. Do not add lower-level detail to an aggregate in the",
        "   top-level SFA bridge.",
        "7. Cross-unit consistency is a separate audit already completed.",
        "",
        "Official Eurostat metadata:",
        "https://ec.europa.eu/eurostat/cache/metadata/en/gov_10dd_esms.htm",
    ])

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    return overall


def main():
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()

    comparisons = build_comparisons(df)
    summary = build_summary(comparisons)
    coverage = build_coverage(df)
    checks = validate_structure(df, comparisons)

    comparisons.to_csv(DETAIL_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    coverage_file = (
        ROOT / "outputs" / "reports"
        / "europe_sfa_adjustment_detail_coverage.csv"
    )

    coverage.to_csv(coverage_file, index=False)

    overall = write_report(
        df,
        comparisons,
        summary,
        coverage,
        checks,
    )

    print("\nEUROSTAT SFA ACCOUNTING RELATIONSHIP DIAGNOSTIC")
    print("=" * 65)
    print(f"Input records: {len(df)}")
    print(f"Comparison records: {len(comparisons)}")
    print(f"Summary rows: {len(summary)}")
    print(f"Structural checks: {len(checks)}")
    print(f"Failed structural checks: {(checks['status'] == 'FAIL').sum()}")
    print(f"Overall structural check: {overall}")

    print("\nARTIFACTS")
    print(DETAIL_FILE)
    print(SUMMARY_FILE)
    print(coverage_file)
    print(REPORT_FILE)

    if overall != "PASS":
        raise SystemExit(
            "Structural check failed. Inspect the report."
        )


if __name__ == "__main__":
    main()
