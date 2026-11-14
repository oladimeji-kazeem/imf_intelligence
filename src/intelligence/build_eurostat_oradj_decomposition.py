from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SFA_FILE = ROOT / "data/processed/europe_eurostat_sfa_reconciliation.csv"
GDP_FILE = ROOT / "data/processed/europe_eurostat_edp_gdp_2019_2024.csv"

OUTPUT_FILE = ROOT / "data/processed/europe_oradj_component_decomposition.csv"
SUMMARY_FILE = ROOT / "data/processed/europe_oradj_component_summary.csv"
COVERAGE_FILE = ROOT / "outputs/reports/europe_oradj_component_coverage.csv"
CROSS_UNIT_FILE = ROOT / "data/processed/europe_oradj_component_cross_unit_check.csv"
REPORT_FILE = ROOT / "outputs/reports/europe_oradj_component_decomposition_report.txt"

EXPECTED_COUNTRIES = 17
EXPECTED_YEARS = list(range(2020, 2025))
EXPECTED_UNITS = ["MIO_NAC", "PC_GDP"]

DETAILS = [
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

REQUIRED_SFA = ["country", "year", "unit", "ORADJ"] + DETAILS
REQUIRED_GDP = ["country_code", "year", "nominal_gdp"]


def load_inputs():
    if not SFA_FILE.exists():
        raise FileNotFoundError(SFA_FILE)

    if not GDP_FILE.exists():
        raise FileNotFoundError(GDP_FILE)

    sfa = pd.read_csv(SFA_FILE)
    gdp = pd.read_csv(GDP_FILE)

    missing_sfa = sorted(set(REQUIRED_SFA) - set(sfa.columns))
    missing_gdp = sorted(set(REQUIRED_GDP) - set(gdp.columns))

    if missing_sfa:
        raise ValueError(f"Missing SFA columns: {missing_sfa}")

    if missing_gdp:
        raise ValueError(f"Missing GDP columns: {missing_gdp}")

    sfa["country"] = (
        sfa["country"].astype(str).str.upper().str.strip()
    )
    sfa["unit"] = (
        sfa["unit"].astype(str).str.upper().str.strip()
    )
    sfa["year"] = pd.to_numeric(
        sfa["year"], errors="raise"
    ).astype(int)

    for col in ["ORADJ"] + DETAILS:
        sfa[col] = pd.to_numeric(sfa[col], errors="coerce")

    gdp["country_code"] = (
        gdp["country_code"].astype(str).str.upper().str.strip()
    )
    gdp["year"] = pd.to_numeric(
        gdp["year"], errors="raise"
    ).astype(int)
    gdp["nominal_gdp"] = pd.to_numeric(
        gdp["nominal_gdp"], errors="coerce"
    )

    return sfa, gdp


def validate_structure(sfa, gdp):
    sfa = sfa[
        sfa["year"].isin(EXPECTED_YEARS)
        & sfa["unit"].isin(EXPECTED_UNITS)
    ].copy()

    if sfa["country"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTRIES} countries; "
            f"found {sfa['country'].nunique()}"
        )

    if sorted(sfa["year"].unique().tolist()) != EXPECTED_YEARS:
        raise ValueError("Unexpected year coverage.")

    if sorted(sfa["unit"].unique().tolist()) != EXPECTED_UNITS:
        raise ValueError("Unexpected unit coverage.")

    if sfa.duplicated(["country", "year", "unit"]).any():
        raise ValueError("Duplicate country-year-unit records.")

    gdp = gdp[gdp["year"].isin(EXPECTED_YEARS)].copy()

    if gdp.duplicated(["country_code", "year"]).any():
        raise ValueError("Duplicate GDP country-year records.")

    return sfa, gdp


def build_decomposition(sfa, gdp):
    data = sfa.merge(
        gdp,
        left_on=["country", "year"],
        right_on=["country_code", "year"],
        how="left",
        validate="many_to_one"
    )

    monetary = data["unit"].eq("MIO_NAC")

    if data.loc[monetary, "nominal_gdp"].isna().any():
        raise ValueError("Missing GDP for MIO_NAC observations.")

    if data.loc[monetary, "nominal_gdp"].le(0).any():
        raise ValueError("Non-positive GDP encountered.")

    source_columns = ["ORADJ"] + DETAILS

    # Convert all components to percentage points of GDP.
    for col in source_columns:
        data[f"{col}_pp"] = np.where(
            monetary,
            data[col] / data["nominal_gdp"] * 100,
            data[col]
        )

    detail_pp = [f"{col}_pp" for col in DETAILS]

    data["complete_detail_set"] = (
        data[detail_pp].notna().all(axis=1)
    )

    data["detail_component_sum_pp"] = (
        data[detail_pp].sum(axis=1, min_count=len(DETAILS))
    )

    data["oradj_pp"] = data["ORADJ_pp"]

    data["oradj_residual_pp"] = (
        data["oradj_pp"] - data["detail_component_sum_pp"]
    )

    data["decomposition_status"] = np.select(
        [
            data["oradj_pp"].isna(),
            ~data["complete_detail_set"]
        ],
        [
            "MISSING_ORADJ",
            "INCOMPLETE_DETAIL_SET"
        ],
        default="COMPLETE"
    )

    # Keep individual component contributions visible.
    rename_map = {
        f"{col}_pp": f"{col.lower()}_pp"
        for col in DETAILS
    }

    data = data.rename(columns=rename_map)

    output_columns = [
        "country",
        "year",
        "unit",
        "oradj_pp",
    ] + [
        f"{col.lower()}_pp" for col in DETAILS
    ] + [
        "detail_component_sum_pp",
        "oradj_residual_pp",
        "complete_detail_set",
        "decomposition_status"
    ]

    return data[output_columns].copy()


def build_coverage(sfa):
    rows = []

    for unit in EXPECTED_UNITS:
        subset = sfa[sfa["unit"].eq(unit)]

        for col in ["ORADJ"] + DETAILS:
            available = int(subset[col].notna().sum())
            expected = len(subset)

            rows.append({
                "unit": unit,
                "indicator": col,
                "available": available,
                "expected": expected,
                "missing": expected - available,
                "coverage_pct": (
                    available / expected * 100 if expected else np.nan
                )
            })

    return pd.DataFrame(rows)


def build_summary(data):
    complete = data[
        data["decomposition_status"].eq("COMPLETE")
    ].copy()

    component_columns = [
        f"{col.lower()}_pp" for col in DETAILS
    ]

    aggregation = {
        "observations": ("year", "count"),
        "cumulative_oradj_pp": ("oradj_pp", "sum"),
        "cumulative_detail_sum_pp": (
            "detail_component_sum_pp", "sum"
        ),
        "cumulative_residual_pp": (
            "oradj_residual_pp", "sum"
        ),
        "mean_absolute_residual_pp": (
            "oradj_residual_pp",
            lambda x: x.abs().mean()
        ),
        "maximum_absolute_residual_pp": (
            "oradj_residual_pp",
            lambda x: x.abs().max()
        )
    }

    for col in component_columns:
        aggregation[f"cumulative_{col}"] = (col, "sum")

    summary = (
        complete.groupby(["country", "unit"], as_index=False)
        .agg(**aggregation)
    )

    return summary


def build_cross_unit_check(data):
    component_columns = [
        "oradj_pp"
    ] + [
        f"{col.lower()}_pp" for col in DETAILS
    ] + [
        "detail_component_sum_pp",
        "oradj_residual_pp"
    ]

    pivot = data.pivot(
        index=["country", "year"],
        columns="unit",
        values=component_columns
    )

    rows = []

    for metric in component_columns:
        left = (metric, "MIO_NAC")
        right = (metric, "PC_GDP")

        if left not in pivot.columns or right not in pivot.columns:
            continue

        paired = pivot[[left, right]].dropna()

        differences = paired[left] - paired[right]

        rows.append({
            "component": metric,
            "paired_observations": len(differences),
            "mean_absolute_difference_pp": differences.abs().mean(),
            "maximum_absolute_difference_pp": differences.abs().max()
        })

    return pd.DataFrame(rows)


def write_report(data, coverage, summary, cross_unit):
    lines = []

    lines.append("EUROSTAT ORADJ COMPONENT DECOMPOSITION")
    lines.append("=" * 78)
    lines.append(f"Input: {SFA_FILE}")
    lines.append(f"GDP input: {GDP_FILE}")
    lines.append(f"Countries: {data['country'].nunique()}")
    lines.append(f"Years: {sorted(data['year'].unique().tolist())}")
    lines.append(f"Units: {sorted(data['unit'].unique().tolist())}")
    lines.append("")
    lines.append("CANDIDATE ADJUSTMENT DETAIL RELATIONSHIP")
    lines.append("-" * 78)
    lines.append("ORADJ = sum of the nine selected adjustment detail series")
    lines.append(
        "This is a complete-case diagnostic, not an assumed universal identity."
    )
    lines.append(
        "All components are expressed as percentage points of current-year GDP."
    )
    lines.append("")
    lines.append("DECOMPOSITION COVERAGE")
    lines.append("-" * 78)

    for unit in EXPECTED_UNITS:
        subset = data[data["unit"].eq(unit)]

        complete = int(
            subset["decomposition_status"].eq("COMPLETE").sum()
        )

        lines.append(
            f"{unit}: {complete}/{len(subset)} complete decompositions"
        )

    lines.append("")
    lines.append("DETAIL INDICATOR COVERAGE")
    lines.append("-" * 78)

    for _, row in coverage.iterrows():
        lines.append(
            f"{row['unit']} | {row['indicator']}: "
            f"{int(row['available'])}/{int(row['expected'])} "
            f"({row['coverage_pct']:.1f}%)"
        )

    lines.append("")
    lines.append("ORADJ RECONCILIATION")
    lines.append("-" * 78)

    for unit in EXPECTED_UNITS:
        subset = data[
            data["unit"].eq(unit)
            & data["decomposition_status"].eq("COMPLETE")
        ]

        if subset.empty:
            continue

        residual = subset["oradj_residual_pp"]

        lines.append(unit)
        lines.append(f"  Comparisons: {len(subset)}")
        lines.append(
            f"  Mean residual: {residual.mean():.8f} pp"
        )
        lines.append(
            f"  Mean absolute residual: {residual.abs().mean():.8f} pp"
        )
        lines.append(
            f"  Median absolute residual: {residual.abs().median():.8f} pp"
        )
        lines.append(
            f"  Maximum absolute residual: {residual.abs().max():.8f} pp"
        )

    lines.append("")
    lines.append("CROSS-UNIT COMPONENT COMPARISON")
    lines.append("-" * 78)

    for _, row in cross_unit.iterrows():
        lines.append(
            f"{row['component']}: "
            f"{int(row['paired_observations'])} paired; "
            f"mean absolute difference "
            f"{row['mean_absolute_difference_pp']:.8f} pp; "
            f"maximum absolute difference "
            f"{row['maximum_absolute_difference_pp']:.8f} pp"
        )

    lines.append("")
    lines.append("COUNTRY SUMMARY")
    lines.append("-" * 78)

    if summary.empty:
        lines.append("No complete country-level decompositions.")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"{row['country']} | {row['unit']} | "
                f"n={int(row['observations'])} | "
                f"cumulative ORADJ={row['cumulative_oradj_pp']:.4f} pp | "
                f"detail sum={row['cumulative_detail_sum_pp']:.4f} pp | "
                f"residual={row['cumulative_residual_pp']:.4f} pp"
            )

    lines.append("")
    lines.append("INTERPRETATION LIMITATIONS")
    lines.append("-" * 78)
    lines.append(
        "1. The selected detail series are tested against ORADJ; "
        "their additivity is not assumed."
    )
    lines.append(
        "2. Only observations with all nine detail values are decomposed."
    )
    lines.append(
        "3. Missing values are not treated as zero."
    )
    lines.append(
        "4. A small residual supports numerical reconciliation, "
        "not causal attribution."
    )
    lines.append(
        "5. Component signs must be interpreted using the official "
        "indicator definitions."
    )
    lines.append(
        "6. Do not add these detail series to the top-level SFA bridge "
        "as well as ORADJ; that would double-count adjustments."
    )
    lines.append(
        "7. Country cumulative sums may cover different numbers of years; "
        "compare only with coverage disclosed."
    )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    sfa, gdp = load_inputs()
    sfa, gdp = validate_structure(sfa, gdp)

    data = build_decomposition(sfa, gdp)
    coverage = build_coverage(sfa)
    summary = build_summary(data)
    cross_unit = build_cross_unit_check(data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    coverage.to_csv(COVERAGE_FILE, index=False)
    cross_unit.to_csv(CROSS_UNIT_FILE, index=False)

    write_report(data, coverage, summary, cross_unit)

    print("EUROSTAT ORADJ COMPONENT DECOMPOSITION")
    print("=" * 68)
    print(f"Source records: {len(sfa)}")
    print(f"Countries: {sfa['country'].nunique()}")
    print(f"Years: {sorted(sfa['year'].unique().tolist())}")
    print(f"Units: {sorted(sfa['unit'].unique().tolist())}")

    for unit in EXPECTED_UNITS:
        subset = data[data["unit"].eq(unit)]
        complete = subset[
            subset["decomposition_status"].eq("COMPLETE")
        ]

        print(
            f"{unit}: {len(complete)}/{len(subset)} "
            "complete decompositions"
        )

        if not complete.empty:
            print(
                "  Maximum absolute residual: "
                f"{complete['oradj_residual_pp'].abs().max():.8f} pp"
            )

    print("")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Coverage: {COVERAGE_FILE}")
    print(f"Cross-unit check: {CROSS_UNIT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
