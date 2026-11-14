from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SFA_FILE = ROOT / "data/processed/europe_eurostat_sfa_reconciliation.csv"
GDP_FILE = ROOT / "data/processed/europe_eurostat_edp_gdp_2019_2024.csv"

PANEL_FILE = ROOT / "data/processed/europe_oradj_category_panel.csv"
COVERAGE_FILE = ROOT / "outputs/reports/europe_oradj_category_coverage.csv"
BALANCED_FILE = ROOT / "data/processed/europe_oradj_balanced_country_summary.csv"
CROSS_UNIT_FILE = ROOT / "data/processed/europe_oradj_category_cross_unit.csv"
REPORT_FILE = ROOT / "outputs/reports/europe_oradj_category_profiles_report.txt"

YEARS = list(range(2020, 2025))
UNITS = ["MIO_NAC", "PC_GDP"]
EXPECTED_COUNTRIES = 17

GROUPS = {
    "excluded_liability_adjustments": [
        "F71_ADJ_LIAB",
        "F8_ADJ_LIAB",
        "FV_ADJ_LIAB",
    ],
    "debt_valuation_adjustments": [
        "ORINV",
        "ORD41A_ADJ",
        "ORRNV",
    ],
    "foreign_currency_valuation": [
        "ORFCD",
    ],
    "other_volume_changes": [
        "KX",
        "K61",
    ],
}

DETAILS = [
    component
    for components in GROUPS.values()
    for component in components
]

REQUIRED_SFA = [
    "country", "year", "unit", "ORADJ"
] + DETAILS


def load_inputs():
    sfa = pd.read_csv(SFA_FILE)
    gdp = pd.read_csv(GDP_FILE)

    missing = sorted(set(REQUIRED_SFA) - set(sfa.columns))
    if missing:
        raise ValueError(f"Missing SFA columns: {missing}")

    required_gdp = {"country_code", "year", "nominal_gdp"}
    missing_gdp = sorted(required_gdp - set(gdp.columns))
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

    sfa = sfa[
        sfa["year"].isin(YEARS)
        & sfa["unit"].isin(UNITS)
    ].copy()

    gdp = gdp[gdp["year"].isin(YEARS)].copy()

    if sfa.duplicated(["country", "year", "unit"]).any():
        raise ValueError("Duplicate SFA country-year-unit records.")

    if gdp.duplicated(["country_code", "year"]).any():
        raise ValueError("Duplicate GDP country-year records.")

    if sfa["country"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError("Unexpected country coverage.")

    if sorted(sfa["year"].unique().tolist()) != YEARS:
        raise ValueError("Unexpected year coverage.")

    if sorted(sfa["unit"].unique().tolist()) != UNITS:
        raise ValueError("Unexpected unit coverage.")

    return sfa, gdp


def convert_to_gdp(sfa, gdp):
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

    for col in source_columns:
        data[f"{col}_pp"] = np.where(
            monetary,
            data[col] / data["nominal_gdp"] * 100,
            data[col]
        )

    return data


def build_category_panel(data):
    result = data[
        ["country", "year", "unit", "ORADJ_pp"]
    ].copy()

    result = result.rename(columns={"ORADJ_pp": "oradj_pp"})

    for component in DETAILS:
        result[f"{component.lower()}_pp"] = data[
            f"{component}_pp"
        ].values

    group_columns = []

    for group_name, components in GROUPS.items():
        columns = [f"{component.lower()}_pp" for component in components]

        group_column = f"{group_name}_pp"

        result[group_column] = result[columns].sum(
            axis=1,
            min_count=len(columns)
        )

        result[f"{group_name}_complete"] = (
            result[columns].notna().all(axis=1)
        )

        group_columns.append(group_column)

    result["all_categories_complete"] = (
        result[group_columns].notna().all(axis=1)
    )

    result["all_details_complete"] = (
        result[
            [f"{component.lower()}_pp" for component in DETAILS]
        ].notna().all(axis=1)
    )

    result["category_sum_pp"] = result[group_columns].sum(
        axis=1,
        min_count=len(group_columns)
    )

    result["oradj_residual_pp"] = (
        result["oradj_pp"] - result["category_sum_pp"]
    )

    result["complete_decomposition"] = (
        result["oradj_pp"].notna()
        & result["all_details_complete"]
    )

    return result


def build_coverage(panel):
    rows = []

    for unit in UNITS:
        subset = panel[panel["unit"].eq(unit)]

        for group_name in GROUPS:
            flag = f"{group_name}_complete"
            available = int(subset[flag].sum())

            rows.append({
                "unit": unit,
                "category": group_name,
                "available": available,
                "expected": len(subset),
                "coverage_pct": (
                    available / len(subset) * 100
                    if len(subset) else np.nan
                )
            })

        complete = int(subset["complete_decomposition"].sum())

        rows.append({
            "unit": unit,
            "category": "all_four_categories",
            "available": complete,
            "expected": len(subset),
            "coverage_pct": (
                complete / len(subset) * 100
                if len(subset) else np.nan
            )
        })

    return pd.DataFrame(rows)


def build_balanced_summary(panel):
    # Use MIO_NAC as the primary analytical source.
    primary = panel[panel["unit"].eq("MIO_NAC")].copy()

    complete = primary[
        primary["complete_decomposition"]
    ].copy()

    # A country qualifies only if all five years are complete.
    counts = (
        complete.groupby("country")["year"]
        .nunique()
    )

    balanced_countries = counts[counts.eq(len(YEARS))].index.tolist()

    balanced = complete[
        complete["country"].isin(balanced_countries)
    ].copy()

    group_columns = [
        f"{group}_pp" for group in GROUPS
    ]

    component_columns = [
        f"{component.lower()}_pp" for component in DETAILS
    ]

    rows = []

    for country, subset in balanced.groupby("country"):
        row = {
            "country": country,
            "observations": len(subset),
            "first_year": int(subset["year"].min()),
            "last_year": int(subset["year"].max()),
            "cumulative_oradj_pp": subset["oradj_pp"].sum(),
            "cumulative_category_sum_pp": subset[
                "category_sum_pp"
            ].sum(),
            "cumulative_oradj_residual_pp": subset[
                "oradj_residual_pp"
            ].sum(),
        }

        for col in group_columns + component_columns:
            row[f"cumulative_{col}"] = subset[col].sum()

        # Gross positive and negative contributions.
        for col in group_columns:
            row[f"gross_positive_{col}"] = subset[col].clip(
                lower=0
            ).sum()

            row[f"gross_negative_{col}"] = subset[col].clip(
                upper=0
            ).sum()

        rows.append(row)

    return pd.DataFrame(rows), balanced_countries, counts


def build_cross_unit_check(panel):
    metrics = [
        "oradj_pp",
        "category_sum_pp",
    ] + [
        f"{group}_pp" for group in GROUPS
    ]

    pivot = panel.pivot(
        index=["country", "year"],
        columns="unit",
        values=metrics
    )

    rows = []

    for metric in metrics:
        left = (metric, "MIO_NAC")
        right = (metric, "PC_GDP")

        paired = pivot[[left, right]].dropna()
        differences = paired[left] - paired[right]

        rows.append({
            "metric": metric,
            "paired_observations": len(differences),
            "mean_absolute_difference_pp": differences.abs().mean(),
            "maximum_absolute_difference_pp": differences.abs().max()
        })

    return pd.DataFrame(rows)


def write_report(panel, coverage, balanced, balanced_countries,
                 counts, cross_unit):
    lines = []

    lines.append("EUROSTAT ORADJ ECONOMIC CATEGORY PROFILES")
    lines.append("=" * 78)
    lines.append(f"Input: {SFA_FILE}")
    lines.append(f"Countries: {panel['country'].nunique()}")
    lines.append(f"Years: {YEARS}")
    lines.append(f"Units: {UNITS}")
    lines.append("")
    lines.append("OFFICIAL ECONOMIC CATEGORY CLASSIFICATION")
    lines.append("-" * 78)

    for group, components in GROUPS.items():
        lines.append(f"{group}: {', '.join(components)}")

    lines.append("")
    lines.append("CATEGORY COVERAGE")
    lines.append("-" * 78)

    for _, row in coverage.iterrows():
        lines.append(
            f"{row['unit']} | {row['category']}: "
            f"{int(row['available'])}/{int(row['expected'])} "
            f"({row['coverage_pct']:.1f}%)"
        )

    lines.append("")
    lines.append("FIVE-YEAR BALANCED PANEL")
    lines.append("-" * 78)
    lines.append(
        f"Countries with complete 2020-2024 detail: "
        f"{len(balanced_countries)}"
    )

    if balanced_countries:
        lines.append("Countries: " + ", ".join(balanced_countries))
    else:
        lines.append(
            "No country has a complete five-year detail panel. "
            "Do not produce five-year country rankings."
        )

    lines.append("")
    lines.append("COMPLETE-YEAR COVERAGE BY COUNTRY")
    lines.append("-" * 78)

    for country, count in counts.sort_values(
        ascending=False
    ).items():
        lines.append(f"{country}: {count}/5 years")

    lines.append("")
    lines.append("BALANCED COUNTRY CUMULATIVE CONTRIBUTIONS")
    lines.append("-" * 78)

    if balanced.empty:
        lines.append("No balanced country results available.")
    else:
        for _, row in balanced.groupby("country").first().reset_index().iterrows():
            lines.append(
                f"{row['country']}: "
                f"ORADJ={row['cumulative_oradj_pp']:.5f} pp; "
                f"category sum={row['cumulative_category_sum_pp']:.5f} pp; "
                f"residual={row['cumulative_oradj_residual_pp']:.6f} pp"
            )

    lines.append("")
    lines.append("CROSS-UNIT CATEGORY COMPARISON")
    lines.append("-" * 78)

    for _, row in cross_unit.iterrows():
        lines.append(
            f"{row['metric']}: "
            f"{int(row['paired_observations'])} paired; "
            f"mean absolute difference="
            f"{row['mean_absolute_difference_pp']:.8f} pp; "
            f"maximum absolute difference="
            f"{row['maximum_absolute_difference_pp']:.8f} pp"
        )

    lines.append("")
    lines.append("INTERPRETATION LIMITATIONS")
    lines.append("-" * 78)
    lines.append(
        "1. Category groupings follow Eurostat's published SFA structure."
    )
    lines.append(
        "2. Source signs are preserved; no sign reversal is applied."
    )
    lines.append(
        "3. A positive category contribution increases the debt transition "
        "under the reported sign convention; a negative contribution "
        "reduces it."
    )
    lines.append(
        "4. Category totals are complete only when every component "
        "within that category is present."
    )
    lines.append(
        "5. Missing values are not zero."
    )
    lines.append(
        "6. Five-year cumulative comparisons are produced only for "
        "countries with all five years complete."
    )
    lines.append(
        "7. These are accounting contributions, not causal estimates."
    )
    lines.append(
        "8. ORADJ remains the aggregate. Do not add category totals "
        "to ORADJ a second time."
    )
    lines.append(
        "9. Large offsetting components should be inspected separately."
    )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    sfa, gdp = load_inputs()
    converted = convert_to_gdp(sfa, gdp)

    panel = build_category_panel(converted)
    coverage = build_coverage(panel)

    balanced_summary, balanced_countries, counts = (
        build_balanced_summary(panel)
    )

    cross_unit = build_cross_unit_check(panel)

    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    panel.to_csv(PANEL_FILE, index=False)
    coverage.to_csv(COVERAGE_FILE, index=False)
    balanced_summary.to_csv(BALANCED_FILE, index=False)
    cross_unit.to_csv(CROSS_UNIT_FILE, index=False)

    write_report(
        panel,
        coverage,
        balanced_summary,
        balanced_countries,
        counts,
        cross_unit
    )

    print("EUROSTAT ORADJ ECONOMIC CATEGORY PROFILES")
    print("=" * 68)
    print(f"Source records: {len(sfa)}")
    print(f"Countries: {sfa['country'].nunique()}")
    print(f"Years: {YEARS}")
    print(f"Units: {UNITS}")
    print(
        "Five-year balanced countries: "
        f"{len(balanced_countries)}"
    )

    for country, count in counts.sort_values(
        ascending=False
    ).items():
        print(f"{country}: {count}/5 complete years")

    print("")
    print(f"Panel: {PANEL_FILE}")
    print(f"Coverage: {COVERAGE_FILE}")
    print(f"Balanced summary: {BALANCED_FILE}")
    print(f"Cross-unit comparison: {CROSS_UNIT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
