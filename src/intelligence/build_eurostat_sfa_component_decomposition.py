from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SFA_FILE = ROOT / "data/processed/europe_eurostat_sfa_reconciliation.csv"
GDP_FILE = ROOT / "data/processed/europe_eurostat_edp_gdp_2019_2024.csv"

OUTPUT_FILE = ROOT / "data/processed/europe_sfa_component_decomposition.csv"
SUMMARY_FILE = ROOT / "data/processed/europe_sfa_component_decomposition_summary.csv"
REPORT_FILE = ROOT / "outputs/reports/europe_sfa_component_decomposition_report.txt"

EXPECTED_COUNTRIES = 17
EXPECTED_YEARS = list(range(2020, 2025))
EXPECTED_UNITS = ["MIO_NAC", "PC_GDP"]

COMPONENTS = ["F_ASS", "ORADJ", "YA3"]
REQUIRED = ["country", "year", "unit", "sfa_calculated"] + COMPONENTS


def load_inputs():
    if not SFA_FILE.exists():
        raise FileNotFoundError(SFA_FILE)

    if not GDP_FILE.exists():
        raise FileNotFoundError(GDP_FILE)

    sfa = pd.read_csv(SFA_FILE)
    gdp = pd.read_csv(GDP_FILE)

    missing_sfa = sorted(set(REQUIRED) - set(sfa.columns))
    if missing_sfa:
        raise ValueError(f"Missing SFA columns: {missing_sfa}")

    required_gdp = {"country_code", "year", "nominal_gdp"}
    missing_gdp = sorted(required_gdp - set(gdp.columns))
    if missing_gdp:
        raise ValueError(f"Missing GDP columns: {missing_gdp}")

    sfa["country"] = sfa["country"].astype(str).str.upper().str.strip()
    sfa["unit"] = sfa["unit"].astype(str).str.upper().str.strip()
    sfa["year"] = pd.to_numeric(sfa["year"], errors="raise").astype(int)

    gdp["country_code"] = (
        gdp["country_code"].astype(str).str.upper().str.strip()
    )
    gdp["year"] = pd.to_numeric(gdp["year"], errors="raise").astype(int)
    gdp["nominal_gdp"] = pd.to_numeric(
        gdp["nominal_gdp"], errors="coerce"
    )

    for col in COMPONENTS + ["sfa_calculated"]:
        sfa[col] = pd.to_numeric(sfa[col], errors="coerce")

    return sfa, gdp


def build_decomposition(sfa, gdp):
    sfa = sfa[
        sfa["year"].isin(EXPECTED_YEARS)
        & sfa["unit"].isin(EXPECTED_UNITS)
    ].copy()

    if sfa.duplicated(["country", "year", "unit"]).any():
        raise ValueError("Duplicate country-year-unit records detected.")

    gdp = gdp[
        gdp["year"].isin(EXPECTED_YEARS)
    ].copy()

    if gdp.duplicated(["country_code", "year"]).any():
        raise ValueError("Duplicate GDP country-year records detected.")

    # The MIO_NAC series must be converted using current-year GDP.
    result = sfa.merge(
        gdp,
        left_on=["country", "year"],
        right_on=["country_code", "year"],
        how="left",
        validate="many_to_one"
    )

    if result.loc[
        result["unit"].eq("MIO_NAC"), "nominal_gdp"
    ].isna().any():
        raise ValueError(
            "Missing GDP match for one or more MIO_NAC observations. "
            "Check country identifiers and GDP coverage."
        )

    if (
        result.loc[result["unit"].eq("MIO_NAC"), "nominal_gdp"]
        .le(0).any()
    ):
        raise ValueError("Non-positive GDP encountered.")

    # Convert monetary values into percentage points of GDP.
    for col in ["sfa_calculated"] + COMPONENTS:
        result[f"{col}_pp"] = np.where(
            result["unit"].eq("MIO_NAC"),
            result[col] / result["nominal_gdp"] * 100,
            result[col]
        )

    result["component_sum_pp"] = (
        result["F_ASS_pp"]
        + result["ORADJ_pp"]
        + result["YA3_pp"]
    )

    result["bridge_residual_pp"] = (
        result["sfa_calculated_pp"] - result["component_sum_pp"]
    )

    result["bridge_status"] = np.where(
        result[["sfa_calculated_pp"] + [
            f"{c}_pp" for c in COMPONENTS
        ]].notna().all(axis=1),
        "COMPLETE",
        "INCOMPLETE"
    )

    result = result.rename(columns={
        "sfa_calculated_pp": "direct_sfa_pp",
        "F_ASS_pp": "financial_asset_acquisition_pp",
        "ORADJ_pp": "other_adjustments_pp",
        "YA3_pp": "statistical_discrepancy_pp"
    })

    output_columns = [
        "country",
        "year",
        "unit",
        "direct_sfa_pp",
        "financial_asset_acquisition_pp",
        "other_adjustments_pp",
        "statistical_discrepancy_pp",
        "component_sum_pp",
        "bridge_residual_pp",
        "bridge_status"
    ]

    return result[output_columns].copy()


def build_summary(data):
    complete = data[data["bridge_status"].eq("COMPLETE")].copy()

    value_columns = [
        "direct_sfa_pp",
        "financial_asset_acquisition_pp",
        "other_adjustments_pp",
        "statistical_discrepancy_pp",
        "component_sum_pp",
        "bridge_residual_pp"
    ]

    summary = (
        complete.groupby(["country", "unit"], as_index=False)
        .agg(
            observations=("year", "count"),
            **{
                f"cumulative_{col}": (col, "sum")
                for col in value_columns
            },
            mean_absolute_bridge_residual_pp=(
                "bridge_residual_pp",
                lambda x: x.abs().mean()
            ),
            maximum_absolute_bridge_residual_pp=(
                "bridge_residual_pp",
                lambda x: x.abs().max()
            )
        )
    )

    return summary


def build_cross_unit_check(data):
    pivot = data.pivot(
        index=["country", "year"],
        columns="unit",
        values=[
            "direct_sfa_pp",
            "financial_asset_acquisition_pp",
            "other_adjustments_pp",
            "statistical_discrepancy_pp"
        ]
    )

    rows = []

    for metric in [
        "direct_sfa_pp",
        "financial_asset_acquisition_pp",
        "other_adjustments_pp",
        "statistical_discrepancy_pp"
    ]:
        if (
            (metric, "MIO_NAC") not in pivot.columns
            or (metric, "PC_GDP") not in pivot.columns
        ):
            continue

        paired = pivot[
            [(metric, "MIO_NAC"), (metric, "PC_GDP")]
        ].dropna()

        differences = (
            paired[(metric, "MIO_NAC")]
            - paired[(metric, "PC_GDP")]
        )

        rows.append({
            "metric": metric,
            "paired_observations": len(differences),
            "mean_absolute_difference_pp": differences.abs().mean(),
            "maximum_absolute_difference_pp": differences.abs().max()
        })

    return pd.DataFrame(rows)


def write_report(sfa, data, summary, cross_unit):
    lines = []

    lines.append("EUROSTAT SFA COMPONENT DECOMPOSITION")
    lines.append("=" * 78)
    lines.append(f"Input: {SFA_FILE}")
    lines.append(f"GDP input: {GDP_FILE}")
    lines.append(f"Countries: {sfa['country'].nunique()}")
    lines.append(f"Years: {sorted(sfa['year'].unique().tolist())}")
    lines.append(f"Units: {sorted(sfa['unit'].unique().tolist())}")
    lines.append("")
    lines.append("OFFICIAL ACCOUNTING STRUCTURE")
    lines.append("-" * 78)
    lines.append(
        "Change in consolidated gross debt = "
        "net borrowing contribution"
    )
    lines.append(
        "    + net acquisition of financial assets"
    )
    lines.append(
        "    + adjustments to other flows and EDP debt definition"
    )
    lines.append(
        "    + statistical discrepancies"
    )
    lines.append("")
    lines.append("SFA COMPONENT BRIDGE")
    lines.append("-" * 78)
    lines.append("Direct SFA = F_ASS + ORADJ + YA3")
    lines.append(
        "All monetary components are expressed as percentage points "
        "of current-year GDP."
    )
    lines.append(
        "MIO_NAC values are divided by current-year nominal GDP; "
        "PC_GDP values are already expressed as percentages of GDP."
    )
    lines.append("")
    lines.append("COVERAGE")
    lines.append("-" * 78)

    for unit in EXPECTED_UNITS:
        unit_data = data[data["unit"].eq(unit)]
        complete = unit_data[
            unit_data["bridge_status"].eq("COMPLETE")
        ]

        lines.append(
            f"{unit}: {len(unit_data)} source records; "
            f"{len(complete)} complete decompositions"
        )

    lines.append("")
    lines.append("BRIDGE RESIDUALS")
    lines.append("-" * 78)

    for unit in EXPECTED_UNITS:
        subset = data[
            data["unit"].eq(unit)
            & data["bridge_status"].eq("COMPLETE")
        ]

        if subset.empty:
            continue

        lines.append(f"{unit}")
        lines.append(
            f"  Comparisons: {len(subset)}"
        )
        lines.append(
            "  Mean absolute residual: "
            f"{subset['bridge_residual_pp'].abs().mean():.8f} pp"
        )
        lines.append(
            "  Maximum absolute residual: "
            f"{subset['bridge_residual_pp'].abs().max():.8f} pp"
        )

    lines.append("")
    lines.append("CROSS-UNIT COMPARISON")
    lines.append("-" * 78)

    if cross_unit.empty:
        lines.append("No paired cross-unit observations.")
    else:
        for _, row in cross_unit.iterrows():
            lines.append(
                f"{row['metric']}: "
                f"{int(row['paired_observations'])} paired observations; "
                f"mean absolute difference "
                f"{row['mean_absolute_difference_pp']:.8f} pp; "
                f"maximum absolute difference "
                f"{row['maximum_absolute_difference_pp']:.8f} pp"
            )

    lines.append("")
    lines.append("INTERPRETATION LIMITATIONS")
    lines.append("-" * 78)
    lines.append(
        "1. This is an accounting decomposition, not a causal model."
    )
    lines.append(
        "2. F_ASS is an aggregate financial-asset acquisition component; "
        "do not add its lower-level subcomponents again."
    )
    lines.append(
        "3. ORADJ is an aggregate adjustment component; its details "
        "must not be added again to the top-level bridge."
    )
    lines.append(
        "4. Missing component values are not treated as zero."
    )
    lines.append(
        "5. Cross-unit comparisons are affected by source rounding."
    )
    lines.append(
        "6. The decomposition covers 17 EU countries and 2020-2024; "
        "it does not cover Norway, Switzerland or the UK."
    )
    lines.append(
        "7. This SFA bridge is separate from the interest-growth "
        "and primary-balance decomposition."
    )
    lines.append(
        "8. A small residual confirms numerical reconciliation, "
        "not independent causal attribution."
    )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    sfa, gdp = load_inputs()

    countries = sorted(sfa["country"].unique())
    years = sorted(sfa["year"].unique())
    units = sorted(sfa["unit"].unique())

    if len(countries) != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTRIES} countries; found {len(countries)}"
        )

    if years != EXPECTED_YEARS:
        raise ValueError(f"Unexpected year coverage: {years}")

    if units != EXPECTED_UNITS:
        raise ValueError(f"Unexpected units: {units}")

    data = build_decomposition(sfa, gdp)
    summary = build_summary(data)
    cross_unit = build_cross_unit_check(data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    cross_unit.to_csv(
        OUTPUT_FILE.parent / "europe_sfa_component_cross_unit_check.csv",
        index=False
    )

    write_report(sfa, data, summary, cross_unit)

    print("EUROSTAT SFA COMPONENT DECOMPOSITION")
    print("=" * 65)
    print(f"Source records: {len(sfa)}")
    print(f"Countries: {len(countries)}")
    print(f"Years: {years}")
    print(f"Units: {units}")
    print(
        "Complete decompositions: "
        f"{(data['bridge_status'] == 'COMPLETE').sum()}"
    )
    print(f"Output: {OUTPUT_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("")
    print("DECOMPOSITION BY UNIT")

    for unit in EXPECTED_UNITS:
        subset = data[
            data["unit"].eq(unit)
            & data["bridge_status"].eq("COMPLETE")
        ]

        if subset.empty:
            continue

        print(
            f"{unit}: {len(subset)} observations; "
            f"maximum absolute residual = "
            f"{subset['bridge_residual_pp'].abs().max():.8f} pp"
        )


if __name__ == "__main__":
    main()