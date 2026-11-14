from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SFA_FILE = ROOT / "data/processed/europe_eurostat_sfa_reconciliation.csv"
IGC_FILE = ROOT / "data/processed/europe_interest_growth_contribution.csv"
GDP_FILE = ROOT / "data/processed/europe_eurostat_edp_gdp_2019_2024.csv"
PB_FILE = ROOT / "data/processed/europe_eurostat_primary_balance.csv"

OUTPUT_FILE = ROOT / "data/processed/europe_sovereign_debt_bridge.csv"
SUMMARY_FILE = ROOT / "data/processed/europe_sovereign_debt_bridge_summary.csv"
COVERAGE_FILE = ROOT / "outputs/reports/europe_sovereign_debt_bridge_coverage.csv"
REPORT_FILE = ROOT / "outputs/reports/europe_sovereign_debt_bridge_report.txt"

EXPECTED_COUNTRIES = 17
EXPECTED_YEARS = [2020, 2021, 2022, 2023, 2024]
EXPECTED_OBSERVATIONS = EXPECTED_COUNTRIES * len(EXPECTED_YEARS)

SFA_COMPONENTS = ["F_ASS", "ORADJ", "YA3"]

ORADJ_CATEGORIES = {
    "excluded_liability_adjustments": [
        "F71_ADJ_LIAB", "F8_ADJ_LIAB", "FV_ADJ_LIAB"
    ],
    "debt_valuation_adjustments": [
        "ORINV", "ORD41A_ADJ", "ORRNV"
    ],
    "foreign_currency_valuation": ["ORFCD"],
    "other_volume_changes": ["KX", "K61"],
}

ALL_ORADJ_DETAILS = [
    component
    for components in ORADJ_CATEGORIES.values()
    for component in components
]


def normalise_code(value):
    """Normalise Eurostat country identifiers."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    aliases = {
        "GREECE": "EL",
        "HELLAS": "EL",
        "CZECHIA": "CZ",
        "CZECH REPUBLIC": "CZ",
        "DENMARK": "DK",
        "GERMANY": "DE",
        "FRANCE": "FR",
        "ITALY": "IT",
        "SPAIN": "ES",
        "PORTUGAL": "PT",
        "POLAND": "PL",
        "ROMANIA": "RO",
        "AUSTRIA": "AT",
        "BELGIUM": "BE",
        "FINLAND": "FI",
        "SWEDEN": "SE",
        "IRELAND": "IE",
        "NETHERLANDS": "NL",
        "THE NETHERLANDS": "NL",
        "HUNGARY": "HU",
    }

    return aliases.get(value, value)


def read_inputs():
    for path in [SFA_FILE, IGC_FILE, GDP_FILE, PB_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    sfa = pd.read_csv(SFA_FILE)
    igc = pd.read_csv(IGC_FILE)
    gdp = pd.read_csv(GDP_FILE)
    pb = pd.read_csv(PB_FILE)

    required = {
        "SFA": {
            "country", "year", "unit", "GD_CH", "B9_T3",
            "F_ASS", "ORADJ", "YA3",
            *ALL_ORADJ_DETAILS,
        },
        "IGC": {
            "country_code", "country", "year",
            "interest_growth_contribution_pp",
            "observed_debt_change_pp",
            "implied_sfa_pp",
        },
        "GDP": {"country_code", "year", "nominal_gdp"},
        "PB": {"country_code", "year", "primary_balance_pct_gdp"},
    }

    for name, frame, columns in [
        ("SFA", sfa, required["SFA"]),
        ("IGC", igc, required["IGC"]),
        ("GDP", gdp, required["GDP"]),
        ("PB", pb, required["PB"]),
    ]:
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} input is missing required columns: {sorted(missing)}"
            )

    sfa["country_code"] = sfa["country"].map(normalise_code)
    igc["country_code"] = igc["country_code"].map(normalise_code)
    gdp["country_code"] = gdp["country_code"].map(normalise_code)
    pb["country_code"] = pb["country_code"].map(normalise_code)

    for frame in [sfa, igc, gdp, pb]:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")

    return sfa, igc, gdp, pb


def check_unique(frame, keys, label):
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, keys].head(10)
        raise ValueError(
            f"{label} contains duplicate keys:\n{examples.to_string(index=False)}"
        )


def build_bridge(sfa, igc, gdp, pb):
    # Retain the complete interest-growth panel as the master country-year set.
    base = igc[
        [
            "country_code",
            "country",
            "year",
            "interest_growth_contribution_pp",
            "observed_debt_change_pp",
            "implied_sfa_pp",
        ]
    ].copy()

    check_unique(base, ["country_code", "year"], "IGC")
    check_unique(gdp, ["country_code", "year"], "GDP")
    check_unique(pb, ["country_code", "year"], "Primary balance")

    # GDP and primary balance are independent inputs used for cross-checking
    # and converting monetary SFA components to percentage points of GDP.
    base = base.merge(
        gdp[["country_code", "year", "nominal_gdp"]],
        on=["country_code", "year"],
        how="left",
        validate="one_to_one",
    )

    base = base.merge(
        pb[
            ["country_code", "year", "primary_balance_pct_gdp"]
        ],
        on=["country_code", "year"],
        how="left",
        validate="one_to_one",
    )

    # Use MIO_NAC for detailed monetary components.
    sfa_money = sfa[sfa["unit"] == "MIO_NAC"].copy()
    check_unique(sfa_money, ["country_code", "year"], "MIO_NAC SFA")

    sfa_columns = [
        "country_code",
        "year",
        "GD_CH",
        "B9_T3",
        "F_ASS",
        "ORADJ",
        "YA3",
        *ALL_ORADJ_DETAILS,
    ]

    base = base.merge(
        sfa_money[sfa_columns],
        on=["country_code", "year"],
        how="left",
        validate="one_to_one",
        indicator="sfa_match",
    )

    # Primary balance: positive = surplus.
    base["primary_contribution_pp"] = (
        -base["primary_balance_pct_gdp"]
    )

    # Direct SFA derived from Eurostat's debt-transition components.
    # GD_CH and B9_T3 are monetary amounts in national currency.
    base["direct_sfa_pp"] = (
        (base["GD_CH"] - base["B9_T3"])
        / base["nominal_gdp"]
        * 100
    )

    # Convert the top-level SFA components to percentage points of GDP.
    for component in SFA_COMPONENTS:
        base[f"{component}_pp"] = (
            base[component] / base["nominal_gdp"] * 100
        )

    base["top_level_sfa_component_sum_pp"] = (
        base["F_ASS_pp"] + base["ORADJ_pp"] + base["YA3_pp"]
    )

    top_level_complete = base[SFA_COMPONENTS].notna().all(axis=1)
    base["top_level_sfa_complete"] = top_level_complete

    # Independently compare the component sum with the direct SFA.
    base["top_level_sfa_component_residual_pp"] = (
        base["top_level_sfa_component_sum_pp"] - base["direct_sfa_pp"]
    )

    # Integrated debt bridge using the top-level SFA component sum.
    base["reconstructed_debt_change_pp"] = (
        base["interest_growth_contribution_pp"]
        + base["primary_contribution_pp"]
        + base["top_level_sfa_component_sum_pp"]
    )

    base["debt_bridge_residual_pp"] = (
        base["observed_debt_change_pp"]
        - base["reconstructed_debt_change_pp"]
    )

    # Alternative bridge using the direct SFA accounting result.
    base["direct_sfa_reconstructed_debt_change_pp"] = (
        base["interest_growth_contribution_pp"]
        + base["primary_contribution_pp"]
        + base["direct_sfa_pp"]
    )

    base["direct_sfa_bridge_residual_pp"] = (
        base["observed_debt_change_pp"]
        - base["direct_sfa_reconstructed_debt_change_pp"]
    )

    # Compare implied SFA from the debt bridge with direct SFA.
    base["implied_vs_direct_sfa_difference_pp"] = (
        base["implied_sfa_pp"] - base["direct_sfa_pp"]
    )

    # Convert detailed ORADJ components to percentage points of GDP.
    for component in ALL_ORADJ_DETAILS:
        base[f"{component}_pp"] = (
            base[component] / base["nominal_gdp"] * 100
        )

    # Build category totals only when every component in the category exists.
    for category, components in ORADJ_CATEGORIES.items():
        output_column = f"{category}_pp"
        complete_column = f"{category}_complete"

        base[complete_column] = base[components].notna().all(axis=1)

        detail_columns = [f"{component}_pp" for component in components]

        base[output_column] = np.where(
            base[complete_column],
            base[detail_columns].sum(axis=1, min_count=len(detail_columns)),
            np.nan,
        )

    base["all_oradj_categories_complete"] = (
        base[
            [f"{category}_complete" for category in ORADJ_CATEGORIES]
        ].all(axis=1)
    )

    category_columns = [
        f"{category}_pp" for category in ORADJ_CATEGORIES
    ]

    base["oradj_category_sum_pp"] = base[category_columns].sum(
        axis=1, min_count=len(category_columns)
    )

    base["oradj_category_residual_pp"] = (
        base["ORADJ_pp"] - base["oradj_category_sum_pp"]
    )

    # Detailed bridge is available only when all four ORADJ categories exist.
    base["detailed_bridge_complete"] = (
        base["top_level_sfa_complete"]
        & base["all_oradj_categories_complete"]
    )

    # Explicitly label missingness rather than treating it as zero.
    base["bridge_status"] = np.select(
        [
            base["top_level_sfa_complete"]
            & base["all_oradj_categories_complete"],
            base["top_level_sfa_complete"],
        ],
        [
            "complete_top_level_and_oradj_detail",
            "complete_top_level_sfa_only",
        ],
        default="incomplete_sfa_components",
    )

    return base


def build_summary(bridge):
    rows = []

    for country_code, group in bridge.groupby("country_code"):
        group = group.sort_values("year")

        valid_top = group[group["top_level_sfa_complete"]]
        valid_detail = group[group["detailed_bridge_complete"]]

        rows.append({
            "country_code": country_code,
            "country": group["country"].iloc[0],
            "expected_years": len(EXPECTED_YEARS),
            "observed_years": len(group),
            "top_level_complete_years": len(valid_top),
            "detailed_complete_years": len(valid_detail),
            "cumulative_debt_change_pp": (
                valid_top["observed_debt_change_pp"].sum()
                if len(valid_top) else np.nan
            ),
            "cumulative_interest_growth_pp": (
                valid_top["interest_growth_contribution_pp"].sum()
                if len(valid_top) else np.nan
            ),
            "cumulative_primary_contribution_pp": (
                valid_top["primary_contribution_pp"].sum()
                if len(valid_top) else np.nan
            ),
            "cumulative_financial_asset_acquisition_pp": (
                valid_top["F_ASS_pp"].sum()
                if len(valid_top) else np.nan
            ),
            "cumulative_oradj_pp": (
                valid_top["ORADJ_pp"].sum()
                if len(valid_top) else np.nan
            ),
            "cumulative_statistical_discrepancy_pp": (
                valid_top["YA3_pp"].sum()
                if len(valid_top) else np.nan
            ),
            "cumulative_bridge_residual_pp": (
                valid_top["debt_bridge_residual_pp"].sum()
                if len(valid_top) else np.nan
            ),
            "cumulative_excluded_liability_adjustments_pp": (
                valid_detail["excluded_liability_adjustments_pp"].sum()
                if len(valid_detail) else np.nan
            ),
            "cumulative_debt_valuation_adjustments_pp": (
                valid_detail["debt_valuation_adjustments_pp"].sum()
                if len(valid_detail) else np.nan
            ),
            "cumulative_foreign_currency_valuation_pp": (
                valid_detail["foreign_currency_valuation_pp"].sum()
                if len(valid_detail) else np.nan
            ),
            "cumulative_other_volume_changes_pp": (
                valid_detail["other_volume_changes_pp"].sum()
                if len(valid_detail) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def build_coverage(bridge):
    rows = []

    for country_code, group in bridge.groupby("country_code"):
        rows.append({
            "country_code": country_code,
            "country": group["country"].iloc[0],
            "expected_years": len(EXPECTED_YEARS),
            "igc_observations": len(group),
            "gdp_observations": group["nominal_gdp"].notna().sum(),
            "primary_balance_observations": (
                group["primary_balance_pct_gdp"].notna().sum()
            ),
            "direct_sfa_observations": group["direct_sfa_pp"].notna().sum(),
            "top_level_complete_years": (
                group["top_level_sfa_complete"].sum()
            ),
            "complete_oradj_detail_years": (
                group["all_oradj_categories_complete"].sum()
            ),
            "complete_detailed_bridge_years": (
                group["detailed_bridge_complete"].sum()
            ),
        })

    return pd.DataFrame(rows)


def validate(bridge):
    checks = []

    def record(name, passed, detail):
        checks.append((name, bool(passed), detail))

    record(
        "Expected IGC observations",
        len(bridge) == EXPECTED_OBSERVATIONS,
        f"{len(bridge)} / {EXPECTED_OBSERVATIONS}",
    )

    record(
        "Expected country count",
        bridge["country_code"].nunique() == EXPECTED_COUNTRIES,
        f"{bridge['country_code'].nunique()} countries",
    )

    years = sorted(bridge["year"].dropna().astype(int).unique().tolist())
    record(
        "Expected years",
        years == EXPECTED_YEARS,
        f"Years found: {years}",
    )

    record(
        "No duplicate country-year records",
        not bridge.duplicated(["country_code", "year"]).any(),
        "Country-year uniqueness",
    )

    record(
        "GDP complete",
        bridge["nominal_gdp"].notna().all(),
        f"Missing: {bridge['nominal_gdp'].isna().sum()}",
    )

    record(
        "Primary balance complete",
        bridge["primary_balance_pct_gdp"].notna().all(),
        f"Missing: {bridge['primary_balance_pct_gdp'].isna().sum()}",
    )

    record(
        "Positive GDP",
        (bridge["nominal_gdp"].dropna() > 0).all(),
        "GDP must be positive",
    )

    top = bridge[bridge["top_level_sfa_complete"]]
    record(
        "Top-level SFA component coverage",
        len(top) > 0,
        f"{len(top)} complete observations",
    )

    if len(top):
        max_component_residual = (
            top["top_level_sfa_component_residual_pp"].abs().max()
        )
        record(
            "Top-level SFA component reconciliation",
            max_component_residual <= 0.10,
            f"Maximum absolute difference: {max_component_residual:.8f} pp",
        )

        max_bridge_residual = top["debt_bridge_residual_pp"].abs().max()
        record(
            "Integrated debt bridge reconciliation",
            max_bridge_residual <= 0.10,
            f"Maximum absolute residual: {max_bridge_residual:.8f} pp",
        )

        max_direct_residual = (
            top["direct_sfa_bridge_residual_pp"].abs().max()
        )
        record(
            "Direct SFA debt bridge reconciliation",
            max_direct_residual <= 0.10,
            f"Maximum absolute residual: {max_direct_residual:.8f} pp",
        )

        max_implied_difference = (
            top["implied_vs_direct_sfa_difference_pp"].abs().max()
        )
        record(
            "Implied versus direct SFA",
            max_implied_difference <= 0.10,
            f"Maximum absolute difference: {max_implied_difference:.8f} pp",
        )

    detail = bridge[bridge["detailed_bridge_complete"]]
    record(
        "Detailed ORADJ category coverage",
        len(detail) > 0,
        f"{len(detail)} complete observations",
    )

    if len(detail):
        max_oradj_residual = (
            detail["oradj_category_residual_pp"].abs().max()
        )
        record(
            "ORADJ category reconciliation",
            max_oradj_residual <= 0.15,
            f"Maximum absolute residual: {max_oradj_residual:.8f} pp",
        )

    return checks


def write_report(bridge, summary, coverage, checks):
    lines = []

    lines.append("EUROPEAN SOVEREIGN DEBT BRIDGE")
    lines.append("=" * 78)
    lines.append(f"Input SFA: {SFA_FILE}")
    lines.append(f"Countries: {bridge['country_code'].nunique()}")
    lines.append(f"Years: {EXPECTED_YEARS}")
    lines.append(f"Master observations: {len(bridge)}")
    lines.append("")

    lines.append("INTEGRATED DEBT BRIDGE")
    lines.append("-" * 78)
    lines.append(
        "Debt change = interest-growth contribution - primary balance "
        "+ financial asset acquisition + ORADJ + statistical discrepancy"
    )
    lines.append(
        "Primary balance is positive for a surplus; its debt contribution "
        "is therefore the negative of the primary balance."
    )
    lines.append("")

    top = bridge[bridge["top_level_sfa_complete"]]
    detailed = bridge[bridge["detailed_bridge_complete"]]

    lines.append("OBSERVATION COVERAGE")
    lines.append("-" * 78)
    lines.append(f"Master IGC observations: {len(bridge)}")
    lines.append(f"Complete top-level SFA bridges: {len(top)}")
    lines.append(f"Complete ORADJ category bridges: {len(detailed)}")
    lines.append(
        f"Countries with at least one complete top-level bridge: "
        f"{top['country_code'].nunique()}"
    )
    lines.append(
        f"Countries with at least one complete detailed bridge: "
        f"{detailed['country_code'].nunique()}"
    )
    lines.append("")

    lines.append("TOP-LEVEL COMPONENT RECONCILIATION")
    lines.append("-" * 78)
    if len(top):
        lines.append(
            "Maximum absolute difference: "
            f"{top['top_level_sfa_component_residual_pp'].abs().max():.8f} pp"
        )
        lines.append(
            "Mean absolute difference: "
            f"{top['top_level_sfa_component_residual_pp'].abs().mean():.8f} pp"
        )
        lines.append(
            "Maximum integrated debt bridge residual: "
            f"{top['debt_bridge_residual_pp'].abs().max():.8f} pp"
        )
        lines.append(
            "Maximum direct-SFA debt bridge residual: "
            f"{top['direct_sfa_bridge_residual_pp'].abs().max():.8f} pp"
        )
    else:
        lines.append("No complete top-level bridges.")
    lines.append("")

    lines.append("COMPLETE DETAILED ORADJ BRIDGES")
    lines.append("-" * 78)
    lines.append(f"Complete country-year observations: {len(detailed)}")
    if len(detailed):
        lines.append(
            "Maximum absolute ORADJ category residual: "
            f"{detailed['oradj_category_residual_pp'].abs().max():.8f} pp"
        )
    lines.append("")

    lines.append("COUNTRY COVERAGE")
    lines.append("-" * 78)
    for row in coverage.sort_values("country_code").itertuples():
        lines.append(
            f"{row.country_code}: top-level {row.top_level_complete_years}/5; "
            f"ORADJ detail {row.complete_oradj_detail_years}/5; "
            f"detailed bridge {row.complete_detailed_bridge_years}/5"
        )
    lines.append("")

    lines.append("FIVE-YEAR BALANCED TOP-LEVEL PANEL")
    lines.append("-" * 78)
    balanced = summary[
        summary["top_level_complete_years"] == len(EXPECTED_YEARS)
    ]

    lines.append(f"Countries with five complete top-level years: {len(balanced)}")
    if len(balanced):
        for row in balanced.sort_values("country_code").itertuples():
            lines.append(
                f"{row.country_code}: debt change={row.cumulative_debt_change_pp:.5f} pp; "
                f"IGC={row.cumulative_interest_growth_pp:.5f}; "
                f"primary contribution={row.cumulative_primary_contribution_pp:.5f}; "
                f"F_ASS={row.cumulative_financial_asset_acquisition_pp:.5f}; "
                f"ORADJ={row.cumulative_oradj_pp:.5f}; "
                f"YA3={row.cumulative_statistical_discrepancy_pp:.5f}; "
                f"residual={row.cumulative_bridge_residual_pp:.6f}"
            )
    lines.append("")

    lines.append("FIVE-YEAR BALANCED DETAILED ORADJ PANEL")
    lines.append("-" * 78)
    balanced_detail = summary[
        summary["detailed_complete_years"] == len(EXPECTED_YEARS)
    ]
    lines.append(
        f"Countries with five complete detailed years: {len(balanced_detail)}"
    )
    if len(balanced_detail):
        for row in balanced_detail.sort_values("country_code").itertuples():
            lines.append(
                f"{row.country_code}: excluded liabilities="
                f"{row.cumulative_excluded_liability_adjustments_pp:.5f}; "
                f"debt valuation={row.cumulative_debt_valuation_adjustments_pp:.5f}; "
                f"FX valuation={row.cumulative_foreign_currency_valuation_pp:.5f}; "
                f"other volume={row.cumulative_other_volume_changes_pp:.5f}"
            )
    lines.append("")

    lines.append("VALIDATION CHECKS")
    lines.append("-" * 78)
    for name, passed, detail_text in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"{status}: {name} — {detail_text}")

    lines.append("")
    lines.append("METHODOLOGICAL LIMITATIONS")
    lines.append("-" * 78)
    limitations = [
        "The master panel is based on the 85 Eurostat interest-growth observations.",
        "SFA components are missing for some country-years; missing values are not zero.",
        "MIO_NAC values are converted using current-year nominal GDP.",
        "The primary balance is observed and not cyclically adjusted.",
        "The interest-growth rate is an opening-debt-implied proxy, not a separately sourced market yield.",
        "SFA and ORADJ components are accounting contributions, not causal estimates.",
        "The integrated bridge does not establish that a component was caused by fiscal policy.",
        "Country cumulative totals should only be compared over common observation windows.",
        "The detailed ORADJ categories are components of ORADJ and must not be added to ORADJ again.",
        "A small reconciliation residual does not independently validate every source definition.",
    ]
    lines.extend(f"{i}. {item}" for i, item in enumerate(limitations, 1))

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    sfa, igc, gdp, pb = read_inputs()

    bridge = build_bridge(sfa, igc, gdp, pb)
    summary = build_summary(bridge)
    coverage = build_coverage(bridge)
    checks = validate(bridge)

    bridge.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    coverage.to_csv(COVERAGE_FILE, index=False)

    write_report(bridge, summary, coverage, checks)

    print("EUROPEAN SOVEREIGN DEBT BRIDGE")
    print("=" * 78)
    print(f"Master observations: {len(bridge)}")
    print(f"Countries: {bridge['country_code'].nunique()}")
    print(f"Top-level complete observations: {bridge['top_level_sfa_complete'].sum()}")
    print(f"Detailed complete observations: {bridge['detailed_bridge_complete'].sum()}")
    print()

    print("VALIDATION")
    print("-" * 78)
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name} — {detail}")

    print()
    print("OUTPUTS")
    print("-" * 78)
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(COVERAGE_FILE)
    print(REPORT_FILE)

    if not all(passed for _, passed, _ in checks):
        raise SystemExit(
            "One or more validation checks failed. Inspect the report before proceeding."
        )

    print()
    print("RESULT: ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()