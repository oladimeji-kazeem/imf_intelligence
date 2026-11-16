from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/europe_sovereign_debt_bridge.csv"

OUTPUT_PANEL = ROOT / "data/processed/europe_debt_driver_diagnostics.csv"
OUTPUT_SUMMARY = ROOT / "data/processed/europe_debt_driver_country_profiles.csv"
OUTPUT_DETAILED = ROOT / "data/processed/europe_debt_driver_oradj_profiles.csv"
OUTPUT_REPORT = ROOT / "outputs/reports/europe_debt_driver_diagnostics_report.txt"

YEARS = [2020, 2021, 2022, 2023, 2024]

TOP_COMPONENTS = {
    "interest_growth": "interest_growth_contribution_pp",
    "primary_balance": "primary_contribution_pp",
    "financial_asset_acquisition": "F_ASS_pp",
    "other_debt_adjustments": "ORADJ_pp",
    "statistical_discrepancy": "YA3_pp",
}

ORADJ_COMPONENTS = [
    "excluded_liability_adjustments_pp",
    "debt_valuation_adjustments_pp",
    "foreign_currency_valuation_pp",
    "other_volume_changes_pp",
]


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "country_code",
        "country",
        "year",
        "top_level_sfa_complete",
        "detailed_bridge_complete",
        "observed_debt_change_pp",
        *TOP_COMPONENTS.values(),
        *ORADJ_COMPONENTS,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["top_level_sfa_complete"] = (
        df["top_level_sfa_complete"].astype(str).str.lower() == "true"
    )
    df["detailed_bridge_complete"] = (
        df["detailed_bridge_complete"].astype(str).str.lower() == "true"
    )

    if df.duplicated(["country_code", "year"]).any():
        raise ValueError("Duplicate country-year observations detected.")

    return df


def classify_dominant_contribution(row, columns):
    positive = {
        name: row[column]
        for name, column in columns.items()
        if pd.notna(row[column]) and row[column] > 0
    }

    negative = {
        name: row[column]
        for name, column in columns.items()
        if pd.notna(row[column]) and row[column] < 0
    }

    largest_positive = (
        max(positive, key=positive.get) if positive else "none"
    )

    largest_negative = (
        min(negative, key=negative.get) if negative else "none"
    )

    return largest_positive, largest_negative


def calculate_country_profiles(df):
    records = []

    for code, group in df.groupby("country_code"):
        group = group.sort_values("year")

        # Only use the balanced five-year panel for cumulative comparisons.
        complete = group[group["top_level_sfa_complete"]].copy()

        if len(complete) != 5 or complete["year"].astype(int).tolist() != YEARS:
            continue

        totals = {}

        for name, column in TOP_COMPONENTS.items():
            totals[name] = complete[column].sum()

        debt_change = complete["observed_debt_change_pp"].sum()

        positive_total = sum(
            max(value, 0) for value in totals.values()
        )

        negative_total = sum(
            min(value, 0) for value in totals.values()
        )

        largest_up, largest_down = classify_dominant_contribution(
            totals,
            {name: name for name in totals},
        )

        # Measure how much gross positive and negative movement offset.
        gross_movement = positive_total + abs(negative_total)

        offset_ratio = (
            1 - abs(debt_change) / gross_movement
            if gross_movement > 0
            else np.nan
        )

        records.append({
            "country_code": code,
            "country": group["country"].iloc[0],
            "complete_years": 5,
            "debt_change_pp": debt_change,
            "interest_growth_pp": totals["interest_growth"],
            "primary_balance_contribution_pp": totals["primary_balance"],
            "financial_asset_acquisition_pp": totals[
                "financial_asset_acquisition"
            ],
            "oradj_pp": totals["other_debt_adjustments"],
            "statistical_discrepancy_pp": totals[
                "statistical_discrepancy"
            ],
            "gross_positive_contributions_pp": positive_total,
            "gross_negative_contributions_pp": negative_total,
            "gross_offsetting_movement_pp": gross_movement,
            "offset_ratio": offset_ratio,
            "largest_upward_contribution": largest_up,
            "largest_downward_contribution": largest_down,
            "reconciliation_error_pp": (
                complete["debt_bridge_residual_pp"].sum()
            ),
        })

    result = pd.DataFrame(records)

    if result.empty:
        raise ValueError("No countries have five complete top-level years.")

    return result.sort_values("debt_change_pp", ascending=False)


def calculate_oradj_profiles(df):
    records = []

    for code, group in df.groupby("country_code"):
        group = group.sort_values("year")

        complete = group[group["detailed_bridge_complete"]].copy()

        if len(complete) != 5 or complete["year"].astype(int).tolist() != YEARS:
            continue

        values = {
            column: complete[column].sum()
            for column in ORADJ_COMPONENTS
        }

        positive = sum(max(value, 0) for value in values.values())
        negative = sum(min(value, 0) for value in values.values())

        records.append({
            "country_code": code,
            "country": group["country"].iloc[0],
            "complete_years": 5,
            **values,
            "gross_positive_oradj_categories_pp": positive,
            "gross_negative_oradj_categories_pp": negative,
            "oradj_category_sum_pp": sum(values.values()),
            "oradj_aggregate_pp": complete["ORADJ_pp"].sum(),
            "category_reconciliation_error_pp": (
                complete["oradj_category_residual_pp"].sum()
            ),
        })

    return pd.DataFrame(records).sort_values("country_code")


def create_annual_diagnostics(df):
    panel = df.copy()

    component_columns = list(TOP_COMPONENTS.values())

    panel["gross_positive_contributions_pp"] = panel[
        component_columns
    ].clip(lower=0).sum(axis=1, min_count=len(component_columns))

    panel["gross_negative_contributions_pp"] = panel[
        component_columns
    ].clip(upper=0).sum(axis=1, min_count=len(component_columns))

    panel["annual_component_sum_pp"] = panel[
        component_columns
    ].sum(axis=1, min_count=len(component_columns))

    panel["annual_bridge_difference_pp"] = (
        panel["observed_debt_change_pp"]
        - panel["annual_component_sum_pp"]
    )

    return panel


def write_report(panel, profiles, oradj, report):
    lines = []

    lines.append("EUROPEAN DEBT DRIVER DIAGNOSTICS")
    lines.append("=" * 78)

    lines.append(f"Annual panel observations: {len(panel)}")
    lines.append(f"Countries: {panel['country_code'].nunique()}")
    lines.append(f"Years: {YEARS}")
    lines.append("")

    lines.append("BALANCED TOP-LEVEL COUNTRY PANEL")
    lines.append("-" * 78)
    lines.append(f"Countries: {len(profiles)}")
    lines.append("")

    lines.append("COUNTRY DEBT CHANGE RANKING")
    lines.append("-" * 78)

    for row in profiles.sort_values(
        "debt_change_pp", ascending=False
    ).itertuples():
        lines.append(
            f"{row.country_code}: debt change={row.debt_change_pp:+.4f} pp; "
            f"IGC={row.interest_growth_pp:+.4f}; "
            f"primary={row.primary_balance_contribution_pp:+.4f}; "
            f"F_ASS={row.financial_asset_acquisition_pp:+.4f}; "
            f"ORADJ={row.oradj_pp:+.4f}; "
            f"YA3={row.statistical_discrepancy_pp:+.4f}"
        )

    lines.append("")
    lines.append("DOMINANT CONTRIBUTIONS")
    lines.append("-" * 78)

    for row in profiles.itertuples():
        lines.append(
            f"{row.country_code}: largest upward={row.largest_upward_contribution}; "
            f"largest downward={row.largest_downward_contribution}; "
            f"offset ratio={row.offset_ratio:.4f}"
        )

    lines.append("")
    lines.append("ORADJ CATEGORY PROFILES")
    lines.append("-" * 78)
    lines.append(f"Five-year balanced countries: {len(oradj)}")

    for row in oradj.itertuples():
        lines.append(
            f"{row.country_code}: excluded liabilities="
            f"{row.excluded_liability_adjustments_pp:+.4f}; "
            f"debt valuation={row.debt_valuation_adjustments_pp:+.4f}; "
            f"FX={row.foreign_currency_valuation_pp:+.4f}; "
            f"other volume={row.other_volume_changes_pp:+.4f}; "
            f"ORADJ={row.oradj_aggregate_pp:+.4f}"
        )

    lines.append("")
    lines.append("INTERPRETATION LIMITATIONS")
    lines.append("-" * 78)

    limitations = [
        "Country cumulative comparisons use five complete years only.",
        "The balanced top-level sample and detailed ORADJ sample differ.",
        "Positive primary contribution represents a primary deficit.",
        "Positive financial asset acquisition can increase gross debt while adding assets.",
        "Interest-growth contribution uses an opening-debt-implied interest-rate proxy.",
        "ORADJ categories are accounting adjustments, not causal estimates.",
        "The offset ratio measures arithmetic cancellation, not economic risk.",
        "No contribution shares are calculated against net debt changes.",
        "No composite sovereign risk score or default probability is inferred.",
    ]

    lines.extend(
        f"{i}. {item}" for i, item in enumerate(limitations, 1)
    )

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_data()

    panel = create_annual_diagnostics(df)
    profiles = calculate_country_profiles(df)
    oradj = calculate_oradj_profiles(df)

    OUTPUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    panel.to_csv(OUTPUT_PANEL, index=False)
    profiles.to_csv(OUTPUT_SUMMARY, index=False)
    oradj.to_csv(OUTPUT_DETAILED, index=False)

    write_report(panel, profiles, oradj, OUTPUT_REPORT)

    print("EUROPEAN DEBT DRIVER DIAGNOSTICS")
    print("=" * 78)
    print(f"Annual observations: {len(panel)}")
    print(f"Balanced top-level countries: {len(profiles)}")
    print(f"Balanced detailed ORADJ countries: {len(oradj)}")
    print()
    print("OUTPUTS")
    print(OUTPUT_PANEL)
    print(OUTPUT_SUMMARY)
    print(OUTPUT_DETAILED)
    print(OUTPUT_REPORT)
    print()
    print("RESULT: COMPLETE")


if __name__ == "__main__":
    main()
