from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/europe_fiscal_risk_input_panel.csv")

OUTPUT = Path(
    "data/processed/europe_fiscal_risk_indicators.csv"
)

REPORT = Path(
    "outputs/reports/europe_fiscal_risk_indicators_report.txt"
)

REFERENCE_YEAR = 2024

def risk_percentile(series, higher_is_riskier=True):
    """
    Calculate relative risk percentile.

    Higher percentile means greater relative risk.
    This is a peer-relative measure, not an absolute risk score.
    """
    return series.rank(
        method="average",
        ascending=higher_is_riskier,
        pct=True
    ) * 100


def main():
    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "year",
        "government_debt",
        "fiscal_balance",
        "primary_balance",
        "debt_change_pp",
        "nominal_gdp_growth_pct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    snapshot = df[
        df["year"] == REFERENCE_YEAR
    ].copy()

    if snapshot.empty:
        raise ValueError(
            f"No records found for {REFERENCE_YEAR}"
        )

    # Directional relative risk indicators.
    # Higher debt = higher relative risk.
    snapshot["debt_burden_percentile"] = risk_percentile(
        snapshot["government_debt"],
        higher_is_riskier=True
    )

    # Lower fiscal balance = higher relative risk.
    snapshot["fiscal_pressure_percentile"] = risk_percentile(
        snapshot["fiscal_balance"],
        higher_is_riskier=False
    )

    # Lower primary balance = higher relative risk.
    snapshot["primary_pressure_percentile"] = risk_percentile(
        snapshot["primary_balance"],
        higher_is_riskier=False
    )

    # Larger positive increases in debt ratio = higher relative risk.
    snapshot["debt_trajectory_percentile"] = risk_percentile(
        snapshot["debt_change_pp"],
        higher_is_riskier=True
    )

    # Nominal GDP growth is contextual.
    # Flag negative nominal growth, but do not rank it as a
    # standalone measure of fiscal sustainability.
    snapshot["nominal_growth_condition"] = (
        snapshot["nominal_gdp_growth_pct"].apply(
            lambda x: (
                "negative_nominal_growth"
                if pd.notna(x) and x < 0
                else (
                    "non_negative_nominal_growth"
                    if pd.notna(x)
                    else "missing"
                )
            )
        )
    )

    # Preserve the exceptional transition observations.
    snapshot["extreme_nominal_growth_flag"] = (
        snapshot["nominal_gdp_growth_pct"] > 100
    )

    # Historical volatility: descriptive context only.
    historical = df[df["year"] <= REFERENCE_YEAR].copy()

    volatility = (
        historical.groupby(["iso", "country"])
        .agg(
            debt_volatility=("government_debt", "std"),
            fiscal_balance_volatility=("fiscal_balance", "std"),
            primary_balance_volatility=("primary_balance", "std"),
            debt_change_volatility=("debt_change_pp", "std"),
            historical_observations=("year", "count")
        )
        .reset_index()
    )

    result = snapshot.merge(
        volatility,
        on=["iso", "country"],
        how="left"
    )

    # Explicitly record the comparison basis.
    result["reference_year"] = REFERENCE_YEAR
    result["risk_measurement_type"] = "peer_relative_percentile"

    output_columns = [
        "iso",
        "country",
        "reference_year",
        "government_debt",
        "debt_burden_percentile",
        "fiscal_balance",
        "fiscal_pressure_percentile",
        "primary_balance",
        "primary_pressure_percentile",
        "debt_change_pp",
        "debt_trajectory_percentile",
        "nominal_gdp_growth_pct",
        "nominal_growth_condition",
        "extreme_nominal_growth_flag",
        "debt_volatility",
        "fiscal_balance_volatility",
        "primary_balance_volatility",
        "debt_change_volatility",
        "historical_observations",
    ]

    result = result[output_columns].sort_values(
        "debt_burden_percentile",
        ascending=False
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT, index=False)

    report = [
        "EUROPEAN FISCAL RISK INDICATORS REPORT",
        "=" * 55,
        f"Reference year: {REFERENCE_YEAR}",
        f"Countries: {result['iso'].nunique()}",
        f"Records: {len(result)}",
        "",
        "Relative percentile availability:",
        f"Debt burden: "
        f"{result['debt_burden_percentile'].notna().sum()}",
        f"Fiscal pressure: "
        f"{result['fiscal_pressure_percentile'].notna().sum()}",
        f"Primary pressure: "
        f"{result['primary_pressure_percentile'].notna().sum()}",
        f"Debt trajectory: "
        f"{result['debt_trajectory_percentile'].notna().sum()}",
        "",
        "Nominal growth conditions:",
        result["nominal_growth_condition"].value_counts().to_string(),
        "",
        "Extreme nominal growth observations above 100%:",
        str(result["extreme_nominal_growth_flag"].sum()),
        "",
        "Historical volatility missing values:",
        result[
            [
                "debt_volatility",
                "fiscal_balance_volatility",
                "primary_balance_volatility",
                "debt_change_volatility",
            ]
        ].isna().sum().to_string(),
        "",
        "Methodological limitations:",
        "- Percentiles are peer-relative, not absolute risk measures.",
        "- Nominal GDP growth is contextual, not independently ranked.",
        "- Historical volatility uses available observations.",
        "- Historical sample lengths differ by country.",
        "- No composite risk score is calculated.",
        "- No sovereign credit rating or default probability is inferred.",
    ]

    REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Report saved: {REPORT}")

    print("\nTOP 10 COUNTRIES BY RELATIVE DEBT BURDEN")
    print(
        result[
            [
                "country",
                "government_debt",
                "debt_burden_percentile",
                "debt_change_pp",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
