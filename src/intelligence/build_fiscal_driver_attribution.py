from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path(
    "data/processed/europe_annual_debt_decomposition.csv"
)

OUTPUT = Path(
    "data/processed/europe_fiscal_driver_attribution.csv"
)

REPORT = Path(
    "outputs/reports/europe_fiscal_driver_attribution_report.txt"
)


def direction(value):
    if value > 0:
        return "increased_debt_ratio"
    if value < 0:
        return "reduced_debt_ratio"
    return "neutral"


def main():
    df = pd.read_csv(INPUT)

    required = [
        "iso",
        "country",
        "year",
        "debt_change_pp",
        "primary_balance_contribution_pp",
        "non_primary_balance_residual_pp",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[df["year"].between(2020, 2024)].copy()

    if len(df) != 100:
        raise ValueError(
            f"Expected 100 observations; found {len(df)}"
        )

    if df.duplicated(["iso", "year"]).any():
        raise ValueError("Duplicate country-year records.")

    numeric = [
        "debt_change_pp",
        "primary_balance_contribution_pp",
        "non_primary_balance_residual_pp",
    ]

    if df[numeric].isna().any().any():
        raise ValueError("Missing contribution values.")

    # Verify the annual accounting identity.
    annual_error = (
        df["debt_change_pp"]
        - df["primary_balance_contribution_pp"]
        - df["non_primary_balance_residual_pp"]
    )

    if not np.allclose(annual_error, 0, atol=1e-10):
        raise ValueError("Annual decomposition failed.")

    # Aggregate five annual transitions.
    result = df.groupby(
        ["iso", "country"], as_index=False
    ).agg(
        observed_debt_change_pp=(
            "debt_change_pp", "sum"
        ),
        cumulative_primary_contribution_pp=(
            "primary_balance_contribution_pp", "sum"
        ),
        cumulative_non_primary_residual_pp=(
            "non_primary_balance_residual_pp", "sum"
        ),
        mean_annual_nominal_gdp_growth_pct=(
            "year", "count"
        ),
        annual_observations=("year", "count"),
    )

    # Add average nominal GDP growth separately.
    growth = pd.read_csv(
        "data/processed/europe_annual_debt_decomposition.csv"
    )

    growth = growth[
        growth["year"].between(2020, 2024)
    ].groupby("iso", as_index=False).agg(
        mean_annual_nominal_gdp_growth_pct=(
            "nominal_gdp_growth_pct", "mean"
        )
    )

    result = result.drop(
        columns=["mean_annual_nominal_gdp_growth_pct"]
    ).merge(
        growth,
        on="iso",
        how="left",
        validate="one_to_one",
    )

    result["reconstructed_debt_change_pp"] = (
        result["cumulative_primary_contribution_pp"]
        + result["cumulative_non_primary_residual_pp"]
    )

    result["reconciliation_error_pp"] = (
        result["observed_debt_change_pp"]
        - result["reconstructed_debt_change_pp"]
    )

    if not np.allclose(
        result["reconciliation_error_pp"],
        0,
        atol=1e-10,
    ):
        raise ValueError(
            "Cumulative contribution reconciliation failed."
        )

    result["primary_contribution_direction"] = (
        result["cumulative_primary_contribution_pp"]
        .apply(direction)
    )

    result["residual_contribution_direction"] = (
        result["cumulative_non_primary_residual_pp"]
        .apply(direction)
    )

    # Relative contribution shares are deliberately omitted.
    # They can be misleading when components offset one another
    # or the observed net debt change is close to zero.

    result = result.sort_values(
        "observed_debt_change_pp",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT, index=False)

    report = [
        "EUROPEAN FISCAL DRIVER ATTRIBUTION",
        "=" * 65,
        "Period: 2019-2024",
        "Transitions: 2020-2024",
        f"Countries: {len(result)}",
        f"Annual observations: {len(df)}",
        "",
        "CUMULATIVE CONTRIBUTIONS",
        result[
            [
                "iso",
                "country",
                "observed_debt_change_pp",
                "cumulative_primary_contribution_pp",
                "cumulative_non_primary_residual_pp",
                "reconciliation_error_pp",
            ]
        ].round(4).to_string(index=False),
        "",
        "PRIMARY CONTRIBUTION DIRECTIONS",
        result[
            "primary_contribution_direction"
        ].value_counts().to_string(),
        "",
        "RESIDUAL CONTRIBUTION DIRECTIONS",
        result[
            "residual_contribution_direction"
        ].value_counts().to_string(),
        "",
        "VALIDATION",
        f"Maximum reconciliation error: "
        f"{result['reconciliation_error_pp'].abs().max():.12f}",
        f"Countries with five transitions: "
        f"{(result['annual_observations'] == 5).sum()}",
        "",
        "LIMITATIONS",
        "- The residual combines multiple economic and accounting effects.",
        "- It is not an independently measured interest-growth contribution.",
        "- Contribution shares are not calculated.",
        "- This is an accounting attribution, not causal identification.",
    ]

    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n".join(report))
    print(f"\nDataset: {OUTPUT}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
