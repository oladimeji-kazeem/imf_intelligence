from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path(
    "data/processed/europe_fiscal_risk_input_panel.csv"
)

ANNUAL_OUTPUT = Path(
    "data/processed/europe_annual_debt_decomposition.csv"
)

SUMMARY_OUTPUT = Path(
    "data/processed/europe_debt_decomposition_summary.csv"
)

REPORT = Path(
    "outputs/reports/europe_annual_debt_decomposition_report.txt"
)

REQUIRED = [
    "iso",
    "country",
    "year",
    "government_debt",
    "primary_balance",
    "nominal_gdp_growth_pct",
    "debt_change_pp",
    "previous_year",
    "previous_debt",
]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    missing_columns = set(REQUIRED) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df = df[df["year"].between(2019, 2024)].copy()

    if len(df) != 120:
        raise ValueError(
            f"Expected 120 records; found {len(df)}."
        )

    if df.duplicated(["iso", "year"]).any():
        raise ValueError("Duplicate country-year records found.")

    numeric_columns = [
        "government_debt",
        "primary_balance",
        "nominal_gdp_growth_pct",
        "debt_change_pp",
        "previous_year",
        "previous_debt",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or invalid numeric values detected."
        )

    # Verify that each annual debt change uses the
    # immediately preceding calendar year's debt.
    if not (df["previous_year"] == df["year"] - 1).all():
        raise ValueError(
            "Non-consecutive lag years detected."
        )

    # Verify the debt-change calculation.
    calculated_change = (
        df["government_debt"] - df["previous_debt"]
    )

    df["debt_change_formula_error"] = (
        df["debt_change_pp"] - calculated_change
    )

    if not np.allclose(
        df["debt_change_formula_error"], 0, atol=0.002
    ):
        raise ValueError(
            "Debt change does not reconcile with debt levels."
        )

    # Annual accounting decomposition.
    df["primary_balance_contribution_pp"] = (
        -df["primary_balance"]
    )

    df["non_primary_balance_residual_pp"] = (
        df["debt_change_pp"] + df["primary_balance"]
    )

    df["reconstructed_debt_change_pp"] = (
        df["primary_balance_contribution_pp"]
        + df["non_primary_balance_residual_pp"]
    )

    df["decomposition_error_pp"] = (
        df["debt_change_pp"]
        - df["reconstructed_debt_change_pp"]
    )

    if not np.allclose(
        df["decomposition_error_pp"], 0, atol=1e-10
    ):
        raise ValueError(
            "Debt decomposition reconciliation failed."
        )

    # Interpret annual contributions without assigning
    # unsupported causal labels to the residual.
    df["primary_balance_effect"] = np.select(
        [
            df["primary_balance_contribution_pp"] > 0,
            df["primary_balance_contribution_pp"] < 0,
        ],
        [
            "increased_debt_ratio",
            "reduced_debt_ratio",
        ],
        default="neutral",
    )

    df["residual_effect"] = np.select(
        [
            df["non_primary_balance_residual_pp"] > 0,
            df["non_primary_balance_residual_pp"] < 0,
        ],
        [
            "increased_debt_ratio",
            "reduced_debt_ratio",
        ],
        default="neutral",
    )

    annual_columns = [
        "iso",
        "country",
        "year",
        "government_debt",
        "previous_debt",
        "nominal_gdp_growth_pct",
        "primary_balance",
        "debt_change_pp",
        "primary_balance_contribution_pp",
        "non_primary_balance_residual_pp",
        "reconstructed_debt_change_pp",
        "decomposition_error_pp",
        "primary_balance_effect",
        "residual_effect",
    ]

    annual = df[annual_columns].sort_values(
        ["iso", "year"]
    ).reset_index(drop=True)

    # Country-level summaries.
    # Do not sum annual percentage-of-GDP contributions:
      # Country-level summaries.
    # The 2019 row measures the 2018-2019 transition.
    # For the 2019-2024 period, use only transitions
    # from 2020 through 2024.

    period = annual[
        annual["year"].between(2020, 2024)
    ].copy()

    if len(period) != 100:
        raise ValueError(
            f"Expected 100 period transitions; found {len(period)}."
        )

    summary = period.groupby(
        ["iso", "country"], as_index=False
    ).agg(
        debt_change_2019_2024_pp=(
            "debt_change_pp", "sum"
        ),
        mean_annual_debt_change_pp=(
            "debt_change_pp", "mean"
        ),
        mean_primary_balance_pct_gdp=(
            "primary_balance", "mean"
        ),
        mean_primary_contribution_pp=(
            "primary_balance_contribution_pp", "mean"
        ),
        mean_non_primary_residual_pp=(
            "non_primary_balance_residual_pp", "mean"
        ),
        mean_nominal_gdp_growth_pct=(
            "nominal_gdp_growth_pct", "mean"
        ),
        annual_observations=("year", "count"),
    )

    # Endpoint comparison: 2024 debt minus 2019 debt.
    endpoint = annual.groupby("iso").agg(
        debt_2019=("government_debt", "first"),
        debt_2024=("government_debt", "last"),
    ).reset_index()

    endpoint["endpoint_debt_change_pp"] = (
        endpoint["debt_2024"] - endpoint["debt_2019"]
    )

    summary = summary.merge(
        endpoint[
            ["iso", "endpoint_debt_change_pp"]
        ],
        on="iso",
        how="left",
        validate="one_to_one",
    )

    summary["period_reconciliation_error_pp"] = (
        summary["debt_change_2019_2024_pp"]
        - summary["endpoint_debt_change_pp"]
    )

    if not np.allclose(
        summary["period_reconciliation_error_pp"],
        0,
        atol=0.002,
    ):
        raise ValueError(
            "Period debt movement does not reconcile."
        )

    if not (summary["annual_observations"] == 5).all():
        raise ValueError(
            "Every country must have five annual transitions."
        )


    ANNUAL_OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )
    SUMMARY_OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )
    REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )

    annual.to_csv(ANNUAL_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    report_lines = [
        "EUROPEAN ANNUAL DEBT DYNAMICS DECOMPOSITION",
        "=" * 65,
        "Period: 2019-2024",
        f"Countries: {annual['iso'].nunique()}",
        f"Annual dataset observations: {len(annual)}",
        f"Period transition observations: {len(period)}"
        "",
        "DECOMPOSITION",
        "Debt change = Primary balance contribution + Residual",
        "Residual = Debt change + Primary balance",
        "",
        "VALIDATION",
        f"Duplicate country-year records: "
        f"{annual.duplicated(['iso', 'year']).sum()}",
        f"Maximum annual reconciliation error: "
        f"{annual['decomposition_error_pp'].abs().max():.12f}",
        f"Maximum period reconciliation error: "
        f"{summary['period_reconciliation_error_pp'].abs().max():.12f}",
        "",
        "MEAN ANNUAL CONTRIBUTIONS",
        f"Mean primary balance contribution: "
        f"{period['primary_balance_contribution_pp'].mean():.4f} pp",
        f"Mean non-primary-balance residual: "
        f"{period['non_primary_balance_residual_pp'].mean():.4f} pp",
        f"Mean annual debt change: "
        f"{period['debt_change_pp'].mean():.4f} pp",
        "",
        "COUNTRY SUMMARY",
        summary.sort_values(
            "debt_change_2019_2024_pp",
            ascending=False
        ).to_string(index=False),
        "",
        "INTERPRETATION LIMITATIONS",
        "- Residual is not a pure interest-growth contribution.",
        "- Interest costs and stock-flow adjustments are not separately identified.",
        "- Annual contributions are expressed in percentage points.",
        "- Annual percentage-of-GDP contributions must not be summed",
        "  as if they were contributions measured against one fixed denominator.",
        "- This is an accounting decomposition, not a causal model.",
        "- No default probability or credit rating is calculated.",
    ]

    REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print(f"\nAnnual dataset: {ANNUAL_OUTPUT}")
    print(f"Country summary: {SUMMARY_OUTPUT}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
