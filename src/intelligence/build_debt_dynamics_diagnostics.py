
from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/europe_multi_year_debt_dynamics.csv"
)

OUTPUT = Path(
    "data/processed/europe_debt_dynamics_diagnostics.csv"
)

REPORT = Path(
    "outputs/reports/europe_debt_dynamics_diagnostics_report.txt"
)

REQUIRED = [
    "iso",
    "country",
    "debt_change_pp_2019_2024",
    "primary_balance_change_pp_2019_2024",
    "debt_change_pp_2021_2024",
    "debt_change_pp_2023_2024",
    "primary_balance_mean_2020_2024",
    "primary_balance_observations_2020_2024",
    "government_debt_mean_2020_2024",
    "government_debt_observations_2020_2024",
    "debt_direction_2019_2024",
    "debt_direction_2021_2024",
    "debt_direction_2023_2024",
    "primary_balance_adjustment_2019_2024",
]


def classify_debt_trajectory(row):
    long_term = row["debt_direction_2019_2024"]
    recovery = row["debt_direction_2021_2024"]
    recent = row["debt_direction_2023_2024"]

    if (
        long_term == "debt_ratio_increased"
        and recovery == "debt_ratio_increased"
        and recent == "debt_ratio_increased"
    ):
        return "persistent_debt_expansion"

    if (
        long_term == "debt_ratio_increased"
        and recovery == "debt_ratio_decreased"
        and recent == "debt_ratio_increased"
    ):
        return "expansion_with_post_pandemic_consolidation"

    if (
        recovery == "debt_ratio_decreased"
        and recent == "debt_ratio_increased"
    ):
        return "consolidation_with_recent_reversal"

    if (
        long_term == "debt_ratio_decreased"
        and recovery == "debt_ratio_decreased"
        and recent == "debt_ratio_decreased"
    ):
        return "sustained_debt_reduction"

    if (
        long_term == "debt_ratio_decreased"
        and recent == "debt_ratio_increased"
    ):
        return "long_term_reduction_recent_increase"

    return "mixed_debt_dynamics"


def classify_primary_adjustment(value):
    if pd.isna(value):
        return "insufficient_data"

    if value > 0:
        return "improved"

    if value < 0:
        return "deteriorated"

    return "unchanged"


def classify_average_primary_balance(value):
    if pd.isna(value):
        return "insufficient_data"

    if value > 0:
        return "average_primary_surplus"

    if value < 0:
        return "average_primary_deficit"

    return "average_primary_balance"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    missing = set(REQUIRED) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["iso"].duplicated().any():
        raise ValueError("Duplicate country records detected.")

    if len(df) != 20:
        raise ValueError(
            f"Expected 20 countries, found {len(df)}."
        )

    numeric_columns = [
        "debt_change_pp_2019_2024",
        "primary_balance_change_pp_2019_2024",
        "debt_change_pp_2021_2024",
        "debt_change_pp_2023_2024",
        "primary_balance_mean_2020_2024",
        "government_debt_mean_2020_2024",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Validate five-year averages.
    if not (
        df["primary_balance_observations_2020_2024"] == 5
    ).all():
        raise ValueError(
            "Incomplete primary balance observations."
        )

    if not (
        df["government_debt_observations_2020_2024"] == 5
    ).all():
        raise ValueError(
            "Incomplete government debt observations."
        )

    # 1. Debt trajectory diagnostics.
    df["debt_trajectory_pattern"] = df.apply(
        classify_debt_trajectory, axis=1
    )

    # 2. Primary balance adjustment.
    df["primary_adjustment_signal"] = df[
        "primary_balance_change_pp_2019_2024"
    ].apply(classify_primary_adjustment)

    # 3. Average primary balance position.
    df["average_primary_balance_position"] = df[
        "primary_balance_mean_2020_2024"
    ].apply(classify_average_primary_balance)

    # 4. Recent debt movement.
    df["recent_debt_movement"] = df[
        "debt_change_pp_2023_2024"
    ].apply(
        lambda x: (
            "debt_ratio_increased"
            if x > 0
            else (
                "debt_ratio_decreased"
                if x < 0
                else "unchanged"
            )
        )
        if pd.notna(x)
        else "insufficient_data"
    )

    # 5. Combined diagnostic observations.
    def combined_diagnostic(row):
        trajectory = row["debt_trajectory_pattern"]
        adjustment = row["primary_adjustment_signal"]

        if (
            trajectory in [
                "persistent_debt_expansion",
                "consolidation_with_recent_reversal",
                "expansion_with_post_pandemic_consolidation",
            ]
            and adjustment == "deteriorated"
        ):
            return "debt_pressure_with_primary_balance_deterioration"

        if (
            trajectory == "sustained_debt_reduction"
            and adjustment == "improved"
        ):
            return "debt_reduction_with_primary_balance_improvement"

        if (
            trajectory in [
                "sustained_debt_reduction",
                "long_term_reduction_recent_increase",
            ]
            and adjustment == "deteriorated"
        ):
            return "debt_reduction_despite_primary_balance_deterioration"

        return "mixed_or_partial_adjustment_signals"

    df["combined_fiscal_diagnostic"] = df.apply(
        combined_diagnostic, axis=1
    )

    # Preserve original measures and diagnostic classifications.
    output_columns = REQUIRED + [
        "debt_trajectory_pattern",
        "primary_adjustment_signal",
        "average_primary_balance_position",
        "recent_debt_movement",
        "combined_fiscal_diagnostic",
    ]

    result = df[output_columns].sort_values(
        "iso"
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT, index=False)

    lines = [
        "EUROPEAN DEBT DYNAMICS DIAGNOSTIC REPORT",
        "=" * 60,
        f"Countries: {len(result)}",
        "",
        "DEBT TRAJECTORY CLASSIFICATIONS",
        result["debt_trajectory_pattern"].value_counts().to_string(),
        "",
        "PRIMARY BALANCE ADJUSTMENT",
        result["primary_adjustment_signal"].value_counts().to_string(),
        "",
        "AVERAGE PRIMARY BALANCE POSITION, 2020-2024",
        result[
            "average_primary_balance_position"
        ].value_counts().to_string(),
        "",
        "COMBINED FISCAL DIAGNOSTICS",
        result[
            "combined_fiscal_diagnostic"
        ].value_counts().to_string(),
        "",
        "COUNTRY DIAGNOSTIC MATRIX",
        result[
            [
                "iso",
                "country",
                "debt_trajectory_pattern",
                "primary_adjustment_signal",
                "average_primary_balance_position",
                "combined_fiscal_diagnostic",
            ]
        ].to_string(index=False),
        "",
        "METHODOLOGICAL LIMITATIONS",
        "- Classifications describe historical observations.",
        "- No probability of default is calculated.",
        "- Primary balance changes are not cyclically adjusted.",
        "- Debt changes are not decomposed into their economic drivers.",
        "- No composite fiscal risk score is calculated.",
        "- The classification rules are descriptive, not causal.",
    ]

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Report saved: {REPORT}")


if __name__ == "__main__":
    main()
