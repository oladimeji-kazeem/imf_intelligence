from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/europe_fiscal_risk_input_panel.csv"
)

OUTPUT = Path(
    "data/processed/europe_multi_year_debt_dynamics.csv"
)

REPORT = Path(
    "outputs/reports/europe_multi_year_debt_dynamics_report.txt"
)

PERIODS = [
    (2019, 2024),
    (2021, 2024),
    (2023, 2024),
]

def get_value(data, iso, year, column):
    match = data[
        (data["iso"] == iso)
        & (data["year"] == year)
    ]

    if match.empty:
        return float("nan")

    return match.iloc[0][column]


def calculate_change(data, iso, start, end, column):
    start_value = get_value(data, iso, start, column)
    end_value = get_value(data, iso, end, column)

    if pd.isna(start_value) or pd.isna(end_value):
        return float("nan")

    return end_value - start_value


def main():
    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "year",
        "government_debt",
        "primary_balance",
        "fiscal_balance",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["year"] = pd.to_numeric(
        df["year"], errors="coerce"
    )

    if df.duplicated(["iso", "year"]).any():
        raise ValueError("Duplicate country-year records found.")

    countries = (
        df[["iso", "country"]]
        .drop_duplicates()
        .sort_values("iso")
    )

    records = []

    for _, country in countries.iterrows():
        iso = country["iso"]
        name = country["country"]

        record = {
            "iso": iso,
            "country": name,
        }

        for start, end in PERIODS:
            suffix = f"{start}_{end}"

            record[f"debt_change_pp_{suffix}"] = (
                calculate_change(
                    df, iso, start, end, "government_debt"
                )
            )

            record[f"primary_balance_change_pp_{suffix}"] = (
                calculate_change(
                    df, iso, start, end, "primary_balance"
                )
            )

            record[f"fiscal_balance_change_pp_{suffix}"] = (
                calculate_change(
                    df, iso, start, end, "fiscal_balance"
                )
            )

        # Average primary balance over 2020–2024.
        recent = df[
            (df["iso"] == iso)
            & df["year"].between(2020, 2024)
        ]

        record["primary_balance_mean_2020_2024"] = (
            recent["primary_balance"].mean()
        )

        record["primary_balance_observations_2020_2024"] = (
            recent["primary_balance"].count()
        )

        # Average government debt over 2020–2024.
        record["government_debt_mean_2020_2024"] = (
            recent["government_debt"].mean()
        )

        record["government_debt_observations_2020_2024"] = (
            recent["government_debt"].count()
        )

        records.append(record)

    result = pd.DataFrame(records)

    # Directional interpretation of debt-ratio movement.
    for period in ["2019_2024", "2021_2024", "2023_2024"]:
        column = f"debt_change_pp_{period}"

        result[f"debt_direction_{period}"] = (
            result[column].apply(
                lambda x: (
                    "debt_ratio_increased"
                    if pd.notna(x) and x > 0
                    else (
                        "debt_ratio_decreased"
                        if pd.notna(x) and x < 0
                        else (
                            "debt_ratio_unchanged"
                            if pd.notna(x)
                            else "insufficient_data"
                        )
                    )
                )
            )
        )

    # Primary balance improvement is a positive change.
    result["primary_balance_adjustment_2019_2024"] = (
        result["primary_balance_change_pp_2019_2024"].apply(
            lambda x: (
                "improved"
                if pd.notna(x) and x > 0
                else (
                    "deteriorated"
                    if pd.notna(x) and x < 0
                    else (
                        "unchanged"
                        if pd.notna(x)
                        else "insufficient_data"
                    )
                )
            )
        )
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT, index=False)

    lines = [
        "EUROPEAN MULTI-YEAR DEBT DYNAMICS REPORT",
        "=" * 60,
        f"Countries: {len(result)}",
        "",
        "Debt ratio changes (percentage points of GDP):",
    ]

    for period in ["2019_2024", "2021_2024", "2023_2024"]:
        column = f"debt_change_pp_{period}"

        lines.extend([
            "",
            f"Period: {period}",
            f"Available observations: {result[column].notna().sum()}",
            f"Debt ratio increased: {(result[column] > 0).sum()}",
            f"Debt ratio decreased: {(result[column] < 0).sum()}",
            f"Unchanged: {(result[column] == 0).sum()}",
            f"Missing: {result[column].isna().sum()}",
        ])

    lines.extend([
        "",
        "Primary balance adjustment, 2019–2024:",
        result[
            "primary_balance_adjustment_2019_2024"
        ].value_counts().to_string(),
        "",
        "Country-level results:",
        result[
            [
                "iso",
                "country",
                "debt_change_pp_2019_2024",
                "debt_change_pp_2021_2024",
                "debt_change_pp_2023_2024",
                "primary_balance_change_pp_2019_2024",
                "primary_balance_mean_2020_2024",
            ]
        ].to_string(index=False),
        "",
        "Methodological limitations:",
        "- Changes are percentage points of GDP, not percentage changes.",
        "- Endpoint comparisons require both endpoint observations.",
        "- A positive primary balance change indicates improvement,",
        "  not necessarily a primary surplus.",
        "- Primary balance is not a direct measure of fiscal capacity.",
        "- No interest-growth differential is calculated.",
        "- No debt sustainability conclusion is inferred.",
    ])

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Report saved: {REPORT}")


if __name__ == "__main__":
    main()
