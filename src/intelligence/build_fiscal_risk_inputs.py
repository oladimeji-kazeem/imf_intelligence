from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/europe_weo_classified.csv")
OUTPUT = Path("data/processed/europe_fiscal_risk_input_panel.csv")
REPORT = Path("outputs/reports/europe_fiscal_risk_data_readiness.txt")

REQUIRED_COLUMNS = {
    "iso", "country", "indicator", "year",
    "value", "observation_status"
}

FISCAL_INDICATORS = [
    "government_debt",
    "fiscal_balance",
    "primary_balance",
    "nominal_gdp_local",
]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Retain actual historical observations only.
    df = df[
        df["observation_status"] == "historical_actual"
    ].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[
        df["indicator"].isin(FISCAL_INDICATORS)
    ].copy()

    # Preserve missing observations; do not replace with zero.
    panel = df.pivot_table(
        index=["iso", "country", "year"],
        columns="indicator",
        values="value",
        aggfunc="first"
    ).reset_index()

    panel.columns.name = None

    for indicator in FISCAL_INDICATORS:
        if indicator not in panel.columns:
            panel[indicator] = float("nan")

    panel = panel.sort_values(
        ["iso", "year"]
    ).reset_index(drop=True)

    # Derive annual changes only where consecutive years exist.
    grouped = panel.groupby("iso", sort=False)

    panel["previous_year"] = grouped["year"].shift(1)
    panel["previous_debt"] = grouped["government_debt"].shift(1)
    panel["previous_nominal_gdp"] = grouped[
        "nominal_gdp_local"
    ].shift(1)

    consecutive = (
        panel["year"] - panel["previous_year"] == 1
    )

    panel["debt_change_pp"] = (
        panel["government_debt"] - panel["previous_debt"]
    ).where(consecutive)

    panel["nominal_gdp_growth_pct"] = (
        (
            panel["nominal_gdp_local"]
            / panel["previous_nominal_gdp"] - 1
        ) * 100
    ).where(
        consecutive
        & panel["previous_nominal_gdp"].notna()
        & (panel["previous_nominal_gdp"] != 0)
    )

    # Convert negative balances into positive deficit magnitudes.
    panel["fiscal_deficit_pct_gdp"] = (
        -panel["fiscal_balance"]
    ).clip(lower=0)

    panel["primary_deficit_pct_gdp"] = (
        -panel["primary_balance"]
    ).clip(lower=0)

    # Common-year comparison snapshot.
    common_year = 2024
    snapshot = panel[panel["year"] == common_year].copy()

    snapshot_columns = [
        "government_debt",
        "fiscal_balance",
        "primary_balance",
        "debt_change_pp",
        "nominal_gdp_growth_pct",
    ]

    complete_snapshot = snapshot.dropna(
        subset=snapshot_columns
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    panel.to_csv(OUTPUT, index=False)

    report = [
        "EUROPEAN FISCAL RISK DATA READINESS REPORT",
        "=" * 48,
        f"Input records: {len(df):,}",
        f"Countries: {df['iso'].nunique()}",
        f"Fiscal risk panel records: {len(panel):,}",
        f"First year: {int(panel['year'].min())}",
        f"Last year: {int(panel['year'].max())}",
        "",
        f"Common comparison year: {common_year}",
        f"Countries in snapshot: {snapshot['iso'].nunique()}",
        f"Countries with complete fiscal snapshot: "
        f"{complete_snapshot['iso'].nunique()}",
        "",
        "Derived indicator availability:",
        f"Debt change available: "
        f"{panel['debt_change_pp'].notna().sum():,}",
        f"Nominal GDP growth available: "
        f"{panel['nominal_gdp_growth_pct'].notna().sum():,}",
        "",
        "Missing values by indicator:",
        panel[
            FISCAL_INDICATORS
            + ["debt_change_pp", "nominal_gdp_growth_pct"]
        ].isna().sum().to_string(),
        "",
        "Methodological notes:",
        "- Historical actual observations only.",
        "- No missing-value imputation.",
        "- Debt change is measured in percentage points of GDP.",
        "- Nominal GDP growth is derived from local-currency GDP.",
        "- Derived changes require consecutive annual observations.",
        "- 2024 is used as the initial common comparison year.",
        "- This dataset is not a sovereign credit rating.",
    ]

    REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Report saved: {REPORT}")


if __name__ == "__main__":
    main()
