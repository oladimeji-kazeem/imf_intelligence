from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/europe_fiscal_risk_indicators.csv"
)

OUTPUT = Path(
    "data/processed/europe_fiscal_risk_signals.csv"
)

REPORT = Path(
    "outputs/reports/europe_fiscal_risk_signals_report.txt"
)

# Percentile columns: higher percentile means greater relative exposure.
RISK_INDICATORS = {
    "government_debt": "debt_burden_percentile",
    "fiscal_balance": "fiscal_pressure_percentile",
    "primary_balance": "primary_pressure_percentile",
    "debt_change": "debt_trajectory_percentile",
}

# Standard deviation columns: higher values indicate greater dispersion.
VOLATILITY_INDICATORS = {
    "government_debt": "debt_volatility",
    "fiscal_balance": "fiscal_balance_volatility",
    "primary_balance": "primary_balance_volatility",
    "debt_change": "debt_change_volatility",
}

def classify_percentile(value):
    if pd.isna(value):
        return "insufficient_data"

    if value <= 100 / 3:
        return "lower_relative_exposure"

    if value <= 200 / 3:
        return "moderate_relative_exposure"

    return "higher_relative_exposure"


def main():
    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "reference_year",
        *RISK_INDICATORS.values(),
        *VOLATILITY_INDICATORS.values(),
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["iso"].duplicated().any():
        raise ValueError(
            "Duplicate countries found in the input dataset."
        )

    records = []

    # A. Peer-relative fiscal exposure signals.
    for indicator, percentile_column in RISK_INDICATORS.items():

        for _, row in df.iterrows():
            value = row[percentile_column]

            records.append({
                "iso": row["iso"],
                "country": row["country"],
                "reference_year": row["reference_year"],
                "risk_dimension": indicator,
                "measurement_type": "peer_relative",
                "measurement_value": value,
                "risk_percentile": value,
                "risk_signal": classify_percentile(value),
            })

    # B. Historical volatility signals.
    for indicator, volatility_column in VOLATILITY_INDICATORS.items():

        percentile_column = f"{indicator}_volatility_percentile"

        df[percentile_column] = (
            df[volatility_column].rank(
                method="average",
                ascending=True,
                pct=True
            ) * 100
        )

        for _, row in df.iterrows():
            value = row[percentile_column]

            records.append({
                "iso": row["iso"],
                "country": row["country"],
                "reference_year": row["reference_year"],
                "risk_dimension": indicator,
                "measurement_type": "historical_volatility",
                "measurement_value": row[volatility_column],
                "risk_percentile": value,
                "risk_signal": classify_percentile(value),
            })

    result = pd.DataFrame(records)

    result["risk_signal"] = result["risk_signal"].replace({
        "higher_relative_exposure": "higher_historical_dispersion"
    }, regex=False) if False else result["risk_signal"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT, index=False)

    lines = [
        "EUROPEAN FISCAL RISK SIGNALS REPORT",
        "=" * 55,
        f"Reference year: {df['reference_year'].iloc[0]}",
        f"Countries: {df['iso'].nunique()}",
        f"Risk dimensions: {result['risk_dimension'].nunique()}",
        f"Total signal records: {len(result)}",
        "",
        "Signals by measurement type:",
        result.groupby(
            ["measurement_type", "risk_signal"]
        ).size().to_string(),
        "",
        "Signals by risk dimension:",
        result.groupby(
            ["risk_dimension", "risk_signal"]
        ).size().to_string(),
        "",
        "Missing percentiles:",
        str(result["risk_percentile"].isna().sum()),
        "",
        "Methodological notes:",
        "- Relative terciles are based on the 20-country peer group.",
        "- Higher exposure means a less favourable peer position.",
        "- Historical volatility is classified separately.",
        "- Volatility is not equivalent to probability of default.",
        "- No composite sovereign risk score is calculated.",
        "- Categories do not represent absolute risk thresholds.",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Report saved: {REPORT}")


if __name__ == "__main__":
    main()
