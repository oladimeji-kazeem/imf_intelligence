from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/europe_fiscal_risk_signals.csv"
)

OUTPUT = Path(
    "data/processed/europe_fiscal_risk_country_profiles.csv"
)

REPORT = Path(
    "outputs/reports/europe_fiscal_risk_country_profiles_report.txt"
)

def main():
    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "reference_year",
        "risk_dimension",
        "measurement_type",
        "measurement_value",
        "risk_percentile",
        "risk_signal",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Each country must have one record for each
    # dimension and measurement type.
    keys = [
        "iso",
        "risk_dimension",
        "measurement_type",
    ]

    duplicates = df.duplicated(keys).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate signal combinations."
        )

    # Create separate matrices for signals, percentiles,
    # and original measurement values.
    index = [
        "iso",
        "country",
        "reference_year",
    ]

    signal_matrix = df.pivot(
        index=index,
        columns=["measurement_type", "risk_dimension"],
        values="risk_signal",
    )

    percentile_matrix = df.pivot(
        index=index,
        columns=["measurement_type", "risk_dimension"],
        values="risk_percentile",
    )

    value_matrix = df.pivot(
        index=index,
        columns=["measurement_type", "risk_dimension"],
        values="measurement_value",
    )

    def flatten_columns(matrix, prefix):
        matrix.columns = [
            f"{prefix}_{measurement}_{dimension}"
            for measurement, dimension in matrix.columns
        ]
        return matrix

    signal_matrix = flatten_columns(
        signal_matrix, "signal"
    )

    percentile_matrix = flatten_columns(
        percentile_matrix, "percentile"
    )

    value_matrix = flatten_columns(
        value_matrix, "value"
    )

    result = pd.concat(
        [
            signal_matrix,
            percentile_matrix,
            value_matrix,
        ],
        axis=1,
    ).reset_index()

    result.columns.name = None

    result = result.sort_values("iso").reset_index(drop=True)

    # Validate expected structure.
    expected_countries = df["iso"].nunique()
    expected_dimensions = df["risk_dimension"].nunique()
    expected_types = df["measurement_type"].nunique()

    expected_signals = expected_dimensions * expected_types

    signal_columns = [
        c for c in result.columns
        if c.startswith("signal_")
    ]

    percentile_columns = [
        c for c in result.columns
        if c.startswith("percentile_")
    ]

    if len(signal_columns) != expected_signals:
        raise ValueError("Unexpected number of signal columns.")

    if len(percentile_columns) != expected_signals:
        raise ValueError("Unexpected number of percentile columns.")

    if result["iso"].duplicated().any():
        raise ValueError("Duplicate countries in final profile.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT, index=False)

    lines = [
        "EUROPEAN FISCAL RISK COUNTRY PROFILES",
        "=" * 55,
        f"Reference year: {df['reference_year'].iloc[0]}",
        f"Countries: {expected_countries}",
        f"Fiscal dimensions: {expected_dimensions}",
        f"Measurement types: {expected_types}",
        f"Expected signals per country: {expected_signals}",
        f"Signal columns: {len(signal_columns)}",
        f"Percentile columns: {len(percentile_columns)}",
        f"Final country profiles: {len(result)}",
        "",
        "Missing signal values:",
        str(result[signal_columns].isna().sum().sum()),
        "",
        "Missing percentile values:",
        str(result[percentile_columns].isna().sum().sum()),
        "",
        "Country-level signal matrix:",
        result[
            ["iso", "country"] + signal_columns
        ].to_string(index=False),
        "",
        "Methodological notes:",
        "- Exposure and historical volatility remain separate.",
        "- Percentiles remain peer-relative.",
        "- No composite fiscal risk score is calculated.",
        "- No sovereign rating or default probability is inferred.",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Report saved: {REPORT}")


if __name__ == "__main__":
    main()
