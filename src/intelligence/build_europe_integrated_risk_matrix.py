
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

SOURCES = {
    "fiscal": DATA / "europe_fiscal_risk_signals.csv",
    "growth": DATA / "europe_growth_risk_classifications.csv",
    "inflation": DATA / "europe_inflation_risk_classifications.csv",
    "external": DATA / "europe_external_risk_classifications.csv",
    "labour": DATA / "europe_labour_risk_classifications.csv",
}

OUTPUT = DATA / "europe_integrated_risk_matrix.csv"
REPORT = REPORTS / "europe_integrated_risk_matrix_report.txt"

EXPECTED_COUNTRIES = 20
EXPECTED_ROWS = 660

VALID_CLASSES = {"higher_risk", "moderate_risk", "lower_risk"}


def standardise_class(value):
    """Convert engine-specific labels to a common classification."""
    value = str(value).strip().lower()

    mapping = {
        "higher_risk": "higher",
        "moderate_risk": "moderate",
        "lower_risk": "lower",
        "higher_relative_exposure": "higher",
        "moderate_relative_exposure": "moderate",
        "lower_relative_exposure": "lower",
    }

    if value not in mapping:
        raise ValueError(f"Unrecognised classification: {value}")

    return mapping[value]


def standardise_percentile(value, scale, context):
    """Convert a percentile to 0-100 without guessing its scale."""
    if pd.isna(value):
        raise ValueError(f"Missing percentile in {context}")

    value = float(value)

    if scale == "0-1":
        if not 0 <= value <= 1:
            raise ValueError(
                f"Percentile outside 0-1 range in {context}: {value}"
            )
        return value * 100

    if scale == "0-100":
        if not 0 <= value <= 100:
            raise ValueError(
                f"Percentile outside 0-100 range in {context}: {value}"
            )
        return value

    raise ValueError(f"Unknown percentile scale: {scale}")


def make_record(
    iso,
    country,
    engine,
    dimension,
    measurement_value,
    original_percentile,
    percentile_scale,
    original_classification,
    measurement_period,
    measurement_type,
    data_coverage,
    source_file,
):
    context = f"{engine}/{iso}/{dimension}"

    return {
        "iso": iso,
        "country": country,
        "risk_engine": engine,
        "risk_dimension": dimension,
        "measurement_value": measurement_value,
        "original_percentile": float(original_percentile),
        "percentile_scale_original": percentile_scale,
        "percentile_0_100": standardise_percentile(
            original_percentile, percentile_scale, context
        ),
        "original_classification": str(original_classification),
        "standardised_classification": standardise_class(
            original_classification
        ),
        "measurement_period": measurement_period,
        "measurement_type": measurement_type,
        "data_coverage": data_coverage,
        "source_file": source_file,
    }


def load_sources():
    frames = {}
    for engine, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        frames[engine] = pd.read_csv(path)
    return frames


def build_matrix(frames):
    records = []

    fiscal = frames["fiscal"]
    growth = frames["growth"]
    inflation = frames["inflation"]
    external = frames["external"]
    labour = frames["labour"]

    # Establish the authoritative ISO-to-country mapping.
    country_map = (
        fiscal[["iso", "country"]]
        .drop_duplicates()
        .set_index("iso")["country"]
        .to_dict()
    )

    if len(country_map) != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTRIES} countries; "
            f"found {len(country_map)} in fiscal source."
        )

    # Check all engines have the same country universe.
    reference_countries = set(country_map)

    for engine, frame in frames.items():
        if "iso" not in frame.columns:
            raise ValueError(f"{engine}: missing ISO column")

        engine_countries = set(frame["iso"].dropna().unique())

        if engine_countries != reference_countries:
            raise ValueError(
                f"{engine}: country universe differs from fiscal source. "
                f"Missing={sorted(reference_countries - engine_countries)}, "
                f"Additional={sorted(engine_countries - reference_countries)}"
            )

    # ---------------------------------------------------------
    # FISCAL: long format, 8 dimensions per country
    # ---------------------------------------------------------
    for _, row in fiscal.iterrows():
        fiscal_dimension = f"{row['measurement_type']}_{row['risk_dimension']}"
        records.append(
            make_record(
                iso=row["iso"],
                country=row["country"],
                engine="fiscal",
                dimension=fiscal_dimension,
                measurement_value=row["measurement_value"],
                original_percentile=row["risk_percentile"],
                percentile_scale="0-100",
                original_classification=row["risk_signal"],
                measurement_period=str(row["reference_year"]),
                measurement_type=row["measurement_type"],
                data_coverage=None,
                source_file=SOURCES["fiscal"].name,
            )
        )

    # ---------------------------------------------------------
    # GROWTH: wide format, 6 dimensions per country
    # ---------------------------------------------------------
    growth_dimensions = {
        "long_run_growth_weakness": {
            "percentile": "long_run_growth_weakness_risk_percentile",
            "classification": "long_run_growth_weakness_risk_class",
            "value": "long_run_mean_growth_pct",
            "period": "long-run historical",
            "type": "mean annual real GDP growth (%)",
            "coverage": "Country-specific historical coverage",
        },
        "recent_growth_weakness": {
            "percentile": "recent_growth_weakness_risk_percentile",
            "classification": "recent_growth_weakness_risk_class",
            "value": "recent_mean_growth_pct",
            "period": "2020-2024",
            "type": "mean annual real GDP growth (%)",
            "coverage": "5 annual observations expected",
        },
        "growth_instability": {
            "percentile": "growth_instability_risk_percentile",
            "classification": "growth_instability_risk_class",
            "value": "volatility_std_2000_2024_pct",
            "period": "2000-2024",
            "type": "Standard deviation of annual real GDP growth (%)",
            "coverage": "Country-specific historical coverage",
        },
        "contraction_frequency": {
            "percentile": "contraction_frequency_risk_percentile",
            "classification": "contraction_frequency_risk_class",
            "value": "contraction_frequency_pct",
            "period": "Historical diagnostic period",
            "type": "Frequency of negative annual growth (%)",
            "coverage": "Country-specific historical coverage",
        },
        "recovery_weakness": {
            "percentile": "recovery_weakness_risk_percentile",
            "classification": "recovery_weakness_risk_class",
            "value": "recovery_failure_rate",
            "period": "Historical contraction episodes",
            "type": "Recovery failure rate",
            "coverage": "See recovery_assessable_episodes",
        },
        "contraction_severity": {
            "percentile": "contraction_severity_risk_percentile",
            "classification": "contraction_severity_risk_class",
            "value": "contraction_severity_risk_percentile",
            "period": "Historical contraction episodes",
            "type": "Composite contraction duration and depth percentile (0-1)",
            "coverage": "See contraction_episode_count",
        },
    }

    for _, row in growth.iterrows():
        for dimension, spec in growth_dimensions.items():
            records.append(
                make_record(
                    iso=row["iso"],
                    country=row["country"],
                    engine="growth",
                    dimension=dimension,
                    measurement_value=row[spec["value"]],
                    original_percentile=row[spec["percentile"]],
                    percentile_scale="0-1",
                    original_classification=row[spec["classification"]],
                    measurement_period=spec["period"],
                    measurement_type=spec["type"],
                    data_coverage=spec["coverage"],
                    source_file=SOURCES["growth"].name,
                )
            )

    # ---------------------------------------------------------
    # INFLATION: wide format, 8 dimensions per country
    # ---------------------------------------------------------
    inflation_dimensions = {
        "recent_inflation_pressure": (
            "recent_inflation_pressure_percentile",
            "recent_inflation_pressure_risk_class",
            "median_inflation_pct_recent",
            "2020-2024",
            "Median CPI inflation (%)",
        ),
        "recent_inflation_volatility_mad": (
            "recent_inflation_volatility_mad_percentile",
            "recent_inflation_volatility_mad_risk_class",
            "mad_inflation_pct_recent",
            "2020-2024",
            "MAD of CPI inflation (%)",
        ),
        "recent_inflation_dispersion_iqr": (
            "recent_inflation_dispersion_iqr_percentile",
            "recent_inflation_dispersion_iqr_risk_class",
            "iqr_inflation_pct_recent",
            "2020-2024",
            "IQR of CPI inflation (%)",
        ),
        "inflation_change_instability": (
            "inflation_change_instability_percentile",
            "inflation_change_instability_risk_class",
            "mean_absolute_annual_change_pp_recent",
            "2020-2024",
            "Mean absolute annual inflation change (pp)",
        ),
        "upward_inflation_shocks": (
            "upward_inflation_shocks_percentile",
            "upward_inflation_shocks_risk_class",
            "recent_upward_inflation_shocks",
            "2020-2024",
            "Count of upward inflation shocks",
        ),
        "historical_deflation_frequency": (
            "historical_deflation_frequency_percentile",
            "historical_deflation_frequency_risk_class",
            "deflation_frequency_long_run",
            "Long-run historical",
            "Frequency of deflation (%)",
        ),
        "historical_inflation_persistence": (
            "historical_inflation_persistence_percentile",
            "historical_inflation_persistence_risk_class",
            "median_inflation_pct_long_run",
            "Long-run historical",
            "Median CPI inflation (%)",
        ),
        "historical_high_inflation_exposure": (
            "historical_high_inflation_exposure_percentile",
            "historical_high_inflation_exposure_risk_class",
            "inflation_ge_10pct_frequency_long_run",
            "Long-run historical",
            "Frequency of inflation at or above 10% (%)",
        ),
    }

    for _, row in inflation.iterrows():
        for dimension, spec in inflation_dimensions.items():
            percentile_col, class_col, value_col, period, measurement = spec

            records.append(
                make_record(
                    iso=row["iso"],
                    country=row["country"],
                    engine="inflation",
                    dimension=dimension,
                    measurement_value=row[value_col],
                    original_percentile=row[percentile_col],
                    percentile_scale="0-1",
                    original_classification=row[class_col],
                    measurement_period=period,
                    measurement_type=measurement,
                    data_coverage="See source diagnostics and input coverage",
                    source_file=SOURCES["inflation"].name,
                )
            )

    # ---------------------------------------------------------
    # EXTERNAL: wide format, classification in dimension column
    # ---------------------------------------------------------
    external_dimensions = {
        "recent_external_imbalance": ("2020-2024", "Current account balance (% GDP)"),
        "historical_deficit_persistence": ("2000-2019", "Frequency of current account deficits"),
        "recent_external_imbalance_volatility": ("2020-2024", "MAD of current account balance"),
        "historical_export_contraction": ("2000-2019", "Frequency of negative export volume growth"),
        "recent_export_contraction": ("2020-2024", "Frequency of negative export volume growth"),
        "historical_export_growth_instability": ("2000-2019", "MAD of export volume growth"),
    }

    for _, row in external.iterrows():
        for dimension, (period, measurement) in external_dimensions.items():
            records.append(
                make_record(
                    iso=row["iso"],
                    country=row["country"],
                    engine="external",
                    dimension=dimension,
                    measurement_value=row[f"{dimension}_value"],
                    original_percentile=row[f"{dimension}_percentile"],
                    percentile_scale="0-1",
                    original_classification=row[dimension],
                    measurement_period=period,
                    measurement_type=measurement,
                    data_coverage="See source diagnostics and input coverage",
                    source_file=SOURCES["external"].name,
                )
            )

    # ---------------------------------------------------------
    # LABOUR: long format, 5 dimensions per country
    # ---------------------------------------------------------
    for _, row in labour.iterrows():
        iso = row["iso"]

        if iso not in country_map:
            raise ValueError(f"Labour ISO {iso} missing from country map")

        records.append(
            make_record(
                iso=iso,
                country=country_map[iso],
                engine="labour",
                dimension=row["dimension"],
                measurement_value=row["value"],
                original_percentile=row["peer_exposure_percentile"],
                percentile_scale="0-100",
                original_classification=row["classification"],
                measurement_period=row["measurement_period"],
                measurement_type=row["measurement"],
                data_coverage=f"{row['eligible_country_count']} eligible countries",
                source_file=SOURCES["labour"].name,
            )
        )

    matrix = pd.DataFrame(records)

    return matrix


def validate_matrix(matrix):
    errors = []

    required_columns = [
        "iso",
        "country",
        "risk_engine",
        "risk_dimension",
        "measurement_value",
        "original_percentile",
        "percentile_0_100",
        "original_classification",
        "standardised_classification",
        "measurement_period",
        "measurement_type",
        "source_file",
    ]

    missing_columns = set(required_columns) - set(matrix.columns)
    if missing_columns:
        errors.append(f"Missing columns: {sorted(missing_columns)}")

    if len(matrix) != EXPECTED_ROWS:
        errors.append(
            f"Expected {EXPECTED_ROWS} rows, found {len(matrix)}"
        )

    if matrix["iso"].nunique() != EXPECTED_COUNTRIES:
        errors.append(
            f"Expected {EXPECTED_COUNTRIES} countries, "
            f"found {matrix['iso'].nunique()}"
        )

    duplicate_keys = matrix.duplicated(
        ["iso", "risk_engine", "risk_dimension"]
    ).sum()

    if duplicate_keys:
        errors.append(f"Duplicate country-engine-dimension keys: {duplicate_keys}")

    if matrix[required_columns].isna().any().any():
        missing = matrix[required_columns].isna().sum()
        missing = missing[missing > 0]
        errors.append(f"Missing required values: {missing.to_dict()}")

    if not matrix["standardised_classification"].isin(
        {"higher", "moderate", "lower"}
    ).all():
        errors.append("Unexpected standardised classification values")

    if not matrix["percentile_0_100"].between(0, 100).all():
        errors.append("Standardised percentile outside 0-100 range")

    expected_by_engine = {
        "fiscal": 160,
        "growth": 120,
        "inflation": 160,
        "external": 120,
        "labour": 100,
    }

    actual_by_engine = matrix["risk_engine"].value_counts().to_dict()

    if actual_by_engine != expected_by_engine:
        errors.append(
            f"Unexpected engine counts. Expected {expected_by_engine}; "
            f"found {actual_by_engine}"
        )

    dimensions_per_country = (
        matrix.groupby("iso")["risk_dimension"].count()
    )

    if not (dimensions_per_country == 33).all():
        errors.append(
            "Not every country has exactly 33 risk dimensions"
        )

    if errors:
        raise ValueError(
            "INTEGRATED MATRIX VALIDATION FAILED:\n- "
            + "\n- ".join(errors)
        )

    return actual_by_engine


def write_report(matrix, engine_counts):
    lines = [
        "=" * 76,
        "EUROPEAN INTEGRATED MACROECONOMIC RISK MATRIX",
        "=" * 76,
        "",
        "VALIDATION STATUS: PASS",
        "",
        f"Total records: {len(matrix)}",
        f"Countries: {matrix['iso'].nunique()}",
        f"Risk engines: {matrix['risk_engine'].nunique()}",
        f"Dimensions per country: {matrix.groupby('iso').size().min()}",
        "",
        "ENGINE RECORD COUNTS",
        "-" * 76,
    ]

    for engine, count in engine_counts.items():
        lines.append(f"{engine:15s} {count:>6}")

    lines.extend([
        "",
        "CLASSIFICATION DISTRIBUTION",
        "-" * 76,
    ])

    distribution = (
        matrix.groupby(["risk_engine", "standardised_classification"])
        .size()
        .unstack(fill_value=0)
    )

    lines.append(distribution.to_string())

    lines.extend([
        "",
        "PERCENTILE STANDARDISATION",
        "-" * 76,
        "Fiscal: original 0-100, retained as 0-100",
        "Labour: original 0-100, retained as 0-100",
        "Growth: original 0-1, multiplied by 100",
        "Inflation: original 0-1, multiplied by 100",
        "External: original 0-1, multiplied by 100",
        "",
        "MISSING SUPPORTING DATA",
        "-" * 76,
        "The seven missing inflation mean deflation episode durations are not",
        "classification dimensions and are not included as risk measurements.",
        "",
        "INTERPRETATION LIMITATIONS",
        "-" * 76,
        "1. Classifications are peer-relative within the selected 20-country group.",
        "2. Percentiles are ranks, not probabilities of crisis or default.",
        "3. Measurement values retain their original units and meanings.",
        "4. Different engines use different historical and recent periods.",
        "5. The matrix is not a composite sovereign or macroeconomic risk score.",
        "6. Missing contextual diagnostics must not be interpreted as zero.",
        "",
        "OUTPUT",
        "-" * 76,
        str(OUTPUT),
    ])

    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("EUROPEAN INTEGRATED MACROECONOMIC RISK MATRIX")
    print("=" * 76)

    frames = load_sources()

    print("\nBuilding canonical risk-dimension records...")
    matrix = build_matrix(frames)

    print("Validating integrated matrix...")
    engine_counts = validate_matrix(matrix)

    # Stable ordering for repeatable output.
    matrix = matrix.sort_values(
        ["iso", "risk_engine", "risk_dimension"]
    ).reset_index(drop=True)

    matrix.to_csv(OUTPUT, index=False)
    write_report(matrix, engine_counts)

    print("\nVALIDATION PASSED")
    print("-" * 76)
    print(f"Total records: {len(matrix)}")
    print(f"Countries: {matrix['iso'].nunique()}")
    print(f"Risk dimensions per country: 33")
    print("\nENGINE COUNTS")

    for engine, count in engine_counts.items():
        print(f"  {engine}: {count}")

    print("\nOUTPUTS CREATED")
    print(f"  {OUTPUT}")
    print(f"  {REPORT}")


if __name__ == "__main__":
    main()
