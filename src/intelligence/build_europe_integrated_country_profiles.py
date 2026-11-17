
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

INPUT = DATA / "europe_integrated_risk_matrix.csv"

COUNTRY_OUTPUT = DATA / "europe_integrated_country_vulnerability_profiles.csv"
ENGINE_OUTPUT = DATA / "europe_integrated_risk_engine_summary.csv"
DIMENSION_OUTPUT = DATA / "europe_integrated_risk_dimension_summary.csv"
REPORT_OUTPUT = REPORTS / "europe_integrated_country_vulnerability_report.txt"

ENGINES = ["fiscal", "growth", "inflation", "external", "labour"]
CLASSES = ["higher", "moderate", "lower"]

EXPECTED_COUNTRIES = 20
EXPECTED_DIMENSIONS = 33


def load_matrix():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "risk_engine",
        "risk_dimension",
        "standardised_classification",
        "percentile_0_100",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def validate_matrix(df):
    errors = []

    if df["iso"].nunique() != EXPECTED_COUNTRIES:
        errors.append("Country count is not 20")

    if len(df) != EXPECTED_COUNTRIES * EXPECTED_DIMENSIONS:
        errors.append("Total record count is not 660")

    if df.duplicated(["iso", "risk_engine", "risk_dimension"]).any():
        errors.append("Duplicate country-engine-dimension keys detected")

    if set(df["standardised_classification"].unique()) != set(CLASSES):
        errors.append("Unexpected classification labels")

    if set(df["risk_engine"].unique()) != set(ENGINES):
        errors.append("Unexpected risk engines")

    dimensions_per_country = df.groupby("iso").size()
    if not (dimensions_per_country == EXPECTED_DIMENSIONS).all():
        errors.append("Some countries do not have 33 dimensions")

    engine_counts = df.groupby(["iso", "risk_engine"]).size()

    expected_per_engine = {
        "fiscal": 8,
        "growth": 6,
        "inflation": 8,
        "external": 6,
        "labour": 5,
    }

    for engine, expected in expected_per_engine.items():
        values = engine_counts.xs(engine, level="risk_engine")
        if not (values == expected).all():
            errors.append(
                f"{engine}: expected {expected} dimensions per country"
            )

    if errors:
        raise ValueError(
            "MATRIX VALIDATION FAILED:\n- " + "\n- ".join(errors)
        )


def build_engine_summary(df):
    summary = (
        df.groupby(
            ["iso", "country", "risk_engine", "standardised_classification"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for label in CLASSES:
        if label not in summary.columns:
            summary[label] = 0

    summary = summary.rename(
        columns={
            "higher": "higher_exposure_count",
            "moderate": "moderate_exposure_count",
            "lower": "lower_exposure_count",
        }
    )

    summary["dimension_count"] = (
        summary["higher_exposure_count"]
        + summary["moderate_exposure_count"]
        + summary["lower_exposure_count"]
    )

    summary["higher_exposure_share_pct"] = (
        summary["higher_exposure_count"]
        / summary["dimension_count"] * 100
    )

    return summary.sort_values(["iso", "risk_engine"])


def build_country_profiles(df, engine_summary):
    base = (
        df.groupby(["iso", "country"])
        .agg(
            total_dimensions=("risk_dimension", "count"),
            median_percentile=("percentile_0_100", "median"),
            minimum_percentile=("percentile_0_100", "min"),
            maximum_percentile=("percentile_0_100", "max"),
        )
        .reset_index()
    )

    overall_counts = (
        df.groupby(["iso", "standardised_classification"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for label in CLASSES:
        if label not in overall_counts.columns:
            overall_counts[label] = 0

    overall_counts = overall_counts.rename(
        columns={
            "higher": "higher_exposure_count",
            "moderate": "moderate_exposure_count",
            "lower": "lower_exposure_count",
        }
    )

    profiles = base.merge(overall_counts, on="iso", validate="one_to_one")

    profiles["higher_exposure_share_pct"] = (
        profiles["higher_exposure_count"]
        / profiles["total_dimensions"] * 100
    )

    # Add engine-specific counts without combining them into a score.
    for engine in ENGINES:
        subset = engine_summary[
            engine_summary["risk_engine"] == engine
        ][
            [
                "iso",
                "higher_exposure_count",
                "moderate_exposure_count",
                "lower_exposure_count",
            ]
        ].copy()

        subset = subset.rename(
            columns={
                "higher_exposure_count": f"{engine}_higher",
                "moderate_exposure_count": f"{engine}_moderate",
                "lower_exposure_count": f"{engine}_lower",
            }
        )

        profiles = profiles.merge(
            subset, on="iso", how="left", validate="one_to_one"
        )

    # Retain the names of higher-exposure dimensions for interpretation.
    higher_dimensions = (
        df[df["standardised_classification"] == "higher"]
        .groupby("iso")["risk_dimension"]
        .agg(lambda values: "; ".join(sorted(values)))
        .rename("higher_exposure_dimensions")
        .reset_index()
    )

    profiles = profiles.merge(
        higher_dimensions, on="iso", how="left", validate="one_to_one"
    )

    profiles["higher_exposure_dimensions"] = (
        profiles["higher_exposure_dimensions"].fillna("")
    )

    profiles["higher_exposure_dimension_count"] = (
        profiles["higher_exposure_dimensions"]
        .apply(lambda value: len(value.split("; ")) if value else 0)
    )

    return profiles.sort_values("iso")


def build_dimension_summary(df):
    summary = (
        df.groupby(
            ["risk_engine", "risk_dimension", "standardised_classification"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for label in CLASSES:
        if label not in summary.columns:
            summary[label] = 0

    summary = summary.rename(
        columns={
            "higher": "higher_country_count",
            "moderate": "moderate_country_count",
            "lower": "lower_country_count",
        }
    )

    summary["countries_classified"] = (
        summary["higher_country_count"]
        + summary["moderate_country_count"]
        + summary["lower_country_count"]
    )

    summary["higher_exposure_share_pct"] = (
        summary["higher_country_count"]
        / summary["countries_classified"] * 100
    )

    return summary.sort_values(["risk_engine", "risk_dimension"])


def write_report(df, profiles, engine_summary, dimension_summary):
    lines = [
        "=" * 76,
        "EUROPEAN INTEGRATED COUNTRY VULNERABILITY PROFILES",
        "=" * 76,
        "",
        "VALIDATION STATUS: PASS",
        f"Countries: {df['iso'].nunique()}",
        f"Integrated risk dimensions: {len(df)}",
        f"Dimensions per country: {EXPECTED_DIMENSIONS}",
        "",
        "COUNTRY HIGHER-EXPOSURE BREADTH",
        "-" * 76,
    ]

    ranked = profiles.sort_values(
        ["higher_exposure_count", "iso"],
        ascending=[False, True],
    )

    for _, row in ranked.iterrows():
        lines.append(
            f"{row['iso']:4s} {row['country'][:24]:24s} "
            f"Higher={int(row['higher_exposure_count']):2d}/33 "
            f"({row['higher_exposure_share_pct']:.1f}%)"
        )

    lines.extend([
        "",
        "ENGINE SUMMARY",
        "-" * 76,
        engine_summary.to_string(index=False),
        "",
        "DIMENSION SUMMARY",
        "-" * 76,
        dimension_summary.to_string(index=False),
        "",
        "INTERPRETATION LIMITATIONS",
        "-" * 76,
        "1. Counts measure the breadth of higher peer-relative signals.",
        "2. Counts are not weighted risk scores or default probabilities.",
        "3. The 33 dimensions are not statistically independent.",
        "4. Percentile medians are descriptive, not composite risk measures.",
        "5. Peer-relative classifications depend on the selected 20 countries.",
        "6. Different dimensions use different measurement periods and units.",
        "7. A lower count does not prove that a country has low absolute risk.",
        "",
        "OUTPUT FILES",
        "-" * 76,
        str(COUNTRY_OUTPUT),
        str(ENGINE_OUTPUT),
        str(DIMENSION_OUTPUT),
    ])

    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("EUROPEAN INTEGRATED COUNTRY VULNERABILITY PROFILES")
    print("=" * 76)

    df = load_matrix()
    validate_matrix(df)

    print("\nBuilding engine-level summaries...")
    engine_summary = build_engine_summary(df)

    print("Building country vulnerability profiles...")
    profiles = build_country_profiles(df, engine_summary)

    print("Building dimension-level summaries...")
    dimension_summary = build_dimension_summary(df)

    # Final output integrity checks.
    if len(profiles) != EXPECTED_COUNTRIES:
        raise ValueError("Country profile output should contain 20 rows")

    if len(engine_summary) != 100:
        raise ValueError("Engine summary should contain 100 rows")

    if len(dimension_summary) != EXPECTED_DIMENSIONS:
        raise ValueError("Dimension summary should contain 33 rows")

    if profiles["higher_exposure_count"].sum() != (
        df["standardised_classification"] == "higher"
    ).sum():
        raise ValueError("Country higher-exposure counts do not reconcile")

    if engine_summary["higher_exposure_count"].sum() != (
        df["standardised_classification"] == "higher"
    ).sum():
        raise ValueError("Engine higher-exposure counts do not reconcile")

    profiles.to_csv(COUNTRY_OUTPUT, index=False)
    engine_summary.to_csv(ENGINE_OUTPUT, index=False)
    dimension_summary.to_csv(DIMENSION_OUTPUT, index=False)

    write_report(df, profiles, engine_summary, dimension_summary)

    print("\nVALIDATION PASSED")
    print("-" * 76)
    print(f"Country profiles: {len(profiles)}")
    print(f"Country-engine summaries: {len(engine_summary)}")
    print(f"Risk-dimension summaries: {len(dimension_summary)}")

    print("\nOUTPUTS CREATED")
    print(f"  {COUNTRY_OUTPUT}")
    print(f"  {ENGINE_OUTPUT}")
    print(f"  {DIMENSION_OUTPUT}")
    print(f"  {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
