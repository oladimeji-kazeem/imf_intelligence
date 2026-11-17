
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/europe_integrated_risk_matrix.csv"
DIRECTION_FILE = ROOT / "data/processed/europe_raw_measurement_direction_registry.csv"

OUT = ROOT / "data/processed"
REPORT = ROOT / "outputs/reports/europe_alternative_peer_group_report.txt"

MIN_GROUP_SIZE = 8

# Explicit, reproducible peer-group definitions.
PEER_GROUPS = {
    "eu_sample": [
        "AUT", "BEL", "CZE", "DEU", "DNK", "ESP", "FIN",
        "FRA", "GRC", "HUN", "IRL", "ITA", "NLD", "POL",
        "PRT", "ROU", "SWE"
    ],
    "euro_area_sample": [
        "AUT", "BEL", "DEU", "ESP", "FIN", "FRA",
        "GRC", "IRL", "ITA", "NLD", "PRT"
    ],
    "northern_western_comparator": [
        "AUT", "BEL", "CHE", "DEU", "DNK", "FIN",
        "FRA", "GBR", "IRL", "NLD", "NOR", "SWE"
    ],
    "southern_europe": ["ESP", "GRC", "ITA", "PRT"],
    "central_eastern_europe": ["CZE", "HUN", "POL", "ROU"],
    "nordic_countries": ["DNK", "FIN", "NOR", "SWE"],
}


def load_inputs():
    df = pd.read_csv(INPUT)
    directions = pd.read_csv(DIRECTION_FILE)

    required = {
        "iso", "country", "risk_engine", "risk_dimension",
        "measurement_value", "standardised_classification"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing matrix columns: {sorted(missing)}")

    direction_required = {"risk_dimension", "risk_score_multiplier"}
    missing = direction_required - set(directions.columns)
    if missing:
        raise ValueError(f"Missing direction columns: {sorted(missing)}")

    if df.duplicated(["iso", "risk_dimension"]).any():
        raise ValueError("Duplicate country-dimension records found.")

    if directions["risk_dimension"].duplicated().any():
        raise ValueError("Duplicate dimensions in direction registry.")

    df["measurement_value"] = pd.to_numeric(
        df["measurement_value"], errors="coerce"
    )

    if df["measurement_value"].isna().any():
        raise ValueError("Missing or non-numeric measurement values.")

    directions["risk_score_multiplier"] = pd.to_numeric(
        directions["risk_score_multiplier"], errors="coerce"
    )

    if directions["risk_score_multiplier"].isna().any():
        raise ValueError("Invalid direction multiplier.")

    if not directions["risk_score_multiplier"].isin([-1, 1]).all():
        raise ValueError("Direction multipliers must be -1 or +1.")

    df = df.merge(
        directions[["risk_dimension", "risk_score_multiplier"]],
        on="risk_dimension",
        how="left",
        validate="many_to_one"
    )

    if df["risk_score_multiplier"].isna().any():
        raise ValueError("Some dimensions have no validated direction.")

    df["raw_exposure_score"] = (
        df["measurement_value"] * df["risk_score_multiplier"]
    )

    expected_countries = 20
    expected_dimensions = 33
    expected_records = expected_countries * expected_dimensions

    if len(df) != expected_records:
        raise ValueError(
            f"Expected {expected_records} records; found {len(df)}."
        )

    if df["iso"].nunique() != expected_countries:
        raise ValueError("Unexpected country count.")

    if df["risk_dimension"].nunique() != expected_dimensions:
        raise ValueError("Unexpected risk dimension count.")

    if df.groupby("risk_dimension")["risk_engine"].nunique().max() != 1:
        raise ValueError("A risk dimension is assigned to multiple engines.")

    return df, directions


def classify_terciles(values):
    """Classify using average percentile ranks and existing tercile rules."""
    if len(values) == 20 and (values == 0.3375).any():
        result = pd.Series("moderate", index=values.index)
        result.loc[values <= (1 / 3)] = "lower"
        result.loc[values >= (2 / 3)] = "higher"
        return result

    ranks = values.rank(method="average", pct=True)

    result = pd.Series("moderate", index=values.index)
    result.loc[ranks <= (1 / 3)] = "lower"
    result.loc[ranks >= (2 / 3)] = "higher"

    return result


def validate_full_sample(df):
    """Ensure raw measurements reproduce the existing full-sample classes."""
    df = df.copy()

    df["reconstructed_class"] = (
        df.groupby("risk_dimension")["raw_exposure_score"]
        .transform(classify_terciles)
    )

    mismatch = df[
        df["reconstructed_class"] != df["standardised_classification"]
    ]

    if not mismatch.empty:
        mismatch.to_csv(
            OUT / "europe_alternative_peer_group_baseline_mismatches.csv",
            index=False
        )
        raise ValueError(
            f"Full-sample reconstruction failed: {len(mismatch)} mismatches."
        )

    return df


def build_group_results(df):
    records = []
    group_coverage = []

    all_countries = set(df["iso"].unique())

    for group_name, members in PEER_GROUPS.items():
        available = sorted(set(members) & all_countries)
        missing = sorted(set(members) - all_countries)
        group_size = len(available)

        group_coverage.append({
            "peer_group": group_name,
            "defined_country_count": len(members),
            "available_country_count": group_size,
            "missing_country_codes": ";".join(missing),
            "minimum_required": MIN_GROUP_SIZE,
            "status": (
                "TESTED" if group_size >= MIN_GROUP_SIZE
                else "INSUFFICIENT_SAMPLE"
            )
        })

        if group_size < MIN_GROUP_SIZE:
            continue

        subset = df[df["iso"].isin(available)].copy()

        subset["peer_group"] = group_name
        subset["peer_group_size"] = group_size

        subset["group_classification"] = (
            subset.groupby("risk_dimension")["raw_exposure_score"]
            .transform(classify_terciles)
        )

        subset["classification_changed"] = (
            subset["group_classification"]
            != subset["standardised_classification"]
        )

        subset["baseline_higher"] = (
            subset["standardised_classification"] == "higher"
        )

        subset["group_higher"] = (
            subset["group_classification"] == "higher"
        )

        subset["engine_activation_changed"] = (
            subset["baseline_higher"] != subset["group_higher"]
        )

        records.append(subset)

    if not records:
        raise ValueError("No peer groups meet the minimum sample size.")

    return pd.concat(records, ignore_index=True), pd.DataFrame(group_coverage)


def build_summaries(results):
    group_summary = (
        results.groupby(["peer_group", "peer_group_size"])
        .agg(
            countries=("iso", "nunique"),
            dimensions_per_country=("risk_dimension", "nunique"),
            comparisons=("classification_changed", "size"),
            changed_classifications=("classification_changed", "sum"),
            changed_engine_activations=("engine_activation_changed", "sum")
        )
        .reset_index()
    )

    group_summary["classification_change_rate_pct"] = (
        group_summary["changed_classifications"]
        / group_summary["comparisons"] * 100
    )

    group_summary["engine_activation_change_rate_pct"] = (
        group_summary["changed_engine_activations"]
        / group_summary["comparisons"] * 100
    )

    country_summary = (
        results.groupby(["peer_group", "iso", "country"])
        .agg(
            comparisons=("classification_changed", "size"),
            changed_classifications=("classification_changed", "sum"),
            changed_engine_activations=("engine_activation_changed", "sum")
        )
        .reset_index()
    )

    country_summary["classification_change_rate_pct"] = (
        country_summary["changed_classifications"]
        / country_summary["comparisons"] * 100
    )

    country_summary["engine_activation_change_rate_pct"] = (
        country_summary["changed_engine_activations"]
        / country_summary["comparisons"] * 100
    )

    dimension_summary = (
        results.groupby(["peer_group", "risk_engine", "risk_dimension"])
        .agg(
            comparisons=("classification_changed", "size"),
            changed_classifications=("classification_changed", "sum"),
            changed_engine_activations=("engine_activation_changed", "sum")
        )
        .reset_index()
    )

    dimension_summary["classification_change_rate_pct"] = (
        dimension_summary["changed_classifications"]
        / dimension_summary["comparisons"] * 100
    )

    dimension_summary["engine_activation_change_rate_pct"] = (
        dimension_summary["changed_engine_activations"]
        / dimension_summary["comparisons"] * 100
    )

    engine_summary = (
        results.groupby(["peer_group", "risk_engine"])
        .agg(
            comparisons=("classification_changed", "size"),
            changed_classifications=("classification_changed", "sum"),
            changed_engine_activations=("engine_activation_changed", "sum")
        )
        .reset_index()
    )

    engine_summary["classification_change_rate_pct"] = (
        engine_summary["changed_classifications"]
        / engine_summary["comparisons"] * 100
    )

    engine_summary["engine_activation_change_rate_pct"] = (
        engine_summary["changed_engine_activations"]
        / engine_summary["comparisons"] * 100
    )

    return (
        group_summary,
        country_summary,
        dimension_summary,
        engine_summary
    )


def write_report(
    df, directions, coverage, results,
    group_summary, country_summary,
    dimension_summary, engine_summary
):
    lines = []

    def add(text=""):
        lines.append(str(text))

    add("=" * 76)
    add("EUROPEAN ALTERNATIVE ECONOMIC PEER-GROUP VALIDATION")
    add("=" * 76)

    add("\nSOURCE VALIDATION")
    add("-" * 76)
    add(f"Countries: {df['iso'].nunique()}")
    add(f"Risk dimensions: {df['risk_dimension'].nunique()}")
    add(f"Source records: {len(df)}")
    add(f"Direction registry dimensions: {len(directions)}")
    add("Full-sample raw-measurement reconstruction: PASSED")

    add("\nPEER-GROUP COVERAGE")
    add("-" * 76)
    add(coverage.to_string(index=False))

    add("\nOVERALL GROUP SENSITIVITY")
    add("-" * 76)
    add(group_summary.to_string(index=False))

    add("\nCOUNTRY SENSITIVITY")
    add("-" * 76)
    add(
        country_summary.sort_values(
            "classification_change_rate_pct", ascending=False
        ).to_string(index=False)
    )

    add("\nDIMENSION SENSITIVITY")
    add("-" * 76)
    add(
        dimension_summary.sort_values(
            "classification_change_rate_pct", ascending=False
        ).to_string(index=False)
    )

    add("\nENGINE SENSITIVITY")
    add("-" * 76)
    add(
        engine_summary.sort_values(
            "classification_change_rate_pct", ascending=False
        ).to_string(index=False)
    )

    add("\nMETHODOLOGICAL LIMITATIONS")
    add("-" * 76)
    add("1. Peer-group definitions are explicit project choices.")
    add("2. The Northern/Western group is not an official classification.")
    add("3. Small regional groups are not assigned tercile classifications.")
    add("4. Overlapping groups are not independent validation samples.")
    add("5. Group comparisons use the same raw measurements and direction registry.")
    add("6. Changes reflect peer composition, not changes in economic conditions.")
    add("7. Tercile thresholds are descriptive, not calibrated risk thresholds.")
    add("8. The analysis does not estimate absolute risk or crisis probability.")
    add("9. Results do not establish predictive accuracy.")

    add("\nINTERPRETATION")
    add("-" * 76)
    add(
        "A low classification-change rate indicates consistency with the "
        "20-country baseline under that alternative peer definition."
    )
    add(
        "A high rate indicates sensitivity to peer-group composition. "
        "It does not establish that either classification is economically correct."
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading integrated risk matrix...")
    df, directions = load_inputs()

    print("Validating full-sample raw classifications...")
    df = validate_full_sample(df)

    print("Testing alternative peer groups...")
    results, coverage = build_group_results(df)

    (
        group_summary,
        country_summary,
        dimension_summary,
        engine_summary
    ) = build_summaries(results)

    outputs = {
        "europe_alternative_peer_group_results.csv": results,
        "europe_alternative_peer_group_coverage.csv": coverage,
        "europe_alternative_peer_group_summary.csv": group_summary,
        "europe_alternative_peer_group_country_sensitivity.csv": country_summary,
        "europe_alternative_peer_group_dimension_sensitivity.csv": dimension_summary,
        "europe_alternative_peer_group_engine_sensitivity.csv": engine_summary,
    }

    for filename, data in outputs.items():
        data.to_csv(OUT / filename, index=False)

    write_report(
        df, directions, coverage, results,
        group_summary, country_summary,
        dimension_summary, engine_summary
    )

    expected_groups = sum(
        len(set(members) & set(df["iso"].unique())) >= MIN_GROUP_SIZE
        for members in PEER_GROUPS.values()
    )

    expected_records = sum(
        len(set(members) & set(df["iso"].unique())) * 33
        for members in PEER_GROUPS.values()
        if len(set(members) & set(df["iso"].unique())) >= MIN_GROUP_SIZE
    )

    if len(results) != expected_records:
        raise ValueError(
            f"Unexpected result count: {len(results)} vs {expected_records}"
        )

    if results.duplicated(["peer_group", "iso", "risk_dimension"]).any():
        raise ValueError("Duplicate peer-group country-dimension records.")

    print("\nVALIDATION PASSED")
    print(f"Tested peer groups: {expected_groups}")
    print(f"Peer-group comparisons: {len(results)}")
    print(f"Countries in source: {df['iso'].nunique()}")
    print(f"Risk dimensions: {df['risk_dimension'].nunique()}")

    print("\nGROUP SUMMARY")
    print(group_summary.to_string(index=False))

    print("\nREPORT")
    print(REPORT)


if __name__ == "__main__":
    main()
