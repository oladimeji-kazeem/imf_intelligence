
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data/processed/"
    "europe_alternative_peer_group_results.csv"
)

OUT = ROOT / "data/processed"
REPORT = (
    ROOT / "outputs/reports/"
    "europe_multi_peer_stability_report.txt"
)

EXPECTED_COUNTRIES = 20
EXPECTED_DIMENSIONS = 33
MIN_PEER_GROUPS = 2

REQUIRED_COLUMNS = {
    "peer_group",
    "iso",
    "country",
    "risk_engine",
    "risk_dimension",
    "group_classification",
    "standardised_classification",
    "classification_changed",
    "baseline_higher",
    "group_higher",
    "engine_activation_changed",
}

VALID_CLASSES = {"higher", "moderate", "lower"}


def load_data():
    df = pd.read_csv(INPUT)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    key = ["peer_group", "iso", "risk_dimension"]

    if df.duplicated(key).any():
        raise ValueError("Duplicate peer-group country-dimension records.")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Missing values found in required fields.")

    if set(df["group_classification"].unique()) - VALID_CLASSES:
        raise ValueError("Unexpected alternative classification labels.")

    if set(df["standardised_classification"].unique()) - VALID_CLASSES:
        raise ValueError("Unexpected baseline classification labels.")

    if df["iso"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError("Unexpected country count.")

    if df["risk_dimension"].nunique() != EXPECTED_DIMENSIONS:
        raise ValueError("Unexpected dimension count.")

    if df.groupby("risk_dimension")["risk_engine"].nunique().max() != 1:
        raise ValueError("A dimension belongs to multiple engines.")

    expected_groups = {
        "eu_sample",
        "euro_area_sample",
        "northern_western_comparator",
    }

    actual_groups = set(df["peer_group"].unique())

    if actual_groups != expected_groups:
        raise ValueError(
            f"Unexpected tested peer groups: {sorted(actual_groups)}"
        )

    # Recalculate boolean fields rather than trusting CSV type inference.
    df["baseline_higher"] = (
        df["standardised_classification"].eq("higher")
    )

    df["group_higher"] = (
        df["group_classification"].eq("higher")
    )

    df["classification_changed"] = (
        df["group_classification"]
        != df["standardised_classification"]
    )

    df["engine_activation_changed"] = (
        df["group_higher"] != df["baseline_higher"]
    )

    return df


def classify_stability(group):
    classes = group["group_classification"].tolist()
    higher_flags = group["group_higher"].tolist()

    peer_count = len(classes)

    if peer_count < MIN_PEER_GROUPS:
        exposure_status = "insufficient_peer_coverage"
        classification_status = "insufficient_peer_coverage"

    else:
        if all(higher_flags):
            exposure_status = "consistent_higher_exposure"
        elif not any(higher_flags):
            exposure_status = "consistent_non_higher_exposure"
        else:
            exposure_status = "peer_sensitive_exposure"

        if len(set(classes)) == 1:
            classification_status = "consistent_same_class"
        else:
            classification_status = "classification_migration"

    return pd.Series({
        "peer_group_count": peer_count,
        "peer_groups_observed": ";".join(sorted(group["peer_group"].unique())),
        "observed_classifications": ";".join(sorted(set(classes))),
        "higher_classification_count": sum(higher_flags),
        "non_higher_classification_count": peer_count - sum(higher_flags),
        "classification_stability": classification_status,
        "exposure_stability": exposure_status,
        "baseline_classification": group["standardised_classification"].iloc[0],
        "baseline_higher": bool(group["baseline_higher"].iloc[0]),
        "mean_group_higher_rate": sum(higher_flags) / peer_count,
    })


def build_country_dimension_stability(df):
    records = []

    for (iso, dimension), group in df.groupby(
        ["iso", "risk_dimension"], sort=True
    ):
        result = classify_stability(group)

        records.append({
            "iso": iso,
            "country": group["country"].iloc[0],
            "risk_engine": group["risk_engine"].iloc[0],
            "risk_dimension": dimension,
            **result.to_dict()
        })

    result = pd.DataFrame(records)

    expected = EXPECTED_COUNTRIES * EXPECTED_DIMENSIONS

    if len(result) != expected:
        raise ValueError(
            f"Expected {expected} country-dimension records; "
            f"found {len(result)}."
        )

    if result.duplicated(["iso", "risk_dimension"]).any():
        raise ValueError("Duplicate country-dimension stability records.")

    return result


def build_country_engine_summary(stability):
    rows = []

    for (iso, country, engine), group in stability.groupby(
        ["iso", "country", "risk_engine"], sort=True
    ):
        assessed = group[
            group["exposure_stability"] != "insufficient_peer_coverage"
        ]

        rows.append({
            "iso": iso,
            "country": country,
            "risk_engine": engine,
            "total_dimensions": len(group),
            "assessed_dimensions": len(assessed),
            "insufficient_dimensions": (
                len(group) - len(assessed)
            ),
            "consistent_higher_dimensions": (
                assessed["exposure_stability"]
                .eq("consistent_higher_exposure").sum()
            ),
            "consistent_non_higher_dimensions": (
                assessed["exposure_stability"]
                .eq("consistent_non_higher_exposure").sum()
            ),
            "peer_sensitive_dimensions": (
                assessed["exposure_stability"]
                .eq("peer_sensitive_exposure").sum()
            ),
            "classification_migrations": (
                assessed["classification_stability"]
                .eq("classification_migration").sum()
            ),
        })

    result = pd.DataFrame(rows)

    if len(result) != EXPECTED_COUNTRIES * 5:
        raise ValueError("Unexpected country-engine summary size.")

    return result


def build_country_summary(stability):
    summary = (
        stability.groupby(["iso", "country"])
        .agg(
            total_dimensions=("risk_dimension", "nunique"),
            peer_assessed_dimensions=(
                "exposure_stability",
                lambda x: (x != "insufficient_peer_coverage").sum()
            ),
            insufficient_dimensions=(
                "exposure_stability",
                lambda x: (x == "insufficient_peer_coverage").sum()
            ),
            consistently_higher_dimensions=(
                "exposure_stability",
                lambda x: (x == "consistent_higher_exposure").sum()
            ),
            consistently_non_higher_dimensions=(
                "exposure_stability",
                lambda x: (x == "consistent_non_higher_exposure").sum()
            ),
            peer_sensitive_dimensions=(
                "exposure_stability",
                lambda x: (x == "peer_sensitive_exposure").sum()
            ),
            classification_migrations=(
                "classification_stability",
                lambda x: (x == "classification_migration").sum()
            ),
            peer_groups_available=("peer_group_count", "max"),
        )
        .reset_index()
    )

    summary["assessed_dimension_coverage_pct"] = (
        summary["peer_assessed_dimensions"]
        / summary["total_dimensions"] * 100
    )

    summary["consistent_higher_share_of_assessed_pct"] = (
        summary["consistently_higher_dimensions"]
        / summary["peer_assessed_dimensions"].replace(0, pd.NA)
        * 100
    )

    return summary


def build_dimension_summary(stability):
    summary = (
        stability.groupby(["risk_engine", "risk_dimension"])
        .agg(
            countries=("iso", "nunique"),
            assessed_countries=(
                "exposure_stability",
                lambda x: (x != "insufficient_peer_coverage").sum()
            ),
            insufficient_countries=(
                "exposure_stability",
                lambda x: (x == "insufficient_peer_coverage").sum()
            ),
            consistently_higher_countries=(
                "exposure_stability",
                lambda x: (x == "consistent_higher_exposure").sum()
            ),
            consistently_non_higher_countries=(
                "exposure_stability",
                lambda x: (x == "consistent_non_higher_exposure").sum()
            ),
            peer_sensitive_countries=(
                "exposure_stability",
                lambda x: (x == "peer_sensitive_exposure").sum()
            ),
            classification_migrations=(
                "classification_stability",
                lambda x: (x == "classification_migration").sum()
            ),
        )
        .reset_index()
    )

    summary["consistent_higher_pct_of_assessed"] = (
        summary["consistently_higher_countries"]
        / summary["assessed_countries"].replace(0, pd.NA)
        * 100
    )

    return summary


def build_peer_group_pair_comparison(df):
    """Compare the three group pairs for countries in both groups."""
    groups = sorted(df["peer_group"].unique())
    rows = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            left_name = groups[i]
            right_name = groups[j]

            left = df[df["peer_group"] == left_name][
                ["iso", "risk_dimension", "group_classification", "group_higher"]
            ].rename(columns={
                "group_classification": "left_class",
                "group_higher": "left_higher"
            })

            right = df[df["peer_group"] == right_name][
                ["iso", "risk_dimension", "group_classification", "group_higher"]
            ].rename(columns={
                "group_classification": "right_class",
                "group_higher": "right_higher"
            })

            merged = left.merge(
                right,
                on=["iso", "risk_dimension"],
                how="inner",
                validate="one_to_one"
            )

            if merged.empty:
                continue

            merged["peer_group_pair"] = f"{left_name} vs {right_name}"
            merged["exact_class_changed"] = (
                merged["left_class"] != merged["right_class"]
            )
            merged["higher_activation_changed"] = (
                merged["left_higher"] != merged["right_higher"]
            )

            rows.append(merged)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def write_report(
    df, stability, country_summary,
    country_engine, dimension_summary, pair_comparison
):
    lines = []

    def add(value=""):
        lines.append(str(value))

    add("=" * 76)
    add("EUROPEAN MULTI-PEER VULNERABILITY STABILITY ASSESSMENT")
    add("=" * 76)

    add("\nSOURCE VALIDATION")
    add("-" * 76)
    add(f"Input records: {len(df)}")
    add(f"Countries: {df['iso'].nunique()}")
    add(f"Dimensions: {df['risk_dimension'].nunique()}")
    add(f"Tested peer groups: {df['peer_group'].nunique()}")
    add(f"Country-dimension stability records: {len(stability)}")

    add("\nEXPOSURE STABILITY DISTRIBUTION")
    add("-" * 76)
    add(
        stability["exposure_stability"]
        .value_counts(dropna=False).to_string()
    )

    add("\nCLASSIFICATION STABILITY DISTRIBUTION")
    add("-" * 76)
    add(
        stability["classification_stability"]
        .value_counts(dropna=False).to_string()
    )

    add("\nCOUNTRY VULNERABILITY STABILITY")
    add("-" * 76)
    add(
        country_summary.sort_values(
            ["peer_assessed_dimensions", "peer_sensitive_dimensions"],
            ascending=[False, False]
        ).to_string(index=False)
    )

    add("\nCOUNTRY-ENGINE STABILITY")
    add("-" * 76)
    add(country_engine.to_string(index=False))

    add("\nDIMENSION STABILITY")
    add("-" * 76)
    add(
        dimension_summary.sort_values(
            ["peer_sensitive_countries", "consistently_higher_countries"],
            ascending=[False, False]
        ).to_string(index=False)
    )

    add("\nDIRECT PEER-GROUP PAIR COMPARISON")
    add("-" * 76)

    if pair_comparison.empty:
        add("No overlapping country-dimension comparisons.")
    else:
        pair_summary = (
            pair_comparison.groupby("peer_group_pair")
            .agg(
                comparisons=("iso", "size"),
                countries=("iso", "nunique"),
                exact_class_changes=("exact_class_changed", "sum"),
                activation_changes=("higher_activation_changed", "sum")
            )
            .reset_index()
        )

        pair_summary["exact_class_change_rate_pct"] = (
            pair_summary["exact_class_changes"]
            / pair_summary["comparisons"] * 100
        )

        pair_summary["activation_change_rate_pct"] = (
            pair_summary["activation_changes"]
            / pair_summary["comparisons"] * 100
        )

        add(pair_summary.to_string(index=False))

    add("\nMETHODOLOGICAL INTERPRETATION")
    add("-" * 76)
    add(
        "1. Consistent higher exposure means all observed eligible peer groups "
        "classify the dimension as higher."
    )
    add(
        "2. Consistent non-higher exposure means no observed eligible peer "
        "group classifies the dimension as higher."
    )
    add(
        "3. Peer-sensitive exposure means higher/non-higher activation changes "
        "across observed groups."
    )
    add(
        "4. Classification migration separately captures changes among "
        "higher, moderate and lower labels."
    )
    add(
        "5. Countries appearing in fewer than two tested groups are marked "
        "insufficient for cross-peer stability assessment."
    )
    add(
        "6. These are descriptive stability labels, not confidence intervals, "
        "probabilities or absolute risk ratings."
    )
    add(
        "7. The three peer groups overlap and are not independent samples."
    )
    add(
        "8. Consistent non-higher exposure does not mean the country has "
        "no economic vulnerability."
    )
    add(
        "9. No composite score, crisis probability or default probability "
        "is estimated."
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading alternative peer-group results...")
    df = load_data()

    print("Building country-dimension stability...")
    stability = build_country_dimension_stability(df)

    print("Building country and engine summaries...")
    country_summary = build_country_summary(stability)
    country_engine = build_country_engine_summary(stability)
    dimension_summary = build_dimension_summary(stability)

    print("Comparing peer-group pairs...")
    pair_comparison = build_peer_group_pair_comparison(df)

    outputs = {
        "europe_multi_peer_country_dimension_stability.csv": stability,
        "europe_multi_peer_country_summary.csv": country_summary,
        "europe_multi_peer_country_engine_summary.csv": country_engine,
        "europe_multi_peer_dimension_summary.csv": dimension_summary,
        "europe_multi_peer_pair_comparison.csv": pair_comparison,
    }

    for filename, data in outputs.items():
        data.to_csv(OUT / filename, index=False)

    write_report(
        df,
        stability,
        country_summary,
        country_engine,
        dimension_summary,
        pair_comparison
    )

    if len(stability) != 660:
        raise ValueError("Expected 660 country-dimension stability records.")

    if len(country_summary) != 20:
        raise ValueError("Expected 20 country profiles.")

    if len(country_engine) != 100:
        raise ValueError("Expected 100 country-engine profiles.")

    if len(dimension_summary) != 33:
        raise ValueError("Expected 33 dimension summaries.")

    if stability.duplicated(["iso", "risk_dimension"]).any():
        raise ValueError("Duplicate stability records.")

    print("\nVALIDATION PASSED")
    print(f"Country-dimension records: {len(stability)}")
    print(f"Country profiles: {len(country_summary)}")
    print(f"Country-engine profiles: {len(country_engine)}")
    print(f"Dimension profiles: {len(dimension_summary)}")
    print(f"Peer-group pair comparisons: {len(pair_comparison)}")

    print("\nEXPOSURE STABILITY")
    print(stability["exposure_stability"].value_counts().to_string())

    print("\nREPORT")
    print(REPORT)


if __name__ == "__main__":
    main()
