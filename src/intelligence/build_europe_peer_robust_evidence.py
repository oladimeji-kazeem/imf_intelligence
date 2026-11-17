
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

PEER_INPUT = DATA / "europe_alternative_peer_group_results.csv"
STABILITY_INPUT = DATA / "europe_multi_peer_country_dimension_stability.csv"

OUTPUT = DATA / "europe_peer_robust_vulnerability_evidence.csv"
COUNTRY_OUTPUT = DATA / "europe_peer_robust_country_summary.csv"
ENGINE_OUTPUT = DATA / "europe_peer_robust_engine_summary.csv"
DIMENSION_OUTPUT = DATA / "europe_peer_robust_dimension_summary.csv"
REPORT_OUTPUT = REPORTS / "europe_peer_robust_evidence_report.txt"

EXPECTED_COUNTRIES = 20
EXPECTED_DIMENSIONS = 33
EXPECTED_RECORDS = 660

REQUIRED_PEER_COLUMNS = {
    "peer_group",
    "iso",
    "country",
    "risk_engine",
    "risk_dimension",
    "group_classification",
    "standardised_classification",
}

REQUIRED_STABILITY_COLUMNS = {
    "iso",
    "country",
    "risk_engine",
    "risk_dimension",
    "peer_group_count",
    "peer_groups_observed",
    "higher_classification_count",
    "non_higher_classification_count",
    "classification_stability",
    "exposure_stability",
    "baseline_classification",
}


def load_inputs():
    peer = pd.read_csv(PEER_INPUT)
    stability = pd.read_csv(STABILITY_INPUT)

    missing_peer = REQUIRED_PEER_COLUMNS - set(peer.columns)
    missing_stability = REQUIRED_STABILITY_COLUMNS - set(stability.columns)

    if missing_peer:
        raise ValueError(
            f"Peer input missing columns: {sorted(missing_peer)}"
        )

    if missing_stability:
        raise ValueError(
            f"Stability input missing columns: {sorted(missing_stability)}"
        )

    peer_key = ["peer_group", "iso", "risk_dimension"]
    stability_key = ["iso", "risk_dimension"]

    if peer.duplicated(peer_key).any():
        raise ValueError("Duplicate peer-group country-dimension records.")

    if stability.duplicated(stability_key).any():
        raise ValueError("Duplicate stability country-dimension records.")

    valid_classes = {"higher", "moderate", "lower"}

    if not set(peer["group_classification"].dropna()).issubset(valid_classes):
        raise ValueError("Unexpected peer classification labels.")

    if not set(peer["standardised_classification"].dropna()).issubset(
        valid_classes
    ):
        raise ValueError("Unexpected baseline classification labels.")

    if len(stability) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} stability records; "
            f"found {len(stability)}."
        )

    if stability["iso"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError("Unexpected country count.")

    if stability["risk_dimension"].nunique() != EXPECTED_DIMENSIONS:
        raise ValueError("Unexpected dimension count.")

    return peer, stability


def build_evidence(peer, stability):
    # Preserve the individual peer-group classifications.
    peer_wide = peer.pivot(
        index=["iso", "risk_dimension"],
        columns="peer_group",
        values="group_classification"
    ).reset_index()

    peer_columns = [
        column for column in peer_wide.columns
        if column not in {"iso", "risk_dimension"}
    ]

    peer_wide = peer_wide.rename(
        columns={column: f"class_{column}" for column in peer_columns}
    )

    evidence = stability.merge(
        peer_wide,
        on=["iso", "risk_dimension"],
        how="left",
        validate="one_to_one"
    )

    # Recalculate support from the actual peer-group classifications.
    class_columns = [
        column for column in evidence.columns
        if column.startswith("class_")
    ]

    evidence["peer_groups_with_classification"] = (
        evidence[class_columns].notna().sum(axis=1)
    )

    evidence["higher_peer_support_count"] = (
        evidence[class_columns].eq("higher").sum(axis=1)
    )

    evidence["non_higher_peer_support_count"] = (
        evidence[class_columns].isin(["moderate", "lower"]).sum(axis=1)
    )

    evidence["higher_peer_support_share"] = (
        evidence["higher_peer_support_count"]
        / evidence["peer_groups_with_classification"].replace(0, pd.NA)
    )

    evidence["supporting_higher_peer_groups"] = evidence.apply(
        lambda row: ";".join(
            column.replace("class_", "")
            for column in class_columns
            if row[column] == "higher"
        ),
        axis=1
    )

    evidence["supporting_non_higher_peer_groups"] = evidence.apply(
        lambda row: ";".join(
            column.replace("class_", "")
            for column in class_columns
            if row[column] in {"moderate", "lower"}
        ),
        axis=1
    )

    # Final evidence label. This is not a numerical risk score.
    label_map = {
        "consistent_higher_exposure": "robust_higher_exposure",
        "consistent_non_higher_exposure": "robust_non_higher_exposure",
        "peer_sensitive_exposure": "peer_sensitive_exposure",
        "insufficient_peer_coverage": "insufficient_peer_coverage",
    }

    evidence["evidence_label"] = (
        evidence["exposure_stability"].map(label_map)
    )

    if evidence["evidence_label"].isna().any():
        raise ValueError("Unmapped exposure stability labels.")

    # Distinguish exposure stability from exact three-class stability.
    evidence["three_class_evidence"] = evidence[
        "classification_stability"
    ].map({
        "consistent_same_class": "same_three_level_class",
        "classification_migration": "three_level_class_migration",
        "insufficient_peer_coverage": "insufficient_peer_coverage",
    })

    # Explicit coverage category.
    evidence["peer_coverage_category"] = evidence[
        "peer_groups_with_classification"
    ].map({
        0: "no_peer_group_coverage",
        1: "one_peer_group_only",
        2: "two_peer_groups",
        3: "three_peer_groups",
    }).fillna("unexpected_peer_group_count")

    if evidence["peer_coverage_category"].eq(
        "unexpected_peer_group_count"
    ).any():
        raise ValueError("Unexpected number of peer groups.")

    # Reconciliation checks.
    if not (
        evidence["peer_groups_with_classification"]
        == evidence["peer_group_count"]
    ).all():
        raise ValueError("Peer-group coverage does not reconcile.")

    if not (
        evidence["higher_peer_support_count"]
        == evidence["higher_classification_count"]
    ).all():
        raise ValueError("Higher-exposure support does not reconcile.")

    if not (
        evidence["non_higher_peer_support_count"]
        == evidence["non_higher_classification_count"]
    ).all():
        raise ValueError("Non-higher support does not reconcile.")

    if not (
        evidence["higher_peer_support_count"]
        + evidence["non_higher_peer_support_count"]
        == evidence["peer_groups_with_classification"]
    ).all():
        raise ValueError("Peer support counts do not reconcile.")

    return evidence


def build_country_summary(evidence):
    rows = []

    for (iso, country), group in evidence.groupby(
        ["iso", "country"], sort=True
    ):
        assessed = group[
            group["evidence_label"] != "insufficient_peer_coverage"
        ]

        rows.append({
            "iso": iso,
            "country": country,
            "total_dimensions": len(group),
            "assessed_dimensions": len(assessed),
            "insufficient_dimensions": (
                len(group) - len(assessed)
            ),
            "robust_higher_dimensions": (
                assessed["evidence_label"]
                .eq("robust_higher_exposure").sum()
            ),
            "peer_sensitive_dimensions": (
                assessed["evidence_label"]
                .eq("peer_sensitive_exposure").sum()
            ),
            "robust_non_higher_dimensions": (
                assessed["evidence_label"]
                .eq("robust_non_higher_exposure").sum()
            ),
            "three_class_migrations": (
                assessed["three_class_evidence"]
                .eq("three_level_class_migration").sum()
            ),
            "three_peer_group_dimensions": (
                group["peer_coverage_category"]
                .eq("three_peer_groups").sum()
            ),
            "two_peer_group_dimensions": (
                group["peer_coverage_category"]
                .eq("two_peer_groups").sum()
            ),
            "one_peer_group_dimensions": (
                group["peer_coverage_category"]
                .eq("one_peer_group_only").sum()
            ),
        })

    result = pd.DataFrame(rows)

    result["assessed_coverage_pct"] = (
        result["assessed_dimensions"]
        / result["total_dimensions"] * 100
    )

    result["robust_higher_share_pct"] = (
        result["robust_higher_dimensions"]
        / result["assessed_dimensions"].replace(0, pd.NA)
        * 100
    )

    result["peer_sensitive_share_pct"] = (
        result["peer_sensitive_dimensions"]
        / result["assessed_dimensions"].replace(0, pd.NA)
        * 100
    )

    return result


def build_group_summary(evidence, group_columns):
    rows = []

    for engine, group in evidence.groupby("risk_engine", sort=True):
        for label in [
            "robust_higher_exposure",
            "peer_sensitive_exposure",
            "robust_non_higher_exposure",
            "insufficient_peer_coverage",
        ]:
            rows.append({
                "risk_engine": engine,
                "evidence_label": label,
                "dimension_records": int(
                    group["evidence_label"].eq(label).sum()
                ),
            })

    return pd.DataFrame(rows)


def build_dimension_summary(evidence):
    rows = []

    for (engine, dimension), group in evidence.groupby(
        ["risk_engine", "risk_dimension"], sort=True
    ):
        assessed = group[
            group["evidence_label"] != "insufficient_peer_coverage"
        ]

        rows.append({
            "risk_engine": engine,
            "risk_dimension": dimension,
            "countries": group["iso"].nunique(),
            "assessed_countries": len(assessed),
            "insufficient_countries": (
                len(group) - len(assessed)
            ),
            "robust_higher_countries": (
                assessed["evidence_label"]
                .eq("robust_higher_exposure").sum()
            ),
            "peer_sensitive_countries": (
                assessed["evidence_label"]
                .eq("peer_sensitive_exposure").sum()
            ),
            "robust_non_higher_countries": (
                assessed["evidence_label"]
                .eq("robust_non_higher_exposure").sum()
            ),
            "three_class_migrations": (
                assessed["three_class_evidence"]
                .eq("three_level_class_migration").sum()
            ),
        })

    return pd.DataFrame(rows)


def write_report(evidence, country, engine, dimension):
    lines = []

    def add(value=""):
        lines.append(str(value))

    add("=" * 76)
    add("EUROPEAN PEER-ROBUST VULNERABILITY EVIDENCE REPORT")
    add("=" * 76)

    add("\nVALIDATION")
    add("-" * 76)
    add(f"Country-dimension records: {len(evidence)}")
    add(f"Countries: {evidence['iso'].nunique()}")
    add(f"Risk dimensions: {evidence['risk_dimension'].nunique()}")
    add(f"Risk engines: {evidence['risk_engine'].nunique()}")

    add("\nEVIDENCE LABEL DISTRIBUTION")
    add("-" * 76)
    add(evidence["evidence_label"].value_counts().to_string())

    add("\nPEER COVERAGE DISTRIBUTION")
    add("-" * 76)
    add(evidence["peer_coverage_category"].value_counts().to_string())

    add("\nCOUNTRY SUMMARY")
    add("-" * 76)
    add(
        country.sort_values(
            ["robust_higher_dimensions", "peer_sensitive_dimensions"],
            ascending=[False, False]
        ).to_string(index=False)
    )

    add("\nENGINE EVIDENCE DISTRIBUTION")
    add("-" * 76)
    add(engine.to_string(index=False))

    add("\nDIMENSION SUMMARY")
    add("-" * 76)
    add(
        dimension.sort_values(
            ["robust_higher_countries", "peer_sensitive_countries"],
            ascending=[False, False]
        ).to_string(index=False)
    )

    add("\nINTERPRETATION AND LIMITATIONS")
    add("-" * 76)
    add(
        "Robust higher exposure means the dimension is higher in every "
        "eligible peer group represented for that country."
    )
    add(
        "Robust non-higher exposure means the dimension is never higher "
        "in the eligible peer groups represented."
    )
    add(
        "Peer-sensitive exposure means the higher/non-higher classification "
        "changes across eligible peer groups."
    )
    add(
        "Insufficient peer coverage means fewer than two peer groups "
        "classify that country-dimension."
    )
    add(
        "Two-group and three-group evidence are both retained, but their "
        "coverage strength is explicitly recorded."
    )
    add(
        "Three-class migration may occur without a change in higher-exposure "
        "activation."
    )
    add(
        "The tested peer groups overlap and are not independent replications."
    )
    add(
        "These labels describe peer-relative classification stability, "
        "not absolute risk, crisis probability, default probability or "
        "predictive accuracy."
    )
    add(
        "No composite risk score is calculated."
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading peer-group and stability data...")
    peer, stability = load_inputs()

    print("Building peer-robust evidence...")
    evidence = build_evidence(peer, stability)

    print("Building country and engine summaries...")
    country = build_country_summary(evidence)
    engine = build_group_summary(evidence, None)
    dimension = build_dimension_summary(evidence)

    evidence.to_csv(OUTPUT, index=False)
    country.to_csv(COUNTRY_OUTPUT, index=False)
    engine.to_csv(ENGINE_OUTPUT, index=False)
    dimension.to_csv(DIMENSION_OUTPUT, index=False)

    write_report(evidence, country, engine, dimension)

    if len(evidence) != EXPECTED_RECORDS:
        raise ValueError("Unexpected evidence record count.")

    if len(country) != EXPECTED_COUNTRIES:
        raise ValueError("Unexpected country summary count.")

    if len(dimension) != EXPECTED_DIMENSIONS:
        raise ValueError("Unexpected dimension summary count.")

    if evidence.duplicated(["iso", "risk_dimension"]).any():
        raise ValueError("Duplicate evidence records.")

    print("\nVALIDATION PASSED")
    print(f"Evidence records: {len(evidence)}")
    print(f"Country summaries: {len(country)}")
    print(f"Engine summaries: {len(engine)}")
    print(f"Dimension summaries: {len(dimension)}")

    print("\nEVIDENCE LABELS")
    print(evidence["evidence_label"].value_counts().to_string())

    print("\nPEER COVERAGE")
    print(evidence["peer_coverage_category"].value_counts().to_string())

    print("\nREPORT")
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
