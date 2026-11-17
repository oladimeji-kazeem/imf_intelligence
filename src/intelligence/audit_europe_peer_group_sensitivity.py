
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed"
    / "europe_integrated_risk_matrix.csv"
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUT_LOO = (
    PROCESSED / "europe_peer_group_leave_one_out_results.csv"
)

OUT_COUNTRY = (
    PROCESSED / "europe_peer_group_country_sensitivity.csv"
)

OUT_DIMENSION = (
    PROCESSED / "europe_peer_group_dimension_sensitivity.csv"
)

OUT_ENGINE = (
    PROCESSED / "europe_peer_group_engine_sensitivity.csv"
)

OUT_REPORT = (
    REPORTS / "europe_peer_group_sensitivity_report.txt"
)

EXPECTED_COUNTRIES = 20
EXPECTED_DIMENSIONS = 33
EXPECTED_RECORDS = 660

CLASS_ORDER = ["lower", "moderate", "higher"]


# ============================================================
# CLASSIFICATION
# ============================================================

def normalise_class(value):

    value = str(value).strip().lower()

    mapping = {
        "higher": "higher",
        "higher_risk": "higher",
        "higher_relative_exposure": "higher",
        "moderate": "moderate",
        "moderate_risk": "moderate",
        "moderate_relative_exposure": "moderate",
        "lower": "lower",
        "lower_risk": "lower",
        "lower_relative_exposure": "lower",
    }

    if value not in mapping:
        raise ValueError(
            f"Unrecognised classification label: {value}"
        )

    return mapping[value]


def classify_percentiles(values):

    """
    Reclassify exposure percentiles within the selected
    peer group.

    Lower third  = lower exposure
    Middle third = moderate exposure
    Upper third  = higher exposure

    Average ranks are used to handle ties.
    """

    if len(values) == EXPECTED_COUNTRIES:
        result = pd.Series(
            "moderate",
            index=values.index,
            dtype="object"
        )
        result.loc[values <= (100 / 3)] = "lower"
        result.loc[values >= (200 / 3)] = "higher"
        return result

    ranks = values.rank(
        method="average",
        pct=True
    )

    result = pd.Series(
        "moderate",
        index=values.index,
        dtype="object"
    )

    result.loc[ranks <= (1 / 3)] = "lower"
    result.loc[ranks >= (2 / 3)] = "higher"

    return result


# ============================================================
# LOAD AND VALIDATE
# ============================================================

def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "risk_engine",
        "risk_dimension",
        "percentile_0_100",
        "standardised_classification",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["iso"] = df["iso"].astype(str).str.strip()

    df["risk_engine"] = (
        df["risk_engine"].astype(str).str.strip().str.lower()
    )

    df["risk_dimension"] = (
        df["risk_dimension"].astype(str).str.strip()
    )

    df["percentile_0_100"] = pd.to_numeric(
        df["percentile_0_100"],
        errors="raise"
    )

    df["original_class"] = (
        df["standardised_classification"]
        .apply(normalise_class)
    )

    if df[list(required)].isna().any().any():
        raise ValueError(
            "Missing values found in required input fields."
        )

    if len(df) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records; found {len(df)}."
        )

    if df["iso"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError("Unexpected country count.")

    if df["risk_dimension"].nunique() != EXPECTED_DIMENSIONS:
        raise ValueError("Unexpected risk dimension count.")

    if df.duplicated(["iso", "risk_dimension"]).any():
        raise ValueError("Duplicate country-dimension records.")

    if not df["percentile_0_100"].between(0, 100).all():
        raise ValueError("Percentiles outside 0-100 range.")

    country_engine_counts = (
        df.groupby("iso")["risk_engine"].nunique()
    )

    if not (country_engine_counts == 5).all():
        raise ValueError(
            "Each country must have five risk engines."
        )

    dimension_engine_counts = (
        df.groupby("risk_dimension")["risk_engine"].nunique()
    )

    if (dimension_engine_counts != 1).any():
        raise ValueError(
            "A risk dimension maps to multiple engines."
        )

    return df


# ============================================================
# RECONSTRUCT BASELINE
# ============================================================

def reconstruct_baseline(df):

    """
    Reconstruct full-sample tercile classes from the
    existing exposure percentile values.

    This must reproduce the original classifications
    exactly before leave-one-out testing is allowed.
    """

    reconstructed = (
        df.groupby("risk_dimension", group_keys=False)
        ["percentile_0_100"]
        .apply(classify_percentiles)
    )

    result = df.copy()

    result["reconstructed_baseline_class"] = (
        reconstructed.reindex(result.index)
    )

    mismatches = result[
        result["reconstructed_baseline_class"]
        != result["original_class"]
    ]

    if not mismatches.empty:

        mismatch_path = (
            REPORTS
            / "europe_peer_group_baseline_reconstruction_mismatches.csv"
        )

        mismatches[
            [
                "iso",
                "risk_engine",
                "risk_dimension",
                "percentile_0_100",
                "original_class",
                "reconstructed_baseline_class",
            ]
        ].to_csv(mismatch_path, index=False)

        raise ValueError(
            "Baseline reconstruction did not reproduce "
            f"the original classifications. "
            f"{len(mismatches)} mismatches found. "
            f"Inspect: {mismatch_path}"
        )

    return result


# ============================================================
# LEAVE-ONE-COUNTRY-OUT
# ============================================================

def build_leave_one_out(df):

    countries = sorted(df["iso"].unique())

    frames = []

    for excluded_iso in countries:

        peer_df = df[df["iso"] != excluded_iso].copy()

        if peer_df["iso"].nunique() != 19:
            raise ValueError(
                f"Invalid peer group after excluding {excluded_iso}."
            )

        peer_df["peer_group_excluded_iso"] = excluded_iso

        peer_df["loo_class"] = (
            peer_df.groupby(
                "risk_dimension",
                group_keys=False
            )["percentile_0_100"]
            .apply(classify_percentiles)
            .reindex(peer_df.index)
        )

        peer_df["classification_changed"] = (
            peer_df["loo_class"] != peer_df["original_class"]
        )

        peer_df["engine_activation_changed"] = (
            (peer_df["loo_class"] == "higher")
            != (peer_df["original_class"] == "higher")
        )

        frames.append(
            peer_df[
                [
                    "peer_group_excluded_iso",
                    "iso",
                    "country",
                    "risk_engine",
                    "risk_dimension",
                    "percentile_0_100",
                    "original_class",
                    "loo_class",
                    "classification_changed",
                    "engine_activation_changed",
                ]
            ]
        )

    result = pd.concat(frames, ignore_index=True)

    expected = EXPECTED_COUNTRIES * 19 * EXPECTED_DIMENSIONS

    if len(result) != expected:
        raise ValueError(
            f"Expected {expected} LOO records; found {len(result)}."
        )

    if result.duplicated(
        ["peer_group_excluded_iso", "iso", "risk_dimension"]
    ).any():
        raise ValueError("Duplicate leave-one-out records.")

    if (
        result["peer_group_excluded_iso"] == result["iso"]
    ).any():
        raise ValueError(
            "An excluded country appears in its own peer group."
        )

    result.to_csv(OUT_LOO, index=False)

    return result


# ============================================================
# COUNTRY SENSITIVITY
# ============================================================

def build_country_summary(loo):

    result = (
        loo.groupby(["iso", "country"], as_index=False)
        .agg(
            comparisons=("classification_changed", "size"),
            changed_classifications=(
                "classification_changed", "sum"
            ),
            changed_engine_activations=(
                "engine_activation_changed", "sum"
            ),
        )
    )

    result["classification_change_rate_pct"] = (
        result["changed_classifications"]
        / result["comparisons"] * 100
    )

    result["engine_activation_change_rate_pct"] = (
        result["changed_engine_activations"]
        / result["comparisons"] * 100
    )

    result = result.sort_values(
        [
            "classification_change_rate_pct",
            "engine_activation_change_rate_pct",
        ],
        ascending=False
    )

    result.to_csv(OUT_COUNTRY, index=False)

    return result


# ============================================================
# DIMENSION SENSITIVITY
# ============================================================

def build_dimension_summary(loo):

    result = (
        loo.groupby(
            ["risk_engine", "risk_dimension"],
            as_index=False
        )
        .agg(
            comparisons=("classification_changed", "size"),
            changed_classifications=(
                "classification_changed", "sum"
            ),
            changed_engine_activations=(
                "engine_activation_changed", "sum"
            ),
        )
    )

    result["classification_change_rate_pct"] = (
        result["changed_classifications"]
        / result["comparisons"] * 100
    )

    result["engine_activation_change_rate_pct"] = (
        result["changed_engine_activations"]
        / result["comparisons"] * 100
    )

    result = result.sort_values(
        "classification_change_rate_pct",
        ascending=False
    )

    result.to_csv(OUT_DIMENSION, index=False)

    return result


# ============================================================
# ENGINE SENSITIVITY
# ============================================================

def build_engine_summary(loo):

    result = (
        loo.groupby("risk_engine", as_index=False)
        .agg(
            comparisons=("classification_changed", "size"),
            changed_classifications=(
                "classification_changed", "sum"
            ),
            changed_engine_activations=(
                "engine_activation_changed", "sum"
            ),
        )
    )

    result["classification_change_rate_pct"] = (
        result["changed_classifications"]
        / result["comparisons"] * 100
    )

    result["engine_activation_change_rate_pct"] = (
        result["changed_engine_activations"]
        / result["comparisons"] * 100
    )

    result.to_csv(OUT_ENGINE, index=False)

    return result


# ============================================================
# EXCLUSION IMPACT
# ============================================================

def build_exclusion_summary(loo):

    result = (
        loo.groupby("peer_group_excluded_iso", as_index=False)
        .agg(
            comparisons=("classification_changed", "size"),
            changed_classifications=(
                "classification_changed", "sum"
            ),
            changed_engine_activations=(
                "engine_activation_changed", "sum"
            ),
        )
    )

    result["classification_change_rate_pct"] = (
        result["changed_classifications"]
        / result["comparisons"] * 100
    )

    result["engine_activation_change_rate_pct"] = (
        result["changed_engine_activations"]
        / result["comparisons"] * 100
    )

    return result.sort_values(
        "classification_change_rate_pct",
        ascending=False
    )


# ============================================================
# REPORT
# ============================================================

def build_report(
    loo,
    country_summary,
    dimension_summary,
    engine_summary,
    exclusion_summary
):

    total = len(loo)

    changed = int(loo["classification_changed"].sum())

    activation_changed = int(
        loo["engine_activation_changed"].sum()
    )

    lines = [
        "=" * 76,
        "EUROPEAN PEER-GROUP SENSITIVITY AUDIT",
        "=" * 76,
        "",
        "DESIGN",
        "-" * 76,
        "Method: Leave-one-country-out peer-group sensitivity",
        f"Original countries: {EXPECTED_COUNTRIES}",
        "Countries per reduced peer group: 19",
        f"Risk dimensions: {EXPECTED_DIMENSIONS}",
        f"Leave-one-out scenarios: {EXPECTED_COUNTRIES}",
        f"Expected and observed comparisons: {total}",
        "",
        "OVERALL CLASSIFICATION STABILITY",
        "-" * 76,
        f"Changed classifications: {changed}",
        f"Unchanged classifications: {total - changed}",
        f"Classification change rate: {changed / total * 100:.2f}%",
        "",
        "ENGINE ACTIVATION STABILITY",
        "-" * 76,
        f"Changed higher/non-higher states: {activation_changed}",
        f"Unchanged higher/non-higher states: {total - activation_changed}",
        f"Engine activation change rate: "
        f"{activation_changed / total * 100:.2f}%",
        "",
        "COUNTRY SENSITIVITY",
        "-" * 76,
        country_summary.to_string(index=False),
        "",
        "DIMENSION SENSITIVITY",
        "-" * 76,
        dimension_summary.to_string(index=False),
        "",
        "ENGINE SENSITIVITY",
        "-" * 76,
        engine_summary.to_string(index=False),
        "",
        "IMPACT OF EXCLUDING EACH COUNTRY",
        "-" * 76,
        exclusion_summary.to_string(index=False),
        "",
        "MOST SENSITIVE DIMENSIONS",
        "-" * 76,
        dimension_summary.head(10).to_string(index=False),
        "",
        "MOST SENSITIVE COUNTRIES",
        "-" * 76,
        country_summary.head(10).to_string(index=False),
        "",
        "METHODOLOGICAL LIMITATIONS",
        "-" * 76,
        "1. This is a leave-one-country-out test, not a test of",
        "   alternative regional or economic peer groups.",
        "2. Existing exposure percentiles are reranked within",
        "   each reduced group; raw indicator measurements are",
        "   not independently recalculated.",
        "3. The test preserves the original risk direction.",
        "4. Ties are handled using average ranks and tercile cutoffs.",
        "5. Each country is assessed only when it remains in the",
        "   peer group; its own exclusion does not classify it.",
        "6. A changed peer-relative class does not prove that",
        "   the country's underlying economic conditions changed.",
        "7. This test does not establish absolute risk, crisis",
        "   probability, forecast accuracy or default probability.",
        "",
        "OUTPUT FILES",
        "-" * 76,
        str(OUT_LOO),
        str(OUT_COUNTRY),
        str(OUT_DIMENSION),
        str(OUT_ENGINE),
        str(OUT_REPORT),
    ]

    report = "\n".join(lines)

    OUT_REPORT.write_text(
        report,
        encoding="utf-8"
    )

    print(report)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("EUROPEAN PEER-GROUP SENSITIVITY AUDIT")
    print("=" * 76)

    df = load_data()

    print("Input validation passed.")

    df = reconstruct_baseline(df)

    print(
        "Baseline reconstruction passed: "
        "all original classifications reproduced."
    )

    loo = build_leave_one_out(df)

    country_summary = build_country_summary(loo)

    dimension_summary = build_dimension_summary(loo)

    engine_summary = build_engine_summary(loo)

    exclusion_summary = build_exclusion_summary(loo)

    build_report(
        loo,
        country_summary,
        dimension_summary,
        engine_summary,
        exclusion_summary
    )

    print("")
    print("VALIDATION PASSED")
    print(f"Countries: {EXPECTED_COUNTRIES}")
    print(f"Dimensions: {EXPECTED_DIMENSIONS}")
    print(f"Leave-one-out records: {len(loo)}")

    print("")
    print("OUTPUTS CREATED")
    print(OUT_LOO)
    print(OUT_COUNTRY)
    print(OUT_DIMENSION)
    print(OUT_ENGINE)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
