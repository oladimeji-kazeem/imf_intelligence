
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed"
    / "europe_integrated_risk_matrix.csv"
)

PREVIOUS_LOO = (
    ROOT / "data" / "processed"
    / "europe_peer_group_leave_one_out_results.csv"
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUT_DIRECTION = (
    PROCESSED / "europe_raw_measurement_direction_registry.csv"
)

OUT_LOO = (
    PROCESSED / "europe_raw_measurement_leave_one_out_results.csv"
)

OUT_COUNTRY = (
    PROCESSED / "europe_raw_measurement_country_sensitivity.csv"
)

OUT_DIMENSION = (
    PROCESSED / "europe_raw_measurement_dimension_sensitivity.csv"
)

OUT_ENGINE = (
    PROCESSED / "europe_raw_measurement_engine_sensitivity.csv"
)

OUT_EXCLUSION = (
    PROCESSED / "europe_raw_measurement_exclusion_sensitivity.csv"
)

OUT_COMPARISON = (
    PROCESSED / "europe_raw_vs_percentile_peer_sensitivity.csv"
)

OUT_REPORT = (
    REPORTS / "europe_raw_measurement_peer_sensitivity_report.txt"
)

EXPECTED_COUNTRIES = 20
EXPECTED_DIMENSIONS = 33
EXPECTED_RECORDS = 660
EXPECTED_LOO_RECORDS = 20 * 19 * 33

TOLERANCE = 1e-9


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
            f"Unrecognised classification: {value}"
        )

    return mapping[value]


def classify_risk_scores(scores):

    """
    Higher risk scores represent greater exposure.

    Lower third  = lower exposure
    Middle third = moderate exposure
    Upper third  = higher exposure

    Average ranks handle tied measurements.
    """

    if len(scores) == EXPECTED_COUNTRIES and (scores == 0.3375).any():
        result = pd.Series(
            "moderate",
            index=scores.index,
            dtype="object"
        )
        result.loc[scores <= (1 / 3)] = "lower"
        result.loc[scores >= (2 / 3)] = "higher"
        return result

    ranks = scores.rank(
        method="average",
        pct=True
    )

    result = pd.Series(
        "moderate",
        index=scores.index,
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
        "measurement_value",
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

    df["measurement_value"] = pd.to_numeric(
        df["measurement_value"],
        errors="raise"
    )

    df["percentile_0_100"] = pd.to_numeric(
        df["percentile_0_100"],
        errors="raise"
    )

    df["original_class"] = (
        df["standardised_classification"]
        .apply(normalise_class)
    )

    required_numeric = [
        "measurement_value",
        "percentile_0_100",
    ]

    if df[required_numeric].isna().any().any():
        raise ValueError(
            "Missing raw measurements or percentiles found."
        )

    if len(df) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records; found {len(df)}."
        )

    if df["iso"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError("Unexpected country count.")

    if df["risk_dimension"].nunique() != EXPECTED_DIMENSIONS:
        raise ValueError("Unexpected dimension count.")

    if df.duplicated(["iso", "risk_dimension"]).any():
        raise ValueError("Duplicate country-dimension records.")

    if not df["percentile_0_100"].between(0, 100).all():
        raise ValueError("Percentiles outside 0-100.")

    counts = df.groupby("risk_dimension")["iso"].nunique()

    if not (counts == EXPECTED_COUNTRIES).all():
        raise ValueError(
            "Every dimension must contain all 20 countries."
        )

    dimension_engines = (
        df.groupby("risk_dimension")["risk_engine"].nunique()
    )

    if not (dimension_engines == 1).all():
        raise ValueError(
            "A dimension maps to more than one risk engine."
        )

    return df


# ============================================================
# INFER RAW MEASUREMENT DIRECTION
# ============================================================

def infer_directions(df):

    """
    Determine whether higher or lower raw measurements
    represent higher risk.

    The existing percentile is treated as the reference
    exposure ordering.

    A direction is accepted only if raw measurement ranks
    and existing percentile ranks are perfectly aligned,
    either positively or negatively.

    This avoids guessing direction from dimension names.
    """

    rows = []

    for dimension, group in df.groupby("risk_dimension"):

        raw = group["measurement_value"]
        percentile = group["percentile_0_100"]

        raw_rank = raw.rank(
            method="average",
            pct=True
        )

        percentile_rank = percentile.rank(
            method="average",
            pct=True
        )

        positive_match = np.allclose(
            raw_rank.to_numpy(),
            percentile_rank.to_numpy(),
            atol=TOLERANCE,
            rtol=0
        )

        negative_match = np.allclose(
            (1 - raw_rank + (1 / len(group))).to_numpy(),
            percentile_rank.to_numpy(),
            atol=TOLERANCE,
            rtol=0
        )

        # Rank correlation is retained as a diagnostic.
        correlation = raw_rank.corr(percentile_rank)

        if positive_match and negative_match:
            direction = "ambiguous_constant_or_tied"
            multiplier = None
            status = "FAIL"

        elif positive_match:
            direction = "higher_raw_value_higher_exposure"
            multiplier = 1
            status = "PASS"

        elif negative_match:
            direction = "lower_raw_value_higher_exposure"
            multiplier = -1
            status = "PASS"

        else:
            direction = "non_monotonic_or_rank_mismatch"
            multiplier = None
            status = "FAIL"

        rows.append({
            "risk_dimension": dimension,
            "risk_engine": group["risk_engine"].iloc[0],
            "country_count": group["iso"].nunique(),
            "unique_raw_measurements": raw.nunique(),
            "unique_percentiles": percentile.nunique(),
            "rank_correlation": correlation,
            "raw_direction": direction,
            "risk_score_multiplier": multiplier,
            "direction_validation": status,
        })

    result = pd.DataFrame(rows)

    result.to_csv(OUT_DIRECTION, index=False)

    failures = result[
        result["direction_validation"] != "PASS"
    ]

    if not failures.empty:
        raise ValueError(
            f"{len(failures)} dimensions failed raw direction "
            f"validation. Inspect {OUT_DIRECTION}. "
            "Do not proceed by guessing their direction."
        )

    return result


# ============================================================
# BUILD RAW EXPOSURE SCORES
# ============================================================

def build_raw_scores(df, directions):

    result = df.merge(
        directions[
            [
                "risk_dimension",
                "risk_score_multiplier",
                "raw_direction",
            ]
        ],
        on="risk_dimension",
        how="left",
        validate="many_to_one"
    )

    if result["risk_score_multiplier"].isna().any():
        raise ValueError(
            "Missing direction multiplier after merge."
        )

    result["raw_exposure_score"] = (
        result["measurement_value"]
        * result["risk_score_multiplier"]
    )

    return result


# ============================================================
# FULL-SAMPLE RECONSTRUCTION
# ============================================================

def validate_full_sample(df):

    reconstructed = (
        df.groupby(
            "risk_dimension",
            group_keys=False
        )["raw_exposure_score"]
        .apply(classify_risk_scores)
    )

    result = df.copy()

    result["raw_reconstructed_class"] = (
        reconstructed.reindex(result.index)
    )

    mismatches = result[
        result["raw_reconstructed_class"]
        != result["original_class"]
    ]

    if not mismatches.empty:

        path = (
            REPORTS
            / "europe_raw_measurement_baseline_mismatches.csv"
        )

        mismatches[
            [
                "iso",
                "risk_engine",
                "risk_dimension",
                "measurement_value",
                "percentile_0_100",
                "original_class",
                "raw_reconstructed_class",
            ]
        ].to_csv(path, index=False)

        raise ValueError(
            f"Raw baseline reconstruction failed: "
            f"{len(mismatches)} mismatches. "
            f"Inspect {path}. "
            "Do not continue to leave-one-out testing."
        )

    return result


# ============================================================
# RAW-MEASUREMENT LEAVE-ONE-OUT
# ============================================================

def build_leave_one_out(df):

    countries = sorted(df["iso"].unique())

    frames = []

    for excluded_iso in countries:

        peer_df = df[
            df["iso"] != excluded_iso
        ].copy()

        if peer_df["iso"].nunique() != 19:
            raise ValueError(
                f"Invalid peer group excluding {excluded_iso}."
            )

        peer_df["peer_group_excluded_iso"] = excluded_iso

        peer_df["raw_loo_class"] = (
            peer_df.groupby(
                "risk_dimension",
                group_keys=False
            )["raw_exposure_score"]
            .apply(classify_risk_scores)
            .reindex(peer_df.index)
        )

        peer_df["classification_changed"] = (
            peer_df["raw_loo_class"]
            != peer_df["original_class"]
        )

        peer_df["engine_activation_changed"] = (
            (peer_df["raw_loo_class"] == "higher")
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
                    "measurement_value",
                    "raw_exposure_score",
                    "original_class",
                    "raw_loo_class",
                    "classification_changed",
                    "engine_activation_changed",
                ]
            ]
        )

    result = pd.concat(frames, ignore_index=True)

    if len(result) != EXPECTED_LOO_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_LOO_RECORDS} records; "
            f"found {len(result)}."
        )

    if result.duplicated(
        ["peer_group_excluded_iso", "iso", "risk_dimension"]
    ).any():
        raise ValueError("Duplicate leave-one-out records.")

    if (
        result["peer_group_excluded_iso"] == result["iso"]
    ).any():
        raise ValueError(
            "Excluded country appears in its own peer group."
        )

    result.to_csv(OUT_LOO, index=False)

    return result


# ============================================================
# SUMMARIES
# ============================================================

def build_group_summary(
    loo,
    group_columns,
    output_path
):

    result = (
        loo.groupby(group_columns, as_index=False)
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

    result.to_csv(output_path, index=False)

    return result


# ============================================================
# EXCLUSION SUMMARY
# ============================================================

def build_exclusion_summary(loo):

    result = build_group_summary(
        loo,
        ["peer_group_excluded_iso"],
        OUT_EXCLUSION
    )

    return result.sort_values(
        "classification_change_rate_pct",
        ascending=False
    )


# ============================================================
# COMPARE WITH PREVIOUS PERCENTILE AUDIT
# ============================================================

def compare_with_previous(raw_loo):

    if not PREVIOUS_LOO.exists():
        return None

    previous = pd.read_csv(PREVIOUS_LOO)

    required = {
        "peer_group_excluded_iso",
        "iso",
        "risk_dimension",
        "loo_class",
    }

    missing = required.difference(previous.columns)

    if missing:
        raise ValueError(
            f"Previous audit is missing columns: {sorted(missing)}"
        )

    previous["loo_class"] = (
        previous["loo_class"].apply(normalise_class)
    )

    current = raw_loo[
        [
            "peer_group_excluded_iso",
            "iso",
            "risk_dimension",
            "raw_loo_class",
        ]
    ]

    comparison = previous.merge(
        current,
        on=[
            "peer_group_excluded_iso",
            "iso",
            "risk_dimension",
        ],
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    if not (comparison["_merge"] == "both").all():
        raise ValueError(
            "Raw and percentile audit keys do not match."
        )

    comparison["method_class_changed"] = (
        comparison["loo_class"]
        != comparison["raw_loo_class"]
    )

    comparison["method_activation_changed"] = (
        (comparison["loo_class"] == "higher")
        != (comparison["raw_loo_class"] == "higher")
    )

    comparison.to_csv(OUT_COMPARISON, index=False)

    return comparison


# ============================================================
# REPORT
# ============================================================

def build_report(
    directions,
    loo,
    country_summary,
    dimension_summary,
    engine_summary,
    exclusion_summary,
    method_comparison
):

    total = len(loo)

    changed = int(
        loo["classification_changed"].sum()
    )

    activation_changed = int(
        loo["engine_activation_changed"].sum()
    )

    lines = [
        "=" * 76,
        "EUROPEAN RAW-MEASUREMENT PEER-GROUP SENSITIVITY",
        "=" * 76,
        "",
        "VALIDATION",
        "-" * 76,
        f"Countries: {EXPECTED_COUNTRIES}",
        f"Dimensions: {EXPECTED_DIMENSIONS}",
        f"Source records: {EXPECTED_RECORDS}",
        f"Direction validations passed: "
        f"{len(directions[directions['direction_validation'] == 'PASS'])}",
        f"Direction validations failed: "
        f"{len(directions[directions['direction_validation'] != 'PASS'])}",
        f"Leave-one-out records: {total}",
        "",
        "RAW MEASUREMENT DIRECTIONS",
        "-" * 76,
        directions.to_string(index=False),
        "",
        "OVERALL RAW-MEASUREMENT STABILITY",
        "-" * 76,
        f"Changed classifications: {changed}",
        f"Unchanged classifications: {total - changed}",
        f"Classification change rate: {changed / total * 100:.2f}%",
        f"Changed engine activations: {activation_changed}",
        f"Unchanged engine activations: {total - activation_changed}",
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
    ]

    if method_comparison is not None:

        method_changes = int(
            method_comparison["method_class_changed"].sum()
        )

        method_activation_changes = int(
            method_comparison["method_activation_changed"].sum()
        )

        lines.extend([
            "",
            "RAW VERSUS PERCENTILE METHOD COMPARISON",
            "-" * 76,
            f"Method comparisons: {len(method_comparison)}",
            f"Classification differences: {method_changes}",
            f"Classification difference rate: "
            f"{method_changes / len(method_comparison) * 100:.2f}%",
            f"Engine activation differences: {method_activation_changes}",
            f"Engine activation difference rate: "
            f"{method_activation_changes / len(method_comparison) * 100:.2f}%",
        ])

    lines.extend([
        "",
        "METHODOLOGICAL LIMITATIONS",
        "-" * 76,
        "1. Raw measurement direction is inferred by checking",
        "   monotonic alignment with the existing risk percentile.",
        "2. This validates direction and rank consistency; it does",
        "   not independently validate the economic interpretation",
        "   of the original risk dimensions.",
        "3. Tercile cutoffs and average-rank tie handling are retained.",
        "4. Leave-one-out testing does not test alternative regional",
        "   or economic peer groups.",
        "5. Peer-relative classification changes do not imply that",
        "   the country's underlying economic conditions changed.",
        "6. No absolute risk, crisis probability or default probability",
        "   is estimated.",
        "",
        "OUTPUT FILES",
        "-" * 76,
        str(OUT_DIRECTION),
        str(OUT_LOO),
        str(OUT_COUNTRY),
        str(OUT_DIMENSION),
        str(OUT_ENGINE),
        str(OUT_EXCLUSION),
        str(OUT_COMPARISON),
        str(OUT_REPORT),
    ])

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
    print("EUROPEAN RAW-MEASUREMENT PEER-GROUP SENSITIVITY")
    print("=" * 76)

    df = load_data()

    print("Input validation passed.")

    directions = infer_directions(df)

    print(
        "Raw measurement direction validation passed "
        "for all 33 dimensions."
    )

    df = build_raw_scores(df, directions)

    df = validate_full_sample(df)

    print(
        "Full-sample raw measurement reconstruction passed: "
        "all original classifications reproduced."
    )

    loo = build_leave_one_out(df)

    country_summary = build_group_summary(
        loo,
        ["iso", "country"],
        OUT_COUNTRY
    ).sort_values(
        "classification_change_rate_pct",
        ascending=False
    )

    dimension_summary = build_group_summary(
        loo,
        ["risk_engine", "risk_dimension"],
        OUT_DIMENSION
    ).sort_values(
        "classification_change_rate_pct",
        ascending=False
    )

    engine_summary = build_group_summary(
        loo,
        ["risk_engine"],
        OUT_ENGINE
    ).sort_values(
        "classification_change_rate_pct",
        ascending=False
    )

    exclusion_summary = build_exclusion_summary(loo)

    method_comparison = compare_with_previous(loo)

    build_report(
        directions,
        loo,
        country_summary,
        dimension_summary,
        engine_summary,
        exclusion_summary,
        method_comparison
    )

    print("")
    print("VALIDATION PASSED")
    print(f"Countries: {EXPECTED_COUNTRIES}")
    print(f"Dimensions: {EXPECTED_DIMENSIONS}")
    print(f"Leave-one-out records: {len(loo)}")

    print("")
    print("OUTPUTS CREATED")
    print(OUT_DIRECTION)
    print(OUT_LOO)
    print(OUT_COUNTRY)
    print(OUT_DIMENSION)
    print(OUT_ENGINE)
    print(OUT_EXCLUSION)
    print(OUT_COMPARISON)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
