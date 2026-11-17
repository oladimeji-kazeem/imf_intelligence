from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "europe_integrated_risk_matrix.csv"
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUTPUT_COUNTRY_ENGINE = (
    PROCESSED / "europe_integrated_cross_engine_vulnerability.csv"
)

OUTPUT_CONCENTRATION = (
    PROCESSED / "europe_integrated_engine_concentration.csv"
)

OUTPUT_ENGINE_PAIRS = (
    PROCESSED / "europe_integrated_engine_pair_cooccurrence.csv"
)

OUTPUT_DIMENSION_PAIRS = (
    PROCESSED / "europe_integrated_dimension_spearman_correlations.csv"
)

OUTPUT_CORRELATION_MATRIX = (
    PROCESSED / "europe_integrated_dimension_correlation_matrix.csv"
)

OUTPUT_REPORT = (
    REPORTS / "europe_integrated_cross_engine_analysis_report.txt"
)


EXPECTED_COUNTRIES = 20
EXPECTED_DIMENSIONS = 33

EXPECTED_ENGINES = {
    "fiscal",
    "growth",
    "inflation",
    "external",
    "labour",
}

HIGHER_LABEL = "higher"


# ============================================================
# LOAD AND VALIDATE
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

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
    df["standardised_classification"] = (
        df["standardised_classification"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["percentile_0_100"] = pd.to_numeric(
        df["percentile_0_100"],
        errors="coerce"
    )

    if df["percentile_0_100"].isna().any():
        raise ValueError(
            "Missing or non-numeric percentile values detected."
        )

    if not df["percentile_0_100"].between(0, 100).all():
        raise ValueError(
            "Percentile values must be between 0 and 100."
        )

    if df.duplicated(
        ["iso", "risk_engine", "risk_dimension"]
    ).any():
        raise ValueError(
            "Duplicate country-engine-dimension records detected."
        )

    country_counts = df.groupby("iso").size()

    if len(country_counts) != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTRIES} countries; "
            f"found {len(country_counts)}."
        )

    if not (country_counts == EXPECTED_DIMENSIONS).all():
        raise ValueError(
            "Not all countries have exactly 33 dimensions."
        )

    actual_engines = set(df["risk_engine"].unique())

    if actual_engines != EXPECTED_ENGINES:
        raise ValueError(
            f"Engine mismatch. Found: {actual_engines}"
        )

    expected_dimensions = (
        df.groupby("risk_dimension")["risk_engine"].nunique()
    )

    if len(expected_dimensions) != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected 33 unique dimensions; "
            f"found {len(expected_dimensions)}."
        )

    df["higher_flag"] = (
        df["standardised_classification"] == HIGHER_LABEL
    ).astype(int)

    return df


# ============================================================
# COUNTRY-ENGINE VULNERABILITY
# ============================================================

def build_country_engine_analysis(df):

    result = (
        df.groupby(
            ["iso", "country", "risk_engine"],
            as_index=False
        )
        .agg(
            higher_exposure_count=("higher_flag", "sum"),
            dimension_count=("risk_dimension", "nunique"),
            percentile_mean=("percentile_0_100", "mean"),
            percentile_median=("percentile_0_100", "median"),
        )
    )

    result["higher_exposure_share_pct"] = (
        result["higher_exposure_count"]
        / result["dimension_count"]
        * 100
    )

    result["engine_has_higher_exposure"] = (
        result["higher_exposure_count"] > 0
    ).astype(int)

    result.to_csv(
        OUTPUT_COUNTRY_ENGINE,
        index=False
    )

    return result


# ============================================================
# ENGINE CONCENTRATION BY COUNTRY
# ============================================================

def build_concentration_analysis(engine_data):

    rows = []

    for (iso, country), group in engine_data.groupby(
        ["iso", "country"]
    ):

        total_higher = int(
            group["higher_exposure_count"].sum()
        )

        engines_with_exposure = int(
            group["engine_has_higher_exposure"].sum()
        )

        if total_higher > 0:

            engine_shares = (
                group["higher_exposure_count"] / total_higher
            )

            concentration_hhi = float(
                (engine_shares ** 2).sum()
            )

            dominant_count = group[
                "higher_exposure_count"
            ].max()

            dominant_engines = sorted(
                group.loc[
                    group["higher_exposure_count"]
                    == dominant_count,
                    "risk_engine"
                ].tolist()
            )

            dominant_engine = ";".join(dominant_engines)

        else:

            concentration_hhi = np.nan
            dominant_engine = "none"

        row = {
            "iso": iso,
            "country": country,
            "total_higher_exposure_dimensions": total_higher,
            "engines_with_higher_exposure": engines_with_exposure,
            "engines_without_higher_exposure": (
                len(EXPECTED_ENGINES) - engines_with_exposure
            ),
            "higher_exposure_engine_concentration_hhi": (
                concentration_hhi
            ),
            "dominant_engine_or_tie": dominant_engine,
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "engines_with_higher_exposure",
            "total_higher_exposure_dimensions",
        ],
        ascending=[False, False]
    )

    result.to_csv(
        OUTPUT_CONCENTRATION,
        index=False
    )

    return result


# ============================================================
# ENGINE PAIR CO-OCCURRENCE
# ============================================================

def build_engine_pair_analysis(engine_data):

    presence = engine_data.pivot(
        index="iso",
        columns="risk_engine",
        values="engine_has_higher_exposure"
    ).fillna(0)

    country_names = (
        engine_data[["iso", "country"]]
        .drop_duplicates()
        .set_index("iso")["country"]
    )

    rows = []

    for engine_a, engine_b in combinations(
        sorted(EXPECTED_ENGINES),
        2
    ):

        both = (
            (presence[engine_a] == 1)
            & (presence[engine_b] == 1)
        )

        either = (
            (presence[engine_a] == 1)
            | (presence[engine_b] == 1)
        )

        both_count = int(both.sum())

        union_count = int(either.sum())

        jaccard = (
            both_count / union_count
            if union_count > 0
            else np.nan
        )

        rows.append({
            "engine_a": engine_a,
            "engine_b": engine_b,
            "countries_with_both": both_count,
            "countries_with_either": union_count,
            "jaccard_cooccurrence": jaccard,
            "countries_with_both_iso": ";".join(
                presence.index[both].tolist()
            ),
            "countries_with_both_names": ";".join(
                country_names.loc[
                    presence.index[both]
                ].tolist()
            ),
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "countries_with_both",
        ascending=False
    )

    result.to_csv(
        OUTPUT_ENGINE_PAIRS,
        index=False
    )

    return result


# ============================================================
# DIMENSION DEPENDENCE
# ============================================================

def build_dimension_correlations(df):

    wide = df.pivot(
        index="iso",
        columns="risk_dimension",
        values="percentile_0_100"
    )

    if wide.shape[1] != EXPECTED_DIMENSIONS:
        raise ValueError(
            "Unexpected number of dimensions in correlation panel."
        )

    correlation_matrix = wide.corr(
        method="spearman"
    )

    correlation_matrix.to_csv(
        OUTPUT_CORRELATION_MATRIX
    )

    rows = []

    for dimension_a, dimension_b in combinations(
        sorted(wide.columns),
        2
    ):

        pair = wide[
            [dimension_a, dimension_b]
        ].dropna()

        if len(pair) < 10:
            continue

        rho, p_value = spearmanr(
            pair[dimension_a],
            pair[dimension_b]
        )

        rows.append({
            "dimension_a": dimension_a,
            "dimension_b": dimension_b,
            "spearman_rho": rho,
            "p_value_exploratory": p_value,
            "countries_observed": len(pair),
            "absolute_correlation": abs(rho),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "absolute_correlation",
            ascending=False
        )

    result.to_csv(
        OUTPUT_DIMENSION_PAIRS,
        index=False
    )

    return result, correlation_matrix


# ============================================================
# REPORT
# ============================================================

def build_report(
    df,
    concentration,
    engine_pairs,
    dimension_pairs
):

    lines = []

    lines.append("=" * 76)
    lines.append(
        "EUROPEAN INTEGRATED CROSS-ENGINE ANALYSIS"
    )
    lines.append("=" * 76)

    lines.append("")
    lines.append("DATA VALIDATION")
    lines.append("-" * 76)

    lines.append(f"Countries: {df['iso'].nunique()}")
    lines.append(f"Dimensions: {df['risk_dimension'].nunique()}")
    lines.append(f"Records: {len(df)}")
    lines.append(f"Engines: {len(EXPECTED_ENGINES)}")

    lines.append("")
    lines.append("COUNTRY CROSS-ENGINE CONCENTRATION")
    lines.append("-" * 76)

    lines.append(
        concentration[
            [
                "iso",
                "total_higher_exposure_dimensions",
                "engines_with_higher_exposure",
                "higher_exposure_engine_concentration_hhi",
                "dominant_engine_or_tie",
            ]
        ].to_string(index=False)
    )

    lines.append("")
    lines.append("ENGINE PAIR CO-OCCURRENCE")
    lines.append("-" * 76)

    lines.append(
        engine_pairs[
            [
                "engine_a",
                "engine_b",
                "countries_with_both",
                "countries_with_either",
                "jaccard_cooccurrence",
            ]
        ].to_string(index=False)
    )

    lines.append("")
    lines.append("STRONGEST DIMENSION CORRELATIONS")
    lines.append("-" * 76)

    if not dimension_pairs.empty:

        lines.append(
            dimension_pairs.head(20)[
                [
                    "dimension_a",
                    "dimension_b",
                    "spearman_rho",
                    "countries_observed",
                ]
            ].to_string(index=False)
        )

    lines.append("")
    lines.append("INTERPRETATION AND LIMITATIONS")
    lines.append("-" * 76)

    lines.append(
        "1. Higher-exposure counts measure peer-relative signal breadth."
    )

    lines.append(
        "2. HHI measures the concentration of higher signals across "
        "the five engines; it is not a measure of economic risk."
    )

    lines.append(
        "3. Engine co-occurrence indicates that two engines have at "
        "least one higher classification in the same country."
    )

    lines.append(
        "4. Jaccard co-occurrence measures overlap relative to the "
        "countries with exposure in either engine."
    )

    lines.append(
        "5. Spearman correlations use country-level percentile "
        "measurements across 20 countries."
    )

    lines.append(
        "6. Correlations are exploratory. Twenty countries provide "
        "a small sample for dependence analysis."
    )

    lines.append(
        "7. Correlation does not establish causal relationships or "
        "prove that two dimensions are redundant."
    )

    lines.append(
        "8. Dimensions may share source data, measurement periods "
        "and underlying economic drivers."
    )

    lines.append(
        "9. No composite risk score, default probability or "
        "country credit ranking is produced."
    )

    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append("-" * 76)

    for path in [
        OUTPUT_COUNTRY_ENGINE,
        OUTPUT_CONCENTRATION,
        OUTPUT_ENGINE_PAIRS,
        OUTPUT_DIMENSION_PAIRS,
        OUTPUT_CORRELATION_MATRIX,
        OUTPUT_REPORT,
    ]:
        lines.append(str(path))

    report = "\n".join(lines)

    OUTPUT_REPORT.write_text(
        report,
        encoding="utf-8"
    )

    print(report)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("EUROPEAN INTEGRATED CROSS-ENGINE ANALYSIS")
    print("=" * 76)

    print("Loading canonical integrated risk matrix...")

    df = load_data()

    print("Validation passed.")
    print("Building country-engine vulnerability analysis...")

    engine_data = build_country_engine_analysis(df)

    print("Building engine concentration analysis...")

    concentration = build_concentration_analysis(
        engine_data
    )

    print("Building engine pair co-occurrence analysis...")

    engine_pairs = build_engine_pair_analysis(
        engine_data
    )

    print("Building dimension dependence analysis...")

    dimension_pairs, correlation_matrix = (
        build_dimension_correlations(df)
    )

    print("Building analytical report...")

    build_report(
        df,
        concentration,
        engine_pairs,
        dimension_pairs
    )

    print("")
    print("VALIDATION PASSED")
    print(f"Countries analysed: {len(concentration)}")
    print(f"Engine pairs analysed: {len(engine_pairs)}")
    print(f"Dimension pairs analysed: {len(dimension_pairs)}")

    print("")
    print("OUTPUTS CREATED")
    print(OUTPUT_COUNTRY_ENGINE)
    print(OUTPUT_CONCENTRATION)
    print(OUTPUT_ENGINE_PAIRS)
    print(OUTPUT_DIMENSION_PAIRS)
    print(OUTPUT_CORRELATION_MATRIX)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()