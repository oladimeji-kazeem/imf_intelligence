from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed"
    / "europe_integrated_risk_matrix.csv"
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUT_ENGINE = (
    PROCESSED / "europe_corrected_engine_concentration.csv"
)

OUT_DIRECTION = (
    PROCESSED / "europe_dimension_directionality_audit.csv"
)

OUT_CORRELATION = (
    PROCESSED / "europe_dimension_dependence_audit.csv"
)

OUT_REPORT = (
    REPORTS / "europe_risk_dependence_audit_report.txt"
)


EXPECTED_ENGINES = {
    "fiscal", "growth", "inflation", "external", "labour"
}


def load_data():

    df = pd.read_csv(INPUT)

    required = [
        "iso",
        "country",
        "risk_engine",
        "risk_dimension",
        "percentile_0_100",
        "standardised_classification",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["risk_engine"] = (
        df["risk_engine"].astype(str).str.lower().str.strip()
    )

    df["standardised_classification"] = (
        df["standardised_classification"]
        .astype(str).str.lower().str.strip()
    )

    df["percentile_0_100"] = pd.to_numeric(
        df["percentile_0_100"],
        errors="coerce"
    )

    if df["percentile_0_100"].isna().any():
        raise ValueError("Missing percentile values.")

    if not df["percentile_0_100"].between(0, 100).all():
        raise ValueError("Percentiles outside 0–100.")

    if df.duplicated(
        ["iso", "risk_engine", "risk_dimension"]
    ).any():
        raise ValueError("Duplicate dimension records.")

    if df["iso"].nunique() != 20:
        raise ValueError("Expected 20 countries.")

    if df["risk_dimension"].nunique() != 33:
        raise ValueError("Expected 33 dimensions.")

    if set(df["risk_engine"].unique()) != EXPECTED_ENGINES:
        raise ValueError("Unexpected risk engines.")

    if not set(df["standardised_classification"]).issubset(
        {"higher", "moderate", "lower"}
    ):
        raise ValueError("Unexpected classification labels.")

    return df


def corrected_engine_concentration(df):

    engine = (
        df.groupby(
            ["iso", "country", "risk_engine"],
            as_index=False
        )
        .agg(
            higher_count=(
                "standardised_classification",
                lambda x: (x == "higher").sum()
            ),
            dimension_count=("risk_dimension", "nunique")
        )
    )

    engine["higher_rate_pct"] = (
        engine["higher_count"]
        / engine["dimension_count"]
        * 100
    )

    engine["has_higher_exposure"] = (
        engine["higher_count"] > 0
    ).astype(int)

    rows = []

    for (iso, country), group in engine.groupby(
        ["iso", "country"]
    ):

        rates = group["higher_rate_pct"].to_numpy()

        rate_total = rates.sum()

        if rate_total > 0:
            shares = rates / rate_total
            corrected_hhi = float(np.sum(shares ** 2))
            dominant_rate = rates.max()
            dominant_engines = sorted(
                group.loc[
                    group["higher_rate_pct"] == dominant_rate,
                    "risk_engine"
                ].tolist()
            )
        else:
            corrected_hhi = np.nan
            dominant_engines = []

        rows.append({
            "iso": iso,
            "country": country,
            "engines_with_higher_exposure": int(
                group["has_higher_exposure"].sum()
            ),
            "total_higher_dimensions": int(
                group["higher_count"].sum()
            ),
            "mean_engine_higher_rate_pct": float(
                group["higher_rate_pct"].mean()
            ),
            "maximum_engine_higher_rate_pct": float(
                group["higher_rate_pct"].max()
            ),
            "corrected_engine_concentration_hhi": (
                corrected_hhi
            ),
            "dominant_engine_or_tie": (
                ";".join(dominant_engines)
                if dominant_engines else "none"
            ),
        })

    result = pd.DataFrame(rows).sort_values(
        ["engines_with_higher_exposure",
         "total_higher_dimensions"],
        ascending=[False, False]
    )

    result.to_csv(OUT_ENGINE, index=False)

    return engine, result


def directionality_audit(df):

    rows = []

    for dimension, group in df.groupby("risk_dimension"):

        higher = group.loc[
            group["standardised_classification"] == "higher",
            "percentile_0_100"
        ]

        other = group.loc[
            group["standardised_classification"] != "higher",
            "percentile_0_100"
        ]

        higher_median = (
            float(higher.median()) if len(higher) else np.nan
        )

        other_median = (
            float(other.median()) if len(other) else np.nan
        )

        difference = higher_median - other_median

        rows.append({
            "risk_dimension": dimension,
            "countries": group["iso"].nunique(),
            "higher_class_count": len(higher),
            "other_class_count": len(other),
            "higher_percentile_median": higher_median,
            "other_percentile_median": other_median,
            "median_difference": difference,
            "directionality_pass": (
                difference > 0
                if pd.notna(difference) else False
            ),
        })

    result = pd.DataFrame(rows).sort_values(
        "median_difference"
    )

    result.to_csv(OUT_DIRECTION, index=False)

    return result


def dependence_audit(df):

    wide = df.pivot(
        index="iso",
        columns="risk_dimension",
        values="percentile_0_100"
    )

    rows = []

    for a, b in combinations(wide.columns, 2):

        pair = wide[[a, b]].dropna()

        if len(pair) < 10:
            continue

        rho, p_value = spearmanr(
            pair[a], pair[b]
        )

        rows.append({
            "dimension_a": a,
            "dimension_b": b,
            "spearman_rho": float(rho),
            "absolute_rho": abs(float(rho)),
            "exploratory_p_value": float(p_value),
            "countries": len(pair),
            "high_dependence_flag": abs(rho) >= 0.80,
        })

    result = pd.DataFrame(rows).sort_values(
        "absolute_rho",
        ascending=False
    )

    result.to_csv(OUT_CORRELATION, index=False)

    return result


def main():

    print("=" * 76)
    print("EUROPEAN RISK DEPENDENCE AUDIT")
    print("=" * 76)

    df = load_data()

    print("Input validation passed.")

    engine, concentration = corrected_engine_concentration(df)

    direction = directionality_audit(df)

    dependence = dependence_audit(df)

    lines = [
        "=" * 76,
        "EUROPEAN RISK DEPENDENCE AUDIT REPORT",
        "=" * 76,
        "",
        "VALIDATION",
        "-" * 76,
        f"Countries: {df['iso'].nunique()}",
        f"Dimensions: {df['risk_dimension'].nunique()}",
        f"Records: {len(df)}",
        "",
        "CORRECTED ENGINE CONCENTRATION",
        "-" * 76,
        concentration.to_string(index=False),
        "",
        "DIRECTIONALITY AUDIT",
        "-" * 76,
        direction.to_string(index=False),
        "",
        "DIRECTIONALITY FAILURES",
        "-" * 76,
        direction.loc[
            ~direction["directionality_pass"]
        ].to_string(index=False),
        "",
        "STRONGEST DIMENSION DEPENDENCIES",
        "-" * 76,
        dependence.head(25).to_string(index=False),
        "",
        "HIGH DEPENDENCE PAIRS (ABSOLUTE RHO >= 0.80)",
        "-" * 76,
        dependence.loc[
            dependence["high_dependence_flag"]
        ].to_string(index=False),
        "",
        "INTERPRETATION LIMITATIONS",
        "-" * 76,
        "1. Corrected HHI uses engine-level higher-exposure rates.",
        "2. HHI measures concentration, not severity or probability.",
        "3. Directionality checks compare higher-class percentiles",
        "   with all other classifications.",
        "4. Correlations are exploratory with only 20 countries.",
        "5. High correlation does not prove redundancy.",
        "6. No dimensions are removed or reweighted by this audit.",
        "7. No composite score or default probability is produced.",
        "",
        "OUTPUT FILES",
        "-" * 76,
        str(OUT_ENGINE),
        str(OUT_DIRECTION),
        str(OUT_CORRELATION),
        str(OUT_REPORT),
    ]

    report = "\n".join(lines)

    OUT_REPORT.write_text(
        report,
        encoding="utf-8"
    )

    print(report)

    print("\nVALIDATION PASSED")
    print(f"Dimension pairs tested: {len(dependence)}")
    print(
        "Directionality failures:",
        int((~direction["directionality_pass"]).sum())
    )


if __name__ == "__main__":
    main()