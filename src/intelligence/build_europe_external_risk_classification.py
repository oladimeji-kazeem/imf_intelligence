
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/europe_external_risk_diagnostics.csv"

OUTPUT = (
    ROOT / "data/processed/"
    "europe_external_risk_classifications.csv"
)

DETAIL_OUTPUT = (
    ROOT / "data/processed/"
    "europe_external_risk_dimension_detail.csv"
)

ROBUSTNESS_OUTPUT = (
    ROOT / "outputs/reports/"
    "europe_external_risk_robustness.csv"
)

REPORT = (
    ROOT / "outputs/reports/"
    "europe_external_risk_classification_report.txt"
)

EXPECTED_COUNTRIES = 20

PERIODS = {
    "modern": "modern_2000_2019",
    "recent": "recent_2020_2024",
}

INDICATORS = {
    "current_account": "BCA_NGDPD",
    "exports": "TX_RPCH",
    "imports": "TM_RPCH",
}


# ============================================================
# CLASSIFICATION DIMENSIONS
# ============================================================

DIMENSIONS = [
    {
        "dimension": "recent_external_imbalance",
        "indicator": "BCA_NGDPD",
        "period": "recent_2020_2024",
        "metric": "median",
        "direction": "lower_is_higher_risk",
        "minimum_observations": 4,
        "minimum_coverage": 0.80,
        "description": (
            "Recent median current-account balance, "
            "percentage of GDP"
        ),
    },
    {
        "dimension": "historical_deficit_persistence",
        "indicator": "BCA_NGDPD",
        "period": "modern_2000_2019",
        "metric": "negative_balance_frequency",
        "direction": "higher_is_higher_risk",
        "minimum_observations": 16,
        "minimum_coverage": 0.80,
        "description": (
            "Frequency of annual current-account deficits"
        ),
    },
    {
        "dimension": "recent_external_imbalance_volatility",
        "indicator": "BCA_NGDPD",
        "period": "recent_2020_2024",
        "metric": "mad",
        "direction": "higher_is_higher_risk",
        "minimum_observations": 4,
        "minimum_coverage": 0.80,
        "description": (
            "Median absolute deviation of recent "
            "current-account balances"
        ),
    },
    {
        "dimension": "historical_export_contraction",
        "indicator": "TX_RPCH",
        "period": "modern_2000_2019",
        "metric": "negative_growth_frequency",
        "direction": "higher_is_higher_risk",
        "minimum_observations": 16,
        "minimum_coverage": 0.80,
        "description": (
            "Frequency of negative export-volume growth"
        ),
    },
    {
        "dimension": "recent_export_contraction",
        "indicator": "TX_RPCH",
        "period": "recent_2020_2024",
        "metric": "negative_growth_frequency",
        "direction": "higher_is_higher_risk",
        "minimum_observations": 4,
        "minimum_coverage": 0.80,
        "description": (
            "Recent frequency of negative export-volume growth"
        ),
    },
    {
        "dimension": "historical_export_growth_instability",
        "indicator": "TX_RPCH",
        "period": "modern_2000_2019",
        "metric": "mad",
        "direction": "higher_is_higher_risk",
        "minimum_observations": 16,
        "minimum_coverage": 0.80,
        "description": (
            "Historical dispersion in export-volume growth"
        ),
    },
]


# ============================================================
# CLASSIFICATION FUNCTIONS
# ============================================================

def classify_peer_exposure(values, direction):
    """
    Classifies eligible countries by peer-relative exposure.

    Lower exposure: bottom third.
    Moderate exposure: middle third.
    Higher exposure: top third.

    Direction determines whether larger or smaller
    underlying values imply greater exposure.
    """

    numeric = pd.to_numeric(values, errors="coerce")

    if direction == "lower_is_higher_risk":
        exposure_values = -numeric
    else:
        exposure_values = numeric

    ranks = exposure_values.rank(
        method="average",
        pct=True
    )

    result = pd.Series(
        "insufficient_data",
        index=values.index,
        dtype="object"
    )

    valid = exposure_values.notna()

    result.loc[valid & (ranks <= 1 / 3)] = "lower_risk"

    result.loc[
        valid & (ranks > 1 / 3) & (ranks <= 2 / 3)
    ] = "moderate_risk"

    result.loc[valid & (ranks > 2 / 3)] = "higher_risk"

    return result, ranks


def safe_spearman(x, y):
    paired = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()

    if len(paired) < 3:
        return np.nan

    if paired["x"].nunique() < 2 or paired["y"].nunique() < 2:
        return np.nan

    return paired["x"].corr(
        paired["y"],
        method="spearman"
    )


# ============================================================
# LOAD AND VALIDATE INPUT
# ============================================================

def load_input():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input diagnostics file not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "indicator_code",
        "period",
        "numeric_observations",
        "numeric_year_coverage",
        "median",
        "mean",
        "mad",
        "iqr",
        "negative_balance_frequency",
        "negative_growth_frequency",
        "largest_consecutive_deficit_run_years",
        "largest_consecutive_negative_growth_run_years",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.duplicated(
        ["iso", "indicator_code", "period"]
    ).any():
        raise ValueError(
            "Duplicate country-indicator-period records found."
        )

    countries = df["iso"].nunique()

    if countries != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTRIES} countries; "
            f"found {countries}."
        )

    return df


# ============================================================
# BUILD DIMENSION DETAIL
# ============================================================

def build_dimension_detail(df):

    records = []

    for dimension in DIMENSIONS:

        subset = df[
            (df["indicator_code"] == dimension["indicator"])
            &
            (df["period"] == dimension["period"])
        ].copy()

        subset["metric_value"] = pd.to_numeric(
            subset[dimension["metric"]],
            errors="coerce"
        )

        subset["numeric_observations"] = pd.to_numeric(
            subset["numeric_observations"],
            errors="coerce"
        )

        subset["numeric_year_coverage"] = pd.to_numeric(
            subset["numeric_year_coverage"],
            errors="coerce"
        )

        eligible = (
            subset["metric_value"].notna()
            &
            (
                subset["numeric_observations"]
                >= dimension["minimum_observations"]
            )
            &
            (
                subset["numeric_year_coverage"]
                >= dimension["minimum_coverage"]
            )
        )

        subset["eligible"] = eligible

        subset["dimension"] = dimension["dimension"]
        subset["description"] = dimension["description"]
        subset["metric"] = dimension["metric"]
        subset["period"] = dimension["period"]
        subset["indicator_code"] = dimension["indicator"]
        subset["direction"] = dimension["direction"]

        subset["risk_classification"] = "insufficient_data"
        subset["peer_exposure_percentile"] = np.nan

        eligible_rows = subset.loc[eligible].copy()

        if len(eligible_rows) >= 5:

            classifications, percentiles = classify_peer_exposure(
                eligible_rows["metric_value"],
                dimension["direction"]
            )

            subset.loc[
                eligible_rows.index,
                "risk_classification"
            ] = classifications

            subset.loc[
                eligible_rows.index,
                "peer_exposure_percentile"
            ] = percentiles

        records.append(
            subset[
                [
                    "iso",
                    "country",
                    "dimension",
                    "indicator_code",
                    "period",
                    "metric",
                    "metric_value",
                    "numeric_observations",
                    "numeric_year_coverage",
                    "eligible",
                    "direction",
                    "peer_exposure_percentile",
                    "risk_classification",
                    "description",
                ]
            ]
        )

    detail = pd.concat(records, ignore_index=True)

    return detail


# ============================================================
# BUILD COUNTRY PROFILES
# ============================================================

def build_country_profiles(detail):

    countries = (
        detail[["iso", "country"]]
        .drop_duplicates()
        .sort_values("iso")
    )

    profiles = countries.copy()

    for dimension in DIMENSIONS:

        name = dimension["dimension"]

        subset = detail[
            detail["dimension"] == name
        ][
            [
                "iso",
                "risk_classification",
                "peer_exposure_percentile",
                "metric_value",
            ]
        ].copy()

        subset = subset.rename(
            columns={
                "risk_classification": name,
                "peer_exposure_percentile": (
                    name + "_percentile"
                ),
                "metric_value": name + "_value",
            }
        )

        profiles = profiles.merge(
            subset,
            on="iso",
            how="left",
            validate="one_to_one"
        )

    return profiles


# ============================================================
# ROBUSTNESS TESTS
# ============================================================

def build_robustness_tests(df):

    tests = []

    # Test 1: Recent current-account mean versus median.
    ca = df[
        (df["indicator_code"] == "BCA_NGDPD")
        &
        (df["period"] == "recent_2020_2024")
    ].copy()

    ca["mean"] = pd.to_numeric(ca["mean"], errors="coerce")
    ca["median"] = pd.to_numeric(ca["median"], errors="coerce")

    tests.append({
        "test": "current_account_mean_vs_median",
        "observations": ca[["mean", "median"]].dropna().shape[0],
        "spearman_rank_correlation": safe_spearman(
            -ca["mean"],
            -ca["median"]
        ),
        "interpretation": (
            "Tests sensitivity of recent imbalance ranking "
            "to the choice of mean or median."
        ),
    })

    # Test 2: Recent current-account MAD versus IQR.
    ca["mad"] = pd.to_numeric(ca["mad"], errors="coerce")
    ca["iqr"] = pd.to_numeric(ca["iqr"], errors="coerce")

    tests.append({
        "test": "current_account_mad_vs_iqr",
        "observations": ca[["mad", "iqr"]].dropna().shape[0],
        "spearman_rank_correlation": safe_spearman(
            ca["mad"],
            ca["iqr"]
        ),
        "interpretation": (
            "Tests whether two dispersion measures produce "
            "similar country rankings."
        ),
    })

    # Test 3: Historical export contraction frequency
    # versus longest consecutive contraction run.
    exports = df[
        (df["indicator_code"] == "TX_RPCH")
        &
        (df["period"] == "modern_2000_2019")
    ].copy()

    exports["negative_growth_frequency"] = pd.to_numeric(
        exports["negative_growth_frequency"],
        errors="coerce"
    )

    exports["longest_run"] = pd.to_numeric(
        exports[
            "largest_consecutive_negative_growth_run_years"
        ],
        errors="coerce"
    )

    tests.append({
        "test": "export_contraction_frequency_vs_longest_run",
        "observations": exports[
            ["negative_growth_frequency", "longest_run"]
        ].dropna().shape[0],
        "spearman_rank_correlation": safe_spearman(
            exports["negative_growth_frequency"],
            exports["longest_run"]
        ),
        "interpretation": (
            "Tests whether contraction frequency and "
            "contraction persistence give similar rankings."
        ),
    })

    return pd.DataFrame(tests)


# ============================================================
# REPORT
# ============================================================

def build_report(df, detail, profiles, robustness):

    lines = []

    lines.append("=" * 76)
    lines.append("EUROPEAN EXTERNAL RISK CLASSIFICATION REPORT")
    lines.append("=" * 76)

    lines.append("")
    lines.append("INPUT")
    lines.append("-" * 76)
    lines.append(str(INPUT))
    lines.append(f"Input records: {len(df):,}")
    lines.append(f"Countries: {df['iso'].nunique()}")
    lines.append(f"Indicators: {df['indicator_code'].nunique()}")

    lines.append("")
    lines.append("CLASSIFICATION OUTPUT")
    lines.append("-" * 76)
    lines.append(f"Dimensions: {len(DIMENSIONS)}")
    lines.append(f"Dimension records: {len(detail)}")
    lines.append(f"Country profiles: {len(profiles)}")

    lines.append("")
    lines.append("CLASSIFICATION DISTRIBUTIONS")
    lines.append("-" * 76)

    for dimension in DIMENSIONS:

        name = dimension["dimension"]

        subset = detail[
            detail["dimension"] == name
        ]

        counts = subset[
            "risk_classification"
        ].value_counts()

        lines.append("")
        lines.append(name)

        for label in [
            "higher_risk",
            "moderate_risk",
            "lower_risk",
            "insufficient_data",
        ]:
            lines.append(
                f"  {label}: {counts.get(label, 0)}"
            )

    lines.append("")
    lines.append("ROBUSTNESS TESTS")
    lines.append("-" * 76)

    for _, row in robustness.iterrows():

        lines.append(
            f"{row['test']}: "
            f"n={row['observations']}, "
            f"Spearman={row['spearman_rank_correlation']}"
        )

    lines.append("")
    lines.append("METHODOLOGICAL LIMITATIONS")
    lines.append("-" * 76)

    limitations = [
        "Classifications are peer-relative, not absolute risk ratings.",
        "Peer rankings depend on the selected 20-country comparison group.",
        "The recent period contains only five annual observations.",
        "Historical deficit frequency does not measure financing vulnerability.",
        "Export volume growth does not measure export value or market share.",
        "Import contraction is retained as context, not a direct risk score.",
        "Current-account balances do not reveal financing composition.",
        "No foreign reserves, external debt maturity or capital-flow data are included.",
        "No external default probability or crisis probability is estimated.",
        "The dimensions are not combined into a composite score.",
    ]

    for item in limitations:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("VALIDATION")
    lines.append("-" * 76)
    lines.append("PASS: Required input fields present.")
    lines.append("PASS: 20-country source coverage.")
    lines.append("PASS: Six dimensions generated.")
    lines.append("PASS: No duplicate country-dimension records.")
    lines.append("PASS: Peer classifications use eligible observations only.")
    lines.append("PASS: Insufficient observations remain unclassified.")

    lines.append("")
    lines.append("OUTPUTS")
    lines.append("-" * 76)
    lines.append(str(OUTPUT))
    lines.append(str(DETAIL_OUTPUT))
    lines.append(str(ROBUSTNESS_OUTPUT))

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("EUROPEAN EXTERNAL RISK CLASSIFICATION ENGINE")
    print("=" * 76)

    df = load_input()

    print(f"Input records: {len(df):,}")
    print(f"Countries: {df['iso'].nunique()}")

    detail = build_dimension_detail(df)

    profiles = build_country_profiles(detail)

    robustness = build_robustness_tests(df)

    # Final structural checks.
    expected_detail_rows = (
        EXPECTED_COUNTRIES * len(DIMENSIONS)
    )

    if len(detail) != expected_detail_rows:
        raise ValueError(
            f"Expected {expected_detail_rows} dimension records; "
            f"found {len(detail)}."
        )

    if detail.duplicated(
        ["iso", "dimension"]
    ).any():
        raise ValueError(
            "Duplicate country-dimension records detected."
        )

    if profiles["iso"].duplicated().any():
        raise ValueError(
            "Duplicate country profiles detected."
        )

    if len(profiles) != EXPECTED_COUNTRIES:
        raise ValueError(
            "Country profile count does not equal 20."
        )

    valid_labels = {
        "higher_risk",
        "moderate_risk",
        "lower_risk",
        "insufficient_data",
    }

    if not set(
        detail["risk_classification"].unique()
    ).issubset(valid_labels):
        raise ValueError(
            "Unexpected classification label detected."
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    detail.to_csv(
        DETAIL_OUTPUT,
        index=False
    )

    profiles.to_csv(
        OUTPUT,
        index=False
    )

    robustness.to_csv(
        ROBUSTNESS_OUTPUT,
        index=False
    )

    build_report(
        df,
        detail,
        profiles,
        robustness
    )

    print("")
    print("OUTPUTS CREATED")
    print("-" * 76)
    print(OUTPUT)
    print(DETAIL_OUTPUT)
    print(ROBUSTNESS_OUTPUT)
    print(REPORT)

    print("")
    print("VALIDATION")
    print("-" * 76)
    print(f"Dimension records: {len(detail)}")
    print(f"Country profiles: {len(profiles)}")
    print("Duplicate country-dimension records: 0")
    print("All structural checks passed.")


if __name__ == "__main__":
    main()
