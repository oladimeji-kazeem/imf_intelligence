from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DIAGNOSTICS_FILE = (
    ROOT / "data/processed/europe_labour_risk_diagnostics.csv"
)
ANNUAL_FILE = (
    ROOT / "data/processed/europe_labour_risk_annual_dynamics.csv"
)

OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

BASELINE_THRESHOLD = 10.0
MIN_HISTORICAL_OBSERVATIONS = 16


def median_absolute_deviation(series):
    series = series.dropna()

    if series.empty:
        return np.nan

    return (series - series.median()).abs().median()


def classify_peer_exposure(values):
    """
    Classify higher raw values as greater peer-relative exposure.

    Percentile ranks:
      <= 1/3: lower
      > 1/3 and < 2/3: moderate
      >= 2/3: higher
    """
    values = pd.to_numeric(values, errors="coerce")
    ranks = values.rank(method="average", pct=True)

    classifications = pd.Series(
        "insufficient_data",
        index=values.index,
        dtype="object",
    )

    valid = values.notna()

    classifications.loc[valid & (ranks <= 1 / 3)] = "lower_risk"
    classifications.loc[
        valid & (ranks >= 2 / 3)
    ] = "higher_risk"

    classifications.loc[
        valid & (ranks > 1 / 3) & (ranks < 2 / 3)
    ] = "moderate_risk"

    return classifications, ranks


def build_country_metrics(diagnostics, annual):
    rows = []

    for iso in sorted(annual["iso"].unique()):
        country_annual = annual[
            annual["iso"] == iso
        ].sort_values("year")

        modern = country_annual[
            country_annual["year"].between(2000, 2019)
        ].copy()

        recent = country_annual[
            country_annual["year"].between(2020, 2024)
        ].copy()

        historical_values = modern[
            "unemployment_rate_pct"
        ].dropna()

        recent_values = recent[
            "unemployment_rate_pct"
        ].dropna()

        historical_changes = modern.loc[
            modern["previous_year"].between(2000, 2019),
            "annual_change_pp"
        ].dropna()

        recent_changes = recent.loc[
            recent["previous_year"].between(2020, 2024),
            "annual_change_pp"
        ].dropna()

        historical_change_mad = median_absolute_deviation(
            historical_changes
        )

        historical_change_mean_abs = (
            historical_changes.abs().mean()
            if not historical_changes.empty else np.nan
        )

        historical_iqr = (
            historical_values.quantile(0.75)
            - historical_values.quantile(0.25)
            if not historical_values.empty else np.nan
        )

        recent_iqr = (
            recent_values.quantile(0.75)
            - recent_values.quantile(0.25)
            if not recent_values.empty else np.nan
        )

        recent_mad = median_absolute_deviation(recent_values)

        row = {
            "iso": iso,

            "recent_numeric_observations": len(recent_values),
            "recent_unemployment_mean_pct": (
                recent_values.mean()
                if not recent_values.empty else np.nan
            ),
            "recent_unemployment_median_pct": (
                recent_values.median()
                if not recent_values.empty else np.nan
            ),
            "recent_unemployment_mad_pct": recent_mad,
            "recent_unemployment_iqr_pct": recent_iqr,

            "historical_numeric_observations": len(
                historical_values
            ),
            "historical_unemployment_mean_pct": (
                historical_values.mean()
                if not historical_values.empty else np.nan
            ),
            "historical_unemployment_median_pct": (
                historical_values.median()
                if not historical_values.empty else np.nan
            ),
            "historical_unemployment_iqr_pct": historical_iqr,

            "historical_change_observations": len(
                historical_changes
            ),
            "historical_change_mad_pp": historical_change_mad,
            "historical_change_mean_absolute_pp": (
                historical_change_mean_abs
            ),

            "recent_change_observations": len(recent_changes),
            "recent_mean_absolute_change_pp": (
                recent_changes.abs().mean()
                if not recent_changes.empty else np.nan
            ),
        }

        # Historical elevated-unemployment frequency.
        for threshold in [8.0, 10.0, 12.0]:
            qualifying = (
                historical_values >= threshold
            ).sum()

            row[
                f"historical_frequency_at_or_above_{int(threshold)}pct"
            ] = (
                qualifying / len(historical_values) * 100
                if len(historical_values) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def add_dimension(
    metrics,
    dimension_name,
    metric_column,
    description,
    measurement_period,
):
    values = metrics[metric_column]

    classifications, ranks = classify_peer_exposure(values)

    detail = pd.DataFrame({
        "iso": metrics["iso"],
        "dimension": dimension_name,
        "measurement": metric_column,
        "measurement_period": measurement_period,
        "description": description,
        "value": values,
        "peer_exposure_percentile": ranks * 100,
        "classification": classifications,
        "eligible_country_count": int(values.notna().sum()),
    })

    return detail


def compare_classifications(
    baseline,
    alternative,
    scenario_name,
    dimension,
):
    merged = pd.DataFrame({
        "baseline": baseline,
        "alternative": alternative,
    }).dropna()

    if merged.empty:
        return {
            "dimension": dimension,
            "scenario": scenario_name,
            "countries_compared": 0,
            "class_changes": 0,
            "class_stability_pct": np.nan,
            "spearman_rank_correlation": np.nan,
        }

    changed = (
        merged["baseline"] != merged["alternative"]
    ).sum()

    return {
        "dimension": dimension,
        "scenario": scenario_name,
        "countries_compared": len(merged),
        "class_changes": int(changed),
        "class_stability_pct": (
            100 * (1 - changed / len(merged))
        ),
        "spearman_rank_correlation": np.nan,
    }


def rank_correlation(a, b):
    combined = pd.concat(
        [
            pd.to_numeric(a, errors="coerce").rename("a"),
            pd.to_numeric(b, errors="coerce").rename("b"),
        ],
        axis=1,
    ).dropna()

    if len(combined) < 3:
        return np.nan

    return combined["a"].corr(
        combined["b"],
        method="spearman",
    )


def main():
    print("=" * 76)
    print("EUROPEAN LABOUR RISK CLASSIFICATION ENGINE")
    print("=" * 76)

    for path in [DIAGNOSTICS_FILE, ANNUAL_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(DIAGNOSTICS_FILE)
    annual = pd.read_csv(ANNUAL_FILE)

    required_annual = {
        "iso",
        "year",
        "unemployment_rate_pct",
        "annual_change_pp",
    }

    missing = required_annual - set(annual.columns)

    if missing:
        raise ValueError(
            f"Annual input missing columns: {sorted(missing)}"
        )

    required_diagnostics = {
        "iso",
        "period",
        "mean_unemployment_pct",
        "median_unemployment_pct",
        "mad_unemployment_pct",
        "iqr_unemployment_pct",
    }

    missing = required_diagnostics - set(diagnostics.columns)

    if missing:
        raise ValueError(
            f"Diagnostics input missing columns: {sorted(missing)}"
        )

    annual["year"] = pd.to_numeric(
        annual["year"], errors="raise"
    ).astype(int)

    annual["unemployment_rate_pct"] = pd.to_numeric(
        annual["unemployment_rate_pct"], errors="coerce"
    )

    annual["annual_change_pp"] = pd.to_numeric(
        annual["annual_change_pp"], errors="coerce"
    )

    if annual.duplicated(["iso", "year"]).any():
        raise ValueError("Duplicate annual country-year records found.")

    countries = sorted(annual["iso"].unique())

    if len(countries) != 20:
        raise ValueError(
            f"Expected 20 countries; found {len(countries)}."
        )

    metrics = build_country_metrics(diagnostics, annual)

    if len(metrics) != 20:
        raise ValueError("Expected 20 country metric profiles.")

    # Require full recent and modern historical coverage.
    if not (metrics["recent_numeric_observations"] == 5).all():
        raise ValueError(
            "Recent period does not have five numeric observations "
            "for every country."
        )

    if not (
        metrics["historical_numeric_observations"] == 20
    ).all():
        raise ValueError(
            "Modern historical period is not complete for every country."
        )

    # Five independent baseline dimensions.
    dimension_specs = [
        (
            "recent_unemployment_pressure",
            "recent_unemployment_median_pct",
            "Median unemployment rate, 2020–2024",
            "2020–2024",
        ),
        (
            "recent_unemployment_volatility",
            "recent_unemployment_mad_pct",
            "Median absolute deviation of unemployment, 2020–2024",
            "2020–2024",
        ),
        (
            "historical_unemployment_pressure",
            "historical_unemployment_median_pct",
            "Median unemployment rate, 2000–2019",
            "2000–2019",
        ),
        (
            "historical_high_unemployment_persistence",
            "historical_frequency_at_or_above_10pct",
            "Frequency of years with unemployment at or above 10%",
            "2000–2019",
        ),
        (
            "historical_unemployment_instability",
            "historical_change_mad_pp",
            "MAD of annual unemployment changes in percentage points",
            "2000–2019",
        ),
    ]

    detail_frames = []

    for name, column, description, period in dimension_specs:
        detail_frames.append(
            add_dimension(
                metrics,
                name,
                column,
                description,
                period,
            )
        )

    detail = pd.concat(detail_frames, ignore_index=True)

    # Robustness: compare alternative measurements.
    robustness_rows = []

    scenario_specs = [
        (
            "recent_unemployment_pressure",
            "recent_unemployment_median_pct",
            "recent_unemployment_mean_pct",
            "recent_mean_vs_median",
        ),
        (
            "recent_unemployment_volatility",
            "recent_unemployment_mad_pct",
            "recent_unemployment_iqr_pct",
            "recent_mad_vs_iqr",
        ),
        (
            "historical_unemployment_pressure",
            "historical_unemployment_median_pct",
            "historical_unemployment_mean_pct",
            "historical_mean_vs_median",
        ),
        (
            "historical_unemployment_instability",
            "historical_change_mad_pp",
            "historical_change_mean_absolute_pp",
            "historical_change_mad_vs_mean_absolute",
        ),
    ]

    for dimension, baseline_col, alternative_col, scenario in scenario_specs:
        baseline_class, _ = classify_peer_exposure(
            metrics[baseline_col]
        )
        alternative_class, _ = classify_peer_exposure(
            metrics[alternative_col]
        )

        result = compare_classifications(
            baseline_class,
            alternative_class,
            scenario,
            dimension,
        )

        result["spearman_rank_correlation"] = rank_correlation(
            metrics[baseline_col],
            metrics[alternative_col],
        )

        robustness_rows.append(result)

    # Persistence threshold sensitivity: 8%, 10%, 12%.
    baseline_persistence_class, _ = classify_peer_exposure(
        metrics[
            "historical_frequency_at_or_above_10pct"
        ]
    )

    for threshold in [8.0, 12.0]:
        column = (
            f"historical_frequency_at_or_above_{int(threshold)}pct"
        )

        alternative_class, _ = classify_peer_exposure(
            metrics[column]
        )

        result = compare_classifications(
            baseline_persistence_class,
            alternative_class,
            f"persistence_threshold_{int(threshold)}pct_vs_10pct",
            "historical_high_unemployment_persistence",
        )

        result["spearman_rank_correlation"] = rank_correlation(
            metrics[
                "historical_frequency_at_or_above_10pct"
            ],
            metrics[column],
        )

        robustness_rows.append(result)

    robustness = pd.DataFrame(robustness_rows)

    # Create country profiles.
    profiles = metrics.copy()

    for dimension in detail["dimension"].unique():
        subset = detail[
            detail["dimension"] == dimension
        ][["iso", "classification"]].rename(
            columns={"classification": dimension}
        )

        profiles = profiles.merge(
            subset,
            on="iso",
            how="left",
            validate="one_to_one",
        )

    # Recent direction is contextual, not a scored dimension.
    recent_diagnostic = diagnostics[
        diagnostics["period"] == "recent_2020_2024"
    ][["iso", "first_last_observed_change_pp"]].copy()

    recent_diagnostic = recent_diagnostic.rename(
        columns={
            "first_last_observed_change_pp":
                "recent_unemployment_change_2020_2024_pp"
        }
    )

    profiles = profiles.merge(
        recent_diagnostic,
        on="iso",
        how="left",
        validate="one_to_one",
    )

    # Validate output structure.
    expected_dimensions = 20 * 5

    if len(detail) != expected_dimensions:
        raise ValueError(
            f"Expected {expected_dimensions} dimension records; "
            f"found {len(detail)}."
        )

    if detail.duplicated(["iso", "dimension"]).any():
        raise ValueError("Duplicate country-dimension classifications.")

    if detail["classification"].isna().any():
        raise ValueError("Missing classification labels detected.")

    if profiles["iso"].nunique() != 20:
        raise ValueError("Country profile count is not 20.")

    # Write outputs.
    profiles_path = (
        OUTPUT_DIR / "europe_labour_risk_country_profiles.csv"
    )
    detail_path = (
        OUTPUT_DIR / "europe_labour_risk_classifications.csv"
    )
    robustness_path = (
        REPORT_DIR / "europe_labour_risk_robustness.csv"
    )
    report_path = (
        REPORT_DIR / "europe_labour_risk_classification_report.txt"
    )

    profiles.to_csv(profiles_path, index=False)
    detail.to_csv(detail_path, index=False)
    robustness.to_csv(robustness_path, index=False)

    # Report dimension distributions.
    distribution_lines = []

    for dimension in detail["dimension"].unique():
        subset = detail[detail["dimension"] == dimension]

        counts = subset["classification"].value_counts()

        distribution_lines.append(
            f"{dimension}: "
            f"higher={counts.get('higher_risk', 0)}, "
            f"moderate={counts.get('moderate_risk', 0)}, "
            f"lower={counts.get('lower_risk', 0)}, "
            f"insufficient={counts.get('insufficient_data', 0)}"
        )

    lines = [
        "=" * 76,
        "EUROPEAN LABOUR RISK CLASSIFICATION REPORT",
        "=" * 76,
        "",
        "INPUTS",
        "-" * 76,
        str(DIAGNOSTICS_FILE),
        str(ANNUAL_FILE),
        "",
        "CLASSIFICATION DESIGN",
        "-" * 76,
        f"Countries: {len(countries)}",
        "Dimensions: 5",
        "Peer-relative classification: lower / moderate / higher exposure",
        "Classification uses percentile ranks within the 20-country group.",
        "No composite labour-risk score is calculated.",
        "",
        "DIMENSION DISTRIBUTIONS",
        "-" * 76,
        *distribution_lines,
        "",
        "ROBUSTNESS TESTS",
        "-" * 76,
        robustness.round(3).to_string(index=False),
        "",
        "RECENT UNEMPLOYMENT DIRECTION — CONTEXT ONLY",
        "-" * 76,
        profiles[
            [
                "iso",
                "recent_unemployment_change_2020_2024_pp",
            ]
        ].sort_values(
            "recent_unemployment_change_2020_2024_pp"
        ).round(3).to_string(index=False),
        "",
        "METHODOLOGICAL LIMITATIONS",
        "-" * 76,
        "Peer-relative classifications are not absolute risk ratings.",
        "The comparison group is limited to the selected 20 countries.",
        "The 10% persistence threshold is a descriptive convention.",
        "Threshold sensitivity is assessed at 8%, 10% and 12%.",
        "MAD and IQR capture different aspects of dispersion.",
        "Unemployment alone does not measure total labour-market slack.",
        "No forecast, crisis probability or composite score is produced.",
        "",
        "VALIDATION",
        "-" * 76,
        "PASS: 20 country profiles created.",
        "PASS: 100 country-dimension classifications created.",
        "PASS: No duplicate country-dimension records.",
        "PASS: Recent period complete for all 20 countries.",
        "PASS: Modern historical period complete for all 20 countries.",
        "PASS: Alternative measurement and threshold tests created.",
        "",
        "OUTPUTS",
        "-" * 76,
        str(profiles_path),
        str(detail_path),
        str(robustness_path),
        str(report_path),
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nLABOUR RISK CLASSIFICATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()