from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# EUROPEAN INFLATION RISK CLASSIFICATION ENGINE
# IMF WEO October 2026
#
# Classifications are peer-relative, not absolute risk ratings.
# Each production dimension is classified independently.
# No composite inflation risk score is calculated.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

DIAGNOSTICS_FILE = INPUT_DIR / "europe_inflation_diagnostics.csv"
DYNAMICS_FILE = INPUT_DIR / "europe_inflation_annual_dynamics.csv"
DEFLATION_FILE = INPUT_DIR / "europe_inflation_deflation_episodes.csv"

CLASSIFICATION_FILE = INPUT_DIR / "europe_inflation_risk_classifications.csv"
DETAIL_FILE = INPUT_DIR / "europe_inflation_risk_dimension_detail.csv"
ROBUSTNESS_FILE = INPUT_DIR / "europe_inflation_risk_robustness.csv"
REPORT_FILE = REPORT_DIR / "europe_inflation_risk_classification_report.txt"

RECENT_PERIOD = "pandemic_post_pandemic_2020_2024"
LONG_RUN_PERIOD = "long_run_1980_2025"
EXPECTED_COUNTRIES = 20

RISK_LABELS = {
    "higher": "higher_risk",
    "moderate": "moderate_risk",
    "lower": "lower_risk",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile_rank(series):
    """Peer percentile; higher metric values mean greater exposure."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True)


def classify_percentile(value):
    if pd.isna(value):
        return "insufficient_data"
    if value >= (2 / 3):
        return RISK_LABELS["higher"]
    if value <= (1 / 3):
        return RISK_LABELS["lower"]
    return RISK_LABELS["moderate"]


def numeric_column(frame, column):
    require(column in frame.columns, f"Missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def build_single_metric_dimension(
    frame,
    dimension,
    metric_column,
    interpretation,
    period,
    minimum_observations,
    metric_label,
):
    values = numeric_column(frame, metric_column)
    percentile = percentile_rank(values)

    result = pd.DataFrame({
        "iso": frame["iso"].values,
        "country": frame["country"].values,
        "dimension": dimension,
        "period": period,
        "measurement_type": "peer_relative",
        "metric_name": metric_label,
        "metric_value": values.values,
        "risk_score_percentile": percentile.values,
        "risk_class": percentile.apply(classify_percentile).values,
        "interpretation": interpretation,
        "minimum_observations_required": minimum_observations,
    })
    result[f"{metric_column}_percentile"] = percentile.values
    return result


def build_robustness_row(
    frame,
    dimension,
    baseline_column,
    alternative_column,
    scenario,
):
    baseline_values = numeric_column(frame, baseline_column)
    alternative_values = numeric_column(frame, alternative_column)

    baseline = percentile_rank(baseline_values)
    alternative = percentile_rank(alternative_values)

    baseline_class = baseline.apply(classify_percentile)
    alternative_class = alternative.apply(classify_percentile)

    valid = baseline.notna() & alternative.notna()
    compared = int(valid.sum())

    correlation = (
        baseline[valid].corr(alternative[valid], method="spearman")
        if compared >= 2 else np.nan
    )
    changed = int(
        (baseline_class[valid].to_numpy()
         != alternative_class[valid].to_numpy()).sum()
    )

    return {
        "dimension": dimension,
        "scenario": scenario,
        "baseline_metric": baseline_column,
        "alternative_metric": alternative_column,
        "countries_compared": compared,
        "classifications_changed": changed,
        "classification_stability_pct": (
            100.0 * (compared - changed) / compared
            if compared else np.nan
        ),
        "spearman_rank_correlation": correlation,
    }


def validate_fraction(frame, column, label):
    values = numeric_column(frame, column)
    require(values.notna().all(), f"{label} contains missing values.")
    require(values.between(0, 1).all(), f"{label} must lie between 0 and 1.")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("EUROPEAN INFLATION RISK CLASSIFICATION ENGINE")
    print("=" * 76)

    # --------------------------------------------------------
    # 1. LOAD AND VALIDATE INPUTS
    # --------------------------------------------------------
    for file in [DIAGNOSTICS_FILE, DYNAMICS_FILE, DEFLATION_FILE]:
        require(file.exists(), f"Input file not found: {file}")

    diagnostics = pd.read_csv(DIAGNOSTICS_FILE)
    dynamics = pd.read_csv(DYNAMICS_FILE)
    episodes = pd.read_csv(DEFLATION_FILE)

    required_diagnostic_columns = [
        "iso", "country", "period", "numeric_observations",
        "mean_inflation_pct", "median_inflation_pct",
        "mad_inflation_pct", "iqr_inflation_pct",
        "mean_absolute_annual_change_pp",
        "maximum_absolute_annual_change_pp",
        "deflation_frequency", "longest_deflation_episode_years",
        "inflation_ge_10pct_frequency", "inflation_ge_20pct_frequency",
    ]
    required_dynamics_columns = [
        "iso", "year", "inflation_change_pp", "is_inflation_change_shock",
    ]
    required_episode_columns = [
        "iso", "start_year", "end_year", "duration_years",
    ]

    for column in required_diagnostic_columns:
        require(column in diagnostics.columns,
                f"Missing diagnostic column: {column}")
    for column in required_dynamics_columns:
        require(column in dynamics.columns,
                f"Missing dynamics column: {column}")
    for column in required_episode_columns:
        require(column in episodes.columns,
                f"Missing episode column: {column}")

    require(
        diagnostics.duplicated(["iso", "period"]).sum() == 0,
        "Duplicate country-period diagnostics.",
    )
    require(
        dynamics.duplicated(["iso", "year"]).sum() == 0,
        "Duplicate annual dynamics records.",
    )

    recent = diagnostics.loc[
        diagnostics["period"] == RECENT_PERIOD
    ].copy()
    long_run = diagnostics.loc[
        diagnostics["period"] == LONG_RUN_PERIOD
    ].copy()

    require(len(recent) == EXPECTED_COUNTRIES,
            "Recent diagnostic panel must contain 20 countries.")
    require(len(long_run) == EXPECTED_COUNTRIES,
            "Long-run diagnostic panel must contain 20 countries.")
    require(recent["iso"].nunique() == EXPECTED_COUNTRIES,
            "Unexpected recent country coverage.")
    require(long_run["iso"].nunique() == EXPECTED_COUNTRIES,
            "Unexpected long-run country coverage.")
    require(
        set(recent["iso"]) == set(long_run["iso"]),
        "Country coverage differs between recent and long-run panels.",
    )
    require(
        recent["numeric_observations"].eq(5).all(),
        "Recent period must contain five numeric annual observations per country.",
    )
    require(
        long_run["numeric_observations"].ge(20).all(),
        "At least 20 long-run observations are required per country.",
    )

    # All production metrics must be available for the 20-country peer set.
    recent_metric_columns = [
        "mean_inflation_pct", "median_inflation_pct",
        "mad_inflation_pct", "iqr_inflation_pct",
        "mean_absolute_annual_change_pp",
        "maximum_absolute_annual_change_pp",
    ]
    long_metric_columns = [
        "median_inflation_pct", "deflation_frequency",
        "longest_deflation_episode_years",
        "inflation_ge_10pct_frequency",
    ]
    for column in recent_metric_columns:
        require(numeric_column(recent, column).notna().all(),
                f"Recent metric contains missing values: {column}")
    for column in long_metric_columns:
        require(numeric_column(long_run, column).notna().all(),
                f"Long-run metric contains missing values: {column}")

    # MAD and IQR are measured in inflation percentage points, not fractions.
    # They are checked for availability above, but must not be constrained to [0, 1].
    validate_fraction(long_run, "deflation_frequency", "Deflation frequency")
    validate_fraction(
        long_run, "inflation_ge_10pct_frequency",
        "Historical >=10% inflation frequency",
    )

    # --------------------------------------------------------
    # 2. BUILD RECENT SHOCK COUNTS
    # --------------------------------------------------------
    recent_dynamics = dynamics.loc[
        pd.to_numeric(dynamics["year"], errors="coerce").between(2020, 2024)
    ].copy()

    recent_dynamics["year"] = pd.to_numeric(
        recent_dynamics["year"], errors="coerce"
    )
    recent_dynamics["inflation_change_pp"] = pd.to_numeric(
        recent_dynamics["inflation_change_pp"], errors="coerce"
    )

    require(
        len(recent_dynamics) == EXPECTED_COUNTRIES * 5,
        "Expected exactly five annual dynamics records per country for 2020-2024.",
    )
    require(
        recent_dynamics.groupby("iso")["year"].nunique().eq(5).all(),
        "Every country must have five distinct annual dynamics years, 2020-2024.",
    )
    require(
        recent_dynamics["inflation_change_pp"].notna().all(),
        "Recent annual inflation changes contain missing values.",
    )

    shock_flag = recent_dynamics["is_inflation_change_shock"]
    if shock_flag.dtype == object:
        shock_flag = (
            shock_flag.astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    require(shock_flag.notna().all(),
            "Shock flag contains values that cannot be interpreted as boolean.")
    recent_dynamics["is_inflation_change_shock"] = shock_flag.astype(bool)

    recent_dynamics["is_upward_shock"] = (
        recent_dynamics["is_inflation_change_shock"]
        & (recent_dynamics["inflation_change_pp"] > 0)
    )
    recent_dynamics["is_downward_shock"] = (
        recent_dynamics["is_inflation_change_shock"]
        & (recent_dynamics["inflation_change_pp"] < 0)
    )

    shock_counts = (
        recent_dynamics.groupby("iso")
        .agg(
            recent_upward_inflation_shocks=("is_upward_shock", "sum"),
            recent_downward_disinflation_shocks=("is_downward_shock", "sum"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 3. BUILD DEFLATION EPISODE PROFILES
    # --------------------------------------------------------
    episodes["duration_years"] = pd.to_numeric(
        episodes["duration_years"], errors="coerce"
    )
    require(
        episodes.empty or episodes["duration_years"].notna().all(),
        "Deflation episode duration contains missing values.",
    )
    require(
        episodes.empty or episodes["duration_years"].gt(0).all(),
        "Deflation episode duration must be positive.",
    )

    if episodes.empty:
        episode_profiles = pd.DataFrame({
            "iso": recent["iso"],
            "deflation_episode_count": 0,
            "longest_deflation_episode_observed": 0,
            "mean_deflation_episode_duration": np.nan,
        })
    else:
        episode_profiles = (
            episodes.groupby("iso")
            .agg(
                deflation_episode_count=("duration_years", "count"),
                longest_deflation_episode_observed=("duration_years", "max"),
                mean_deflation_episode_duration=("duration_years", "mean"),
            )
            .reset_index()
        )

    country_frame = recent[["iso", "country"]].copy()
    episode_profiles = country_frame.merge(
        episode_profiles,
        on="iso",
        how="left",
        validate="one_to_one",
    )

    for column in [
        "deflation_episode_count",
        "longest_deflation_episode_observed",
    ]:
        episode_profiles[column] = episode_profiles[column].fillna(0)

    # --------------------------------------------------------
    # 4. MERGE COUNTRY PROFILES
    # --------------------------------------------------------
    profile = recent.merge(
        long_run,
        on=["iso", "country"],
        how="inner",
        suffixes=("_recent", "_long_run"),
        validate="one_to_one",
    )

    profile = profile.merge(
        shock_counts, on="iso", how="left", validate="one_to_one"
    )
    profile = profile.merge(
        episode_profiles,
        on=["iso", "country"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_episodes"),
    )

    for column in [
        "recent_upward_inflation_shocks",
        "recent_downward_disinflation_shocks",
        "deflation_episode_count",
        "longest_deflation_episode_observed",
    ]:
        profile[column] = profile[column].fillna(0)

    require(len(profile) == EXPECTED_COUNTRIES,
            "Merged country profile must contain 20 countries.")
    require(profile["iso"].nunique() == EXPECTED_COUNTRIES,
            "Merged country profile contains duplicate or missing countries.")

    # --------------------------------------------------------
    # 5. DEFINE EIGHT SEPARATE PRODUCTION DIMENSIONS
    # --------------------------------------------------------
    # The production classification uses one primary metric per dimension.
    # Alternative measures remain visible in the robustness output.
    dimension_definitions = [
        {
            "dimension": "recent_inflation_pressure",
            "column": "median_inflation_pct_recent",
            "interpretation": (
                "Higher median CPI inflation in 2020-2024 indicates greater "
                "recent price pressure. The mean is retained as a sensitivity measure."
            ),
            "period": "2020-2024",
            "minimum": 5,
            "label": "Median annual CPI inflation, percent",
        },
        {
            "dimension": "recent_inflation_volatility_mad",
            "column": "mad_inflation_pct_recent",
            "interpretation": (
                "Higher median absolute deviation of annual CPI inflation in "
                "2020-2024 indicates greater recent dispersion around the median."
            ),
            "period": "2020-2024",
            "minimum": 5,
            "label": "Median absolute deviation of annual CPI inflation",
        },
        {
            "dimension": "recent_inflation_dispersion_iqr",
            "column": "iqr_inflation_pct_recent",
            "interpretation": (
                "Higher interquartile range of annual CPI inflation in 2020-2024 "
                "indicates greater dispersion across the middle half of observations. "
                "This is a separate diagnostic, not a substitute for MAD."
            ),
            "period": "2020-2024",
            "minimum": 5,
            "label": "Interquartile range of annual CPI inflation",
        },
        {
            "dimension": "inflation_change_instability",
            "column": "mean_absolute_annual_change_pp_recent",
            "interpretation": (
                "Higher mean absolute annual changes indicate greater movement "
                "in the inflation rate, regardless of whether inflation rises or falls. "
                "This is not directional inflation acceleration."
            ),
            "period": "2020-2024",
            "minimum": 4,
            "label": "Mean absolute annual change in inflation, percentage points",
        },
        {
            "dimension": "upward_inflation_shocks",
            "column": "recent_upward_inflation_shocks",
            "interpretation": (
                "More statistically unusual positive annual changes in inflation "
                "during 2020-2024 indicate greater observed upward-shock frequency. "
                "Shock thresholds are based on each country's historical distribution."
            ),
            "period": "2020-2024",
            "minimum": 5,
            "label": "Count of upward inflation shocks",
        },
        {
            "dimension": "historical_deflation_frequency",
            "column": "deflation_frequency_long_run",
            "interpretation": (
                "A higher share of years with negative annual CPI inflation "
                "during 1980-2025 indicates greater historical deflation frequency. "
                "Episode duration is reported separately."
            ),
            "period": "1980-2025",
            "minimum": 20,
            "label": "Frequency of annual deflation",
        },
        {
            "dimension": "historical_inflation_persistence",
            "column": "median_inflation_pct_long_run",
            "interpretation": (
                "Higher long-run median annual CPI inflation indicates a higher "
                "historical central level of inflation. It does not measure extreme episodes."
            ),
            "period": "1980-2025",
            "minimum": 20,
            "label": "Median annual CPI inflation, percent",
        },
        {
            "dimension": "historical_high_inflation_exposure",
            "column": "inflation_ge_10pct_frequency_long_run",
            "interpretation": (
                "A higher share of years with annual CPI inflation at or above "
                "10% indicates greater historical high-inflation exposure. "
                "The 10% threshold is descriptive, not a universal crisis threshold."
            ),
            "period": "1980-2025",
            "minimum": 20,
            "label": "Frequency of annual CPI inflation at or above 10%",
        },
    ]

    detail_records = []
    for definition in dimension_definitions:
        dimension_frame = profile[["iso", "country", definition["column"]]].copy()
        result = build_single_metric_dimension(
            frame=dimension_frame,
            dimension=definition["dimension"],
            metric_column=definition["column"],
            interpretation=definition["interpretation"],
            period=definition["period"],
            minimum_observations=definition["minimum"],
            metric_label=definition["label"],
        )
        detail_records.append(result)

    detail = pd.concat(detail_records, ignore_index=True)

    # --------------------------------------------------------
    # 6. BUILD COUNTRY CLASSIFICATION TABLE
    # --------------------------------------------------------
    classification = profile[["iso", "country"]].copy()

    for dimension in detail["dimension"].unique():
        subset = detail.loc[
            detail["dimension"] == dimension,
            ["iso", "risk_score_percentile", "risk_class"],
        ].copy()
        subset = subset.rename(columns={
            "risk_score_percentile": f"{dimension}_percentile",
            "risk_class": f"{dimension}_risk_class",
        })
        classification = classification.merge(
            subset, on="iso", how="left", validate="one_to_one"
        )

    # Contextual measures are deliberately not converted into classifications.
    contextual_map = {
        "mean_inflation_pct_recent": "mean_inflation_pct_recent",
        "median_inflation_pct_recent": "median_inflation_pct_recent",
        "mad_inflation_pct_recent": "mad_inflation_pct_recent",
        "iqr_inflation_pct_recent": "iqr_inflation_pct_recent",
        "mean_absolute_annual_change_pp_recent":
            "mean_absolute_annual_change_pp_recent",
        "maximum_absolute_annual_change_pp_recent":
            "maximum_absolute_annual_change_pp_recent",
        "median_inflation_pct_long_run": "median_inflation_pct_long_run",
        "inflation_ge_10pct_frequency_long_run":
            "inflation_ge_10pct_frequency_long_run",
        "inflation_ge_20pct_frequency_long_run":
            "inflation_ge_20pct_frequency_long_run",
        "recent_upward_inflation_shocks": "recent_upward_inflation_shocks",
        "recent_downward_disinflation_shocks":
            "recent_downward_disinflation_shocks",
        "deflation_frequency_long_run": "deflation_frequency_long_run",
        "deflation_episode_count": "deflation_episode_count",
        "longest_deflation_episode_observed":
            "longest_deflation_episode_observed",
        "mean_deflation_episode_duration": "mean_deflation_episode_duration",
    }

    for output_column, profile_column in contextual_map.items():
        require(profile_column in profile.columns,
                f"Contextual field missing from merged profile: {profile_column}")
        classification[output_column] = profile[profile_column].values

    # --------------------------------------------------------
    # 7. ROBUSTNESS AUDIT
    # --------------------------------------------------------
    # Each comparison tests an alternative measure against the production
    # measure. It does not establish predictive accuracy.
    robustness_scenarios = [
        (
            "recent_inflation_pressure",
            "median_inflation_pct_recent",
            "mean_inflation_pct_recent",
            "mean_vs_median",
        ),
        (
            "recent_inflation_dispersion",
            "mad_inflation_pct_recent",
            "iqr_inflation_pct_recent",
            "mad_vs_iqr",
        ),
        (
            "inflation_change_instability",
            "mean_absolute_annual_change_pp_recent",
            "maximum_absolute_annual_change_pp_recent",
            "mean_absolute_change_vs_maximum_absolute_change",
        ),
        (
            "historical_deflation_frequency",
            "deflation_frequency_long_run",
            "longest_deflation_episode_observed",
            "frequency_vs_longest_episode",
        ),
        (
            "historical_inflation_exposure",
            "median_inflation_pct_long_run",
            "inflation_ge_10pct_frequency_long_run",
            "median_vs_high_inflation_frequency",
        ),
    ]

    robustness_rows = []
    for dimension, baseline, alternative, scenario in robustness_scenarios:
        robustness_rows.append(
            build_robustness_row(
                profile, dimension, baseline, alternative, scenario
            )
        )
    robustness = pd.DataFrame(robustness_rows)

    # --------------------------------------------------------
    # 8. VALIDATION
    # --------------------------------------------------------
    expected_dimensions = len(dimension_definitions)
    require(len(classification) == EXPECTED_COUNTRIES,
            "Expected 20 country classification profiles.")
    require(classification["iso"].nunique() == EXPECTED_COUNTRIES,
            "Duplicate or missing country profiles.")
    require(
        len(detail) == EXPECTED_COUNTRIES * expected_dimensions,
        f"Expected {EXPECTED_COUNTRIES * expected_dimensions} "
        "country-dimension records.",
    )
    require(detail.duplicated(["iso", "dimension"]).sum() == 0,
            "Duplicate country-dimension records.")
    require(
        detail["risk_class"].isin([
            "higher_risk", "moderate_risk",
            "lower_risk", "insufficient_data",
        ]).all(),
        "Unexpected classification label.",
    )
    require(
        detail["risk_score_percentile"].between(0, 1).all(),
        "Risk percentile outside 0-1 range.",
    )

    class_columns = [
        column for column in classification.columns
        if column.endswith("_risk_class")
    ]
    require(
        classification[class_columns].notna().all().all(),
        "Missing risk classifications.",
    )
    require(
        classification[class_columns].isin([
            "higher_risk", "moderate_risk", "lower_risk",
            "insufficient_data",
        ]).all().all(),
        "Unexpected country-level classification label.",
    )
    require(
        (classification["recent_upward_inflation_shocks"] >= 0).all(),
        "Negative upward shock count.",
    )
    require(
        (classification["recent_downward_disinflation_shocks"] >= 0).all(),
        "Negative downward shock count.",
    )
    require(
        (classification["deflation_episode_count"] >= 0).all(),
        "Negative deflation episode count.",
    )
    require(
        (classification["longest_deflation_episode_observed"] >= 0).all(),
        "Negative deflation episode duration.",
    )
    require(len(robustness) == len(robustness_scenarios),
            "Unexpected number of robustness scenarios.")

    # --------------------------------------------------------
    # 9. SAVE OUTPUTS
    # --------------------------------------------------------
    classification.to_csv(CLASSIFICATION_FILE, index=False)
    detail.to_csv(DETAIL_FILE, index=False)
    robustness.to_csv(ROBUSTNESS_FILE, index=False)

    # --------------------------------------------------------
    # 10. REPORT
    # --------------------------------------------------------
    report = [
        "=" * 76,
        "EUROPEAN INFLATION RISK CLASSIFICATION REPORT",
        "=" * 76,
        "",
        "INPUTS",
        "-" * 76,
        str(DIAGNOSTICS_FILE),
        str(DYNAMICS_FILE),
        str(DEFLATION_FILE),
        "",
        "CLASSIFICATION DESIGN",
        "-" * 76,
        "Peer-relative tercile classifications.",
        "Higher percentile means greater measured exposure.",
        "Higher risk: percentile >= 2/3.",
        "Moderate risk: percentile > 1/3 and < 2/3.",
        "Lower risk: percentile <= 1/3.",
        "Each production dimension is classified independently.",
        "No composite inflation risk score is calculated.",
        "No absolute inflation-risk threshold is asserted.",
        "",
        "EIGHT PRODUCTION DIMENSIONS",
        "-" * 76,
        "1. Recent inflation pressure: median CPI inflation, 2020-2024.",
        "2. Recent inflation volatility: MAD, 2020-2024.",
        "3. Recent inflation dispersion: IQR, 2020-2024.",
        "4. Inflation-change instability: mean absolute annual change, 2020-2024.",
        "5. Upward inflation shocks: count of unusual positive changes, 2020-2024.",
        "6. Historical deflation frequency: share of negative-inflation years, 1980-2025.",
        "7. Historical inflation persistence: long-run median CPI inflation, 1980-2025.",
        "8. Historical high-inflation exposure: frequency >=10%, 1980-2025.",
        "",
        "INTERPRETATION NOTES",
        "-" * 76,
        "MAD and IQR are separate dimensions; neither is averaged into the other.",
        "Mean inflation is a sensitivity measure; median is the production pressure metric.",
        "Mean absolute change measures movement, not directional acceleration.",
        "Deflation frequency is classified; episode duration is reported separately.",
        "Long-run median and >=10% frequency are separate historical dimensions.",
        "The 10% threshold is descriptive, not a universal crisis threshold.",
        "",
        "COUNTRY CLASSIFICATIONS",
        "-" * 76,
    ]

    class_columns = [
        column for column in classification.columns
        if column.endswith("_risk_class")
    ]
    for row in classification.sort_values("iso").itertuples(index=False):
        values = [
            f"{column.removesuffix('_risk_class')}={getattr(row, column)}"
            for column in class_columns
        ]
        report.append(f"{row.iso} {row.country}: " + "; ".join(values))

    report.extend(["", "CLASSIFICATION DISTRIBUTIONS", "-" * 76])
    for dimension in detail["dimension"].unique():
        subset = detail.loc[detail["dimension"] == dimension]
        counts = subset["risk_class"].value_counts()
        report.append(f"{dimension}:")
        for label in [
            "higher_risk", "moderate_risk",
            "lower_risk", "insufficient_data",
        ]:
            report.append(f"  {label}: {int(counts.get(label, 0))}")

    report.extend([
        "",
        "RECENT SHOCK DIRECTION",
        "-" * 76,
        "Total upward shocks, 2020-2024: "
        f"{int(classification['recent_upward_inflation_shocks'].sum())}",
        "Total downward disinflation shocks, 2020-2024: "
        f"{int(classification['recent_downward_disinflation_shocks'].sum())}",
        "",
        "ROBUSTNESS AUDIT",
        "-" * 76,
    ])

    for row in robustness.itertuples(index=False):
        report.append(
            f"{row.dimension} | {row.scenario}: "
            f"changed={row.classifications_changed}/"
            f"{row.countries_compared}; "
            f"stability={row.classification_stability_pct:.1f}%; "
            f"Spearman={row.spearman_rank_correlation:.3f}"
        )

    report.extend([
        "",
        "METHODOLOGICAL LIMITATIONS",
        "-" * 76,
        "Peer-relative classes are not absolute economic risk ratings.",
        "Five annual observations provide limited evidence on persistence.",
        "MAD and IQR capture different aspects of dispersion.",
        "Shock detection depends on country-specific historical distributions.",
        "A statistical shock is not automatically an economic crisis.",
        "Mean absolute annual change does not identify upward acceleration.",
        "Deflation episode duration is descriptive and separate from frequency.",
        "Long-run inflation measures can reflect historical monetary transitions.",
        "The 10% threshold is descriptive, not a universal crisis threshold.",
        "No causal inference, forecast, or default probability is estimated.",
        "Robustness comparisons measure sensitivity, not predictive accuracy.",
        "",
        "VALIDATION",
        "-" * 76,
        "PASS: All three input files found.",
        "PASS: Required columns present.",
        "PASS: No duplicate country-period diagnostics.",
        "PASS: No duplicate annual dynamics records.",
        "PASS: Recent and long-run panels cover the same 20 countries.",
        "PASS: Five recent numeric annual observations per country.",
        "PASS: Recent shock panel contains five years per country.",
        "PASS: Required production metrics contain no missing values.",
        "PASS: Frequency metrics lie between zero and one.",
        "PASS: 20 country profiles generated.",
        f"PASS: {EXPECTED_COUNTRIES * expected_dimensions} "
        "country-dimension records generated.",
        "PASS: No duplicate country-dimension records.",
        "PASS: Risk percentiles lie between zero and one.",
        "PASS: Risk classification labels valid.",
        "PASS: Shock counts non-negative.",
        "PASS: Deflation episode metrics non-negative.",
        "PASS: Five robustness scenarios generated.",
        "",
        "OUTPUTS CREATED",
        "-" * 76,
        str(CLASSIFICATION_FILE),
        str(DETAIL_FILE),
        str(ROBUSTNESS_FILE),
        str(REPORT_FILE),
        "",
        "=" * 76,
        "EUROPEAN INFLATION RISK CLASSIFICATION VALIDATION PASSED",
        "=" * 76,
    ])

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[-75:]))


if __name__ == "__main__":
    main()
