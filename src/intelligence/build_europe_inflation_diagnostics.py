
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# EUROPEAN INFLATION DIAGNOSTICS
# Source: IMF WEO October 2026
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/europe_inflation_risk_input_panel.csv"

OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSTICS_FILE = OUTPUT_DIR / "europe_inflation_diagnostics.csv"
DYNAMICS_FILE = OUTPUT_DIR / "europe_inflation_annual_dynamics.csv"
DEFLATION_FILE = OUTPUT_DIR / "europe_inflation_deflation_episodes.csv"
REPORT_FILE = REPORT_DIR / "europe_inflation_diagnostics_report.txt"

VALUE_COL = "inflation_pct"

PERIODS = {
    "long_run_1980_2025": (1980, 2025),
    "modern_2000_2019": (2000, 2019),
    "recent_pre_pandemic_2015_2019": (2015, 2019),
    "pandemic_post_pandemic_2020_2024": (2020, 2024),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def consecutive_episode_records(group, condition_column, episode_type):
    """
    Build episodes from consecutive calendar years satisfying
    a Boolean condition. Missing observations and year gaps
    break an episode.
    """
    records = []

    group = group.sort_values("year").copy()
    active = []

    def save_episode(years, values):
        if not years:
            return

        records.append({
            "iso": group["iso"].iloc[0],
            "country": group["country"].iloc[0],
            "episode_type": episode_type,
            "start_year": int(years[0]),
            "end_year": int(years[-1]),
            "duration_years": len(years),
            "worst_inflation_pct": float(min(values)),
            "best_inflation_pct": float(max(values)),
            "mean_inflation_pct": float(np.mean(values)),
        })

    for row in group.itertuples(index=False):
        year = int(row.year)
        value = getattr(row, VALUE_COL)
        condition = bool(getattr(row, condition_column))

        if condition and pd.notna(value):
            if active and year != active[-1][0] + 1:
                save_episode(
                    [item[0] for item in active],
                    [item[1] for item in active],
                )
                active = []

            active.append((year, float(value)))
        else:
            save_episode(
                [item[0] for item in active],
                [item[1] for item in active],
            )
            active = []

    save_episode(
        [item[0] for item in active],
        [item[1] for item in active],
    )

    return records


def calculate_diagnostics(group, period_name, start_year, end_year):
    group = group.sort_values("year").copy()
    observed = group.dropna(subset=[VALUE_COL]).copy()

    values = observed[VALUE_COL]
    n = len(values)

    expected_years = end_year - start_year + 1
    coverage = n / expected_years if expected_years else np.nan

    if n == 0:
        return {
            "period": period_name,
            "period_start": start_year,
            "period_end": end_year,
            "expected_years": expected_years,
            "numeric_observations": 0,
            "coverage_ratio": 0.0,
        }

    # Inflation level and dispersion
    mean_value = values.mean()
    median_value = values.median()
    std_value = values.std(ddof=1) if n >= 2 else np.nan
    mad_value = (values - median_value).abs().median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    # Annual changes require consecutive calendar years.
    observed = observed.sort_values("year")
    observed["previous_year"] = observed["year"].shift(1)
    observed["previous_inflation"] = observed[VALUE_COL].shift(1)

    consecutive = (
        observed["year"] - observed["previous_year"]
    ) == 1

    changes = (
        observed.loc[consecutive, VALUE_COL]
        - observed.loc[consecutive, "previous_inflation"]
    )

    # Lag-one autocorrelation on consecutive annual observations.
    lag_pairs = observed.loc[consecutive, [VALUE_COL, "previous_inflation"]]

    if len(lag_pairs) >= 3:
        persistence = lag_pairs[VALUE_COL].corr(
            lag_pairs["previous_inflation"]
        )
    else:
        persistence = np.nan

    # Deflation episodes within the selected period.
    deflation_records = consecutive_episode_records(
        observed,
        "is_deflation",
        "deflation",
    )

    # Positive inflation episodes are descriptive, not a measure
    # of excessive inflation.
    positive_records = consecutive_episode_records(
        observed,
        "is_positive_inflation",
        "positive_inflation",
    )

    deflation_years = int((values < 0).sum())
    positive_years = int((values > 0).sum())

    return {
        "period": period_name,
        "period_start": start_year,
        "period_end": end_year,
        "expected_years": expected_years,
        "numeric_observations": n,
        "missing_observations": expected_years - n,
        "coverage_ratio": coverage,

        "mean_inflation_pct": mean_value,
        "median_inflation_pct": median_value,
        "std_inflation_pct": std_value,
        "mad_inflation_pct": mad_value,
        "iqr_inflation_pct": q3 - q1,
        "minimum_inflation_pct": values.min(),
        "maximum_inflation_pct": values.max(),

        "positive_inflation_years": positive_years,
        "positive_inflation_frequency": positive_years / n,
        "deflation_years": deflation_years,
        "deflation_frequency": deflation_years / n,
        "deflation_episode_count": len(deflation_records),
        "longest_deflation_episode_years": (
            max(
                (x["duration_years"] for x in deflation_records),
                default=0,
            )
        ),

        "positive_inflation_episode_count": len(positive_records),
        "longest_positive_inflation_episode_years": (
            max(
                (x["duration_years"] for x in positive_records),
                default=0,
            )
        ),

        "consecutive_annual_change_observations": len(changes),
        "mean_annual_inflation_change_pp": (
            changes.mean() if len(changes) else np.nan
        ),
        "mean_absolute_annual_change_pp": (
            changes.abs().mean() if len(changes) else np.nan
        ),
        "maximum_absolute_annual_change_pp": (
            changes.abs().max() if len(changes) else np.nan
        ),
        "lag_one_inflation_autocorrelation": persistence,

        "inflation_ge_10pct_frequency": (
            observed["is_high_inflation_10pct"].sum() / n
        ),
        "inflation_ge_20pct_frequency": (
            observed["is_high_inflation_20pct"].sum() / n
        ),
    }


def build_annual_dynamics(data):
    frames = []

    for iso, group in data.groupby("iso"):
        group = group.sort_values("year").copy()

        group["previous_year"] = group["year"].shift(1)
        group["previous_inflation_pct"] = group[VALUE_COL].shift(1)

        group["is_consecutive_year"] = (
            group["year"] - group["previous_year"]
        ) == 1

        group["inflation_change_pp"] = np.where(
            group["is_consecutive_year"],
            group[VALUE_COL] - group["previous_inflation_pct"],
            np.nan,
        )

        group["acceleration_direction"] = np.select(
            [
                group["inflation_change_pp"] > 0,
                group["inflation_change_pp"] < 0,
            ],
            [
                "accelerating",
                "decelerating",
            ],
            default="unchanged_or_unavailable",
        )

        # Country-specific robust shock detection.
        valid_changes = group["inflation_change_pp"].dropna()

        if len(valid_changes) >= 5:
            change_median = valid_changes.median()
            mad = (valid_changes - change_median).abs().median()

            if mad > 0:
                robust_scale = 1.4826 * mad
                threshold = 3.0 * robust_scale
                method = "median_absolute_deviation_3_scaled"
            else:
                deviations = (valid_changes - change_median).abs()
                threshold = deviations.quantile(0.95)
                method = "empirical_95th_percentile_fallback"

            group["inflation_change_robust_score"] = (
                group["inflation_change_pp"] - change_median
            ) / (
                1.4826 * mad if mad > 0 else np.nan
            )

            group["inflation_change_shock_threshold_pp"] = threshold
            group["inflation_change_shock_method"] = method

            group["is_inflation_change_shock"] = (
                group["inflation_change_pp"].notna()
                & (
                    (group["inflation_change_pp"] - change_median).abs()
                    > threshold
                )
            )
        else:
            group["inflation_change_robust_score"] = np.nan
            group["inflation_change_shock_threshold_pp"] = np.nan
            group["inflation_change_shock_method"] = (
                "insufficient_history"
            )
            group["is_inflation_change_shock"] = False

        frames.append(group)

    result = pd.concat(frames, ignore_index=True)

    return result


def main():
    print("=" * 76)
    print("EUROPEAN INFLATION DIAGNOSTICS")
    print("=" * 76)

    require(INPUT_FILE.exists(), f"Input file not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE)

    required_columns = [
        "iso",
        "country",
        "year",
        "inflation_pct",
        "unit",
        "scale",
        "source",
        "source_vintage",
        "observation_status",
        "is_missing_inflation",
        "is_deflation",
        "is_high_inflation_10pct",
        "is_high_inflation_20pct",
    ]

    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    require(
        not missing_columns,
        f"Required columns missing: {missing_columns}",
    )

    data["year"] = pd.to_numeric(data["year"], errors="raise")
    data[VALUE_COL] = pd.to_numeric(data[VALUE_COL], errors="coerce")

    require(data["iso"].notna().all(), "Missing country ISO codes.")
    require(data["country"].notna().all(), "Missing country names.")
    require(data["year"].notna().all(), "Missing years.")

    data["year"] = data["year"].astype(int)

    # Scope: historical actual observations only.
    require(
        set(data["observation_status"].dropna().unique())
        == {"historical_actual"},
        "Unexpected observation status found. Review input scope.",
    )

    require(
        set(data["unit"].dropna().unique()) == {"Percent"},
        "Unexpected inflation unit found.",
    )

    require(
        set(data["scale"].dropna().unique()) == {"Units"},
        "Unexpected inflation scale found.",
    )

    duplicate_count = int(data.duplicated(["iso", "year"]).sum())
    require(duplicate_count == 0, "Duplicate country-year observations.")

    # Verify that missingness and deflation flags agree with values.
    missing_flag_mismatches = int(
        (
            data["is_missing_inflation"]
            != data[VALUE_COL].isna()
        ).sum()
    )

    require(
        missing_flag_mismatches == 0,
        "Missingness flag does not agree with inflation values.",
    )

    deflation_flag_mismatches = int(
        (
            data["is_deflation"]
            != (data[VALUE_COL] < 0)
        ).sum()
    )

    require(
        deflation_flag_mismatches == 0,
        "Deflation flag does not agree with inflation values.",
    )

    data["is_positive_inflation"] = data[VALUE_COL] > 0

    countries = (
        data[["iso", "country"]]
        .drop_duplicates()
        .sort_values("iso")
    )

    require(len(countries) == 20, "Expected 20 European countries.")

    # --------------------------------------------------------
    # 1. COUNTRY-PERIOD DIAGNOSTICS
    # --------------------------------------------------------

    diagnostic_records = []

    for country in countries.itertuples(index=False):
        country_data = data[data["iso"] == country.iso]

        for period_name, (start_year, end_year) in PERIODS.items():
            period_data = country_data[
                country_data["year"].between(start_year, end_year)
            ].copy()

            record = calculate_diagnostics(
                period_data,
                period_name,
                start_year,
                end_year,
            )

            record["iso"] = country.iso
            record["country"] = country.country

            diagnostic_records.append(record)

    diagnostics = pd.DataFrame(diagnostic_records)

    diagnostics = diagnostics[
        [
            "iso",
            "country",
            "period",
            "period_start",
            "period_end",
        ]
        + [
            col for col in diagnostics.columns
            if col not in {
                "iso",
                "country",
                "period",
                "period_start",
                "period_end",
            }
        ]
    ]

    # --------------------------------------------------------
    # 2. ANNUAL INFLATION DYNAMICS AND SHOCKS
    # --------------------------------------------------------

    dynamics = build_annual_dynamics(data)

    # --------------------------------------------------------
    # 3. FULL-HISTORY DEFLATION EPISODES
    # --------------------------------------------------------

    deflation_records = []

    for _, group in data.groupby("iso"):
        deflation_records.extend(
            consecutive_episode_records(
                group.dropna(subset=[VALUE_COL]),
                "is_deflation",
                "deflation",
            )
        )

    deflation_episodes = pd.DataFrame(deflation_records)

    if deflation_episodes.empty:
        deflation_episodes = pd.DataFrame(
            columns=[
                "iso",
                "country",
                "episode_type",
                "start_year",
                "end_year",
                "duration_years",
                "worst_inflation_pct",
                "best_inflation_pct",
                "mean_inflation_pct",
            ]
        )

    # --------------------------------------------------------
    # 4. VALIDATION
    # --------------------------------------------------------

    require(
        len(diagnostics) == 80,
        "Expected 80 country-period diagnostic records.",
    )

    require(
        diagnostics.duplicated(["iso", "period"]).sum() == 0,
        "Duplicate country-period diagnostic records.",
    )

    require(
        len(dynamics) == len(data),
        "Annual dynamics row count differs from input.",
    )

    require(
        dynamics.duplicated(["iso", "year"]).sum() == 0,
        "Duplicate annual dynamics records.",
    )

    require(
        (diagnostics["deflation_frequency"].dropna().between(0, 1)).all(),
        "Deflation frequency outside 0-1 range.",
    )

    require(
        (diagnostics["positive_inflation_frequency"].dropna().between(0, 1)).all(),
        "Positive inflation frequency outside 0-1 range.",
    )

    require(
        (diagnostics["coverage_ratio"].dropna().between(0, 1)).all(),
        "Coverage ratio outside 0-1 range.",
    )

    require(
        (deflation_episodes["duration_years"] > 0).all(),
        "Invalid deflation episode duration.",
    )

    require(
        (
            deflation_episodes["end_year"]
            >= deflation_episodes["start_year"]
        ).all(),
        "Deflation episode end precedes start.",
    )

    # --------------------------------------------------------
    # 5. SAVE OUTPUTS
    # --------------------------------------------------------

    diagnostics.to_csv(DIAGNOSTICS_FILE, index=False)
    dynamics.to_csv(DYNAMICS_FILE, index=False)
    deflation_episodes.to_csv(DEFLATION_FILE, index=False)

    # --------------------------------------------------------
    # 6. REPORT
    # --------------------------------------------------------

    numeric = data.dropna(subset=[VALUE_COL])

    shock_counts = (
        dynamics.groupby("iso")["is_inflation_change_shock"]
        .sum()
        .sort_values(ascending=False)
    )

    report_lines = [
        "=" * 76,
        "EUROPEAN INFLATION DIAGNOSTICS REPORT",
        "=" * 76,
        "",
        "SOURCE AND COVERAGE",
        "-" * 76,
        f"Input: {INPUT_FILE}",
        f"Source: {data['source'].iloc[0]}",
        f"Vintage: {data['source_vintage'].iloc[0]}",
        f"Countries: {data['iso'].nunique()}",
        f"Historical records: {len(data)}",
        f"Numeric observations: {len(numeric)}",
        f"Missing numeric observations: {data[VALUE_COL].isna().sum()}",
        f"Duplicate country-year records: {duplicate_count}",
        "",
        "DIAGNOSTIC DESIGN",
        "-" * 76,
        "Inflation level: mean, median, minimum and maximum.",
        "Dispersion: standard deviation, MAD and IQR.",
        "Persistence: lag-one autocorrelation and positive-inflation episode duration.",
        "Acceleration: annual changes in inflation, in percentage points.",
        "Deflation: negative annual inflation and consecutive episode duration.",
        "Shock detection: country-specific robust distribution of annual changes.",
        "No missing inflation values were imputed.",
        "",
        "IMPORTANT INTERPRETATION NOTES",
        "-" * 76,
        "Positive inflation is not automatically excessive inflation.",
        "Deflation frequency is not a complete measure of economic weakness.",
        "Autocorrelation is descriptive and does not establish causation.",
        "Inflation shocks are statistical outliers, not necessarily economic crises.",
        "The 10% and 20% flags are supplementary fixed-threshold indicators.",
        "Historical averages can be dominated by high-inflation transition episodes.",
        "Country-period comparisons remain sensitive to differing historical regimes.",
        "",
        "COUNTRY-PERIOD DIAGNOSTICS",
        "-" * 76,
        f"Records generated: {len(diagnostics)}",
        f"Periods: {len(PERIODS)}",
        "",
        "2020-2024 INFLATION LEVELS",
        "-" * 76,
    ]

    recent = diagnostics[
        diagnostics["period"] == "pandemic_post_pandemic_2020_2024"
    ].sort_values("mean_inflation_pct", ascending=False)

    for row in recent.itertuples(index=False):
        report_lines.append(
            f"{row.iso} {row.country}: "
            f"mean={row.mean_inflation_pct:.3f}%; "
            f"median={row.median_inflation_pct:.3f}%; "
            f"max={row.maximum_inflation_pct:.3f}%; "
            f"deflation years={row.deflation_years}"
        )

    report_lines.extend([
        "",
        "DEFLATION EPISODES",
        "-" * 76,
        f"Total episodes: {len(deflation_episodes)}",
        "",
        "COUNTRIES WITH MOST STATISTICAL INFLATION SHOCKS",
        "-" * 76,
    ])

    for iso, count in shock_counts.head(10).items():
        report_lines.append(f"{iso}: {int(count)}")

    report_lines.extend([
        "",
        "VALIDATION",
        "-" * 76,
        "PASS: Required columns present.",
        "PASS: Historical actual observations only.",
        "PASS: Inflation unit and scale validated.",
        "PASS: No duplicate country-year observations.",
        "PASS: Missingness flags agree with numeric values.",
        "PASS: Deflation flags agree with numeric values.",
        "PASS: 20 countries represented.",
        "PASS: 80 country-period records generated.",
        "PASS: Annual dynamics preserve input row count.",
        "PASS: Deflation episode durations are valid.",
        "",
        "OUTPUTS CREATED",
        "-" * 76,
        str(DIAGNOSTICS_FILE),
        str(DYNAMICS_FILE),
        str(DEFLATION_FILE),
        str(REPORT_FILE),
        "",
        "=" * 76,
        "INFLATION DIAGNOSTICS VALIDATION PASSED",
        "=" * 76,
    ])

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines[-35:]))


if __name__ == "__main__":
    main()
