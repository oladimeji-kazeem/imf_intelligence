from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/europe_weo_classified.csv"
OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

INDICATOR = "LUR"
HIGH_UNEMPLOYMENT_THRESHOLD = 10.0

PERIODS = {
    "long_run_1980_2025": (1980, 2025),
    "modern_2000_2019": (2000, 2019),
    "pre_pandemic_2015_2019": (2015, 2019),
    "recent_2020_2024": (2020, 2024),
}


def mad(series):
    """Median absolute deviation around the median."""
    median = series.median()
    return (series - median).abs().median()


def longest_consecutive_run(years):
    """Longest run of consecutive calendar years."""
    if not years:
        return 0

    longest = 1
    current = 1
    previous = years[0]

    for year in years[1:]:
        if year == previous + 1:
            current += 1
        else:
            current = 1

        longest = max(longest, current)
        previous = year

    return longest


def build_annual_dynamics(df):
    rows = []

    for iso, group in df.groupby("iso"):
        group = group.sort_values("year").reset_index(drop=True)

        for i, row in group.iterrows():
            value = row["value"]
            year = int(row["year"])

            previous_value = np.nan
            previous_year = np.nan
            annual_change = np.nan

            if i > 0:
                previous = group.iloc[i - 1]

                if (
                    year == int(previous["year"]) + 1
                    and pd.notna(value)
                    and pd.notna(previous["value"])
                ):
                    previous_value = float(previous["value"])
                    previous_year = int(previous["year"])
                    annual_change = float(value) - previous_value

            rows.append({
                "iso": iso,
                "year": year,
                "unemployment_rate_pct": value,
                "previous_year": previous_year,
                "previous_unemployment_rate_pct": previous_value,
                "annual_change_pp": annual_change,
                "annual_change_direction": (
                    "increase" if annual_change > 0
                    else "decrease" if annual_change < 0
                    else "no_change" if pd.notna(annual_change)
                    else "unavailable"
                ),
                "high_unemployment_flag": (
                    bool(value >= HIGH_UNEMPLOYMENT_THRESHOLD)
                    if pd.notna(value) else np.nan
                ),
            })

    return pd.DataFrame(rows)


def build_high_unemployment_episodes(annual):
    episodes = []

    for iso, group in annual.groupby("iso"):
        group = group.sort_values("year").reset_index(drop=True)

        active = []
        previous_year = None

        def save_episode(years, values):
            if not years:
                return

            episodes.append({
                "iso": iso,
                "start_year": years[0],
                "end_year": years[-1],
                "duration_years": len(years),
                "mean_unemployment_pct": float(np.mean(values)),
                "maximum_unemployment_pct": float(np.max(values)),
                "threshold_pct": HIGH_UNEMPLOYMENT_THRESHOLD,
            })

        for _, row in group.iterrows():
            year = int(row["year"])
            value = row["unemployment_rate_pct"]

            qualifies = (
                pd.notna(value)
                and value >= HIGH_UNEMPLOYMENT_THRESHOLD
            )

            consecutive = (
                previous_year is not None
                and year == previous_year + 1
            )

            if not consecutive or not qualifies:
                if active:
                    save_episode(
                        [item[0] for item in active],
                        [item[1] for item in active],
                    )
                    active = []

            if qualifies:
                active.append((year, float(value)))

            previous_year = year

        if active:
            save_episode(
                [item[0] for item in active],
                [item[1] for item in active],
            )

    columns = [
        "iso", "start_year", "end_year", "duration_years",
        "mean_unemployment_pct", "maximum_unemployment_pct",
        "threshold_pct",
    ]

    return pd.DataFrame(episodes, columns=columns)


def build_period_diagnostics(annual):
    rows = []

    for iso, country_data in annual.groupby("iso"):
        country_data = country_data.sort_values("year")

        for period_name, (start_year, end_year) in PERIODS.items():
            period = country_data[
                country_data["year"].between(start_year, end_year)
            ].copy()

            expected_years = end_year - start_year + 1
            observed = period["unemployment_rate_pct"].dropna()
            numeric_count = len(observed)

            changes = period.loc[
                period["previous_year"].between(start_year, end_year), 
                "annual_change_pp",
            ].dropna()

            high_years = period.loc[
                period["unemployment_rate_pct"]
                >= HIGH_UNEMPLOYMENT_THRESHOLD,
                "year",
            ].astype(int).tolist()

            high_frequency = (
                len(high_years) / numeric_count * 100
                if numeric_count else np.nan
            )

            first_observed = (
                period.loc[
                    period["unemployment_rate_pct"].notna()
                ].iloc[0]
                if numeric_count else None
            )

            last_observed = (
                period.loc[
                    period["unemployment_rate_pct"].notna()
                ].iloc[-1]
                if numeric_count else None
            )

            first_last_change = np.nan
            if first_observed is not None and last_observed is not None:
                first_last_change = (
                    float(last_observed["unemployment_rate_pct"])
                    - float(first_observed["unemployment_rate_pct"])
                )

            rows.append({
                "iso": iso,
                "period": period_name,
                "period_start": start_year,
                "period_end": end_year,
                "expected_years": expected_years,
                "numeric_observations": numeric_count,
                "missing_observations": expected_years - numeric_count,
                "coverage_pct": numeric_count / expected_years * 100,
                "first_numeric_year": (
                    int(period.loc[
                        period["unemployment_rate_pct"].notna(),
                        "year",
                    ].min()) if numeric_count else np.nan
                ),
                "last_numeric_year": (
                    int(period.loc[
                        period["unemployment_rate_pct"].notna(),
                        "year",
                    ].max()) if numeric_count else np.nan
                ),
                "mean_unemployment_pct": (
                    observed.mean() if numeric_count else np.nan
                ),
                "median_unemployment_pct": (
                    observed.median() if numeric_count else np.nan
                ),
                "std_unemployment_pct": (
                    observed.std(ddof=1) if numeric_count > 1 else np.nan
                ),
                "mad_unemployment_pct": (
                    mad(observed) if numeric_count else np.nan
                ),
                "iqr_unemployment_pct": (
                    observed.quantile(0.75) - observed.quantile(0.25)
                    if numeric_count else np.nan
                ),
                "minimum_unemployment_pct": (
                    observed.min() if numeric_count else np.nan
                ),
                "maximum_unemployment_pct": (
                    observed.max() if numeric_count else np.nan
                ),
                "high_unemployment_years": len(high_years),
                "high_unemployment_frequency_pct": high_frequency,
                "longest_high_unemployment_run_years": (
                    longest_consecutive_run(high_years)
                ),
                "mean_annual_change_pp": (
                    changes.mean() if len(changes) else np.nan
                ),
                "mean_absolute_annual_change_pp": (
                    changes.abs().mean() if len(changes) else np.nan
                ),
                "maximum_absolute_annual_change_pp": (
                    changes.abs().max() if len(changes) else np.nan
                ),
                "first_last_observed_change_pp": first_last_change,
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 76)
    print("EUROPEAN LABOUR RISK DIAGNOSTICS")
    print("=" * 76)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(INPUT)

    required = {
        "iso", "year", "weo_code", "value", "observation_status"
    }
    missing_columns = required - set(source.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = source[
        (source["weo_code"] == INDICATOR)
        & (source["observation_status"] == "historical_actual")
    ].copy()

    if df.empty:
        raise ValueError("No historical LUR observations found.")

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if df.duplicated(["iso", "year"]).any():
        duplicates = df.loc[
            df.duplicated(["iso", "year"], keep=False),
            ["iso", "year"],
        ]
        raise ValueError(
            "Duplicate country-year observations detected:\n"
            f"{duplicates.to_string(index=False)}"
        )

    countries = sorted(df["iso"].unique())

    if len(countries) != 20:
        raise ValueError(
            f"Expected 20 countries; found {len(countries)}."
        )

    if (df["value"].dropna() < 0).any():
        raise ValueError("Negative unemployment rates detected.")

    annual = build_annual_dynamics(df)
    diagnostics = build_period_diagnostics(annual)
    episodes = build_high_unemployment_episodes(annual)

    # Output files
    annual_path = OUTPUT_DIR / "europe_labour_risk_annual_dynamics.csv"
    diagnostics_path = OUTPUT_DIR / "europe_labour_risk_diagnostics.csv"
    episodes_path = OUTPUT_DIR / "europe_labour_risk_high_unemployment_episodes.csv"
    report_path = REPORT_DIR / "europe_labour_risk_diagnostics_report.txt"

    annual.to_csv(annual_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    episodes.to_csv(episodes_path, index=False)

    # Validation
    expected_diagnostics = len(countries) * len(PERIODS)

    if len(diagnostics) != expected_diagnostics:
        raise ValueError(
            f"Expected {expected_diagnostics} diagnostic rows; "
            f"found {len(diagnostics)}."
        )

    if diagnostics.duplicated(["iso", "period"]).any():
        raise ValueError("Duplicate country-period diagnostics detected.")

    recent = diagnostics[
        diagnostics["period"] == "recent_2020_2024"
    ]

    if len(recent) != 20:
        raise ValueError("Recent period does not contain 20 countries.")

    if not (recent["numeric_observations"] == 5).all():
        raise ValueError(
            "Recent 2020–2024 period is not complete for all countries."
        )

    if not np.allclose(
        recent["coverage_pct"].to_numpy(),
        100.0,
    ):
        raise ValueError("Recent period coverage is not 100%.")

    # Report
    lines = [
        "=" * 76,
        "EUROPEAN LABOUR RISK DIAGNOSTICS REPORT",
        "=" * 76,
        "",
        "INPUT",
        "-" * 76,
        f"Source: {INPUT}",
        f"Indicator: {INDICATOR}",
        f"Countries: {len(countries)}",
        f"Annual records: {len(annual)}",
        f"Numeric observations: {annual['unemployment_rate_pct'].notna().sum()}",
        f"Missing observations: {annual['unemployment_rate_pct'].isna().sum()}",
        "",
        "PERIOD DIAGNOSTICS",
        "-" * 76,
        f"Country-period records: {len(diagnostics)}",
        diagnostics[
            [
                "period",
                "numeric_observations",
                "missing_observations",
                "coverage_pct",
            ]
        ].groupby("period").agg(
            ["min", "max", "mean"]
        ).round(2).to_string(),
        "",
        "RECENT UNEMPLOYMENT PRESSURE — 2020–2024",
        "-" * 76,
        recent[
            [
                "iso",
                "median_unemployment_pct",
                "mad_unemployment_pct",
                "iqr_unemployment_pct",
                "first_last_observed_change_pp",
            ]
        ].sort_values(
            "median_unemployment_pct",
            ascending=False,
        ).round(3).to_string(index=False),
        "",
        "HISTORICAL ELEVATED UNEMPLOYMENT — 2000–2019",
        "-" * 76,
        diagnostics[
            diagnostics["period"] == "modern_2000_2019"
        ][
            [
                "iso",
                "high_unemployment_frequency_pct",
                "longest_high_unemployment_run_years",
                "median_unemployment_pct",
            ]
        ].sort_values(
            "high_unemployment_frequency_pct",
            ascending=False,
        ).round(3).to_string(index=False),
        "",
        "HIGH-UNEMPLOYMENT EPISODES",
        "-" * 76,
        f"Threshold: {HIGH_UNEMPLOYMENT_THRESHOLD:.1f}% unemployment",
        f"Episode records: {len(episodes)}",
        (
            episodes.sort_values(
                ["duration_years", "mean_unemployment_pct"],
                ascending=[False, False],
            ).head(20).round(3).to_string(index=False)
            if not episodes.empty else "No episodes identified."
        ),
        "",
        "METHODOLOGICAL NOTES",
        "-" * 76,
        "Unemployment rates are expressed as percentages of the labour force.",
        "Annual changes are percentage-point changes, not percentage growth.",
        "MAD and IQR measure dispersion, not the probability of a crisis.",
        "The 10% threshold is a descriptive screening convention.",
        "Missing observations and calendar-year gaps break consecutive runs.",
        "Historical coverage differs by country.",
        "Unemployment alone does not measure total labour-market slack.",
        "No composite risk score or unemployment forecast is produced.",
        "",
        "VALIDATION",
        "-" * 76,
        "PASS: Required source columns present.",
        "PASS: LUR historical observations extracted.",
        "PASS: 20 countries identified.",
        "PASS: No duplicate country-year records.",
        "PASS: 80 country-period diagnostic records created.",
        "PASS: Recent 2020–2024 panel is complete for all 20 countries.",
        "PASS: No negative numeric unemployment rates.",
        "",
        "OUTPUTS",
        "-" * 76,
        str(annual_path),
        str(diagnostics_path),
        str(episodes_path),
        str(report_path),
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nDIAGNOSTICS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()