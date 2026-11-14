from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PROFILE_FILE = ROOT / "data/processed/europe_country_indicator_profiles.csv"
SNAPSHOT_FILE = ROOT / "data/processed/europe_latest_economic_snapshot.csv"
COVERAGE_FILE = ROOT / "outputs/reports/europe_historical_coverage.csv"

OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

# Indicators for which a directional ranking is reasonably interpretable.
# Current account and trade growth are deliberately excluded.
RANKING_RULES = {
    "real_gdp_growth": {
        "direction": "higher_is_better",
        "ascending": False,
        "interpretation": "Higher real GDP growth",
    },
    "inflation": {
        "direction": "lower_is_better",
        "ascending": True,
        "interpretation": "Lower CPI inflation",
    },
    "government_debt": {
        "direction": "lower_is_better",
        "ascending": True,
        "interpretation": "Lower government debt-to-GDP",
    },
    "fiscal_balance": {
        "direction": "higher_is_better",
        "ascending": False,
        "interpretation": "Stronger overall fiscal balance",
    },
    "primary_balance": {
        "direction": "higher_is_better",
        "ascending": False,
        "interpretation": "Stronger primary fiscal balance",
    },
    "unemployment": {
        "direction": "lower_is_better",
        "ascending": True,
        "interpretation": "Lower unemployment",
    },
}


def stop(message):
    sys.exit(f"\nERROR: {message}")


# --------------------------------------------------
# 1. Load input datasets
# --------------------------------------------------

print("\nEUROPEAN ECONOMIC COMPARISON ENGINE")
print("=" * 60)

for path in [PROFILE_FILE, SNAPSHOT_FILE, COVERAGE_FILE]:
    if not path.exists():
        stop(f"Required input file not found: {path}")

profiles = pd.read_csv(PROFILE_FILE)
snapshot = pd.read_csv(SNAPSHOT_FILE)
coverage = pd.read_csv(COVERAGE_FILE)

print("\nInput records:")
print("Profiles:", len(profiles))
print("Latest snapshot:", len(snapshot))
print("Coverage:", len(coverage))


# --------------------------------------------------
# 2. Validate required columns
# --------------------------------------------------

profile_required = [
    "iso", "country", "indicator", "weo_code",
    "observations", "first_observation_year",
    "latest_observation_year", "mean", "median",
    "standard_deviation", "minimum", "maximum",
    "unit", "scale",
]

snapshot_required = [
    "iso", "country", "indicator", "weo_code",
    "latest_observation_year", "latest_value",
    "unit", "scale",
]

coverage_required = [
    "iso", "country", "indicator",
    "available_observations", "first_year", "latest_year",
]

for name, frame, required in [
    ("Profiles", profiles, profile_required),
    ("Snapshot", snapshot, snapshot_required),
    ("Coverage", coverage, coverage_required),
]:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        stop(f"{name} is missing columns: {missing}")


# --------------------------------------------------
# 3. Validate keys and numeric fields
# --------------------------------------------------

KEY = ["iso", "indicator"]

for name, frame in [
    ("Profiles", profiles),
    ("Snapshot", snapshot),
    ("Coverage", coverage),
]:
    if frame.duplicated(KEY).any():
        stop(f"Duplicate country-indicator records in {name}.")

numeric_columns = {
    "profiles": [
        "observations", "first_observation_year",
        "latest_observation_year", "mean", "median",
        "standard_deviation", "minimum", "maximum",
    ],
    "snapshot": ["latest_observation_year", "latest_value"],
    "coverage": [
        "available_observations", "first_year", "latest_year"
    ],
}

for column in numeric_columns["profiles"]:
    profiles[column] = pd.to_numeric(profiles[column], errors="coerce")

for column in numeric_columns["snapshot"]:
    snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")

for column in numeric_columns["coverage"]:
    coverage[column] = pd.to_numeric(coverage[column], errors="coerce")

if snapshot["latest_value"].isna().any():
    stop("Latest snapshot contains missing economic values.")

if profiles["observations"].isna().any():
    stop("Invalid observation counts in profiles.")

if coverage["available_observations"].isna().any():
    stop("Invalid observation counts in coverage.")


# --------------------------------------------------
# 4. Match the three datasets
# --------------------------------------------------

profiles = profiles.rename(
    columns={
        "latest_observation_year": "profile_latest_year",
        "first_observation_year": "profile_first_year",
        "observations": "profile_observations",
    }
)

snapshot = snapshot.rename(
    columns={
        "latest_observation_year": "snapshot_year",
    }
)

coverage = coverage.rename(
    columns={
        "first_year": "coverage_first_year",
        "latest_year": "coverage_latest_year",
        "available_observations": "coverage_observations",
    }
)

merge_key = ["iso", "country", "indicator", "weo_code", "unit", "scale"]

comparison = profiles.merge(
    snapshot,
    on=merge_key,
    how="inner",
    validate="one_to_one",
)

comparison = comparison.merge(
    coverage,
    on=["iso", "country", "indicator"],
    how="inner",
    validate="one_to_one",
)

if len(comparison) != 220:
    stop(
        f"Expected 220 matched profiles; obtained {len(comparison)}."
    )

# Verify that the three datasets describe the same sample.
comparison["snapshot_year_consistent"] = (
    comparison["snapshot_year"]
    == comparison["profile_latest_year"]
)

comparison["coverage_years_consistent"] = (
    (comparison["profile_first_year"] == comparison["coverage_first_year"])
    &
    (comparison["profile_latest_year"] == comparison["coverage_latest_year"])
)

comparison["observation_count_consistent"] = (
    comparison["profile_observations"]
    == comparison["coverage_observations"]
)

# --------------------------------------------------
# 5. Create latest economic country comparison
# --------------------------------------------------

comparison = comparison.sort_values(
    ["indicator", "country"]
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    OUTPUT_DIR / "europe_country_comparison.csv",
    index=False,
)


# --------------------------------------------------
# 6. Generate indicator-specific rankings
# --------------------------------------------------

ranking_frames = []

for indicator, rule in RANKING_RULES.items():

    subset = comparison[
        comparison["indicator"] == indicator
    ].copy()

    if subset.empty:
        print(f"WARNING: No records found for {indicator}")
        continue

    # Rank only countries whose latest observation is from
    # the same calendar year.
    for observation_year, group in subset.groupby("snapshot_year"):

        group = group.copy()

        group["rank"] = group["latest_value"].rank(
            method="min",
            ascending=rule["ascending"],
        ).astype(int)

        group["ranking_indicator"] = indicator
        group["ranking_direction"] = rule["direction"]
        group["ranking_interpretation"] = rule["interpretation"]
        group["ranking_basis"] = "Latest historical value; same-year comparison"
        group["reference_year"] = int(observation_year)
        group["countries_compared"] = len(group)

        ranking_frames.append(
            group[
                [
                    "iso", "country", "ranking_indicator",
                    "latest_value", "unit", "scale",
                    "reference_year", "rank",
                    "ranking_direction",
                    "ranking_interpretation",
                    "ranking_basis", "countries_compared",
                ]
            ]
        )

if ranking_frames:
    rankings = pd.concat(ranking_frames, ignore_index=True)
else:
    rankings = pd.DataFrame()

rankings.to_csv(
    OUTPUT_DIR / "europe_economic_rankings.csv",
    index=False,
)


# --------------------------------------------------
# 7. Historical economic volatility
# --------------------------------------------------

volatility = comparison[
    [
        "iso", "country", "indicator",
        "standard_deviation", "mean",
        "profile_observations",
        "profile_first_year", "profile_latest_year",
        "unit", "scale",
    ]
].copy()

# Coefficient of variation is not meaningful where the mean
# is zero or close to zero, or where values may be negative.
volatility["coefficient_of_variation"] = pd.NA

valid_cv = (
    volatility["mean"].notna()
    &
    (volatility["mean"].abs() > 1e-10)
    &
    (volatility["mean"] > 0)
)

volatility.loc[valid_cv, "coefficient_of_variation"] = (
    volatility.loc[valid_cv, "standard_deviation"]
    / volatility.loc[valid_cv, "mean"].abs()
)

# Lower standard deviation indicates lower absolute historical
# variability within the same indicator. It does not prove resilience.
volatility["variability_order"] = (
    volatility.groupby("indicator")["standard_deviation"]
    .rank(method="min", ascending=True)
)

volatility["variability_interpretation"] = (
    "Lower standard deviation means lower historical variability; "
    "sample windows may differ."
)

volatility.to_csv(
    OUTPUT_DIR / "europe_economic_volatility.csv",
    index=False,
)


# --------------------------------------------------
# 8. Generate data integrity report
# --------------------------------------------------

checks = {
    "Matched profile records": len(comparison),
    "Snapshot year inconsistencies": int(
        (~comparison["snapshot_year_consistent"]).sum()
    ),
    "Coverage year inconsistencies": int(
        (~comparison["coverage_years_consistent"]).sum()
    ),
    "Observation count inconsistencies": int(
        (~comparison["observation_count_consistent"]).sum()
    ),
    "Ranking records": len(rankings),
}

report_file = REPORT_DIR / "europe_comparative_intelligence_report.txt"

with open(report_file, "w", encoding="utf-8") as report:

    report.write("EUROPEAN ECONOMIC COMPARATIVE INTELLIGENCE\n")
    report.write("=" * 60 + "\n\n")

    report.write("DATASET SUMMARY\n")
    report.write("-" * 60 + "\n")

    report.write(f"Countries: {comparison['iso'].nunique()}\n")
    report.write(f"Indicators: {comparison['indicator'].nunique()}\n")
    report.write(f"Country-indicator comparisons: {len(comparison)}\n")

    report.write("\nDATA INTEGRITY CHECKS\n")
    report.write("-" * 60 + "\n")

    for label, value in checks.items():
        report.write(f"{label}: {value}\n")

    report.write("\nLATEST SNAPSHOT YEAR DISTRIBUTION\n")
    report.write("-" * 60 + "\n")

    year_distribution = (
        comparison.groupby(
            ["indicator", "snapshot_year"]
        ).size()
    )

    report.write(year_distribution.to_string() + "\n")

    report.write("\nRANKING COVERAGE\n")
    report.write("-" * 60 + "\n")

    if not rankings.empty:
        ranking_coverage = (
            rankings.groupby(
                ["ranking_indicator", "reference_year"]
            )["iso"].nunique()
        )
        report.write(ranking_coverage.to_string() + "\n")

    report.write("\nMETHODOLOGICAL NOTES\n")
    report.write("-" * 60 + "\n")

    report.write(
        "Rankings are calculated separately by indicator and reference year.\n"
    )
    report.write(
        "Historical volatility is based on the available historical sample.\n"
    )
    report.write(
        "Historical sample windows differ between countries and indicators.\n"
    )
    report.write(
        "Current account and trade growth are not given directional rankings.\n"
    )
    report.write(
        "Nominal GDP is retained as economic size, not treated as a "
        "direct measure of economic strength.\n"
    )
    report.write(
        "These rankings are descriptive comparisons, not sovereign credit ratings.\n"
    )


# --------------------------------------------------
# 9. Display execution summary
# --------------------------------------------------

print("\nDATA INTEGRITY CHECKS")
print("---------------------")

for label, value in checks.items():
    print(f"{label}: {value}")

print("\nLATEST SNAPSHOT YEAR DISTRIBUTION")
print("---------------------------------")

print(
    comparison.groupby(
        ["indicator", "snapshot_year"]
    ).size().to_string()
)

print("\nRANKING COVERAGE")
print("----------------")

if not rankings.empty:
    print(
        rankings.groupby(
            ["ranking_indicator", "reference_year"]
        )["iso"].nunique().to_string()
    )

print("\nOUTPUT FILES")
print("------------")

for filename in [
    "europe_country_comparison.csv",
    "europe_economic_rankings.csv",
    "europe_economic_volatility.csv",
]:
    print(OUTPUT_DIR / filename)

print(report_file)

print("\nCOMPARATIVE INTELLIGENCE BUILD COMPLETED.")
