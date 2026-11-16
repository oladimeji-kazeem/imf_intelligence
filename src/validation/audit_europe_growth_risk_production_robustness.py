
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

GROWTH_FILE = ROOT / "data/processed/europe_growth_risk_input_panel.csv"
EPISODES_FILE = ROOT / "data/processed/europe_growth_contraction_episodes.csv"
BASELINE_FILE = ROOT / "data/processed/europe_growth_risk_classifications.csv"

OUTPUT_FILE = ROOT / "data/processed/europe_growth_risk_production_robustness.csv"
DETAIL_FILE = ROOT / "data/processed/europe_growth_risk_production_robustness_detail.csv"
REPORT_FILE = ROOT / "outputs/reports/europe_growth_risk_production_robustness_report.txt"

CLASS_ORDER = {
    "lower_risk": 1,
    "moderate_risk": 2,
    "higher_risk": 3,
}

CLASS_LABELS = ["lower_risk", "moderate_risk", "higher_risk"]

EXPECTED_COUNTRIES = 20
MIN_RECOVERY_EPISODES = [3, 5]


# ============================================================
# GENERAL HELPERS
# ============================================================

def require_columns(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def classify_risk_percentile(percentile):
    if pd.isna(percentile):
        return np.nan
    if percentile >= (2 / 3):
        return "higher_risk"
    if percentile <= (1 / 3):
        return "lower_risk"
    return "moderate_risk"


def percentile_rank(values):
    """
    Higher values represent higher risk.
    Average ranks are used for ties.
    """
    values = pd.Series(values, dtype="float64")
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.notna()

    if valid.sum() > 0:
        result.loc[valid] = values.loc[valid].rank(
            method="average",
            pct=True,
            ascending=True
        )

    return result


def build_scenario(
    metrics,
    scenario_name,
    dimension,
    risk_metric,
    baseline_percentile_column,
    baseline_class_column,
    note
):
    """
    metrics must contain ISO and risk_metric.
    Higher risk_metric must always mean greater risk.
    """
    frame = metrics[["ISO", risk_metric]].copy()
    frame = frame.rename(columns={risk_metric: "alternative_risk_metric"})

    frame["scenario"] = scenario_name
    frame["dimension"] = dimension
    frame["alternative_risk_percentile"] = percentile_rank(
        frame["alternative_risk_metric"]
    )
    frame["alternative_risk_class"] = (
        frame["alternative_risk_percentile"].apply(classify_risk_percentile)
    )

    frame = frame.merge(
        baseline[
            [
                "iso",
                baseline_percentile_column,
                baseline_class_column
            ]
        ],
        left_on="ISO",
        right_on="iso",
        how="left",
        validate="one_to_one"
    )

    frame = frame.rename(
        columns={
            baseline_percentile_column: "production_risk_percentile",
            baseline_class_column: "production_risk_class"
        }
    )

    frame["class_changed"] = np.where(
        frame["alternative_risk_class"].notna()
        & frame["production_risk_class"].notna(),
        frame["alternative_risk_class"] != frame["production_risk_class"],
        np.nan
    )

    frame["note"] = note

    return frame[
        [
            "ISO",
            "scenario",
            "dimension",
            "alternative_risk_metric",
            "alternative_risk_percentile",
            "alternative_risk_class",
            "production_risk_percentile",
            "production_risk_class",
            "class_changed",
            "note"
        ]
    ]


def compare_scenario(detail, scenario_name):
    subset = detail.loc[detail["scenario"] == scenario_name].copy()

    comparable = subset.dropna(
        subset=[
            "alternative_risk_percentile",
            "production_risk_percentile",
            "alternative_risk_class",
            "production_risk_class"
        ]
    )

    n = len(comparable)

    if n >= 3:
        correlation = spearmanr(
            comparable["alternative_risk_percentile"],
            comparable["production_risk_percentile"]
        ).statistic
    else:
        correlation = np.nan

    changed = int(comparable["class_changed"].sum()) if n else 0
    unchanged = n - changed

    return {
        "scenario": scenario_name,
        "dimension": subset["dimension"].iloc[0],
        "countries_available": len(subset),
        "countries_compared": n,
        "countries_unchanged": unchanged,
        "countries_changed": changed,
        "classification_stability_pct": (
            unchanged / n * 100 if n else np.nan
        ),
        "spearman_vs_production_percentile": correlation,
    }


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

for file_path in [GROWTH_FILE, EPISODES_FILE, BASELINE_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required input file not found: {file_path}")

growth = pd.read_csv(GROWTH_FILE)
episodes = pd.read_csv(EPISODES_FILE)
baseline = pd.read_csv(BASELINE_FILE)

require_columns(
    growth,
    ["ISO", "country", "year", "growth_pct", "is_contraction"],
    "Growth input panel"
)

require_columns(
    episodes,
    [
        "ISO",
        "country",
        "contraction_years",
        "worst_annual_growth_pct",
        "recovery_status"
    ],
    "Contraction episode file"
)

require_columns(
    baseline,
    [
        "iso",
        "country",
        "long_run_growth_weakness_risk_percentile",
        "long_run_growth_weakness_risk_class",
        "recent_growth_weakness_risk_percentile",
        "recent_growth_weakness_risk_class",
        "growth_instability_risk_percentile",
        "growth_instability_risk_class",
        "contraction_frequency_risk_percentile",
        "contraction_frequency_risk_class",
        "contraction_severity_risk_percentile",
        "contraction_severity_risk_class",
        "recovery_weakness_risk_percentile",
        "recovery_weakness_risk_class",
        "recovery_assessable_episodes",
        "recovery_failures",
        "recovery_failure_rate"
    ],
    "Production classification file"
)

growth["ISO"] = growth["ISO"].astype(str).str.upper()
episodes["ISO"] = episodes["ISO"].astype(str).str.upper()
baseline["iso"] = baseline["iso"].astype(str).str.upper()

growth["year"] = pd.to_numeric(growth["year"], errors="coerce")
growth["growth_pct"] = pd.to_numeric(
    growth["growth_pct"], errors="coerce"
)

episodes["contraction_years"] = pd.to_numeric(
    episodes["contraction_years"], errors="coerce"
)
episodes["worst_annual_growth_pct"] = pd.to_numeric(
    episodes["worst_annual_growth_pct"], errors="coerce"
)

if baseline["iso"].duplicated().any():
    raise ValueError("Duplicate country records in production classification file.")

if growth.duplicated(["ISO", "year"]).any():
    raise ValueError("Duplicate ISO-year records in growth input panel.")

if episodes.duplicated(["ISO", "contraction_start_year"]).any():
    raise ValueError("Duplicate country-contraction-start records in episode file.")

if set(baseline["iso"]) != set(growth["ISO"].unique()):
    raise ValueError("Country coverage differs between baseline and growth input.")

if set(baseline["iso"]) != set(episodes["ISO"].unique()):
    raise ValueError("Country coverage differs between baseline and episode file.")

baseline = baseline.sort_values("iso").reset_index(drop=True)

growth = growth.loc[
    growth["growth_pct"].notna()
].copy()

print("=" * 70)
print("EUROPEAN GROWTH RISK PRODUCTION ROBUSTNESS AUDIT")
print("=" * 70)
print(f"Growth observations: {len(growth)}")
print(f"Countries: {growth['ISO'].nunique()}")
print(f"Contraction episodes: {len(episodes)}")
print(f"Production country profiles: {len(baseline)}")


# ============================================================
# SCENARIO 1 — LONG-RUN MEDIAN GROWTH
# ============================================================

long_run = growth.loc[
    growth["year"].between(1980, 2025)
].copy()

long_run_metrics = (
    long_run.groupby("ISO", as_index=False)
    .agg(long_run_mean=("growth_pct", "mean"),
         long_run_median=("growth_pct", "median"),
         observations=("growth_pct", "count"))
)

scenario_long_median = build_scenario(
    long_run_metrics.assign(risk_metric=-long_run_metrics["long_run_median"]),
    "long_run_median_growth",
    "long_run_growth_weakness",
    "risk_metric",
    "long_run_growth_weakness_risk_percentile",
    "long_run_growth_weakness_risk_class",
    "Uses median rather than mean growth over 1980-2025."
)


# ============================================================
# SCENARIO 2 — LONG-RUN MEAN EXCLUDING IRELAND 2015
# ============================================================

long_run_ex_ireland = long_run.loc[
    ~((long_run["ISO"] == "IRL") & (long_run["year"] == 2015))
].copy()

long_run_ex_metrics = (
    long_run_ex_ireland.groupby("ISO", as_index=False)
    .agg(mean_growth=("growth_pct", "mean"),
         observations=("growth_pct", "count"))
)

scenario_ireland = build_scenario(
    long_run_ex_metrics.assign(risk_metric=-long_run_ex_metrics["mean_growth"]),
    "long_run_mean_ex_ireland_2015",
    "long_run_growth_weakness",
    "risk_metric",
    "long_run_growth_weakness_risk_percentile",
    "long_run_growth_weakness_risk_class",
    "Recalculates long-run mean after excluding Ireland's 2015 observation."
)


# ============================================================
# SCENARIO 3 — RECENT MEDIAN GROWTH
# ============================================================

recent = growth.loc[
    growth["year"].between(2020, 2024)
].copy()

recent_metrics = (
    recent.groupby("ISO", as_index=False)
    .agg(recent_mean=("growth_pct", "mean"),
         recent_median=("growth_pct", "median"),
         observations=("growth_pct", "count"))
)

scenario_recent_median = build_scenario(
    recent_metrics.assign(risk_metric=-recent_metrics["recent_median"]),
    "recent_median_growth",
    "recent_growth_weakness",
    "risk_metric",
    "recent_growth_weakness_risk_percentile",
    "recent_growth_weakness_risk_class",
    "Uses median rather than mean growth over 2020-2024."
)


# ============================================================
# SCENARIO 4 — MAD VOLATILITY
# ============================================================

volatility = growth.loc[
    growth["year"].between(2000, 2024)
].copy()

volatility_metrics = (
    volatility.groupby("ISO", as_index=False)
    .agg(
        standard_deviation=("growth_pct", "std"),
        median_growth=("growth_pct", "median")
    )
)

volatility_metrics["mad"] = (
    volatility.groupby("ISO")["growth_pct"]
    .apply(
        lambda values: np.median(
            np.abs(values - np.median(values))
        )
    )
    .values
)

scenario_mad = build_scenario(
    volatility_metrics.assign(risk_metric=volatility_metrics["mad"]),
    "volatility_mad",
    "growth_instability",
    "risk_metric",
    "growth_instability_risk_percentile",
    "growth_instability_risk_class",
    "Uses median absolute deviation instead of standard deviation."
)


# ============================================================
# SCENARIO 5 — COMMON-PERIOD CONTRACTION FREQUENCY
# ============================================================

frequency_period = growth.loc[
    growth["year"].between(2000, 2024)
].copy()

frequency_metrics = (
    frequency_period.groupby("ISO", as_index=False)
    .agg(
        contraction_years=("is_contraction", "sum"),
        observed_years=("growth_pct", "count")
    )
)

frequency_metrics["risk_metric"] = (
    frequency_metrics["contraction_years"]
    / frequency_metrics["observed_years"]
)

scenario_frequency = build_scenario(
    frequency_metrics,
    "contraction_frequency_2000_2024",
    "contraction_frequency",
    "risk_metric",
    "contraction_frequency_risk_percentile",
    "contraction_frequency_risk_class",
    "Uses contraction frequency over the common 2000-2024 period."
)


# ============================================================
# SCENARIO 6 — MEDIAN-BASED CONTRACTION SEVERITY
# ============================================================

episode_metrics = (
    episodes.groupby("ISO", as_index=False)
    .agg(
        mean_duration=("contraction_years", "mean"),
        median_duration=("contraction_years", "median"),
        mean_worst_growth=("worst_annual_growth_pct", "mean"),
        median_worst_growth=("worst_annual_growth_pct", "median"),
        episode_count=("contraction_years", "count")
    )
)

# Positive depth means a more severe negative growth outcome.
episode_metrics["median_depth"] = -episode_metrics["median_worst_growth"]

duration_percentile = percentile_rank(
    episode_metrics.set_index("ISO")["median_duration"]
)
depth_percentile = percentile_rank(
    episode_metrics.set_index("ISO")["median_depth"]
)

severity_metrics = episode_metrics.copy()
severity_metrics["duration_percentile"] = (
    severity_metrics["ISO"].map(duration_percentile)
)
severity_metrics["depth_percentile"] = (
    severity_metrics["ISO"].map(depth_percentile)
)
severity_metrics["risk_metric"] = (
    severity_metrics["duration_percentile"]
    + severity_metrics["depth_percentile"]
) / 2

scenario_severity = build_scenario(
    severity_metrics,
    "contraction_median_severity",
    "contraction_severity",
    "risk_metric",
    "contraction_severity_risk_percentile",
    "contraction_severity_risk_class",
    "Averages peer percentiles of median episode duration and median worst-year depth."
)


# ============================================================
# SCENARIOS 7 AND 8 — RECOVERY MINIMUM SAMPLE TESTS
# ============================================================

recovery_metrics = baseline[
    [
        "iso",
        "recovery_assessable_episodes",
        "recovery_failures",
        "recovery_failure_rate"
    ]
].copy()

recovery_metrics = recovery_metrics.rename(columns={"iso": "ISO"})

recovery_scenarios = []

for minimum in MIN_RECOVERY_EPISODES:
    eligible = recovery_metrics.loc[
        recovery_metrics["recovery_assessable_episodes"] >= minimum
    ].copy()

    eligible["risk_metric"] = eligible["recovery_failure_rate"]

    scenario = build_scenario(
        eligible,
        f"recovery_minimum_{minimum}_assessable_episodes",
        "recovery_weakness",
        "risk_metric",
        "recovery_weakness_risk_percentile",
        "recovery_weakness_risk_class",
        (
            "Reclassifies only countries with at least "
            f"{minimum} assessable contraction episodes."
        )
    )

    recovery_scenarios.append(scenario)


# ============================================================
# COMBINE SCENARIOS
# ============================================================

scenario_frames = [
    scenario_long_median,
    scenario_ireland,
    scenario_recent_median,
    scenario_mad,
    scenario_frequency,
    scenario_severity,
    *recovery_scenarios
]

detail = pd.concat(scenario_frames, ignore_index=True)

summary_rows = []

for scenario_name in detail["scenario"].unique():
    summary_rows.append(compare_scenario(detail, scenario_name))

summary = pd.DataFrame(summary_rows)


# ============================================================
# ADD PRODUCTION BASELINE REFERENCE
# ============================================================

baseline_dimensions = [
    (
        "long_run_growth_weakness",
        "long_run_growth_weakness_risk_percentile",
        "long_run_growth_weakness_risk_class"
    ),
    (
        "recent_growth_weakness",
        "recent_growth_weakness_risk_percentile",
        "recent_growth_weakness_risk_class"
    ),
    (
        "growth_instability",
        "growth_instability_risk_percentile",
        "growth_instability_risk_class"
    ),
    (
        "contraction_frequency",
        "contraction_frequency_risk_percentile",
        "contraction_frequency_risk_class"
    ),
    (
        "contraction_severity",
        "contraction_severity_risk_percentile",
        "contraction_severity_risk_class"
    ),
    (
        "recovery_weakness",
        "recovery_weakness_risk_percentile",
        "recovery_weakness_risk_class"
    )
]

baseline_summary = []

for dimension, percentile_col, class_col in baseline_dimensions:
    values = baseline[percentile_col].dropna()

    baseline_summary.append({
        "scenario": "production_baseline",
        "dimension": dimension,
        "countries_available": len(values),
        "countries_compared": len(values),
        "countries_unchanged": len(values),
        "countries_changed": 0,
        "classification_stability_pct": 100.0 if len(values) else np.nan,
        "spearman_vs_production_percentile": 1.0 if len(values) else np.nan
    })

baseline_summary = pd.DataFrame(baseline_summary)


# ============================================================
# VALIDATION
# ============================================================

checks = []

checks.append((
    "Expected 20-country production baseline",
    len(baseline) == EXPECTED_COUNTRIES
))

checks.append((
    "Eight alternative scenarios generated",
    detail["scenario"].nunique() == 8
))

checks.append((
    "No duplicate country-scenario records",
    not detail.duplicated(["ISO", "scenario"]).any()
))

checks.append((
    "All six production dimensions represented",
    set(detail["dimension"].unique()) == {
        "long_run_growth_weakness",
        "recent_growth_weakness",
        "growth_instability",
        "contraction_frequency",
        "contraction_severity",
        "recovery_weakness"
    }
))

checks.append((
    "Alternative classes use permitted labels",
    set(detail["alternative_risk_class"].dropna().unique()).issubset(
        set(CLASS_LABELS)
    )
))

checks.append((
    "Production classes use permitted labels",
    set(detail["production_risk_class"].dropna().unique()).issubset(
        set(CLASS_LABELS)
    )
))

checks.append((
    "No negative episode durations",
    (episodes["contraction_years"].dropna() > 0).all()
))

checks.append((
    "Recovery thresholds reduce or preserve eligible coverage",
    len(
        recovery_metrics.loc[
            recovery_metrics["recovery_assessable_episodes"] >= 5
        ]
    )
    <=
    len(
        recovery_metrics.loc[
            recovery_metrics["recovery_assessable_episodes"] >= 3
        ]
    )
))

checks.append((
    "Every scenario summary has a valid comparison count",
    (summary["countries_compared"] <= EXPECTED_COUNTRIES).all()
))

# Validate all alternative scenario records reconcile to the output.
checks.append((
    "Scenario output row count reconciles",
    len(detail) == sum(
        len(frame) for frame in scenario_frames
    )
))

# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

summary_output = pd.concat(
    [baseline_summary, summary],
    ignore_index=True
)

detail.to_csv(DETAIL_FILE, index=False)
summary_output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

lines = []

lines.append("=" * 78)
lines.append("EUROPEAN GROWTH RISK PRODUCTION ROBUSTNESS AUDIT")
lines.append("=" * 78)
lines.append("")
lines.append(f"Countries: {baseline['iso'].nunique()}")
lines.append(f"Growth observations: {len(growth)}")
lines.append(f"Contraction episodes: {len(episodes)}")
lines.append(f"Alternative scenario records: {len(detail)}")
lines.append("")
lines.append("PRODUCTION CLASSIFICATION ROBUSTNESS")
lines.append("-" * 78)

for _, row in summary.iterrows():
    lines.append(
        f"{row['scenario']}: "
        f"dimension={row['dimension']}; "
        f"compared={row['countries_compared']}; "
        f"changed={row['countries_changed']}; "
        f"stability={row['classification_stability_pct']:.1f}% "
        if pd.notna(row["classification_stability_pct"])
        else (
            f"{row['scenario']}: "
            f"dimension={row['dimension']}; "
            f"compared={row['countries_compared']}; "
            "stability=not available"
        )
    )

    lines[-1] += (
        f"; Spearman={row['spearman_vs_production_percentile']:.4f}"
        if pd.notna(row["spearman_vs_production_percentile"])
        else "; Spearman=not available"
    )

lines.append("")
lines.append("COUNTRY CLASSIFICATION CHANGES")
lines.append("-" * 78)

changed_rows = detail.loc[
    detail["class_changed"] == True
].sort_values(["dimension", "scenario", "ISO"])

if changed_rows.empty:
    lines.append("No alternative scenario changed a production classification.")
else:
    for _, row in changed_rows.iterrows():
        lines.append(
            f"{row['ISO']} | {row['scenario']} | "
            f"{row['production_risk_class']} -> "
            f"{row['alternative_risk_class']}"
        )

lines.append("")
lines.append("RECOVERY SAMPLE COVERAGE")
lines.append("-" * 78)

for minimum in MIN_RECOVERY_EPISODES:
    eligible = recovery_metrics.loc[
        recovery_metrics["recovery_assessable_episodes"] >= minimum
    ]
    lines.append(
        f"Minimum {minimum} assessable episodes: "
        f"{len(eligible)} of {len(recovery_metrics)} countries eligible"
    )

lines.append("")
lines.append("DIMENSIONS NOT FULLY TESTED")
lines.append("-" * 78)
lines.append(
    "Recovery tests change minimum sample eligibility but do not establish "
    "statistical precision or confidence intervals."
)
lines.append(
    "Contraction frequency common-period comparison changes the time window; "
    "it is not a test of an alternative frequency definition."
)
lines.append(
    "No absolute-risk thresholds or out-of-sample predictive tests are included."
)
lines.append(
    "Peer-relative tercile classes remain dependent on the selected country group."
)

lines.append("")
lines.append("VALIDATION")
lines.append("-" * 78)

for label, passed in checks:
    lines.append(f"{label}: {'PASS' if passed else 'FAIL'}")

all_passed = all(passed for _, passed in checks)
lines.append("")
lines.append(
    "VALIDATION PASSED" if all_passed else "VALIDATION FAILED"
)

REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

print("")
print("SCENARIO COMPARISONS")
print(summary.to_string(index=False))

print("")
print("VALIDATION")
for label, passed in checks:
    print(f"{label}: {'PASS' if passed else 'FAIL'}")

print("")
print("OUTPUTS CREATED")
print(OUTPUT_FILE)
print(DETAIL_FILE)
print(REPORT_FILE)

print("")
print("VALIDATION PASSED" if all_passed else "VALIDATION FAILED")

if not all_passed:
    raise SystemExit(1)
