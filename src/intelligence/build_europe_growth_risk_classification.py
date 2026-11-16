
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PANEL = ROOT / "data/processed/europe_growth_risk_input_panel.csv"
INPUT_EPISODES = ROOT / "data/processed/europe_growth_contraction_episodes.csv"
INPUT_RECOVERY = ROOT / "data/processed/europe_growth_recovery_summary.csv"

OUTPUT_CLASSIFICATIONS = (
    ROOT / "data/processed/europe_growth_risk_classifications.csv"
)
OUTPUT_DETAIL = (
    ROOT / "data/processed/europe_growth_risk_dimension_detail.csv"
)
OUTPUT_REPORT = (
    ROOT / "outputs/reports/europe_growth_risk_classification_report.txt"
)

LONG_RUN_START = 1980
LONG_RUN_END = 2025
RECENT_START = 2020
RECENT_END = 2024
VOLATILITY_START = 2000
VOLATILITY_END = 2024

MIN_LONG_RUN_OBSERVATIONS = 20
MIN_VOLATILITY_OBSERVATIONS = 20


# ============================================================
# GENERAL UTILITIES
# ============================================================

def normalise_name(value):
    return (
        str(value).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "pct")
        .replace("/", "_")
    )


def find_column(df, candidates, description, required=True):
    lookup = {normalise_name(col): col for col in df.columns}

    for candidate in candidates:
        key = normalise_name(candidate)
        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            f"Could not identify {description}.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def classify_percentile(value):
    if pd.isna(value):
        return "insufficient_data"

    if value >= 2 / 3:
        return "higher_risk"

    if value <= 1 / 3:
        return "lower_risk"

    return "moderate_risk"


def peer_percentile(values):
    """
    Average percentile rank across the available peer group.
    Higher values always represent higher risk.
    """
    return values.rank(method="average", pct=True)


def write_report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# LOAD AND STANDARDISE INPUTS
# ============================================================

def load_growth_panel():
    if not INPUT_PANEL.exists():
        raise FileNotFoundError(INPUT_PANEL)

    df = pd.read_csv(INPUT_PANEL)

    iso_col = find_column(
        df,
        ["ISO", "iso", "iso_code", "country_iso", "country_code"],
        "country ISO code",
    )
    country_col = find_column(
        df,
        ["country", "country_name", "name"],
        "country name",
        required=False,
    )
    year_col = find_column(
        df,
        ["year", "time", "period"],
        "calendar year",
    )
    growth_col = find_column(
        df,
        [
            "real_gdp_growth_pct",
            "real_gdp_growth",
            "growth_pct",
            "value",
            "obs_value",
        ],
        "real GDP growth value",
    )

    result = pd.DataFrame({
        "iso": df[iso_col].astype(str).str.upper().str.strip(),
        "year": numeric_series(df[year_col]),
        "growth_pct": numeric_series(df[growth_col]),
    })

    if country_col:
        result["country"] = df[country_col].astype(str).str.strip()
    else:
        result["country"] = result["iso"]

    result = result.dropna(subset=["iso", "year"])
    result["year"] = result["year"].astype(int)

    if result.duplicated(["iso", "year"]).any():
        duplicates = result.loc[
            result.duplicated(["iso", "year"], keep=False)
        ]
        raise ValueError(
            "Duplicate country-year records in growth input:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return result


def load_episode_data():
    if not INPUT_EPISODES.exists():
        raise FileNotFoundError(INPUT_EPISODES)

    df = pd.read_csv(INPUT_EPISODES)

    iso_col = find_column(
        df, ["ISO", "iso", "iso_code"], "episode ISO code"
    )
    duration_col = find_column(
        df,
        ["contraction_years", "episode_duration"],
        "contraction duration",
    )
    worst_col = find_column(
        df,
        ["worst_annual_growth_pct", "worst_annual_growth"],
        "worst annual growth",
    )

    result = pd.DataFrame({
        "iso": df[iso_col].astype(str).str.upper().str.strip(),
        "contraction_years": numeric_series(df[duration_col]),
        "worst_annual_growth_pct": numeric_series(df[worst_col]),
    })

    return result


def load_recovery_data():
    if not INPUT_RECOVERY.exists():
        raise FileNotFoundError(INPUT_RECOVERY)

    df = pd.read_csv(INPUT_RECOVERY)

    required = {
        "iso": ["ISO", "iso", "iso_code"],
        "episodes": ["contraction_episodes"],
        "recovered": ["recovered_episodes"],
        "not_recovered": ["not_recovered_within_5_years"],
        "unassessable": ["unassessable_episodes"],
        "incomplete": ["incomplete_follow_up"],
    }

    selected = {
        key: find_column(df, candidates, key)
        for key, candidates in required.items()
    }

    result = pd.DataFrame({
        "iso": df[selected["iso"]].astype(str).str.upper().str.strip(),
        "contraction_episodes": numeric_series(
            df[selected["episodes"]]
        ),
        "recovered_episodes": numeric_series(
            df[selected["recovered"]]
        ),
        "not_recovered_within_5_years": numeric_series(
            df[selected["not_recovered"]]
        ),
        "unassessable_episodes": numeric_series(
            df[selected["unassessable"]]
        ),
        "incomplete_follow_up": numeric_series(
            df[selected["incomplete"]]
        ),
    })

    if result["iso"].duplicated().any():
        raise ValueError("Duplicate ISO codes in recovery summary.")

    return result


# ============================================================
# BUILD COUNTRY-LEVEL METRICS
# ============================================================

def build_country_metrics(panel, episodes, recovery):
    countries = (
        panel[["iso", "country"]]
        .drop_duplicates("iso")
        .sort_values("iso")
        .reset_index(drop=True)
    )

    rows = []

    for _, country_row in countries.iterrows():
        iso = country_row["iso"]
        country = country_row["country"]

        data = panel.loc[panel["iso"] == iso].copy()
        data = data.sort_values("year")

        long_run = data.loc[
            data["year"].between(LONG_RUN_START, LONG_RUN_END),
            "growth_pct",
        ].dropna()

        recent = data.loc[
            data["year"].between(RECENT_START, RECENT_END),
            "growth_pct",
        ].dropna()

        volatility = data.loc[
            data["year"].between(VOLATILITY_START, VOLATILITY_END),
            "growth_pct",
        ].dropna()

        observed = data.loc[
            data["year"].between(LONG_RUN_START, LONG_RUN_END),
            "growth_pct",
        ].dropna()

        negative_years = int((observed < 0).sum())

        episodes_country = episodes.loc[episodes["iso"] == iso]

        mean_duration = episodes_country[
            "contraction_years"
        ].mean()

        mean_worst_growth = episodes_country[
            "worst_annual_growth_pct"
        ].mean()

        # Positive risk magnitude: more negative annual growth
        # produces a larger downside severity measure.
        mean_contraction_depth = (
            -mean_worst_growth
            if pd.notna(mean_worst_growth)
            else np.nan
        )

        recovery_row = recovery.loc[recovery["iso"] == iso]

        if len(recovery_row) != 1:
            raise ValueError(
                f"Expected one recovery summary for {iso}; "
                f"found {len(recovery_row)}."
            )

        recovery_row = recovery_row.iloc[0]

        recovered = recovery_row["recovered_episodes"]
        not_recovered = recovery_row[
            "not_recovered_within_5_years"
        ]

        assessable = recovered + not_recovered

        recovery_failure_rate = (
            not_recovered / assessable
            if assessable > 0
            else np.nan
        )

        rows.append({
            "iso": iso,
            "country": country,

            "long_run_mean_growth_pct": long_run.mean(),
            "long_run_median_growth_pct": long_run.median(),
            "long_run_observations": len(long_run),

            "recent_mean_growth_pct": recent.mean(),
            "recent_median_growth_pct": recent.median(),
            "recent_observations": len(recent),

            "volatility_std_2000_2024_pct": volatility.std(ddof=1),
            "volatility_mad_2000_2024_pct": (
                (volatility - volatility.median()).abs().median()
                if len(volatility) else np.nan
            ),
            "volatility_observations": len(volatility),

            "contraction_years": negative_years,
            "observed_growth_years": len(observed),
            "contraction_frequency_pct": (
                negative_years / len(observed) * 100
                if len(observed) else np.nan
            ),

            "mean_contraction_duration_years": mean_duration,
            "mean_worst_annual_growth_pct": mean_worst_growth,
            "mean_contraction_depth_pct": mean_contraction_depth,
            "contraction_episode_count": len(episodes_country),

            "recovery_assessable_episodes": assessable,
            "recovery_failures": not_recovered,
            "recovery_failure_rate": recovery_failure_rate,
            "recovery_episodes": recovery_row["contraction_episodes"],
            "unassessable_episodes": recovery_row["unassessable_episodes"],
            "incomplete_follow_up": recovery_row["incomplete_follow_up"],
        })

    return pd.DataFrame(rows)


# ============================================================
# CLASSIFY RISK DIMENSIONS
# ============================================================

def classify_dimensions(metrics):
    result = metrics.copy()

    dimensions = {
        "long_run_growth_weakness": {
            "source": "long_run_mean_growth_pct",
            "reverse": True,
            "minimum": MIN_LONG_RUN_OBSERVATIONS,
            "count": "long_run_observations",
        },
        "recent_growth_weakness": {
            "source": "recent_mean_growth_pct",
            "reverse": True,
            "minimum": 5,
            "count": "recent_observations",
        },
        "growth_instability": {
            "source": "volatility_std_2000_2024_pct",
            "reverse": False,
            "minimum": MIN_VOLATILITY_OBSERVATIONS,
            "count": "volatility_observations",
        },
        "contraction_frequency": {
            "source": "contraction_frequency_pct",
            "reverse": False,
            "minimum": MIN_LONG_RUN_OBSERVATIONS,
            "count": "observed_growth_years",
        },
        "recovery_weakness": {
            "source": "recovery_failure_rate",
            "reverse": False,
            "minimum": 1,
            "count": "recovery_assessable_episodes",
        },
    }

    detail_rows = []

    for dimension, settings in dimensions.items():
        source = settings["source"]
        raw = result[source].copy()

        # Lower growth means higher risk for growth performance.
        risk_values = -raw if settings["reverse"] else raw

        enough_data = (
            result[settings["count"]] >= settings["minimum"]
        ) & raw.notna()

        risk_values = risk_values.where(enough_data)

        percentiles = peer_percentile(risk_values)

        result[f"{dimension}_risk_percentile"] = percentiles
        result[f"{dimension}_risk_class"] = percentiles.apply(
            classify_percentile
        )

        for index, row in result.iterrows():
            detail_rows.append({
                "iso": row["iso"],
                "country": row["country"],
                "dimension": dimension,
                "source_metric": source,
                "source_value": row[source],
                "risk_percentile": percentiles.loc[index],
                "risk_class": classify_percentile(
                    percentiles.loc[index]
                ),
                "peer_group_size": int(risk_values.notna().sum()),
            })

    # Contraction severity combines duration and depth.
    duration = result["mean_contraction_duration_years"]
    depth = result["mean_contraction_depth_pct"]

    duration_pct = peer_percentile(duration)
    depth_pct = peer_percentile(depth)

    severity_percentile = pd.concat(
        [duration_pct, depth_pct], axis=1
    ).mean(axis=1, skipna=True)

    result["contraction_severity_risk_percentile"] = (
        severity_percentile
    )
    result["contraction_severity_risk_class"] = (
        severity_percentile.apply(classify_percentile)
    )

    for index, row in result.iterrows():
        detail_rows.append({
            "iso": row["iso"],
            "country": row["country"],
            "dimension": "contraction_severity",
            "source_metric": "mean_duration_and_mean_depth",
            "source_value": np.nan,
            "risk_percentile": severity_percentile.loc[index],
            "risk_class": classify_percentile(
                severity_percentile.loc[index]
            ),
            "peer_group_size": int(severity_percentile.notna().sum()),
        })

    detail = pd.DataFrame(detail_rows)

    return result, detail


# ============================================================
# VALIDATION
# ============================================================

def validate_outputs(metrics, detail, panel, episodes, recovery):
    checks = []

    checks.append((
        "One country profile per ISO",
        not metrics["iso"].duplicated().any()
    ))

    checks.append((
        "Expected 20-country coverage",
        metrics["iso"].nunique() == 20
    ))

    checks.append((
        "No duplicate country-dimension records",
        not detail.duplicated(["iso", "dimension"]).any()
    ))

    checks.append((
        "All six risk dimensions represented",
        detail["dimension"].nunique() == 6
    ))

    checks.append((
        "No impossible contraction frequencies",
        metrics["contraction_frequency_pct"].dropna().between(0, 100).all()
    ))

    checks.append((
        "Recovery failure rates within 0-1",
        metrics["recovery_failure_rate"].dropna().between(0, 1).all()
    ))

    checks.append((
        "Recovery counts reconcile",
        (
            recovery["contraction_episodes"]
            == recovery["recovered_episodes"]
            + recovery["not_recovered_within_5_years"]
            + recovery["unassessable_episodes"]
            + recovery["incomplete_follow_up"]
        ).all()
    ))

    

    failed = [name for name, passed in checks if not passed]

    print("\nVALIDATION")
    print("-" * 70)

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if failed:
        raise ValueError(
            "Validation failed: " + "; ".join(failed)
        )

    return checks


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("EUROPEAN GROWTH RISK CLASSIFICATION ENGINE")
    print("=" * 70)

    panel = load_growth_panel()
    episodes = load_episode_data()
    recovery = load_recovery_data()

    print(f"Growth observations: {len(panel)}")
    print(f"Growth countries: {panel['iso'].nunique()}")
    print(f"Contraction episodes: {len(episodes)}")
    print(f"Recovery country profiles: {len(recovery)}")

    if set(panel["iso"]) != set(recovery["iso"]):
        raise ValueError(
            "Country coverage differs between growth panel "
            "and recovery summary."
        )

    metrics = build_country_metrics(
        panel, episodes, recovery
    )

    classifications, detail = classify_dimensions(metrics)

    validate_outputs(
        classifications, detail, panel, episodes, recovery
    )

    OUTPUT_CLASSIFICATIONS.parent.mkdir(
        parents=True, exist_ok=True
    )
    OUTPUT_DETAIL.parent.mkdir(
        parents=True, exist_ok=True
    )

    classifications.to_csv(
        OUTPUT_CLASSIFICATIONS, index=False
    )
    detail.to_csv(OUTPUT_DETAIL, index=False)

    lines = [
        "EUROPEAN GROWTH RISK CLASSIFICATION REPORT",
        "=" * 70,
        "",
        "METHODOLOGY",
        "-" * 70,
        "Historical actual annual real GDP growth observations only.",
        f"Long-run period: {LONG_RUN_START}-{LONG_RUN_END}.",
        f"Recent period: {RECENT_START}-{RECENT_END}.",
        f"Volatility period: {VOLATILITY_START}-{VOLATILITY_END}.",
        "Risk classes are peer-relative tercile classifications.",
        "Higher risk percentiles indicate greater relative exposure.",
        "No composite growth risk score is calculated.",
        "Contraction severity combines average episode duration",
        "and average worst annual growth.",
        "Recovery weakness excludes unassessable and incomplete episodes",
        "from its failure-rate denominator.",
        "",
        "COUNTRY COVERAGE",
        "-" * 70,
        f"Countries: {classifications['iso'].nunique()}",
        f"Growth observations: {len(panel)}",
        f"Contraction episodes: {len(episodes)}",
        "",
        "RISK CLASS DISTRIBUTIONS",
        "-" * 70,
    ]

    dimensions = [
        "long_run_growth_weakness",
        "recent_growth_weakness",
        "growth_instability",
        "contraction_frequency",
        "contraction_severity",
        "recovery_weakness",
    ]

    for dimension in dimensions:
        column = f"{dimension}_risk_class"
        counts = classifications[column].value_counts(dropna=False)

        lines.append(dimension.upper())

        for category in [
            "higher_risk",
            "moderate_risk",
            "lower_risk",
            "insufficient_data",
        ]:
            lines.append(
                f"  {category}: {int(counts.get(category, 0))}"
            )

        lines.append("")

    lines.extend([
        "IMPORTANT LIMITATIONS",
        "-" * 70,
        "Peer-relative classifications are not absolute risk thresholds.",
        "The country peer group contains 20 selected European economies.",
        "Historical coverage is not identical across all countries.",
        "Long-run mean growth can be affected by extreme observations.",
        "Standard deviation is sensitive to outliers.",
        "Contraction episodes use annual rather than quarterly data.",
        "Recovery is GDP-level recovery, not a full resilience measure.",
        "The classification is not a sovereign rating or default probability.",
        "",
        "OUTPUTS",
        "-" * 70,
        str(OUTPUT_CLASSIFICATIONS),
        str(OUTPUT_DETAIL),
        str(OUTPUT_REPORT),
    ])

    write_report(OUTPUT_REPORT, lines)

    print("\nOUTPUTS CREATED")
    print("-" * 70)
    print(OUTPUT_CLASSIFICATIONS)
    print(OUTPUT_DETAIL)
    print(OUTPUT_REPORT)

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
