
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PANEL_FILE = ROOT / "data/processed/europe_growth_risk_input_panel.csv"
EPISODES_FILE = ROOT / "data/processed/europe_growth_contraction_episodes.csv"
BASELINE_FILE = ROOT / "data/processed/europe_growth_risk_classifications.csv"

OUTPUT_COMPARISON = (
    ROOT / "data/processed/europe_growth_risk_sensitivity.csv"
)
OUTPUT_REPORT = (
    ROOT / "outputs/reports/europe_growth_risk_sensitivity_report.txt"
)


def find_column(df, candidates, description):
    lookup = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Cannot identify {description}. "
        f"Available columns: {list(df.columns)}"
    )


def percentile_class(values):
    ranks = values.rank(method="average", pct=True)

    classes = pd.Series(
        "insufficient_data",
        index=values.index,
        dtype="object",
    )

    classes.loc[ranks <= 1 / 3] = "lower_risk"
    classes.loc[ranks >= 2 / 3] = "higher_risk"

    middle = (
        ranks.notna()
        & (ranks > 1 / 3)
        & (ranks < 2 / 3)
    )
    classes.loc[middle] = "moderate_risk"

    return ranks, classes


def load_panel():
    df = pd.read_csv(PANEL_FILE)

    iso = find_column(
        df, ["iso", "ISO", "iso_code"], "ISO code"
    )
    year = find_column(
        df, ["year", "time"], "year"
    )
    growth = find_column(
        df,
        [
            "real_gdp_growth_pct",
            "real_gdp_growth",
            "growth_pct",
            "value",
            "obs_value",
        ],
        "growth value",
    )

    result = pd.DataFrame({
        "iso": df[iso].astype(str).str.upper().str.strip(),
        "year": pd.to_numeric(df[year], errors="coerce"),
        "growth_pct": pd.to_numeric(
            df[growth], errors="coerce"
        ),
    }).dropna(subset=["iso", "year"])

    result["year"] = result["year"].astype(int)

    if result.duplicated(["iso", "year"]).any():
        raise ValueError("Duplicate country-year growth records.")

    return result


def load_episodes():
    df = pd.read_csv(EPISODES_FILE)

    iso = find_column(
        df, ["iso", "ISO", "iso_code"], "episode ISO"
    )
    duration = find_column(
        df, ["contraction_years", "episode_duration"],
        "episode duration",
    )
    worst = find_column(
        df,
        ["worst_annual_growth_pct", "worst_annual_growth"],
        "worst annual growth",
    )

    result = pd.DataFrame({
        "iso": df[iso].astype(str).str.upper().str.strip(),
        "duration": pd.to_numeric(
            df[duration], errors="coerce"
        ),
        "worst_growth": pd.to_numeric(
            df[worst], errors="coerce"
        ),
    })

    return result


def country_growth_metrics(panel):
    rows = []

    for iso, group in panel.groupby("iso"):
        long_run = group.loc[
            group["year"].between(1980, 2025),
            "growth_pct",
        ].dropna()

        volatility = group.loc[
            group["year"].between(2000, 2024),
            "growth_pct",
        ].dropna()

        ireland_sensitivity = group.loc[
            ~(
                (group["year"] == 2015)
                & (group["iso"] == "IRL")
            )
            & group["year"].between(1980, 2025),
            "growth_pct",
        ].dropna()

        rows.append({
            "iso": iso,
            "long_run_mean": long_run.mean(),
            "long_run_median": long_run.median(),
            "long_run_mean_ex_ireland_2015": (
                ireland_sensitivity.mean()
            ),
            "long_run_observations": len(long_run),
            "volatility_std": volatility.std(ddof=1),
            "volatility_mad": (
                (volatility - volatility.median())
                .abs()
                .median()
            ),
            "volatility_observations": len(volatility),
        })

    return pd.DataFrame(rows)


def country_episode_metrics(episodes):
    rows = []

    for iso, group in episodes.groupby("iso"):
        duration = group["duration"].dropna()
        depth = (-group["worst_growth"]).dropna()

        rows.append({
            "iso": iso,
            "mean_duration": duration.mean(),
            "median_duration": duration.median(),
            "mean_depth": depth.mean(),
            "median_depth": depth.median(),
            "episode_count": len(group),
        })

    return pd.DataFrame(rows)


def add_scenario(
    output_rows,
    metrics,
    scenario,
    metric,
    reverse=False,
):
    values = metrics.set_index("iso")[metric].copy()

    if reverse:
        values = -values

    percentile, classification = percentile_class(values)

    for iso in values.index:
        output_rows.append({
            "iso": iso,
            "scenario": scenario,
            "metric": metric,
            "metric_value": (
                metrics.set_index("iso").loc[iso, metric]
            ),
            "risk_percentile": percentile.loc[iso],
            "risk_class": classification.loc[iso],
        })


def compare_scenarios(detail):
    wide = detail.pivot(
        index="iso",
        columns="scenario",
        values="risk_percentile",
    )

    comparisons = []

    scenario_names = list(wide.columns)

    for i, first in enumerate(scenario_names):
        for second in scenario_names[i + 1:]:
            pair = wide[[first, second]].dropna()

            correlation = (
                pair[first].corr(pair[second], method="spearman")
                if len(pair) >= 3
                else np.nan
            )

            first_class = detail.loc[
                detail["scenario"] == first
            ].set_index("iso")["risk_class"]

            second_class = detail.loc[
                detail["scenario"] == second
            ].set_index("iso")["risk_class"]

            common = first_class.index.intersection(
                second_class.index
            )

            changes = int(
                (
                    first_class.loc[common]
                    != second_class.loc[common]
                ).sum()
            )

            comparisons.append({
                "scenario_a": first,
                "scenario_b": second,
                "spearman_rank_correlation": correlation,
                "countries_with_changed_class": changes,
                "countries_compared": len(common),
            })

    return pd.DataFrame(comparisons)


def main():
    print("=" * 70)
    print("EUROPEAN GROWTH RISK SENSITIVITY AUDIT")
    print("=" * 70)

    panel = load_panel()
    episodes = load_episodes()

    growth_metrics = country_growth_metrics(panel)
    episode_metrics = country_episode_metrics(episodes)

    output_rows = []

    # Long-run growth: higher risk means lower growth.
    add_scenario(
        output_rows,
        growth_metrics,
        "long_run_mean_baseline",
        "long_run_mean",
        reverse=True,
    )

    add_scenario(
        output_rows,
        growth_metrics,
        "long_run_median",
        "long_run_median",
        reverse=True,
    )

    add_scenario(
        output_rows,
        growth_metrics,
        "long_run_mean_ex_ireland_2015",
        "long_run_mean_ex_ireland_2015",
        reverse=True,
    )

    # Volatility: higher values mean greater instability.
    add_scenario(
        output_rows,
        growth_metrics,
        "volatility_standard_deviation",
        "volatility_std",
    )

    add_scenario(
        output_rows,
        growth_metrics,
        "volatility_median_absolute_deviation",
        "volatility_mad",
    )

    # Contraction duration and depth.
    add_scenario(
        output_rows,
        episode_metrics,
        "contraction_mean_duration",
        "mean_duration",
    )

    add_scenario(
        output_rows,
        episode_metrics,
        "contraction_median_duration",
        "median_duration",
    )

    add_scenario(
        output_rows,
        episode_metrics,
        "contraction_mean_depth",
        "mean_depth",
    )

    add_scenario(
        output_rows,
        episode_metrics,
        "contraction_median_depth",
        "median_depth",
    )

    detail = pd.DataFrame(output_rows)
    comparisons = compare_scenarios(detail)

    OUTPUT_COMPARISON.parent.mkdir(
        parents=True, exist_ok=True
    )
    detail.to_csv(OUTPUT_COMPARISON, index=False)

    lines = [
        "EUROPEAN GROWTH RISK SENSITIVITY AUDIT",
        "=" * 70,
        "",
        "PURPOSE",
        "-" * 70,
        "Tests sensitivity of peer-relative growth risk rankings",
        "to alternative measurement choices.",
        "This audit does not overwrite production classifications.",
        "",
        "SCENARIOS",
        "-" * 70,
        "Long-run mean versus median growth.",
        "Long-run mean excluding Ireland's 2015 observation.",
        "Standard deviation versus median absolute deviation.",
        "Mean versus median contraction duration.",
        "Mean versus median contraction depth.",
        "",
        "SCENARIO COMPARISONS",
        "-" * 70,
    ]

    for _, row in comparisons.iterrows():
        lines.append(
            f"{row['scenario_a']} vs {row['scenario_b']}: "
            f"Spearman={row['spearman_rank_correlation']:.4f}; "
            f"class changes="
            f"{row['countries_with_changed_class']}/"
            f"{row['countries_compared']}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 70,
        "High rank correlation indicates similar country ordering.",
        "Class changes indicate sensitivity around tercile boundaries.",
        "A robust measure can reduce outlier influence but may hide",
        "genuine extreme economic events.",
        "Ireland 2015 is excluded only in its designated sensitivity",
        "scenario; the original observation remains in the source data.",
        "",
        "OUTPUT",
        "-" * 70,
        str(OUTPUT_COMPARISON),
    ])

    OUTPUT_REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )
    OUTPUT_REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Countries: {growth_metrics['iso'].nunique()}")
    print(f"Growth observations: {len(panel)}")
    print(f"Contraction episodes: {len(episodes)}")
    print(f"Sensitivity records: {len(detail)}")

    print("\nSCENARIO COMPARISONS")
    print(comparisons.to_string(index=False))

    print("\nOUTPUTS CREATED")
    print(OUTPUT_COMPARISON)
    print(OUTPUT_REPORT)

    if growth_metrics["iso"].nunique() != 20:
        raise ValueError("Expected 20 countries.")

    if episodes["iso"].nunique() != 20:
        raise ValueError("Episode data does not cover 20 countries.")

    if detail.duplicated(["iso", "scenario"]).any():
        raise ValueError("Duplicate country-scenario records.")

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
