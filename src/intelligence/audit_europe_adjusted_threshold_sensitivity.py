
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed"
    / "europe_correlation_adjusted_engine_rates.csv"
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUT_ENGINE = (
    PROCESSED / "europe_adjusted_threshold_engine_sensitivity.csv"
)

OUT_COUNTRY = (
    PROCESSED / "europe_adjusted_threshold_country_profiles.csv"
)

OUT_TRANSITIONS = (
    PROCESSED / "europe_adjusted_threshold_transitions.csv"
)

OUT_SUMMARY = (
    PROCESSED / "europe_adjusted_threshold_scenario_summary.csv"
)

OUT_REPORT = (
    REPORTS / "europe_adjusted_threshold_sensitivity_report.txt"
)

EXPECTED_COUNTRIES = 20

EXPECTED_ENGINE_COUNTS = {
    "fiscal": 6,
    "growth": 6,
    "inflation": 7,
    "external": 6,
    "labour": 5,
}

SCENARIOS = [
    {
        "name": "baseline_one_signal",
        "rule": "minimum_count",
        "threshold": 1,
    },
    {
        "name": "minimum_two_signals",
        "rule": "minimum_count",
        "threshold": 2,
    },
    {
        "name": "minimum_25_percent",
        "rule": "minimum_rate",
        "threshold": 0.25,
    },
    {
        "name": "minimum_50_percent",
        "rule": "minimum_rate",
        "threshold": 0.50,
    },
]

FOCUS_COUNTRIES = ["GRC", "ROU", "PRT", "ITA", "POL"]


# ============================================================
# TYPOLOGY
# ============================================================

def assign_typology(engine_count):

    if engine_count == 0:
        return "no_higher_engine_signal"

    if engine_count <= 2:
        return "concentrated_exposure"

    if engine_count <= 4:
        return "multi_engine_exposure"

    return "broad_based_exposure"


# ============================================================
# LOAD AND VALIDATE
# ============================================================

def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "risk_engine",
        "adjusted_higher_count",
        "adjusted_signal_count",
        "adjusted_higher_rate",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["iso"] = df["iso"].astype(str).str.strip()
    df["risk_engine"] = (
        df["risk_engine"].astype(str).str.lower().str.strip()
    )

    numeric_columns = [
        "adjusted_higher_count",
        "adjusted_signal_count",
        "adjusted_higher_rate",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df.isna().any().any():
        raise ValueError("Missing values found in input.")

    if df["iso"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError("Unexpected country count.")

    if len(df) != EXPECTED_COUNTRIES * 5:
        raise ValueError("Expected 100 country-engine records.")

    if df.duplicated(["iso", "risk_engine"]).any():
        raise ValueError("Duplicate country-engine records.")

    if set(df["risk_engine"].unique()) != set(
        EXPECTED_ENGINE_COUNTS
    ):
        raise ValueError("Unexpected risk engine set.")

    for engine, expected_count in EXPECTED_ENGINE_COUNTS.items():

        subset = df[df["risk_engine"] == engine]

        if not (
            subset["adjusted_signal_count"] == expected_count
        ).all():
            raise ValueError(
                f"Unexpected adjusted signal count for {engine}."
            )

    if (
        df["adjusted_higher_count"]
        > df["adjusted_signal_count"]
    ).any():
        raise ValueError("Higher count exceeds signal count.")

    calculated_rate = (
        df["adjusted_higher_count"]
        / df["adjusted_signal_count"]
    )

    if not (
        (calculated_rate - df["adjusted_higher_rate"]).abs()
        < 1e-9
    ).all():
        raise ValueError("Adjusted rate reconciliation failed.")

    return df


# ============================================================
# APPLY THRESHOLD SCENARIOS
# ============================================================

def build_engine_scenarios(df):

    frames = []

    for scenario in SCENARIOS:

        current = df.copy()

        if scenario["rule"] == "minimum_count":

            active = (
                current["adjusted_higher_count"]
                >= scenario["threshold"]
            )

        elif scenario["rule"] == "minimum_rate":

            active = (
                current["adjusted_higher_rate"]
                >= scenario["threshold"]
            )

        else:
            raise ValueError("Unknown scenario rule.")

        current["scenario"] = scenario["name"]
        current["threshold_rule"] = scenario["rule"]
        current["threshold_value"] = scenario["threshold"]
        current["engine_active"] = active.astype(int)

        frames.append(current)

    result = pd.concat(frames, ignore_index=True)

    result.to_csv(OUT_ENGINE, index=False)

    return result


# ============================================================
# BUILD COUNTRY PROFILES
# ============================================================

def build_country_profiles(engine_scenarios):

    result = (
        engine_scenarios.groupby(
            ["scenario", "iso", "country"],
            as_index=False
        )
        .agg(
            active_engine_count=("engine_active", "sum"),
            total_higher_signals=("adjusted_higher_count", "sum"),
            total_adjusted_signals=("adjusted_signal_count", "sum"),
        )
    )

    result["vulnerability_typology"] = (
        result["active_engine_count"].apply(assign_typology)
    )

    result["higher_signal_share_pct"] = (
        result["total_higher_signals"]
        / result["total_adjusted_signals"] * 100
    )

    result.to_csv(OUT_COUNTRY, index=False)

    return result


# ============================================================
# BASELINE TRANSITIONS
# ============================================================

def build_transitions(country_profiles):

    baseline = (
        country_profiles[
            country_profiles["scenario"] == "baseline_one_signal"
        ][
            [
                "iso",
                "country",
                "active_engine_count",
                "vulnerability_typology",
            ]
        ]
        .rename(columns={
            "active_engine_count": "baseline_engine_count",
            "vulnerability_typology": "baseline_typology",
        })
    )

    rows = []

    for scenario in country_profiles["scenario"].unique():

        current = (
            country_profiles[
                country_profiles["scenario"] == scenario
            ][
                [
                    "iso",
                    "active_engine_count",
                    "vulnerability_typology",
                ]
            ]
            .rename(columns={
                "active_engine_count": "scenario_engine_count",
                "vulnerability_typology": "scenario_typology",
            })
        )

        merged = baseline.merge(
            current,
            on="iso",
            validate="one_to_one"
        )

        merged["scenario"] = scenario

        merged["engine_count_changed"] = (
            merged["baseline_engine_count"]
            != merged["scenario_engine_count"]
        )

        merged["typology_changed"] = (
            merged["baseline_typology"]
            != merged["scenario_typology"]
        )

        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)

    result.to_csv(OUT_TRANSITIONS, index=False)

    return result


# ============================================================
# SCENARIO SUMMARY
# ============================================================

def build_summary(engine_scenarios, country_profiles, transitions):

    rows = []

    for scenario in country_profiles["scenario"].unique():

        countries = country_profiles[
            country_profiles["scenario"] == scenario
        ]

        engines = engine_scenarios[
            engine_scenarios["scenario"] == scenario
        ]

        changes = transitions[
            transitions["scenario"] == scenario
        ]

        rows.append({
            "scenario": scenario,
            "broad_based": int(
                (countries["vulnerability_typology"]
                 == "broad_based_exposure").sum()
            ),
            "multi_engine": int(
                (countries["vulnerability_typology"]
                 == "multi_engine_exposure").sum()
            ),
            "concentrated": int(
                (countries["vulnerability_typology"]
                 == "concentrated_exposure").sum()
            ),
            "no_higher_signal": int(
                (countries["vulnerability_typology"]
                 == "no_higher_engine_signal").sum()
            ),
            "active_engine_records": int(
                engines["engine_active"].sum()
            ),
            "countries_with_engine_count_change": int(
                changes["engine_count_changed"].sum()
            ),
            "countries_with_typology_change": int(
                changes["typology_changed"].sum()
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(OUT_SUMMARY, index=False)

    return result


# ============================================================
# REPORT
# ============================================================

def build_report(
    engine_scenarios,
    country_profiles,
    transitions,
    summary
):

    lines = [
        "=" * 76,
        "EUROPEAN DEPENDENCE-ADJUSTED THRESHOLD SENSITIVITY",
        "=" * 76,
        "",
        "VALIDATION",
        "-" * 76,
        f"Countries: {EXPECTED_COUNTRIES}",
        f"Engines: {len(EXPECTED_ENGINE_COUNTS)}",
        f"Adjusted country-engine records: {len(engine_scenarios) // 4}",
        f"Scenario engine records: {len(engine_scenarios)}",
        f"Country-scenario records: {len(country_profiles)}",
        "",
        "SCENARIO DISTRIBUTIONS",
        "-" * 76,
        summary.to_string(index=False),
        "",
        "TYPOLOGY TRANSITIONS FROM BASELINE",
        "-" * 76,
    ]

    for scenario in summary["scenario"]:

        subset = transitions[
            transitions["scenario"] == scenario
        ]

        lines.append(
            f"\n{scenario}: "
            f"{int(subset['typology_changed'].sum())}/"
            f"{len(subset)} typology changes; "
            f"{int(subset['engine_count_changed'].sum())}/"
            f"{len(subset)} engine-count changes"
        )

        changed = subset[subset["typology_changed"]]

        if not changed.empty:
            lines.append(
                changed[
                    [
                        "iso",
                        "baseline_typology",
                        "scenario_typology",
                        "baseline_engine_count",
                        "scenario_engine_count",
                    ]
                ].to_string(index=False)
            )

    lines.extend([
        "",
        "ENGINE ACTIVATION BY SCENARIO",
        "-" * 76,
    ])

    activation = (
        engine_scenarios.groupby(
            ["scenario", "risk_engine"]
        )["engine_active"]
        .sum()
        .reset_index(name="countries_with_active_engine")
    )

    lines.append(activation.to_string(index=False))

    lines.extend([
        "",
        "FOCUS COUNTRY COMPARISON",
        "-" * 76,
    ])

    focus = country_profiles[
        country_profiles["iso"].isin(FOCUS_COUNTRIES)
    ]

    focus_pivot = focus.pivot(
        index="iso",
        columns="scenario",
        values="active_engine_count"
    )

    lines.append(focus_pivot.to_string())

    focus_types = focus.pivot(
        index="iso",
        columns="scenario",
        values="vulnerability_typology"
    )

    lines.append("\nFOCUS COUNTRY TYPOLOGIES")
    lines.append(focus_types.to_string())

    lines.extend([
        "",
        "METHODOLOGICAL LIMITATIONS",
        "-" * 76,
        "1. Thresholds are exploratory sensitivity scenarios.",
        "2. No threshold has been calibrated against observed",
        "   sovereign distress or macroeconomic crisis outcomes.",
        "3. Engine counts are not severity measures.",
        "4. A country can have zero higher peer-relative signals",
        "   while still having an absolute economic vulnerability.",
        "5. Peer-group sensitivity and temporal stability remain",
        "   separate validation requirements.",
        "6. No composite risk score or default probability is created.",
        "",
        "OUTPUT FILES",
        "-" * 76,
        str(OUT_ENGINE),
        str(OUT_COUNTRY),
        str(OUT_TRANSITIONS),
        str(OUT_SUMMARY),
        str(OUT_REPORT),
    ])

    report = "\n".join(lines)

    OUT_REPORT.write_text(
        report,
        encoding="utf-8"
    )

    print(report)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("DEPENDENCE-ADJUSTED THRESHOLD SENSITIVITY")
    print("=" * 76)

    df = load_data()

    print("Input validation passed.")

    engine_scenarios = build_engine_scenarios(df)

    country_profiles = build_country_profiles(
        engine_scenarios
    )

    transitions = build_transitions(country_profiles)

    summary = build_summary(
        engine_scenarios,
        country_profiles,
        transitions
    )

    build_report(
        engine_scenarios,
        country_profiles,
        transitions,
        summary
    )

    print("")
    print("VALIDATION PASSED")
    print(f"Engine scenario records: {len(engine_scenarios)}")
    print(f"Country scenario records: {len(country_profiles)}")
    print(f"Transition records: {len(transitions)}")

    print("")
    print("OUTPUTS CREATED")
    print(OUT_ENGINE)
    print(OUT_COUNTRY)
    print(OUT_TRANSITIONS)
    print(OUT_SUMMARY)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
