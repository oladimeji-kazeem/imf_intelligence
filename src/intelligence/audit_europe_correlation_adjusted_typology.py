
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed"
    / "europe_integrated_risk_matrix.csv"
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUT_DIMENSION_MAP = (
    PROCESSED / "europe_correlation_adjusted_dimension_map.csv"
)

OUT_PAIR_AUDIT = (
    PROCESSED / "europe_correlated_pair_country_audit.csv"
)

OUT_ENGINE_RATES = (
    PROCESSED / "europe_correlation_adjusted_engine_rates.csv"
)

OUT_COUNTRY_PROFILES = (
    PROCESSED / "europe_correlation_adjusted_country_profiles.csv"
)

OUT_COMPARISON = (
    PROCESSED / "europe_correlation_adjusted_typology_comparison.csv"
)

OUT_CONCENTRATION = (
    PROCESSED / "europe_correlation_adjusted_engine_concentration.csv"
)

OUT_REPORT = (
    REPORTS / "europe_correlation_adjusted_typology_report.txt"
)

EXPECTED_COUNTRIES = 20
EXPECTED_DIMENSIONS = 33
EXPECTED_ENGINES = {
    "fiscal", "growth", "inflation", "external", "labour"
}

# Highly correlated pairs within the same engine.
# These are sensitivity groupings, not declarations of redundancy.
CORRELATED_PAIRS = {
    "fiscal": [
        (
            "fiscal_balance_volatility_pair",
            "historical_volatility_fiscal_balance",
            "historical_volatility_primary_balance",
        ),
        (
            "fiscal_balance_pressure_pair",
            "peer_relative_fiscal_balance",
            "peer_relative_primary_balance",
        ),
    ],
    "inflation": [
        (
            "historical_inflation_persistence_pair",
            "historical_high_inflation_exposure",
            "historical_inflation_persistence",
        ),
    ],
}

FOCUS_COUNTRIES = ["GRC", "ROU", "PRT", "ITA", "POL"]


# ============================================================
# TYPOLOGY RULE
# ============================================================

def assign_typology(active_engine_count):

    if active_engine_count == 0:
        return "no_higher_engine_signal"

    if active_engine_count <= 2:
        return "concentrated_exposure"

    if active_engine_count <= 4:
        return "multi_engine_exposure"

    return "broad_based_exposure"


# ============================================================
# LOAD AND VALIDATE INPUT
# ============================================================

def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "iso",
        "country",
        "risk_engine",
        "risk_dimension",
        "standardised_classification",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    for col in [
        "iso",
        "country",
        "risk_engine",
        "risk_dimension",
        "standardised_classification",
    ]:
        df[col] = df[col].astype(str).str.strip()

    df["risk_engine"] = df["risk_engine"].str.lower()

    df["standardised_classification"] = (
        df["standardised_classification"].str.lower()
    )

    if len(df) != EXPECTED_COUNTRIES * EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected 660 rows, found {len(df)}."
        )

    if df["iso"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError("Unexpected country count.")

    if df["risk_dimension"].nunique() != EXPECTED_DIMENSIONS:
        raise ValueError("Unexpected dimension count.")

    if set(df["risk_engine"].unique()) != EXPECTED_ENGINES:
        raise ValueError("Unexpected engine names.")

    if df.duplicated(
        ["iso", "risk_engine", "risk_dimension"]
    ).any():
        raise ValueError("Duplicate country-engine-dimension rows.")

    if not set(
        df["standardised_classification"].unique()
    ).issubset({"higher", "moderate", "lower"}):
        raise ValueError("Unexpected classification labels.")

    df["higher_flag"] = (
        df["standardised_classification"] == "higher"
    ).astype(int)

    return df


# ============================================================
# BUILD DIMENSION-TO-GROUP MAPPING
# ============================================================

def build_dimension_map(df):

    mapping = {}

    for engine, pairs in CORRELATED_PAIRS.items():

        for group_name, dimension_a, dimension_b in pairs:

            for dimension in [dimension_a, dimension_b]:

                if dimension in mapping:
                    raise ValueError(
                        f"Dimension assigned to multiple groups: {dimension}"
                    )

                mapping[dimension] = {
                    "engine": engine,
                    "group": group_name,
                    "paired": True,
                }

    rows = []

    for _, row in df[
        ["risk_engine", "risk_dimension"]
    ].drop_duplicates().iterrows():

        engine = row["risk_engine"]
        dimension = row["risk_dimension"]

        if dimension in mapping:

            item = mapping[dimension]

            if item["engine"] != engine:
                raise ValueError(
                    f"Engine mismatch for dimension {dimension}"
                )

            group_name = item["group"]
            paired = True

        else:
            group_name = dimension
            paired = False

        rows.append({
            "risk_engine": engine,
            "risk_dimension": dimension,
            "adjusted_signal_group": group_name,
            "part_of_correlated_pair": paired,
        })

    result = pd.DataFrame(rows)

    # Ensure every specified pair exists exactly once.
    for engine, pairs in CORRELATED_PAIRS.items():

        for group_name, dimension_a, dimension_b in pairs:

            for dimension in [dimension_a, dimension_b]:

                match = result[
                    (result["risk_engine"] == engine)
                    & (result["risk_dimension"] == dimension)
                ]

                if len(match) != 1:
                    raise ValueError(
                        f"Expected exactly one mapping for {dimension}; "
                        f"found {len(match)}"
                    )

    result.to_csv(OUT_DIMENSION_MAP, index=False)

    return result


# ============================================================
# AUDIT CORRELATED PAIRS BY COUNTRY
# ============================================================

def build_pair_audit(df):

    rows = []

    for engine, pairs in CORRELATED_PAIRS.items():

        for pair_name, dimension_a, dimension_b in pairs:

            a = df[
                (df["risk_engine"] == engine)
                & (df["risk_dimension"] == dimension_a)
            ][["iso", "country", "higher_flag"]].rename(
                columns={"higher_flag": "dimension_a_higher"}
            )

            b = df[
                (df["risk_engine"] == engine)
                & (df["risk_dimension"] == dimension_b)
            ][["iso", "higher_flag"]].rename(
                columns={"higher_flag": "dimension_b_higher"}
            )

            merged = a.merge(
                b,
                on="iso",
                validate="one_to_one"
            )

            if len(merged) != EXPECTED_COUNTRIES:
                raise ValueError(
                    f"Incomplete pair coverage: {pair_name}"
                )

            merged["engine"] = engine
            merged["pair_name"] = pair_name
            merged["dimension_a"] = dimension_a
            merged["dimension_b"] = dimension_b

            merged["pair_signal_higher"] = (
                (
                    merged["dimension_a_higher"]
                    + merged["dimension_b_higher"]
                ) > 0
            ).astype(int)

            merged["both_higher"] = (
                (
                    merged["dimension_a_higher"] == 1
                )
                & (
                    merged["dimension_b_higher"] == 1
                )
            )

            merged["one_higher_only"] = (
                merged["dimension_a_higher"]
                + merged["dimension_b_higher"] == 1
            )

            rows.append(merged)

    result = pd.concat(rows, ignore_index=True)

    result.to_csv(OUT_PAIR_AUDIT, index=False)

    return result


# ============================================================
# BUILD CORRELATION-ADJUSTED ENGINE SIGNALS
# ============================================================

def build_adjusted_engine_rates(df, dimension_map):

    work = df.merge(
        dimension_map,
        on=["risk_engine", "risk_dimension"],
        how="left",
        validate="many_to_one",
    )

    if work["adjusted_signal_group"].isna().any():
        raise ValueError("Unmapped dimensions found.")

    # Within each country and engine, collapse each pair into
    # one binary signal. If either member is higher, the
    # combined signal is higher.
    grouped = (
        work.groupby(
            [
                "iso",
                "country",
                "risk_engine",
                "adjusted_signal_group",
            ],
            as_index=False
        )
        .agg(
            adjusted_higher_flag=("higher_flag", "max"),
            source_dimension_count=("risk_dimension", "nunique"),
        )
    )

    rates = (
        grouped.groupby(
            ["iso", "country", "risk_engine"],
            as_index=False
        )
        .agg(
            adjusted_higher_count=(
                "adjusted_higher_flag", "sum"
            ),
            adjusted_signal_count=(
                "adjusted_signal_group", "nunique"
            ),
            source_dimension_count=(
                "source_dimension_count", "sum"
            ),
        )
    )

    rates["adjusted_higher_rate"] = (
        rates["adjusted_higher_count"]
        / rates["adjusted_signal_count"]
    )

    rates["adjusted_higher_rate_pct"] = (
        rates["adjusted_higher_rate"] * 100
    )

    # Original engine rate for comparison.
    original = (
        df.groupby(
            ["iso", "country", "risk_engine"],
            as_index=False
        )
        .agg(
            original_higher_count=("higher_flag", "sum"),
            original_dimension_count=("risk_dimension", "nunique"),
        )
    )

    original["original_higher_rate"] = (
        original["original_higher_count"]
        / original["original_dimension_count"]
    )

    rates = rates.merge(
        original,
        on=["iso", "country", "risk_engine"],
        validate="one_to_one"
    )

    rates["higher_rate_change_pp"] = (
        rates["adjusted_higher_rate"]
        - rates["original_higher_rate"]
    ) * 100

    rates["adjusted_engine_active"] = (
        rates["adjusted_higher_count"] >= 1
    )

    rates.to_csv(OUT_ENGINE_RATES, index=False)

    return rates


# ============================================================
# BUILD COUNTRY TYPOLOGY COMPARISON
# ============================================================

def build_country_profiles(df, rates):

    original_engine = (
        df.groupby(
            ["iso", "country", "risk_engine"],
            as_index=False
        )
        .agg(
            original_higher_count=("higher_flag", "sum")
        )
    )

    original_country = (
        original_engine.groupby(
            ["iso", "country"],
            as_index=False
        )
        .agg(
            original_active_engine_count=(
                "original_higher_count",
                lambda x: int((x >= 1).sum())
            ),
            original_higher_dimension_count=(
                "original_higher_count", "sum"
            ),
        )
    )

    adjusted_country = (
        rates.groupby(
            ["iso", "country"],
            as_index=False
        )
        .agg(
            adjusted_active_engine_count=(
                "adjusted_engine_active", "sum"
            ),
            adjusted_higher_signal_count=(
                "adjusted_higher_count", "sum"
            ),
            adjusted_signal_count=(
                "adjusted_signal_count", "sum"
            ),
        )
    )

    result = original_country.merge(
        adjusted_country,
        on=["iso", "country"],
        validate="one_to_one"
    )

    result["original_typology"] = (
        result["original_active_engine_count"]
        .apply(assign_typology)
    )

    result["adjusted_typology"] = (
        result["adjusted_active_engine_count"]
        .apply(assign_typology)
    )

    result["typology_changed"] = (
        result["original_typology"]
        != result["adjusted_typology"]
    )

    result["original_higher_share_pct"] = (
        result["original_higher_dimension_count"]
        / EXPECTED_DIMENSIONS * 100
    )

    result["adjusted_higher_signal_share_pct"] = (
        result["adjusted_higher_signal_count"]
        / result["adjusted_signal_count"] * 100
    )

    result.to_csv(OUT_COUNTRY_PROFILES, index=False)

    result.to_csv(OUT_COMPARISON, index=False)

    return result


# ============================================================
# CORRELATION-ADJUSTED ENGINE CONCENTRATION
# ============================================================

def build_concentration(rates):

    rows = []

    for (iso, country), group in rates.groupby(
        ["iso", "country"]
    ):

        # Include all five engines, including zero-exposure engines.
        if len(group) != len(EXPECTED_ENGINES):
            raise ValueError(
                f"Expected five engines for {iso}."
            )

        total_rate = group["adjusted_higher_rate"].sum()

        if total_rate > 0:
            shares = (
                group["adjusted_higher_rate"] / total_rate
            )

            hhi = float((shares ** 2).sum())

            max_rate = group["adjusted_higher_rate"].max()

            dominant = ";".join(
                sorted(
                    group.loc[
                        np.isclose(
                            group["adjusted_higher_rate"],
                            max_rate
                        ),
                        "risk_engine"
                    ].tolist()
                )
            )

        else:
            hhi = np.nan
            dominant = "none"
            max_rate = 0.0

        rows.append({
            "iso": iso,
            "country": country,
            "active_engines": int(
                group["adjusted_engine_active"].sum()
            ),
            "adjusted_higher_signals": int(
                group["adjusted_higher_count"].sum()
            ),
            "mean_engine_higher_rate_pct": (
                group["adjusted_higher_rate"].mean() * 100
            ),
            "maximum_engine_higher_rate_pct": (
                max_rate * 100
            ),
            "adjusted_engine_concentration_hhi": hhi,
            "dominant_engine_or_tie": dominant,
        })

    result = pd.DataFrame(rows)

    result.to_csv(OUT_CONCENTRATION, index=False)

    return result


# ============================================================
# REPORT
# ============================================================

def build_report(
    df,
    dimension_map,
    pair_audit,
    rates,
    profiles,
    concentration
):

    lines = []

    lines.extend([
        "=" * 76,
        "EUROPEAN CORRELATION-ADJUSTED VULNERABILITY TYPOLOGY AUDIT",
        "=" * 76,
        "",
        "VALIDATION",
        "-" * 76,
        f"Countries: {df['iso'].nunique()}",
        f"Original dimensions: {df['risk_dimension'].nunique()}",
        f"Original records: {len(df)}",
        f"Engines: {df['risk_engine'].nunique()}",
        f"Correlated pairs collapsed: {len(CORRELATED_PAIRS['fiscal']) + len(CORRELATED_PAIRS['inflation'])}",
        f"Adjusted signal groups per country: {dimension_map.groupby('risk_engine')['adjusted_signal_group'].nunique().sum()}",
        "",
        "CORRELATED PAIR AUDIT",
        "-" * 76,
    ])

    for pair_name, group in pair_audit.groupby("pair_name"):

        lines.append(f"\n{pair_name}")

        lines.append(
            f"Both dimensions higher: {int(group['both_higher'].sum())}"
        )

        lines.append(
            f"Exactly one dimension higher: "
            f"{int(group['one_higher_only'].sum())}"
        )

        lines.append(
            f"Neither dimension higher: "
            f"{int((group['pair_signal_higher'] == 0).sum())}"
        )

        lines.append(
            "Countries with both higher: "
            + ", ".join(
                group.loc[group["both_higher"], "iso"].tolist()
            )
        )

    lines.extend([
        "",
        "TYPOLOGY COMPARISON",
        "-" * 76,
    ])

    comparison = pd.crosstab(
        profiles["original_typology"],
        profiles["adjusted_typology"],
        margins=True
    )

    lines.append(comparison.to_string())

    lines.append(
        f"\nCountries changing typology: "
        f"{int(profiles['typology_changed'].sum())}/"
        f"{len(profiles)}"
    )

    changed = profiles[profiles["typology_changed"]]

    if not changed.empty:
        lines.append("\nCOUNTRIES WITH TYPOLOGY CHANGES")
        lines.append(
            changed[
                [
                    "iso",
                    "original_typology",
                    "adjusted_typology",
                    "original_active_engine_count",
                    "adjusted_active_engine_count",
                    "original_higher_dimension_count",
                    "adjusted_higher_signal_count",
                ]
            ].to_string(index=False)
        )

    lines.extend([
        "",
        "COUNTRY ENGINE EXPOSURE COMPARISON",
        "-" * 76,
    ])

    focus = profiles[
        profiles["iso"].isin(FOCUS_COUNTRIES)
    ].copy()

    lines.append(
        focus[
            [
                "iso",
                "original_active_engine_count",
                "adjusted_active_engine_count",
                "original_higher_dimension_count",
                "adjusted_higher_signal_count",
                "original_typology",
                "adjusted_typology",
            ]
        ].to_string(index=False)
    )

    lines.extend([
        "",
        "ADJUSTED ENGINE CONCENTRATION",
        "-" * 76,
    ])

    lines.append(
        concentration.sort_values(
            "mean_engine_higher_rate_pct",
            ascending=False
        ).to_string(index=False)
    )

    lines.extend([
        "",
        "ENGINE RATE COMPARISON",
        "-" * 76,
    ])

    lines.append(
        rates[
            [
                "iso",
                "risk_engine",
                "original_higher_count",
                "original_dimension_count",
                "adjusted_higher_count",
                "adjusted_signal_count",
                "original_higher_rate",
                "adjusted_higher_rate",
                "higher_rate_change_pp",
            ]
        ].to_string(index=False)
    )

    lines.extend([
        "",
        "METHODOLOGICAL INTERPRETATION",
        "-" * 76,
        "1. Correlated pairs are collapsed into one binary signal.",
        "2. A pair is higher if either member is higher.",
        "3. The adjustment reduces the opportunity to count two",
        "   closely related dimensions as two separate signals.",
        "4. It does not prove that the indicators are redundant.",
        "5. Pairwise correlation is based on a 20-country sample.",
        "6. The cross-engine external-inflation correlation is not",
        "   collapsed because it spans different risk engines.",
        "7. The adjusted HHI measures concentration of adjusted",
        "   engine higher-exposure rates, not risk severity.",
        "8. No composite risk score or default probability is created.",
        "",
        "OUTPUT FILES",
        "-" * 76,
        str(OUT_DIMENSION_MAP),
        str(OUT_PAIR_AUDIT),
        str(OUT_ENGINE_RATES),
        str(OUT_COUNTRY_PROFILES),
        str(OUT_COMPARISON),
        str(OUT_CONCENTRATION),
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
    print("CORRELATION-ADJUSTED TYPOLOGY AUDIT")
    print("=" * 76)

    df = load_data()

    print("Input validation passed.")

    dimension_map = build_dimension_map(df)

    pair_audit = build_pair_audit(df)

    rates = build_adjusted_engine_rates(
        df,
        dimension_map
    )

    profiles = build_country_profiles(
        df,
        rates
    )

    concentration = build_concentration(rates)

    build_report(
        df,
        dimension_map,
        pair_audit,
        rates,
        profiles,
        concentration
    )

    print("")
    print("VALIDATION PASSED")
    print(f"Countries: {len(profiles)}")
    print(f"Original dimensions: {df['risk_dimension'].nunique()}")
    print(f"Adjusted signal groups: {dimension_map['adjusted_signal_group'].nunique()}")
    print(
        "Countries with changed typology: "
        f"{int(profiles['typology_changed'].sum())}"
    )

    print("")
    print("OUTPUTS CREATED")
    print(OUT_DIMENSION_MAP)
    print(OUT_PAIR_AUDIT)
    print(OUT_ENGINE_RATES)
    print(OUT_COUNTRY_PROFILES)
    print(OUT_COMPARISON)
    print(OUT_CONCENTRATION)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
