from pathlib import Path
from itertools import combinations

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"
OUTPUT = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SHEET_NAME = "Debt_2025"
HEADER_ROW = 64

AGGREGATE_FIELDS = [
    "DEBT_ADVANCED_ECON_2025",
    "DEBT_EME_2025",
    "DEBT_HIPC_2025",
    "DEBT_OTHER_2025",
    "DEBT_EXAE_2025",
    "DEBT_SSA_2025",
    "DEBT_DEFRATE_SOVEREIGNS_2025",
    "DEBT_TOTAL_DEF_SOVEREIGNS_2025",
    "DEBT_TOTAL_NUMB_SOVEREIGNS_2025",
    "DEBT_DEFRATE_WORLD_GDP_2025",
    "DEBT_ARREARS_TOTAL_SOVEREIGNS_2025",
    "DEBT_ARREARS_DEFRATE_CONV_DEF_2025",
    "DEBT_ARREARS_DEFRATE_WORLD_GDP_2025",
    "DEBT_ARREARS_DEFRATE_SOVEREIGNS_2025",
    "DEBT_GROSS_WORLD_PUBLIC_DEBT_2025",
    "DEBT_DEFRATE_EME_OTHER_GDP_2025",
    "DEBT_DEFRATE_WORLD_PUBLIC_DEBT_2025",
    "DEBT_DEFRATE_CHINA_2025",
    "DEBT_GROSS_EME_OTHER_GDP_2025",
    "DEBT_GROSS_CHINA_2025",
    "DEBT_DEFRATE_PARIS_CLUB_DEBT_2025",
    "DEBT_PARIS_CLUB_DEBT_2025",
    "DEBT_GROSS_WORLD_PRODUCT_2025",
]

EXPECTED_YEARS = set(range(1960, 2025))


def write_csv(df, filename):
    df.to_csv(OUTPUT / filename, index=False)


def main():
    print("=" * 78)
    print("BoC-BOE AGGREGATE SERIES VALUE-INTEGRITY AUDIT")
    print("=" * 78)

    df = pd.read_excel(
        SOURCE,
        sheet_name=SHEET_NAME,
        header=HEADER_ROW,
    )

    required = [
        "DEBT_COUNTRY",
        "DEBT_COUNTRY_GROUP",
        "DEBT_YEAR",
    ] + AGGREGATE_FIELDS

    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    world = df.loc[
        (df["DEBT_COUNTRY"] == "World")
        & (df["DEBT_COUNTRY_GROUP"] == "World")
    ].copy()

    if world.empty:
        raise ValueError("No World observations found.")

    world["DEBT_YEAR"] = pd.to_numeric(
        world["DEBT_YEAR"], errors="coerce"
    )

    world = world.sort_values("DEBT_YEAR")

    if world["DEBT_YEAR"].duplicated().any():
        duplicates = world.loc[
            world["DEBT_YEAR"].duplicated(keep=False),
            ["DEBT_COUNTRY", "DEBT_YEAR"],
        ]
        raise ValueError(
            "Duplicate World-year rows found:\n"
            + duplicates.to_string(index=False)
        )

    observed_years = set(
        world["DEBT_YEAR"].dropna().astype(int).tolist()
    )

    missing_world_years = sorted(EXPECTED_YEARS - observed_years)
    unexpected_world_years = sorted(observed_years - EXPECTED_YEARS)

    print(f"World observations: {len(world)}")
    print(f"First year: {int(world['DEBT_YEAR'].min())}")
    print(f"Last year: {int(world['DEBT_YEAR'].max())}")
    print(f"Missing World years: {missing_world_years}")
    print(f"Unexpected World years: {unexpected_world_years}")

    # ---------------------------------------------------------
    # 1. Per-series value and coverage profile
    # ---------------------------------------------------------
    profile_rows = []

    numeric_series = {}

    for field in AGGREGATE_FIELDS:
        raw = world[field]
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_series[field] = numeric

        populated_mask = raw.notna()
        numeric_mask = numeric.notna()

        populated_years = world.loc[
            numeric_mask, "DEBT_YEAR"
        ].astype(int).tolist()

        first_year = min(populated_years) if populated_years else None
        last_year = max(populated_years) if populated_years else None

        expected_in_range = (
            set(range(first_year, last_year + 1))
            if first_year is not None
            else set()
        )

        gaps = sorted(expected_in_range - set(populated_years))

        non_numeric_count = int(
            (populated_mask & ~numeric_mask).sum()
        )

        profile_rows.append({
            "series_id": field,
            "world_rows": len(world),
            "populated_cells": int(populated_mask.sum()),
            "numeric_cells": int(numeric_mask.sum()),
            "missing_cells": int(raw.isna().sum()),
            "non_numeric_populated_cells": non_numeric_count,
            "first_year": first_year,
            "last_year": last_year,
            "internal_missing_year_count": len(gaps),
            "internal_missing_years": ";".join(map(str, gaps)),
            "zero_count": int((numeric == 0).sum()),
            "negative_count": int((numeric < 0).sum()),
            "positive_count": int((numeric > 0).sum()),
            "minimum": numeric.min(),
            "maximum": numeric.max(),
            "mean": numeric.mean(),
            "unique_numeric_values": int(numeric.nunique()),
        })

    profile = pd.DataFrame(profile_rows)
    write_csv(profile, "boc_boe_aggregate_series_value_profile.csv")

    # ---------------------------------------------------------
    # 2. Annual observation-level values
    # ---------------------------------------------------------
    observation_columns = ["DEBT_COUNTRY", "DEBT_COUNTRY_GROUP", "DEBT_YEAR"]
    observation_columns += AGGREGATE_FIELDS

    observations = world[observation_columns].copy()
    write_csv(
        observations,
        "boc_boe_aggregate_series_integrity_observations.csv",
    )

    # ---------------------------------------------------------
    # 3. Exact duplicate series detection
    # ---------------------------------------------------------
    duplicate_rows = []

    for field_a, field_b in combinations(AGGREGATE_FIELDS, 2):
        a = numeric_series[field_a]
        b = numeric_series[field_b]

        common = a.notna() & b.notna()
        common_count = int(common.sum())

        if common_count == 0:
            continue

        differences = (a[common] - b[common]).abs()
        exact_equal = bool((differences == 0).all())

        correlation = (
            a[common].corr(b[common])
            if common_count >= 2
            else None
        )

        duplicate_rows.append({
            "series_a": field_a,
            "series_b": field_b,
            "common_observations": common_count,
            "exactly_equal_on_common_years": exact_equal,
            "maximum_absolute_difference": differences.max(),
            "correlation": correlation,
        })

    pairwise = pd.DataFrame(duplicate_rows)

    if not pairwise.empty:
        pairwise = pairwise.sort_values(
            ["exactly_equal_on_common_years", "common_observations"],
            ascending=[False, False],
        )

    write_csv(
        pairwise,
        "boc_boe_aggregate_series_pairwise_comparison.csv",
    )

    exact_pairs = (
        pairwise.loc[pairwise["exactly_equal_on_common_years"]]
        if not pairwise.empty
        else pd.DataFrame()
    )

    # ---------------------------------------------------------
    # 4. Report
    # ---------------------------------------------------------
    report = []
    report.append("=" * 78)
    report.append("BoC-BOE AGGREGATE SERIES VALUE-INTEGRITY AUDIT")
    report.append("=" * 78)
    report.append("")
    report.append("SOURCE")
    report.append(str(SOURCE))
    report.append("")
    report.append("WORLD ENTITY CHECK")
    report.append(f"World rows: {len(world)}")
    report.append(f"Observed years: {int(world['DEBT_YEAR'].min())}-{int(world['DEBT_YEAR'].max())}")
    report.append(f"Missing years in 1960-2024: {missing_world_years}")
    report.append(f"Unexpected years: {unexpected_world_years}")
    report.append("")
    report.append("SERIES VALUE PROFILE")
    report.append(profile.to_string(index=False))
    report.append("")
    report.append("EXACT DUPLICATE SERIES PAIRS")
    if exact_pairs.empty:
        report.append("No exact duplicate pairs detected on common observations.")
    else:
        report.append(exact_pairs.to_string(index=False))
    report.append("")
    report.append("HIGH CORRELATION PAIRS")
    if pairwise.empty:
        report.append("No pairwise comparisons available.")
    else:
        high_corr = pairwise.loc[
            pairwise["correlation"].abs() >= 0.999
        ].sort_values("correlation", ascending=False)
        if high_corr.empty:
            report.append("No pairs with absolute correlation >= 0.999.")
        else:
            report.append(high_corr.to_string(index=False))
    report.append("")
    report.append("INTERPRETATION LIMITATIONS")
    report.append(
        "- Numeric validity does not establish semantic validity or units."
    )
    report.append(
        "- Exact equality on common years identifies numerical duplication, "
        "not necessarily identical documented definitions."
    )
    report.append(
        "- High correlation does not establish that two series measure the same thing."
    )
    report.append(
        "- Missing values are not treated as zero."
    )
    report.append(
        "- No default events, probabilities, or severity scores are inferred."
    )
    report.append("")

    report_path = REPORTS / "boc_boe_aggregate_series_value_integrity_audit.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("")
    print("OUTPUTS")
    print(f"Profile: {OUTPUT / 'boc_boe_aggregate_series_value_profile.csv'}")
    print(f"Observations: {OUTPUT / 'boc_boe_aggregate_series_integrity_observations.csv'}")
    print(f"Pairwise comparisons: {OUTPUT / 'boc_boe_aggregate_series_pairwise_comparison.csv'}")
    print(f"Report: {report_path}")
    print("")
    print("AUDIT COMPLETED.")


if __name__ == "__main__":
    main()