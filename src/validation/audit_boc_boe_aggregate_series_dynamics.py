from pathlib import Path
from itertools import combinations

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"
OUTPUT = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

FIELDS = [
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


def main():
    df = pd.read_excel(
        SOURCE,
        sheet_name="Debt_2025",
        header=64,
    )

    world = df.loc[
        (df["DEBT_COUNTRY"] == "World")
        & (df["DEBT_COUNTRY_GROUP"] == "World")
    ].copy()

    world["DEBT_YEAR"] = pd.to_numeric(
        world["DEBT_YEAR"], errors="coerce"
    )

    world = world.sort_values("DEBT_YEAR").set_index("DEBT_YEAR")

    data = world[FIELDS].apply(pd.to_numeric, errors="coerce")

    # First differences: change from one year to the next.
    differences = data.diff()

    rows = []

    for a, b in combinations(FIELDS, 2):
        level_pair = data[[a, b]].dropna()
        diff_pair = differences[[a, b]].dropna()

        level_pearson = (
            level_pair[a].corr(level_pair[b], method="pearson")
            if len(level_pair) >= 3 else None
        )

        level_spearman = (
            level_pair[a].corr(level_pair[b], method="spearman")
            if len(level_pair) >= 3 else None
        )

        diff_pearson = (
            diff_pair[a].corr(diff_pair[b], method="pearson")
            if len(diff_pair) >= 3 else None
        )

        diff_spearman = (
            diff_pair[a].corr(diff_pair[b], method="spearman")
            if len(diff_pair) >= 3 else None
        )

        rows.append({
            "series_a": a,
            "series_b": b,
            "level_n": len(level_pair),
            "level_pearson": level_pearson,
            "level_spearman": level_spearman,
            "difference_n": len(diff_pair),
            "difference_pearson": diff_pearson,
            "difference_spearman": diff_spearman,
            "pearson_absolute_change": (
                abs(level_pearson) - abs(diff_pearson)
                if level_pearson is not None
                and diff_pearson is not None
                else None
            ),
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "level_pearson",
        key=lambda x: x.abs(),
        ascending=False,
    )

    output_file = OUTPUT / "boc_boe_aggregate_series_dynamics.csv"
    result.to_csv(output_file, index=False)

    high_level = result.loc[
        result["level_pearson"].abs() >= 0.90
    ]

    high_level.to_csv(
        OUTPUT / "boc_boe_aggregate_high_level_correlations.csv",
        index=False,
    )

    high_diff = result.loc[
        result["difference_pearson"].abs() >= 0.90
    ]

    high_diff.to_csv(
        OUTPUT / "boc_boe_aggregate_high_difference_correlations.csv",
        index=False,
    )

    report = [
        "=" * 78,
        "BoC-BOE AGGREGATE SERIES DYNAMICS AUDIT",
        "=" * 78,
        "",
        f"World annual observations: {len(data)}",
        f"Series assessed: {len(FIELDS)}",
        f"Pairwise comparisons: {len(result)}",
        "",
        "LEVEL CORRELATION >= 0.90",
        high_level.to_string(index=False),
        "",
        "FIRST-DIFFERENCE CORRELATION >= 0.90",
        high_diff.to_string(index=False),
        "",
        "INTERPRETATION",
        "- First differences measure annual changes, not levels.",
        "- High level correlation that weakens after differencing may reflect shared trends.",
        "- High first-difference correlation indicates stronger short-run co-movement,",
        "  but does not establish causation or identical definitions.",
        "- No series is removed or declared redundant by this audit.",
        "- No stationarity, cointegration or structural-break test is performed.",
    ]

    report_file = (
        REPORTS / "boc_boe_aggregate_series_dynamics_audit.txt"
    )
    report_file.write_text("\n".join(report), encoding="utf-8")

    print("AUDIT COMPLETED.")
    print(f"Full results: {output_file}")
    print(f"High level correlations: "
          f"{OUTPUT / 'boc_boe_aggregate_high_level_correlations.csv'}")
    print(f"High difference correlations: "
          f"{OUTPUT / 'boc_boe_aggregate_high_difference_correlations.csv'}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()