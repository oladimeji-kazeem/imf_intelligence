
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "data/processed/"
    "boc_boe_europe_positive_default_observations.csv"
)

OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

CREDITOR_FIELDS = [
    "DEBT_IMF_2025",
    "DEBT_IBRD_2025",
    "DEBT_IDA_2025",
    "DEBT_IADB_2025",
    "DEBT_PARIS_CLUB_2025",
    "DEBT_CHINA_2025",
    "DEBT_OTHER_OFFICIAL_CREDITORS_2025",
    "DEBT_PRIVATE_CREDITORS_2025",
]

INSTRUMENT_FIELDS = [
    "DEBT_FC_BANK_LOANS_2025",
    "DEBT_FC_BONDS_2025",
    "DEBT_LC_DEBT_2025",
    "DEBT_FISCAL_ARREARS_2025",
]

DETAIL_OUTPUT = (
    OUTPUT_DIR / "boc_boe_component_missingness_by_country.csv"
)

ROW_OUTPUT = (
    OUTPUT_DIR / "boc_boe_component_row_availability.csv"
)

COOBS_OUTPUT = (
    OUTPUT_DIR / "boc_boe_component_coavailability.csv"
)

REPORT_OUTPUT = (
    REPORT_DIR / "boc_boe_component_missingness_audit.txt"
)


def classify(value):
    if pd.isna(value):
        return "MISSING"
    if value == 0:
        return "ZERO"
    if value > 0:
        return "POSITIVE"
    return "NEGATIVE"


def main():

    df = pd.read_csv(SOURCE)

    creditor_fields = [
        c for c in CREDITOR_FIELDS if c in df.columns
    ]

    instrument_fields = [
        c for c in INSTRUMENT_FIELDS if c in df.columns
    ]

    components = creditor_fields + instrument_fields

    if not components:
        raise ValueError("No component fields found.")

    for col in components:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    detail_rows = []

    for country, group in df.groupby("DEBT_COUNTRY"):

        for component in components:

            values = group[component]

            detail_rows.append({
                "country": country,
                "component_group": (
                    "CREDITOR"
                    if component in creditor_fields
                    else "INSTRUMENT"
                ),
                "component": component,
                "observations": len(group),
                "positive": int((values > 0).sum()),
                "zero": int((values == 0).sum()),
                "missing": int(values.isna().sum()),
                "negative": int((values < 0).sum()),
                "availability_pct": round(
                    values.notna().mean() * 100, 2
                ),
            })

    detail = pd.DataFrame(detail_rows)

    detail.to_csv(DETAIL_OUTPUT, index=False)

    # Row-level availability.
    row_data = df[
        ["DEBT_COUNTRY", "DEBT_YEAR", "DEBT_TOTAL_2025"]
    ].copy()

    row_data["creditor_components_available"] = (
        df[creditor_fields].notna().sum(axis=1)
    )

    row_data["instrument_components_available"] = (
        df[instrument_fields].notna().sum(axis=1)
    )

    row_data["creditor_components_missing"] = (
        len(creditor_fields)
        - row_data["creditor_components_available"]
    )

    row_data["instrument_components_missing"] = (
        len(instrument_fields)
        - row_data["instrument_components_available"]
    )

    row_data.to_csv(ROW_OUTPUT, index=False)

    # Pairwise co-availability.
    # Counts rows where both component fields are populated.
    coobs_rows = []

    for fields, group_name in [
        (creditor_fields, "CREDITOR"),
        (instrument_fields, "INSTRUMENT"),
    ]:

        for i, field_a in enumerate(fields):
            for field_b in fields[i + 1:]:

                coobs_rows.append({
                    "component_group": group_name,
                    "component_a": field_a,
                    "component_b": field_b,
                    "both_available": int(
                        (
                            df[field_a].notna()
                            & df[field_b].notna()
                        ).sum()
                    ),
                    "both_positive": int(
                        (
                            (df[field_a] > 0)
                            & (df[field_b] > 0)
                        ).sum()
                    ),
                })

    coobs = pd.DataFrame(coobs_rows)
    coobs.to_csv(COOBS_OUTPUT, index=False)

    lines = [
        "BOC–BOE COMPONENT MISSINGNESS AUDIT",
        "=" * 70,
        "",
        f"Positive country-year observations: {len(df)}",
        f"Countries: {df['DEBT_COUNTRY'].nunique()}",
        "",
        "ROW-LEVEL COMPONENT AVAILABILITY",
        "-" * 70,
        f"Creditor fields available per row: "
        f"min={row_data['creditor_components_available'].min()}, "
        f"median={row_data['creditor_components_available'].median()}, "
        f"max={row_data['creditor_components_available'].max()}",
        f"Instrument fields available per row: "
        f"min={row_data['instrument_components_available'].min()}, "
        f"median={row_data['instrument_components_available'].median()}, "
        f"max={row_data['instrument_components_available'].max()}",
        "",
        "COMPONENT AVAILABILITY BY COUNTRY",
        "-" * 70,
    ]

    for country in sorted(df["DEBT_COUNTRY"].unique()):

        lines.append(f"\n{country}")

        subset = detail[detail["country"] == country]

        for _, row in subset.iterrows():
            lines.append(
                f"  {row['component']} | "
                f"positive={row['positive']} | "
                f"zero={row['zero']} | "
                f"missing={row['missing']} | "
                f"availability={row['availability_pct']}%"
            )

    lines.extend([
        "",
        "PAIRWISE COMPONENT CO-AVAILABILITY",
        "-" * 70,
    ])

    for _, row in coobs.sort_values(
        ["component_group", "both_available"],
        ascending=[True, False]
    ).iterrows():

        lines.append(
            f"{row['component_group']} | "
            f"{row['component_a']} + {row['component_b']} | "
            f"both_available={row['both_available']} | "
            f"both_positive={row['both_positive']}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 70,
        "Missing values are not treated as zero.",
        "Availability does not establish component completeness.",
        "Pairwise availability does not establish additivity.",
        "No total reconciliation is attempted in this audit.",
        "",
        "OUTPUT FILES",
        "-" * 70,
        str(DETAIL_OUTPUT),
        str(ROW_OUTPUT),
        str(COOBS_OUTPUT),
        str(REPORT_OUTPUT),
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
