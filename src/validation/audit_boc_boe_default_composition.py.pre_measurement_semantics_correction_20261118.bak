
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/processed/boc_boe_europe_positive_default_observations.csv"
OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    OUTPUT_DIR / "boc_boe_default_composition_reconciliation.csv"
)

COMPONENT_OUTPUT = (
    OUTPUT_DIR / "boc_boe_default_component_profile.csv"
)

REPORT_OUTPUT = (
    REPORT_DIR / "boc_boe_default_composition_audit.txt"
)


def main():

    df = pd.read_csv(SOURCE)

    required = {
        "DEBT_COUNTRY",
        "DEBT_YEAR",
        "DEBT_TOTAL_2025",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required fields: {sorted(missing)}"
        )

    creditor_fields = [
        col for col in CREDITOR_FIELDS if col in df.columns
    ]

    instrument_fields = [
        col for col in INSTRUMENT_FIELDS if col in df.columns
    ]

    if not creditor_fields or not instrument_fields:
        raise ValueError(
            "Expected creditor or instrument fields are missing."
        )

    numeric_fields = [
        "DEBT_TOTAL_2025",
        *creditor_fields,
        *instrument_fields,
    ]

    for col in numeric_fields:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DEBT_YEAR"] = pd.to_numeric(
        df["DEBT_YEAR"], errors="coerce"
    )

    # Work only with positive total debt-in-default observations.
    df = df[df["DEBT_TOTAL_2025"] > 0].copy()

    if df.empty:
        raise ValueError("No positive observations found.")

    # Do not silently treat missing components as zero.
    df["creditor_components_missing"] = (
        df[creditor_fields].isna().sum(axis=1)
    )

    df["instrument_components_missing"] = (
        df[instrument_fields].isna().sum(axis=1)
    )

    # Subtotals are calculated only where all components are present.
    df["creditor_components_complete"] = (
        df["creditor_components_missing"] == 0
    )

    df["instrument_components_complete"] = (
        df["instrument_components_missing"] == 0
    )

    df["creditor_subtotal"] = (
        df[creditor_fields].sum(axis=1, min_count=len(creditor_fields))
    )

    df["instrument_subtotal"] = (
        df[instrument_fields].sum(
            axis=1, min_count=len(instrument_fields)
        )
    )

    df["creditor_residual"] = (
        df["DEBT_TOTAL_2025"] - df["creditor_subtotal"]
    )

    df["instrument_residual"] = (
        df["DEBT_TOTAL_2025"] - df["instrument_subtotal"]
    )

    df["creditor_reconciliation_pct"] = (
        df["creditor_subtotal"] / df["DEBT_TOTAL_2025"] * 100
    )

    df["instrument_reconciliation_pct"] = (
        df["instrument_subtotal"] / df["DEBT_TOTAL_2025"] * 100
    )

    df.to_csv(DETAIL_OUTPUT, index=False)

    # Component profile: count positive observations and missing values.
    component_rows = []

    for group_name, fields in [
        ("CREDITOR", creditor_fields),
        ("INSTRUMENT", instrument_fields),
    ]:

        for field in fields:

            values = df[field]

            component_rows.append({
                "classification_group": group_name,
                "component": field,
                "positive_observations": int(
                    (values > 0).sum()
                ),
                "zero_observations": int(
                    (values == 0).sum()
                ),
                "missing_observations": int(
                    values.isna().sum()
                ),
                "maximum_observed_value": (
                    values.max() if values.notna().any() else None
                ),
            })

    component_profile = pd.DataFrame(component_rows)

    component_profile.to_csv(
        COMPONENT_OUTPUT, index=False
    )

    # Reconciliation counts.
    creditor_complete = df[df["creditor_components_complete"]]
    instrument_complete = df[df["instrument_components_complete"]]

    tolerance = 0.01

    creditor_matches = (
        creditor_complete["creditor_residual"].abs() <= tolerance
    )

    instrument_matches = (
        instrument_complete["instrument_residual"].abs() <= tolerance
    )

    lines = [
        "BOC–BOE DEFAULT COMPOSITION AUDIT",
        "=" * 70,
        "",
        f"Positive country-year observations: {len(df)}",
        f"Countries represented: {df['DEBT_COUNTRY'].nunique()}",
        "",
        "CREDITOR COMPONENT RECONCILIATION",
        "-" * 70,
        f"Complete component observations: {len(creditor_complete)}",
        f"Matching total within tolerance: {int(creditor_matches.sum())}",
        f"Non-matching totals: {int((~creditor_matches).sum())}",
        f"Observations with missing components: "
        f"{int((~df['creditor_components_complete']).sum())}",
        "",
        "INSTRUMENT COMPONENT RECONCILIATION",
        "-" * 70,
        f"Complete component observations: {len(instrument_complete)}",
        f"Matching total within tolerance: {int(instrument_matches.sum())}",
        f"Non-matching totals: {int((~instrument_matches).sum())}",
        f"Observations with missing components: "
        f"{int((~df['instrument_components_complete']).sum())}",
        "",
        "COMPONENT PROFILE",
        "-" * 70,
    ]

    for _, row in component_profile.iterrows():
        lines.append(
            f"{row['classification_group']} | "
            f"{row['component']} | "
            f"positive={row['positive_observations']} | "
            f"zero={row['zero_observations']} | "
            f"missing={row['missing_observations']} | "
            f"maximum={row['maximum_observed_value']}"
        )

    lines.extend([
        "",
        "INTERPRETATION LIMITATIONS",
        "-" * 70,
        "Creditor and instrument classifications are separate dimensions.",
        "Their subtotals must not be added together.",
        "A residual does not automatically indicate an error.",
        "Missing components are not treated as zero.",
        "Reconciliation does not establish economic exclusivity.",
        "Annual observations are stocks, not independent default events.",
        "",
        "OUTPUT FILES",
        "-" * 70,
        str(DETAIL_OUTPUT),
        str(COMPONENT_OUTPUT),
        str(REPORT_OUTPUT),
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
