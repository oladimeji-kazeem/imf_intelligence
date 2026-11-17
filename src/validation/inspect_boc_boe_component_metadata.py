
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "data/external/"
    "boc_boe_sovereign_default_database_2025.xlsx"
)

OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PEERS = [
    "Greece",
    "Hungary",
    "Ireland",
    "Poland",
    "Portugal",
    "Romania",
    "United Kingdom",
]

METADATA_OUTPUT = (
    REPORT_DIR / "boc_boe_series_metadata_inspection.txt"
)

SAMPLE_OUTPUT = (
    OUTPUT_DIR / "boc_boe_original_positive_observations.csv"
)


def main():

    print("Opening original BoC–BoE workbook...")

    wb = load_workbook(
        SOURCE,
        read_only=True,
        data_only=True
    )

    if "Debt_2025" not in wb.sheetnames:
        raise ValueError("Debt_2025 sheet not found.")

    ws = wb["Debt_2025"]

    lines = [
        "BOC–BOE SERIES METADATA INSPECTION",
        "=" * 75,
        "",
        "NON-EMPTY CELLS IN METADATA ROWS 1–64",
        "-" * 75,
    ]

    for row in ws.iter_rows(
        min_row=1,
        max_row=64,
        values_only=False
    ):
        entries = []

        for cell in row:
            if cell.value is not None:
                entries.append(
                    f"{cell.coordinate}={cell.value}"
                )

        if entries:
            lines.append(" | ".join(entries))

    lines.extend([
        "",
        "OBSERVATION TABLE HEADER",
        "-" * 75,
    ])

    headers = [
        cell.value
        for cell in ws[65]
    ]

    for index, header in enumerate(headers, start=1):
        if header is not None:
            lines.append(f"{index}: {header}")

    METADATA_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    wb.close()

    # Independently extract positive observations using pandas.
    df = pd.read_excel(
        SOURCE,
        sheet_name="Debt_2025",
        header=64
    )

    df["DEBT_COUNTRY"] = (
        df["DEBT_COUNTRY"]
        .astype("string")
        .str.strip()
    )

    df["DEBT_TOTAL_2025"] = pd.to_numeric(
        df["DEBT_TOTAL_2025"],
        errors="coerce"
    )

    sample = df[
        df["DEBT_COUNTRY"].isin(PEERS)
        & (df["DEBT_TOTAL_2025"] > 0)
    ].copy()

    sample = sample.sort_values(
        ["DEBT_COUNTRY", "DEBT_YEAR"]
    )

    sample.to_csv(
        SAMPLE_OUTPUT,
        index=False
    )

    print("\nMETADATA INSPECTION COMPLETED")
    print(f"Metadata report: {METADATA_OUTPUT}")
    print(f"Original positive records: {len(sample)}")
    print(f"Original countries: {sample['DEBT_COUNTRY'].nunique()}")
    print(f"Original observations: {SAMPLE_OUTPUT}")


if __name__ == "__main__":
    main()
