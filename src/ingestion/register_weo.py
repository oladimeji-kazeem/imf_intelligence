from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPORT_DIR = PROJECT_ROOT / "outputs" / "exports"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FIND WEO FILE
# --------------------------------------------------

def find_weo_file():

    files = list(RAW_DIR.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(
            "No Excel file found in data/raw/. "
            "Place your downloaded IMF WEO file there."
        )

    if len(files) > 1:
        print("Multiple Excel files found:")

        for index, file in enumerate(files, start=1):
            print(f"{index}. {file.name}")

        raise ValueError(
            "Keep only the intended WEO file in data/raw/ "
            "for this initial registration."
        )

    return files[0]


# --------------------------------------------------
# VALIDATE FILE
# --------------------------------------------------

def validate_excel(file_path):

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if file_path.stat().st_size == 0:
        raise ValueError("Excel file is empty.")

    excel = pd.ExcelFile(file_path)

    print("\nWORKBOOK INSPECTION")
    print("-------------------")

    print("Filename:", file_path.name)
    print("Worksheets:", excel.sheet_names)

    if "Countries" not in excel.sheet_names:
        raise ValueError(
            "Expected IMF WEO worksheet 'Countries' not found."
        )

    data = pd.read_excel(
        file_path,
        sheet_name="Countries"
    )

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    column_mapping = {
        "INDICATOR.ID": "WEO Subject Code",
        "INDICATOR": "Subject Descriptor",
        "UNIT": "Units",
        "SCALE": "Scale"
    }
    for modern_col, legacy_col in column_mapping.items():
        if modern_col in data.columns and legacy_col not in data.columns:
            data[legacy_col] = data[modern_col]

    expected_columns = [
        "WEO Subject Code",
        "Subject Descriptor",
        "Units",
        "Scale"
    ]

    missing = [
        column for column in expected_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}"
        )

    if data.empty:
        raise ValueError(
            "Countries worksheet contains no records."
        )

    print("\nDATA DIMENSIONS")
    print("-------------------")
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    return data, excel.sheet_names


# --------------------------------------------------
# CHECKSUM
# --------------------------------------------------

def calculate_checksum(file_path):

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# --------------------------------------------------
# CREATE INDICATOR CATALOGUE
# --------------------------------------------------

def create_indicator_catalogue(data):

    catalogue = (
        data[
            [
                "WEO Subject Code",
                "Subject Descriptor",
                "Units",
                "Scale"
            ]
        ]
        .drop_duplicates()
        .sort_values("WEO Subject Code")
    )

    output_file = (
        EXPORT_DIR / "weo_indicator_catalogue.csv"
    )

    catalogue.to_csv(
        output_file,
        index=False
    )

    print("\nINDICATOR CATALOGUE")
    print("-------------------")
    print("Unique indicators:", len(catalogue))
    print("Saved:", output_file)

    return catalogue


# --------------------------------------------------
# CREATE METADATA
# --------------------------------------------------

def create_metadata(file_path, data, sheets):

    metadata = {
        "source": "International Monetary Fund",
        "dataset": "World Economic Outlook",
        "vintage": "To be confirmed from workbook",
        "retrieved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "filename": file_path.name,
        "sha256": calculate_checksum(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "worksheets": sheets,
        "countries_sheet_rows": len(data),
        "countries_sheet_columns": len(data.columns)
    }

    output_file = RAW_DIR / "weo_download_metadata.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metadata, file, indent=4)

    print("\nMETADATA")
    print("-------------------")
    print("SHA-256:", metadata["sha256"])
    print("Saved:", output_file)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("IMF ECONOMIC INTELLIGENCE LAB")
    print("MANUAL WEO FILE REGISTRATION")
    print("=" * 60)

    file_path = find_weo_file()

    data, sheets = validate_excel(file_path)

    create_indicator_catalogue(data)

    create_metadata(
        file_path,
        data,
        sheets
    )

    print("\nRegistration completed successfully.")


if __name__ == "__main__":
    main()
