from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd
import requests


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPORT_DIR = PROJECT_ROOT / "outputs" / "exports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

VINTAGE = "October 2026"

SOURCE_URL = (
    "https://data.imf.org/-/media/iData/External-Storage/"
    "Documents/0FA3DC49A04D446E9D69DBEB29979E1C/"
    "en/WEOOct2026all.xlsx"
)

FILE_PATH = RAW_DIR / "WEOOct2026all.xlsx"


# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

def download_weo():

    print("Starting IMF WEO download...")

    response = requests.get(
        SOURCE_URL,
        headers={"User-Agent": "curl/8.4.0"},
        timeout=180
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if "spreadsheet" not in content_type and \
       "octet-stream" not in content_type:

        raise ValueError(
            f"Unexpected download content type: {content_type}"
        )

    if not response.content.startswith(b"PK"):
        raise ValueError(
            "Downloaded file does not appear to be an Excel file."
        )

    if FILE_PATH.exists():
        raise FileExistsError(
            f"File already exists: {FILE_PATH}\n"
            "Preserve the existing snapshot. "
            "Use a separate filename for a new download."
        )

    FILE_PATH.write_bytes(response.content)

    print(f"Downloaded: {FILE_PATH}")
    print(f"File size: {FILE_PATH.stat().st_size:,} bytes")

    return FILE_PATH


# --------------------------------------------------
# CHECKSUM
# --------------------------------------------------

def calculate_checksum(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# INSPECT WORKBOOK
# --------------------------------------------------

def inspect_workbook(file_path):

    excel = pd.ExcelFile(file_path)

    print("\nAvailable worksheets:")

    for sheet in excel.sheet_names:
        print(f"- {sheet}")

    if "Countries" not in excel.sheet_names:
        raise ValueError(
            "Expected Countries worksheet not found."
        )

    countries = pd.read_excel(
        file_path,
        sheet_name="Countries"
    )

    countries.columns = [
        str(column).strip()
        for column in countries.columns
    ]

    column_mapping = {
        "INDICATOR.ID": "WEO Subject Code",
        "INDICATOR": "Subject Descriptor",
        "UNIT": "Units",
        "SCALE": "Scale"
    }
    for modern_col, legacy_col in column_mapping.items():
        if modern_col in countries.columns and legacy_col not in countries.columns:
            countries[legacy_col] = countries[modern_col]

    print("\nCountries worksheet dimensions:")
    print(f"Rows: {len(countries):,}")
    print(f"Columns: {len(countries.columns):,}")

    print("\nColumn names:")

    for column in countries.columns:
        print(column)

    return countries, excel.sheet_names


# --------------------------------------------------
# INDICATOR CATALOGUE
# --------------------------------------------------

def create_indicator_catalogue(data):

    required_columns = [
        "WEO Subject Code",
        "Subject Descriptor",
        "Units",
        "Scale"
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing expected WEO columns: {missing}"
        )

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

    print(f"\nIndicator catalogue saved: {output_file}")

    return catalogue


# --------------------------------------------------
# INGESTION METADATA
# --------------------------------------------------

def save_metadata(file_path, sheets, data):

    metadata = {
        "source": "International Monetary Fund",
        "dataset": "World Economic Outlook",
        "vintage": VINTAGE,
        "retrieved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_url": SOURCE_URL,
        "filename": file_path.name,
        "sha256": calculate_checksum(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "worksheets": sheets,
        "countries_sheet_rows": len(data),
        "countries_sheet_columns": len(data.columns)
    }

    metadata_file = (
        RAW_DIR / "weo_download_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(f"Metadata saved: {metadata_file}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    file_path = download_weo()

    data, sheets = inspect_workbook(file_path)

    create_indicator_catalogue(data)

    save_metadata(
        file_path,
        sheets,
        data
    )

    print("\nIMF WEO ingestion completed successfully.")


if __name__ == "__main__":
    main()
