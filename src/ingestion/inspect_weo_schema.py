from pathlib import Path
import pandas as pd
import re


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
EXPORT_DIR = ROOT / "outputs" / "exports"
REPORT_DIR = ROOT / "outputs" / "reports"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD WEO WORKBOOK
# --------------------------------------------------

def load_weo():

    files = list(RAW_DIR.glob("*.xlsx"))

    if len(files) != 1:
        raise ValueError(
            "Expected exactly one Excel file in data/raw."
        )

    file_path = files[0]

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
        "SCALE": "Scale",
        "COUNTRY.ID": "ISO"
    }
    for modern_col, legacy_col in column_mapping.items():
        if modern_col in data.columns and legacy_col not in data.columns:
            data[legacy_col] = data[modern_col]

    return data, file_path


# --------------------------------------------------
# INSPECT COLUMN STRUCTURE
# --------------------------------------------------

def inspect_columns(data):

    print("\nWEO COLUMN STRUCTURE")
    print("=" * 60)

    for number, column in enumerate(
        data.columns, start=1
    ):
        print(f"{number:02d}. {column}")

    year_columns = [
        column for column in data.columns
        if re.fullmatch(r"\d{4}", str(column))
    ]

    print("\nYEAR COVERAGE")
    print("=" * 60)

    if year_columns:
        print("First year:", year_columns[0])
        print("Last year:", year_columns[-1])
        print("Number of year columns:", len(year_columns))
    else:
        print("No year columns detected.")


# --------------------------------------------------
# SEARCH INDICATOR DESCRIPTIONS
# --------------------------------------------------

def find_indicator_candidates(data):

    patterns = {
        "real_gdp_growth": r"real gdp growth",
        "nominal_gdp": r"gross domestic product.*current prices",
        "inflation": r"inflation.*average consumer prices",
        "government_debt": r"general government gross debt",
        "fiscal_balance": r"general government.*net lending|net borrowing",
        "government_revenue": r"general government.*revenue",
        "government_expenditure": r"general government.*expenditure",
        "unemployment": r"unemployment rate",
        "current_account": r"current account balance",
        "exports": r"exports of goods and services",
        "imports": r"imports of goods and services"
    }

    required = [
        "WEO Subject Code",
        "Subject Descriptor",
        "Units",
        "Scale"
    ]

    for column in required:
        if column not in data.columns:
            raise ValueError(
                f"Missing expected column: {column}"
            )

    catalogue = (
        data[required]
        .drop_duplicates()
        .copy()
    )

    catalogue["Subject Descriptor"] = (
        catalogue["Subject Descriptor"]
        .astype(str)
        .str.strip()
    )

    results = []

    for internal_name, pattern in patterns.items():

        matches = catalogue[
            catalogue["Subject Descriptor"].str.contains(
                pattern,
                case=False,
                regex=True,
                na=False
            )
        ].copy()

        matches.insert(
            0,
            "Internal Indicator",
            internal_name
        )

        results.append(matches)

    candidates = pd.concat(
        results,
        ignore_index=True
    )

    output_file = (
        EXPORT_DIR / "weo_indicator_candidates.csv"
    )

    candidates.to_csv(
        output_file,
        index=False
    )

    print("\nINDICATOR CANDIDATES")
    print("=" * 60)

    if candidates.empty:
        print("No candidate indicators found.")
    else:
        print(
            candidates.to_string(index=False)
        )

    print("\nCandidate catalogue saved:")
    print(output_file)

    return candidates


# --------------------------------------------------
# INSPECT COUNTRY IDENTIFIERS
# --------------------------------------------------

def inspect_countries(data):

    print("\nCOUNTRY IDENTIFIER STRUCTURE")
    print("=" * 60)

    for column in data.columns:

        if column.lower() in [
            "iso",
            "weo country code",
            "country",
            "weo country name",
            "country.id"
        ]:
            print(f"\nColumn: {column}")

            print(
                data[column]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(15)
                .to_list()
            )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    data, file_path = load_weo()

    print("Workbook:", file_path.name)
    print("Records:", len(data))

    inspect_columns(data)

    candidates = find_indicator_candidates(data)

    inspect_countries(data)

    report = REPORT_DIR / "weo_schema_report.txt"

    with open(report, "w", encoding="utf-8") as file:

        file.write("IMF WEO SCHEMA INSPECTION\n")
        file.write("=" * 60 + "\n\n")

        file.write("Workbook: " + file_path.name + "\n\n")

        file.write("COLUMNS\n")

        for column in data.columns:
            file.write(str(column) + "\n")

        file.write("\nINDICATOR CANDIDATES\n")
        file.write(
            candidates.to_string(index=False)
            if not candidates.empty
            else "No candidates found."
        )

    print("\nSchema inspection completed.")
    print("Report:", report)


if __name__ == "__main__":
    main()
