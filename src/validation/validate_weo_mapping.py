
from pathlib import Path
import hashlib
import pandas as pd
import yaml

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

WORKBOOK = ROOT / "data" / "raw" / "WEOOct2026all.xlsx"
MAPPING_FILE = ROOT / "config" / "weo_indicator_mapping.yaml"

OUTPUT_DIR = ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "weo_mapping_validation.txt"

# --------------------------------------------------
# LOAD CONFIGURATION
# --------------------------------------------------

print("=" * 65)
print("IMF ECONOMIC INTELLIGENCE LAB")
print("WEO INDICATOR MAPPING VALIDATION")
print("=" * 65)

if not WORKBOOK.exists():
    raise FileNotFoundError(f"WEO workbook not found: {WORKBOOK}")

if not MAPPING_FILE.exists():
    raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")

with open(MAPPING_FILE, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

indicators = config["indicators"]

# --------------------------------------------------
# LOAD WORKBOOK
# --------------------------------------------------

df = pd.read_excel(
    WORKBOOK,
    sheet_name="Countries",
    engine="openpyxl"
)

df.columns = [str(col).strip() for col in df.columns]

column_mapping = {
    "INDICATOR.ID": "WEO Subject Code",
    "INDICATOR": "Subject Descriptor",
    "UNIT": "Units",
    "SCALE": "Scale",
    "COUNTRY.ID": "ISO"
}
for modern_col, legacy_col in column_mapping.items():
    if modern_col in df.columns and legacy_col not in df.columns:
        df[legacy_col] = df[modern_col]

# --------------------------------------------------
# WORKBOOK STRUCTURE
# --------------------------------------------------

print("\n1. WORKBOOK STRUCTURE")
print("-" * 65)

print(f"Filename: {WORKBOOK.name}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

blank_columns = [
    str(col) for col in df.columns
    if str(col).startswith("Unnamed:")
]

duplicate_columns = (
    pd.Index(df.columns)[pd.Index(df.columns).duplicated()]
    .tolist()
)

print(f"Unnamed columns: {len(blank_columns)}")
print(f"Duplicate column names: {len(duplicate_columns)}")

print("\nFirst 10 columns:")
for col in df.columns[:10]:
    print(f"- {col}")

print("\nLast 10 columns:")
for col in df.columns[-10:]:
    print(f"- {col}")

# --------------------------------------------------
# VERIFY REQUIRED SOURCE COLUMNS
# --------------------------------------------------

required_columns = [
    "WEO Subject Code",
    "Subject Descriptor",
    "Units",
    "Scale"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required WEO columns missing: {missing_columns}"
    )

# --------------------------------------------------
# VALIDATE INDICATOR MAPPINGS
# --------------------------------------------------

print("\n2. INDICATOR MAPPING VALIDATION")
print("-" * 65)

results = []

for internal_name, details in indicators.items():

    code = details["weo_code"]

    matched = df[
        df["WEO Subject Code"].astype(str).str.strip() == code
    ]

    if matched.empty:
        results.append({
            "internal_name": internal_name,
            "weo_code": code,
            "status": "FAILED - CODE NOT FOUND",
            "descriptor": None,
            "units": None,
            "scale": None
        })

        print(f"FAILED: {internal_name} ({code})")
        continue

    if len(matched) > 1:

        unique_descriptors = matched["Subject Descriptor"].nunique()
        unique_units = matched["Units"].nunique()
        unique_scales = matched["Scale"].nunique()

        if (
            unique_descriptors <= 1
            and unique_units <= 1
            and unique_scales <= 1
        ):
            status = f"PASSED - {len(matched)} COUNTRY RECORDS"
        else:
            status = "WARNING - INCONSISTENT METADATA"

    else:
        status = "PASSED - SINGLE RECORD"

    source = matched.iloc[0]

    descriptor = str(source["Subject Descriptor"]).strip()
    units = str(source["Units"]).strip()
    scale = str(source["Scale"]).strip()

    results.append({
        "internal_name": internal_name,
        "weo_code": code,
        "status": status,
        "descriptor": descriptor,
        "units": units,
        "scale": scale
    })

    print(f"\nIndicator: {internal_name}")
    print(f"WEO code: {code}")
    print(f"Status: {status}")
    print(f"Descriptor: {descriptor}")
    print(f"Units: {units}")
    print(f"Scale: {scale}")

# --------------------------------------------------
# VALIDATION SUMMARY
# --------------------------------------------------

results_df = pd.DataFrame(results)

passed = results_df["status"].str.startswith("PASSED").sum()
failed = results_df["status"].str.startswith("FAILED").sum()
warnings = results_df["status"].str.startswith("WARNING").sum()

print("\n3. VALIDATION SUMMARY")
print("-" * 65)

print(f"Total mapped indicators: {len(indicators)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Warnings: {warnings}")

# --------------------------------------------------
# WORKBOOK FINGERPRINT
# --------------------------------------------------

sha256 = hashlib.sha256()

with open(WORKBOOK, "rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        sha256.update(chunk)

# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as report:

    report.write("IMF WEO MAPPING VALIDATION REPORT\n")
    report.write("=" * 65 + "\n\n")

    report.write(f"Workbook: {WORKBOOK.name}\n")
    report.write(f"Rows: {df.shape[0]}\n")
    report.write(f"Columns: {df.shape[1]}\n")
    report.write(f"SHA-256: {sha256.hexdigest()}\n\n")

    report.write(f"Passed: {passed}\n")
    report.write(f"Failed: {failed}\n")
    report.write(f"Warnings: {warnings}\n\n")

    report.write(results_df.to_string(index=False))

    report.write("\n\nCOLUMN DIAGNOSTICS\n")
    report.write(f"Unnamed columns: {blank_columns}\n")
    report.write(f"Duplicate columns: {duplicate_columns}\n")

print("\n4. REPORT")
print("-" * 65)
print(f"Saved: {REPORT_FILE}")

# --------------------------------------------------
# FINAL STATUS
# --------------------------------------------------

if failed == 0 and warnings == 0:
    print("\nMAPPING VALIDATION PASSED.")
else:
    print("\nMAPPING VALIDATION REQUIRES REVIEW.")

print("=" * 65)

print("\n3. COUNTRY-INDICATOR DUPLICATE CHECK")
print("-" * 65)

duplicate_keys = df.duplicated(
    subset=["ISO", "WEO Subject Code"],
    keep=False
)

duplicate_records = df.loc[duplicate_keys]

print(f"Duplicate country-indicator records: {len(duplicate_records)}")

if not duplicate_records.empty:
    print(duplicate_records[
        ["ISO", "COUNTRY", "WEO Subject Code"]
    ].head(20).to_string(index=False))
else:
    print("No duplicate country-indicator records detected.")
