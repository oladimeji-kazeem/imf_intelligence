from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/external/kiel_monthly_sovereign_default_1970_2025.xlsx.xlsx"
DATA = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

CSV_OUT = DATA / "kiel_europe_peer_coverage.csv"
REPORT_OUT = REPORTS / "kiel_europe_peer_coverage_audit.txt"

PEERS = {
    "AUT": "Austria",
    "BEL": "Belgium",
    "CHE": "Switzerland",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "GRC": "Greece",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ITA": "Italy",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "SWE": "Sweden",
}

# Explicit source-code to canonical peer-code crosswalk.
WDI_TO_PEER_ISO3 = {
    "ROM": "ROU",
}

SHEET = "DATASET Defaults & Restruct."
HEADER_ROW = 6
FIRST_DATA_ROW = 7

REQUIRED_COLUMNS = [
    "Case nr",
    "Country case",
    "WDI code",
    "Start of default or restructuring process: default or announcement",
    "End of restructuring: completion of exchange",
    "Strictly preemptive",
    "Weakly preemptive",
    "Post-default",
    "Default date",
    "Announcement",
    "No exact start date (missing default month)",
]

if not SOURCE.is_file():
    raise SystemExit(f"STOP: source workbook not found: {SOURCE}")

for output in (CSV_OUT, REPORT_OUT):
    if output.exists():
        raise SystemExit(f"STOP: refusing to overwrite existing output: {output}")

wb = load_workbook(SOURCE, read_only=True, data_only=True)

try:
    if SHEET not in wb.sheetnames:
        raise SystemExit(f"STOP: required sheet not found: {SHEET}")

    ws = wb[SHEET]

    header_values = next(
        ws.iter_rows(
            min_row=HEADER_ROW,
            max_row=HEADER_ROW,
            values_only=True,
        )
    )
    headers = [
        str(value).strip() if value is not None else ""
        for value in header_values
    ]

    missing_headers = [
        column for column in REQUIRED_COLUMNS
        if column not in headers
    ]
    if missing_headers:
        raise SystemExit(
            f"STOP: missing required source columns: {missing_headers}"
        )

    index = {column: headers.index(column) for column in REQUIRED_COLUMNS}

    source_records = []
    all_case_numbers = []
    invalid_peer_codes = []
    invalid_flags = []
    invalid_dates = []

    flag_columns = [
        "Strictly preemptive",
        "Weakly preemptive",
        "Post-default",
        "No exact start date (missing default month)",
    ]

    for excel_row, row in enumerate(
        ws.iter_rows(min_row=FIRST_DATA_ROW, values_only=True),
        start=FIRST_DATA_ROW,
    ):
        country_value = row[index["Country case"]]
        if country_value is None or str(country_value).strip() == "":
            continue

        country = str(country_value).strip()
        source_code_value = row[index["WDI code"]]
        source_code = (
            str(source_code_value).strip()
            if source_code_value is not None else ""
        )
        peer_code = WDI_TO_PEER_ISO3.get(source_code, source_code)

        case_number = row[index["Case nr"]]
        if case_number is not None:
            all_case_numbers.append(case_number)

        flags = {}
        for column in flag_columns:
            value = row[index[column]]
            if value not in (0, 1):
                invalid_flags.append({
                    "excel_row": excel_row,
                    "case_number": case_number,
                    "column": column,
                    "value": repr(value),
                })
            flags[column] = value

        start = row[index[REQUIRED_COLUMNS[3]]]
        end = row[index[REQUIRED_COLUMNS[4]]]
        default_date = row[index["Default date"]]
        announcement = row[index["Announcement"]]

        for column, value in (
            ("start", start),
            ("end", end),
            ("default_date", default_date),
            ("announcement", announcement),
        ):
            if value is not None and not isinstance(value, datetime):
                invalid_dates.append({
                    "excel_row": excel_row,
                    "case_number": case_number,
                    "field": column,
                    "value": repr(value),
                })

        if peer_code in PEERS and PEERS[peer_code] != country:
            invalid_peer_codes.append({
                "excel_row": excel_row,
                "country": country,
                "source_code": source_code,
                "mapped_peer_code": peer_code,
                "expected_country": PEERS[peer_code],
            })

        source_records.append({
            "excel_row": excel_row,
            "case_number": case_number,
            "country": country,
            "source_wdi_code": source_code,
            "peer_iso3": peer_code if peer_code in PEERS else "",
            "start": start,
            "end": end,
            "strict": flags["Strictly preemptive"],
            "weak": flags["Weakly preemptive"],
            "post": flags["Post-default"],
            "default_date": default_date,
            "announcement": announcement,
            "no_exact_start": flags[
                "No exact start date (missing default month)"
            ],
        })

finally:
    wb.close()

if invalid_flags:
    raise SystemExit(
        "STOP: non-binary or missing classification/date-quality flags found. "
        f"Examples: {invalid_flags[:10]}"
    )

if invalid_dates:
    raise SystemExit(
        "STOP: non-date values found in date columns. "
        f"Examples: {invalid_dates[:10]}"
    )

if invalid_peer_codes:
    raise SystemExit(
        "STOP: country/code mismatch found. "
        f"Examples: {invalid_peer_codes[:10]}"
    )

if len(all_case_numbers) != len(set(all_case_numbers)):
    raise SystemExit("STOP: duplicate case numbers found in source records.")

peer_records = [
    record for record in source_records
    if record["peer_iso3"] in PEERS
]

# Ensure the explicit crosswalk has been exercised and checked.
rom_records = [
    record for record in peer_records
    if record["source_wdi_code"] == "ROM"
]
if len(rom_records) != 3 or any(
    record["peer_iso3"] != "ROU" for record in rom_records
):
    raise SystemExit(
        "STOP: expected exactly three ROM records mapped to ROU; "
        f"observed {len(rom_records)}."
    )

rows = []

for iso3, country in PEERS.items():
    records = [
        record for record in peer_records
        if record["peer_iso3"] == iso3
    ]

    starts = [record["start"] for record in records if record["start"]]
    ends = [record["end"] for record in records if record["end"]]

    strict_count = sum(record["strict"] == 1 for record in records)
    weak_count = sum(record["weak"] == 1 for record in records)
    post_count = sum(record["post"] == 1 for record in records)

    overlap_strict_weak = sum(
        record["strict"] == 1 and record["weak"] == 1
        for record in records
    )
    overlap_post_preemptive = sum(
        record["post"] == 1
        and (record["strict"] == 1 or record["weak"] == 1)
        for record in records
    )

    chronology_inversions = sum(
        record["start"] is not None
        and record["end"] is not None
        and record["end"] < record["start"]
        for record in records
    )

    source_codes = sorted({
        record["source_wdi_code"] for record in records
    })

    rows.append({
        "peer_iso3": iso3,
        "country": country,
        "source_record_count": len(records),
        "source_wdi_codes": ";".join(source_codes),
        "first_start_date": min(starts).date().isoformat() if starts else "",
        "last_start_date": max(starts).date().isoformat() if starts else "",
        "missing_start_dates": sum(r["start"] is None for r in records),
        "missing_end_dates": sum(r["end"] is None for r in records),
        "missing_default_dates": sum(
            r["default_date"] is None for r in records
        ),
        "missing_announcements": sum(
            r["announcement"] is None for r in records
        ),
        "imprecise_start_dates": sum(
            r["no_exact_start"] == 1 for r in records
        ),
        "strictly_preemptive_raw_flags": strict_count,
        "weakly_preemptive_raw_flags": weak_count,
        "post_default_raw_flags": post_count,
        "strict_weak_overlap": overlap_strict_weak,
        "post_preemptive_overlap": overlap_post_preemptive,
        "chronology_inversions": chronology_inversions,
        "coverage_status": (
            "RESTRUCTURING_RECORDS_FOUND"
            if records else "NO_RECORD_IN_KIEL_MAIN_DATASET"
        ),
        "interpretation": (
            "Restructuring-process records identified; raw classifications "
            "are source flags, not independently verified default findings"
            if records else
            "No matching main-dataset record; not evidence of zero defaults"
        ),
    })

df = pd.DataFrame(rows)

if len(df) != 20 or df["peer_iso3"].nunique() != 20:
    raise SystemExit("STOP: output does not contain exactly 20 unique peers.")

if int(df["source_record_count"].sum()) != len(peer_records):
    raise SystemExit("STOP: peer case counts do not reconcile to source rows.")

if int(df["source_record_count"].sum()) != 12:
    raise SystemExit(
        "STOP: expected 12 peer records from prior source inspection; "
        f"observed {int(df['source_record_count'].sum())}."
    )

# Source-wide quality diagnostics.
strict_weak_overlap_total = sum(
    record["strict"] == 1 and record["weak"] == 1
    for record in source_records
)
post_preemptive_overlap_total = sum(
    record["post"] == 1
    and (record["strict"] == 1 or record["weak"] == 1)
    for record in source_records
)
chronology_inversions_total = sum(
    record["start"] is not None
    and record["end"] is not None
    and record["end"] < record["start"]
    for record in source_records
)
imprecise_starts_total = sum(
    record["no_exact_start"] == 1 for record in source_records
)

# README count is 196; workbook's observed main-sheet records are 197.
README_REPORTED_CASE_COUNT = 196
OBSERVED_MAIN_CASE_COUNT = len(source_records)

DATA.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df.to_csv(CSV_OUT, index=False)

lines = [
    "=" * 100,
    "KIEL EUROPEAN PEER RESTRUCTURING-PROCESS COVERAGE AUDIT",
    "=" * 100,
    f"Source workbook: {SOURCE.name}",
    f"Source sheet: {SHEET}",
    f"Main-sheet records observed: {OBSERVED_MAIN_CASE_COUNT}",
    f"README-reported main-sheet records: {README_REPORTED_CASE_COUNT}",
    f"README/workbook count discrepancy: {OBSERVED_MAIN_CASE_COUNT - README_REPORTED_CASE_COUNT}",
    f"Peer countries assessed: {len(df)}",
    f"Peer countries with records: {(df['source_record_count'] > 0).sum()}",
    f"Peer main-sheet records: {int(df['source_record_count'].sum())}",
    "",
    "PEER COUNTRY COVERAGE",
    "-" * 100,
]

for _, row in df.iterrows():
    lines.append(
        f"{row['peer_iso3']} | {row['country']} | "
        f"cases={row['source_record_count']} | "
        f"strict={row['strictly_preemptive_raw_flags']} | "
        f"weak={row['weakly_preemptive_raw_flags']} | "
        f"post={row['post_default_raw_flags']} | "
        f"missing_default_dates={row['missing_default_dates']} | "
        f"imprecise_starts={row['imprecise_start_dates']} | "
        f"{row['coverage_status']}"
    )

lines.extend([
    "",
    "SOURCE-WIDE CLASSIFICATION AND DATE QUALITY",
    "-" * 100,
    f"Strict/weak overlapping flags: {strict_weak_overlap_total}",
    f"Post-default/preemptive overlapping flags: {post_preemptive_overlap_total}",
    f"Chronology inversions (end before start): {chronology_inversions_total}",
    f"Imprecise start-date flags: {imprecise_starts_total}",
    "",
    "CLASSIFICATION INTERPRETATION",
    "-" * 100,
    "1. Raw strict, weak and post-default flags are retained independently.",
    "2. Strictly preemptive and weakly preemptive flags overlap in the source;",
    "   they must not be treated as mutually exclusive categories.",
    "3. The source README definitions and observed flag combinations require",
    "   semantic reconciliation before exclusive classification is attempted.",
    "4. A post-default flag reflects the source's coding definition; it is not",
    "   an independent verification of payment default by this audit.",
    "",
    "COVERAGE AND ATTRIBUTION",
    "-" * 100,
    "1. The dataset records restructuring processes involving foreign private",
    "   creditors; it is not a comprehensive census of sovereign defaults.",
    "2. No matching record does not establish that a country had no default.",
    "3. The source WDI code ROM is explicitly mapped to peer ISO3 ROU for Romania.",
    "4. Original source codes are retained in the peer coverage CSV.",
    "5. Multiple restructuring records for one country remain separate cases.",
    "6. Ongoing cases listed separately in the workbook are excluded from the",
    "   completed main-sheet case count.",
    "",
    "SOURCE LIMITATIONS",
    "-" * 100,
    "1. The workbook README states 196 main cases; the main sheet contains 197.",
    "2. Some source start dates are explicitly flagged as imprecise.",
    "3. The dataset's coverage and coding rules exclude some continuous defaults",
    "   without an implemented debt exchange.",
    "4. This audit assesses source coverage and internal coding properties; it",
    "   does not independently validate the underlying historical events.",
    "",
    "OUTPUT",
    "-" * 100,
    str(CSV_OUT),
    str(REPORT_OUT),
    "",
    "KIEL PEER COVERAGE AUDIT COMPLETED.",
])

report = "\n".join(lines)
REPORT_OUT.write_text(report, encoding="utf-8")
print(report)
