
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "data" / "processed"
    / "europe_banking_crisis_duration_audit.csv"
)

OUTPUT = (
    ROOT / "data" / "processed"
    / "europe_banking_crisis_duration_verified.csv"
)

REPORT = (
    ROOT / "outputs" / "reports"
    / "europe_banking_crisis_duration_verification_report.txt"
)

# Findings checked against IMF Working Paper 2026/094,
# Appendix I, Table A1.
#
# True  = source explicitly marks end year with footnote 5/.
# False = source does not mark the episode with footnote 5/.

SOURCE_MARKERS = {
    ("AUT", 2008): True,
    ("FIN", 1991): False,
    ("GRC", 2008): True,
    ("HUN", 1991): True,
    ("HUN", 2008): True,
    ("IRL", 2008): True,
    ("PRT", 2008): True,
}

if not SOURCE.exists():
    raise FileNotFoundError(f"Duration audit not found: {SOURCE}")

data = pd.read_csv(SOURCE)

required = [
    "event_id",
    "iso",
    "onset_year",
    "end_year",
    "duration_years_inclusive",
]

missing = [c for c in required if c not in data.columns]
if missing:
    raise ValueError(f"Missing required fields: {missing}")

data["iso"] = data["iso"].astype(str).str.strip().str.upper()
data["onset_year"] = pd.to_numeric(
    data["onset_year"], errors="raise"
).astype(int)

data["end_year"] = pd.to_numeric(
    data["end_year"], errors="raise"
).astype(int)

data["duration_years_inclusive"] = pd.to_numeric(
    data["duration_years_inclusive"], errors="raise"
).astype(int)

if data["event_id"].duplicated().any():
    raise ValueError("Duplicate event IDs in duration audit.")

if len(data) != 13:
    raise ValueError(f"Expected 13 records, found {len(data)}.")

data["source_marker_key"] = list(
    zip(data["iso"], data["onset_year"])
)

# All seven original five-year candidates must be mapped.
five_year_candidates = data[
    data["duration_years_inclusive"] == 5
]

candidate_keys = set(five_year_candidates["source_marker_key"])
mapped_keys = set(SOURCE_MARKERS)

if candidate_keys != mapped_keys:
    raise ValueError(
        "Source marker mapping does not match the seven "
        f"five-year candidates. Missing: {candidate_keys - mapped_keys}; "
        f"unexpected: {mapped_keys - candidate_keys}"
    )

data["imf_appendix_footnote_5_marker"] = data[
    "source_marker_key"
].map(SOURCE_MARKERS)

data["imf_appendix_footnote_5_marker"] = (
    data["imf_appendix_footnote_5_marker"]
    .map({True: "yes", False: "no"})
    .fillna("not_applicable")
)

def classify(row):
    duration = row["duration_years_inclusive"]
    marker = row["imf_appendix_footnote_5_marker"]

    if marker == "yes":
        return "confirmed_truncated_by_imf_footnote"

    if duration == 5 and marker == "no":
        return "five_year_duration_not_marked_as_truncated"

    if duration < 5:
        return "within_five_year_limit_not_marked_truncated"

    return "requires_investigation"

data["verified_truncation_status"] = data.apply(
    classify, axis=1
)

data["source_verification_reference"] = (
    "Laeven and Valencia (2026), IMF WP/26/094, "
    "Appendix I, Table A1, footnote 5/"
)

data["actual_resolution_year"] = pd.NA
data["actual_resolution_year_status"] = (
    "not_established_by_this_audit"
)

# Reconciliation checks.
confirmed = data[
    data["verified_truncation_status"]
    == "confirmed_truncated_by_imf_footnote"
]

unmarked_five_year = data[
    data["verified_truncation_status"]
    == "five_year_duration_not_marked_as_truncated"
]

within = data[
    data["verified_truncation_status"]
    == "within_five_year_limit_not_marked_truncated"
]

if len(confirmed) != 6:
    raise ValueError(
        f"Expected six confirmed truncations; found {len(confirmed)}."
    )

if len(unmarked_five_year) != 1:
    raise ValueError(
        "Expected one five-year record without the footnote marker."
    )

if len(within) != 6:
    raise ValueError(
        f"Expected six shorter episodes; found {len(within)}."
    )

if len(data) != len(confirmed) + len(unmarked_five_year) + len(within):
    raise ValueError("Duration categories do not reconcile.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

data.drop(columns=["source_marker_key"]).to_csv(
    OUTPUT, index=False, encoding="utf-8"
)

saved = pd.read_csv(OUTPUT)

if len(saved) != 13:
    raise ValueError("Saved output does not contain 13 records.")

if saved["event_id"].duplicated().any():
    raise ValueError("Duplicate event IDs in saved output.")

lines = [
    "=" * 76,
    "EUROPEAN BANKING-CRISIS DURATION VERIFICATION REPORT",
    "=" * 76,
    "",
    "SOURCE",
    "-" * 76,
    "Laeven and Valencia (2026), IMF WP/26/094",
    "Appendix I, Table A1, footnote 5/",
    "",
    "VERIFICATION RESULTS",
    "-" * 76,
    f"Banking-crisis records: {len(data)}",
    f"Confirmed truncated episodes: {len(confirmed)}",
    f"Five-year episodes without truncation marker: {len(unmarked_five_year)}",
    f"Episodes shorter than five years: {len(within)}",
    "",
    "INDIVIDUAL RESULTS",
    "-" * 76,
]

for _, row in data.sort_values(
    ["iso", "onset_year"]
).iterrows():
    lines.append(
        f"{row['iso']} | "
        f"{row['onset_year']}-{row['end_year']} | "
        f"{row['duration_years_inclusive']} years | "
        f"footnote 5/: {row['imf_appendix_footnote_5_marker']} | "
        f"{row['verified_truncation_status']}"
    )

lines.extend([
    "",
    "METHODOLOGICAL CONTROLS",
    "-" * 76,
    "- Footnote status is based on IMF Appendix I, Table A1.",
    "- Five-year duration alone is not treated as proof of truncation.",
    "- Finland remains unmarked despite its five-year duration.",
    "- No end dates have been extended or changed.",
    "- Actual resolution years remain unestablished.",
    "- Original event registry remains unchanged.",
    "",
    "OUTPUT",
    "-" * 76,
    str(OUTPUT),
    str(REPORT),
    "",
    "STATUS: SOURCE-FOOTNOTE VERIFICATION PASSED",
])

REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
