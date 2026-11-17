
from pathlib import Path
import pandas as pd
import re
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PATH = (
    ROOT / "data" / "external" /
    "europe_historical_stress_events.csv"
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUTPUT_EVENTS = (
    PROCESSED /
    "europe_sovereign_default_event_adjudication.csv"
)

OUTPUT_COUNTRIES = (
    PROCESSED /
    "europe_sovereign_default_event_adjudication_country_summary.csv"
)

OUTPUT_REPORT = (
    REPORTS /
    "europe_sovereign_default_event_adjudication.txt"
)

RULESET_VERSION = "1.0"


# ============================================================
# NORMALISATION
# ============================================================

def normalise_column(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower()
    ).strip("_")


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalise_value(value):
    return clean_text(value).lower().replace(" ", "_")


def require_column(df, candidates, label):

    for candidate in candidates:

        candidate = normalise_column(candidate)

        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Required {label} column not found. "
        f"Expected: {candidates}. "
        f"Available: {list(df.columns)}"
    )


# ============================================================
# EVENT-TYPE ADJUDICATION
# ============================================================

EVENT_TYPE_RULES = {

    "sovereign_default": (
        "EXPLICIT_DEFAULT_LABEL_IN_REGISTRY",
        "The source registry explicitly labels this event as a "
        "sovereign default. Independent source adjudication remains required."
    ),

    "sovereign_restructuring": (
        "REGISTRY_RECORDED_SOVEREIGN_RESTRUCTURING",
        "The source registry records a sovereign debt restructuring. "
        "Contractual payment default is not independently inferred."
    ),

    "financial_assistance": (
        "REGISTRY_RECORDED_FINANCIAL_ASSISTANCE",
        "Official financial assistance is recorded. "
        "This is not independently proof of sovereign default."
    ),

    "systemic_banking_crisis": (
        "REGISTRY_RECORDED_BANKING_CRISIS",
        "A systemic banking crisis is recorded. "
        "This is not independently proof of sovereign default."
    ),

    "related_financial_stress": (
        "REGISTRY_RECORDED_RELATED_FINANCIAL_STRESS",
        "Related financial stress is recorded. "
        "Sovereign-default status is not established."
    ),

    "sovereign_debt_distress": (
        "REGISTRY_RECORDED_RELATED_FINANCIAL_STRESS",
        "Sovereign debt distress is recorded. "
        "Default status requires further evidence."
    ),

    "currency_crisis": (
        "REGISTRY_RECORDED_RELATED_FINANCIAL_STRESS",
        "A currency crisis is recorded. "
        "This is not independently proof of sovereign default."
    ),

    "debt_crisis": (
        "REGISTRY_RECORDED_RELATED_FINANCIAL_STRESS",
        "A debt crisis is recorded. "
        "Specific sovereign-default status requires verification."
    ),
}



# ============================================================
# ADJUDICATION FUNCTION
# ============================================================

def adjudicate_event(row, columns):

    event_type = normalise_value(
        row[columns["event_type"]]
    )

    verification = normalise_value(
        row[columns["verification"]]
    )

    inclusion = normalise_value(
        row[columns["inclusion"]]
    )

    event_id = clean_text(
        row[columns["event_id"]]
    )

    # --------------------------------------------------------
    # Basic record validation
    # --------------------------------------------------------

    if not event_id:

        return (
            "INSUFFICIENT_EVENT_EVIDENCE",
            "REJECTED_MISSING_EVENT_ID",
            "Event record has no unique event identifier."
        )

    if inclusion != "include":

        return (
            "INSUFFICIENT_EVENT_EVIDENCE",
            "NOT_INCLUDED_IN_REGISTRY",
            "Registry record is not marked for inclusion."
        )

    if verification != "verified":

        return (
            "INSUFFICIENT_EVENT_EVIDENCE",
            "NOT_VERIFIED",
            "Event record does not have verified status."
        )

    # --------------------------------------------------------
    # Explicit event-type rules
    # --------------------------------------------------------

    if event_type not in EVENT_TYPE_RULES:

        return (
            "INSUFFICIENT_EVENT_EVIDENCE",
            "UNRECOGNISED_EVENT_TYPE",
            f"Event type '{event_type}' requires manual adjudication."
        )

    classification, rationale = EVENT_TYPE_RULES[event_type]

    return (
        classification,
        "RULE_BASED_VERIFIED_EVENT",
        rationale
    )


# ============================================================
# LOAD REGISTRY
# ============================================================

def load_registry():

    if not REGISTRY_PATH.exists():

        raise FileNotFoundError(
            f"Historical registry not found: {REGISTRY_PATH}"
        )

    df = pd.read_csv(REGISTRY_PATH)

    df.columns = [
        normalise_column(column)
        for column in df.columns
    ]

    columns = {

        "event_id": require_column(
            df,
            ["event_id"],
            "event identifier"
        ),

        "iso": require_column(
            df,
            ["iso", "iso3"],
            "country ISO code"
        ),

        "country": require_column(
            df,
            ["country", "country_name"],
            "country"
        ),

        "event_type": require_column(
            df,
            ["event_type"],
            "event type"
        ),

        "event_name": require_column(
            df,
            ["event_name", "name"],
            "event name"
        ),

        "verification": require_column(
            df,
            ["verification_status"],
            "verification status"
        ),

        "inclusion": require_column(
            df,
            ["inclusion_decision"],
            "inclusion decision"
        ),

        "source_organisation": require_column(
            df,
            ["source_organisation", "source"],
            "source organisation"
        ),

        "source_title": require_column(
            df,
            ["source_title"],
            "source title"
        ),

        "source_url": require_column(
            df,
            ["source_url"],
            "source URL"
        ),

        "evidence_note": require_column(
            df,
            ["evidence_note"],
            "evidence note"
        ),
    }

    return df, columns


# ============================================================
# BUILD EVENT-LEVEL ADJUDICATION
# ============================================================

def build_event_adjudication():

    registry, columns = load_registry()

    original_count = len(registry)

    # --------------------------------------------------------
    # Registry integrity checks
    # --------------------------------------------------------

    event_ids = (
        registry[columns["event_id"]]
        .astype(str)
        .str.strip()
    )

    if event_ids.eq("").any():

        raise ValueError(
            "Registry contains blank event identifiers."
        )

    if event_ids.duplicated().any():

        duplicates = event_ids[
            event_ids.duplicated(keep=False)
        ].tolist()

        raise ValueError(
            f"Duplicate event IDs detected: {duplicates}"
        )

    if registry[columns["iso"]].isna().any():

        raise ValueError(
            "Registry contains missing ISO country codes."
        )

    # --------------------------------------------------------
    # Adjudicate each event
    # --------------------------------------------------------

    classifications = []
    rule_statuses = []
    rationales = []

    for _, row in registry.iterrows():

        classification, rule_status, rationale = (
            adjudicate_event(row, columns)
        )

        classifications.append(classification)
        rule_statuses.append(rule_status)
        rationales.append(rationale)

    # --------------------------------------------------------
    # Preserve source fields and append adjudication fields
    # --------------------------------------------------------

    output = registry.copy()

    output["adjudication_classification"] = classifications

    output["adjudication_rule_status"] = rule_statuses

    output["adjudication_rationale"] = rationales

    output["adjudication_ruleset_version"] = RULESET_VERSION

    output["default_proof_status"] = output[
        "adjudication_classification"
    ].map({

         "EXPLICIT_DEFAULT_LABEL_IN_REGISTRY":
            "EXPLICIT_REGISTRY_LABEL_REQUIRES_SOURCE_REVIEW",

        "REGISTRY_RECORDED_SOVEREIGN_RESTRUCTURING":
            "RESTRUCTURING_RECORDED_PAYMENT_DEFAULT_NOT_ESTABLISHED",

        "REGISTRY_RECORDED_FINANCIAL_ASSISTANCE":
            "NOT_ESTABLISHED",

        "REGISTRY_RECORDED_BANKING_CRISIS":
            "NOT_ESTABLISHED",

        "REGISTRY_RECORDED_RELATED_FINANCIAL_STRESS":
            "NOT_ESTABLISHED",

        "INSUFFICIENT_EVENT_EVIDENCE":
            "UNDETERMINED",

    })

    # --------------------------------------------------------
    # Classification reconciliation
    # --------------------------------------------------------

    if len(output) != original_count:

        raise RuntimeError(
            "Event-level record count changed during adjudication."
        )

    if output[columns["event_id"]].nunique() != original_count:

        raise RuntimeError(
            "Event identifier uniqueness failed."
        )

    if output["adjudication_classification"].isna().any():

        raise RuntimeError(
            "Some event records were not adjudicated."
        )

    # --------------------------------------------------------
    # Save event-level output
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_EVENTS,
        index=False
    )

    return output, columns


# ============================================================
# COUNTRY SUMMARY
# ============================================================

def build_country_summary(output, columns):

    iso_col = columns["iso"]
    country_col = columns["country"]

    classifications = sorted(
        output["adjudication_classification"].unique()
    )

    rows = []

    for iso in sorted(
        output[iso_col].astype(str).str.upper().unique()
    ):

        subset = output[
            output[iso_col]
            .astype(str)
            .str.upper()
            .eq(iso)
        ]

        country_values = (
            subset[country_col]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(country_values) != 1:

            raise ValueError(
                f"Inconsistent country names for ISO {iso}: "
                f"{country_values.tolist()}"
            )

        country = country_values[0]

        row = {
            "iso3": iso,
            "country": country,
            "registry_event_records": len(subset),
            "distinct_event_ids": subset[columns["event_id"]].nunique(),
            "verified_event_records": int(
                subset[columns["verification"]]
                .astype(str)
                .str.lower()
                .eq("verified")
                .sum()
            ),
            "included_event_records": int(
                subset[columns["inclusion"]]
                .astype(str)
                .str.lower()
                .eq("include")
                .sum()
            ),
            "sovereign_default_records": int(
                subset["adjudication_classification"]
                .eq("EXPLICIT_DEFAULT_LABEL_IN_REGISTRY")
                .sum()
            ),
            "sovereign_restructuring_records": int(
                subset["adjudication_classification"]
                .eq("REGISTRY_RECORDED_SOVEREIGN_RESTRUCTURING")
                .sum()
            ),
            "financial_assistance_records": int(
                subset["adjudication_classification"]
                .eq("REGISTRY_RECORDED_FINANCIAL_ASSISTANCE")
                .sum()
            ),
            "banking_crisis_records": int(
                subset["adjudication_classification"]
                .eq("REGISTRY_RECORDED_BANKING_CRISIS")
                .sum()
            ),
            "related_financial_stress_records": int(
                subset["adjudication_classification"]
                .eq("REGISTRY_RECORDED_RELATED_FINANCIAL_STRESS")
                .sum()
            ),
            "insufficient_evidence_records": int(
                subset["adjudication_classification"]
                .eq("INSUFFICIENT_EVENT_EVIDENCE")
                .sum()
            ),
        }

        for classification in classifications:

            column = (
                "count_" +
                classification.lower()
            )

            row[column] = int(
                subset["adjudication_classification"]
                .eq(classification)
                .sum()
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    if summary["registry_event_records"].sum() != len(output):

        raise RuntimeError(
            "Country summary does not reconcile with event records."
        )

    return summary


# ============================================================
# REPORT
# ============================================================

def write_report(output, summary, columns):

    classification_counts = (
        output["adjudication_classification"]
        .value_counts()
        .sort_index()
    )

    country_default_records = int(
        summary["sovereign_default_records"].sum()
    )

    country_restructuring_records = int(
        summary["sovereign_restructuring_records"].sum()
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as report:

        report.write("=" * 76 + "\n")
        report.write(
            "EUROPEAN SOVEREIGN DEFAULT EVENT ADJUDICATION\n"
        )
        report.write("=" * 76 + "\n\n")

        report.write("INPUT AND RECONCILIATION\n")
        report.write("-" * 76 + "\n")

        report.write(
            f"Registry event records: {len(output)}\n"
        )

        report.write(
            f"Unique event IDs: "
            f"{output[columns['event_id']].nunique()}\n"
        )

        report.write(
            f"Countries represented: "
            f"{summary['iso3'].nunique()}\n"
        )

        report.write(
            f"Ruleset version: {RULESET_VERSION}\n\n"
        )

        report.write("EVENT ADJUDICATION DISTRIBUTION\n")
        report.write("-" * 76 + "\n")

        for classification, count in classification_counts.items():

            report.write(
                f"{classification}: {count}\n"
            )

        report.write("\nSOVEREIGN DEFAULT AND RESTRUCTURING\n")
        report.write("-" * 76 + "\n")

        report.write(
            f"Explicit sovereign-default records: "
            f"{country_default_records}\n"
        )

        report.write(
            f"Confirmed sovereign-restructuring records: "
            f"{country_restructuring_records}\n"
        )

        report.write(
            "Restructuring records are not automatically "
            "classified as contractual payment defaults.\n\n"
        )

        report.write("COUNTRY SUMMARY\n")
        report.write("-" * 76 + "\n")

        for _, row in summary.iterrows():

            report.write(
                f"{row['iso3']} | {row['country']} | "
                f"Events: {row['registry_event_records']} | "
                f"Defaults: {row['sovereign_default_records']} | "
                f"Restructurings: "
                f"{row['sovereign_restructuring_records']} | "
                f"Financial assistance: "
                f"{row['financial_assistance_records']} | "
                f"Banking crises: "
                f"{row['banking_crisis_records']}\n"
            )

        report.write("\nMETHODOLOGICAL LIMITATIONS\n")
        report.write("-" * 76 + "\n")

        limitations = [

            "Adjudication is based on the existing registry event_type, "
            "verification_status and inclusion_decision fields.",

            "Verified event status does not independently prove sovereign default.",

            "Sovereign restructuring is not automatically equivalent to "
            "a missed contractual payment.",

            "Official financial assistance is not automatically sovereign default.",

            "Systemic banking crises are not automatically sovereign default.",

            "Unrecognised event types require manual review.",

            "No new external evidence has been independently verified by this script.",

            "The source registry is preserved and is not modified.",

            "No country default probability, severity score or ranking is produced.",

        ]

        for limitation in limitations:

            report.write(f"- {limitation}\n")


# ============================================================
# MAIN
# ============================================================

def main():

    output, columns = build_event_adjudication()

    summary = build_country_summary(
        output,
        columns
    )

    summary.to_csv(
        OUTPUT_COUNTRIES,
        index=False
    )

    write_report(
        output,
        summary,
        columns
    )

    print("=" * 76)
    print(
        "EUROPEAN SOVEREIGN DEFAULT EVENT ADJUDICATION"
    )
    print("=" * 76)

    print(
        f"Registry event records: {len(output)}"
    )

    print(
        f"Unique event IDs: "
        f"{output[columns['event_id']].nunique()}"
    )

    print(
        f"Countries represented: "
        f"{summary['iso3'].nunique()}"
    )

    print("\nEVENT ADJUDICATION DISTRIBUTION")

    print(
        output["adjudication_classification"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCOUNTRY SUMMARY")

    print(
        summary[
            [
                "iso3",
                "country",
                "sovereign_default_records",
                "sovereign_restructuring_records",
                "financial_assistance_records",
                "banking_crisis_records",
            ]
        ].to_string(index=False)
    )

    print("\nOUTPUTS")
    print(OUTPUT_EVENTS)
    print(OUTPUT_COUNTRIES)
    print(OUTPUT_REPORT)

    print("\nEVENT ADJUDICATION COMPLETED.")


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr
        )

        raise
