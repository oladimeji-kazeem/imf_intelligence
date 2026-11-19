
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

ESM_FILE = EXTERNAL_DIR / "esm_domestic_sovereign_defaults.xlsx"
REGISTRY_FILE = EXTERNAL_DIR / "europe_historical_stress_events.csv"

SOURCE_FILES = {
    "boc_boe": [
        "boc_boe_sovereign_default_database_2025.xlsx",
    ],
    "kiel": [
        "kiel_monthly_sovereign_default_1970_2025.xlsx.xlsx",
    ],
    "esm": ["esm_domestic_sovereign_defaults.xlsx"],
}

ALIASES = {
    "country": ["country", "country_name", "nation"],
    "start": [
        "event_start_date",
        "start date",
        "start_date",
        "date_start",
        "onset_year",
        "start_year",
    ],
    "end": [
        "event_end_date",
        "end date",
        "end_date",
        "date_end",
        "end_year",
    ],
    "event_id": ["event_id", "episode_id", "id"],
    "event_type": ["event_type", "type", "crisis_type"],
}


def normalize_name(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def find_column(df, aliases):
    normalized = {normalize_name(c): c for c in df.columns}

    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]

    return None


def normalize_date(value):
    """Convert dates and year-only values into timestamps."""
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, (int, float)):
        year = int(value)
        if 1800 <= year <= 2100:
            return pd.Timestamp(year=year, month=1, day=1)

    return pd.to_datetime(value, errors="coerce")


def locate_source(candidates):
    for filename in candidates:
        path = EXTERNAL_DIR / filename
        if path.exists():
            return path
    return None


def inspect_sources():
    results = []

    for source, candidates in SOURCE_FILES.items():
        path = locate_source(candidates)

        results.append({
            "source": source,
            "expected_files": "; ".join(candidates),
            "status": "FOUND" if path else "NOT_FOUND",
            "actual_path": str(path) if path else "",
        })

    return pd.DataFrame(results)


def load_esm():
    if not ESM_FILE.exists():
        raise FileNotFoundError(
            f"ESM dataset not found: {ESM_FILE}"
        )

    df = pd.read_excel(
        ESM_FILE,
        sheet_name="2022 version"
    )

    df.columns = [str(c).strip() for c in df.columns]

    country_col = find_column(df, ALIASES["country"])
    start_col = find_column(df, ALIASES["start"])
    end_col = find_column(df, ALIASES["end"])

    required = {
        "country": country_col,
        "start": start_col,
        "end": end_col,
    }

    missing = [k for k, v in required.items() if v is None]

    if missing:
        raise ValueError(
            f"Missing required ESM columns: {missing}"
        )

    df["country_original"] = df[country_col]
    df["country_key"] = df[country_col].map(normalize_name)
    df["start_normalized"] = df[start_col].map(normalize_date)
    df["end_normalized"] = df[end_col].map(normalize_date)

    df["date_valid"] = (
        df["start_normalized"].notna()
        & df["end_normalized"].notna()
        & (
            df["end_normalized"]
            >= df["start_normalized"]
        )
    )

    df["source"] = "ESM domestic-law restructuring"

    return df


def load_registry():
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Historical stress registry not found: {REGISTRY_FILE}"
        )

    df = pd.read_csv(REGISTRY_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    country_col = find_column(df, ALIASES["country"])
    start_col = find_column(df, ALIASES["start"])
    end_col = find_column(df, ALIASES["end"])

    if country_col is None:
        raise ValueError(
            "Registry has no identifiable country column."
        )

    df["country_original"] = df[country_col]
    df["country_key"] = df[country_col].map(normalize_name)

    df["registry_start"] = (
        df[start_col].map(normalize_date)
        if start_col else pd.NaT
    )

    onset_col = find_column(df, ["onset_year", "start_year", "year"])
    if onset_col and onset_col != start_col:
        df["registry_start"] = df["registry_start"].fillna(
            df[onset_col].map(normalize_date)
        )

    df["registry_end"] = (
        df[end_col].map(normalize_date)
        if end_col else pd.NaT
    )

    end_year_col = find_column(df, ["end_year", "year_end"])
    if end_year_col and end_year_col != end_col:
        df["registry_end"] = df["registry_end"].fillna(
            df[end_year_col].map(normalize_date)
        )

    df["registry_end"] = df["registry_end"].fillna(df["registry_start"])

    return df


ISO_TO_COUNTRY = {
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


def identify_european_countries(registry, explicit_countries=None):
    """
    Prefer an explicitly supplied 20-country universe.
    Registry countries are a fallback only and must not
    be mistaken for the complete European universe.
    """
    if explicit_countries:
        countries = [
            ISO_TO_COUNTRY.get(str(c).strip().upper(), str(c).strip())
            for c in explicit_countries
        ]
        universe_source = "explicit_configuration"
    else:
        countries = sorted(
            registry["country_original"]
            .dropna()
            .astype(str)
            .unique()
        )
        universe_source = "registry_fallback_incomplete"

    return countries, universe_source


def intervals_overlap(a_start, a_end, b_start, b_end):
    if any(pd.isna(x) for x in [
        a_start, a_end, b_start, b_end
    ]):
        return False

    return a_start <= b_end and b_start <= a_end


def reconcile_events(esm, registry, countries):
    country_keys = {normalize_name(c) for c in countries}

    esm_europe = esm[
        esm["country_key"].isin(country_keys)
    ].copy()

    matches = []

    for esm_idx, esm_row in esm_europe.iterrows():
        candidates = registry[
            registry["country_key"]
            == esm_row["country_key"]
        ]

        matched_ids = []
        matched_rows = []

        for reg_idx, reg_row in candidates.iterrows():
            if intervals_overlap(
                esm_row["start_normalized"],
                esm_row["end_normalized"],
                reg_row["registry_start"],
                reg_row["registry_end"],
            ):
                matched_ids.append(str(reg_idx))
                matched_rows.append(reg_row)

        if matched_rows:
            match_status = "TEMPORAL_OVERLAP"
        elif len(candidates):
            match_status = "SAME_COUNTRY_NO_TEMPORAL_OVERLAP"
        else:
            match_status = "NO_REGISTRY_COUNTRY_MATCH"

        matches.append({
            "esm_row_index": esm_idx,
            "country": esm_row["country_original"],
            "start_date": esm_row["start_normalized"],
            "end_date": esm_row["end_normalized"],
            "instrument": esm_row.get("Instrument"),
            "amount_mln_usd": esm_row.get("Amount (mln US$)"),
            "npv_loss_pct": esm_row.get("NPV loss (%)"),
            "date_valid": esm_row["date_valid"],
            "registry_match_status": match_status,
            "registry_row_indices": ";".join(matched_ids),
            "registry_match_count": len(matched_ids),
        })

    return esm_europe, pd.DataFrame(matches)


def build_country_coverage(countries, esm_europe):
    rows = []

    for country in countries:
        key = normalize_name(country)
        subset = esm_europe[
            esm_europe["country_key"] == key
        ]

        rows.append({
            "country": country,
            "esm_records": len(subset),
            "valid_date_records": int(
                subset["date_valid"].sum()
            ),
            "invalid_date_records": int(
                (~subset["date_valid"]).sum()
            ),
            "esm_evidence_status": (
                "RECORDS_FOUND"
                if len(subset)
                else "NO_RECORD_IN_ESM"
            ),
            "interpretation": (
                "Domestic-law restructuring evidence identified"
                if len(subset)
                else "No record identified; not evidence of zero defaults"
            ),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Audit European sovereign default source coverage."
    )

    parser.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Explicit country universe, ideally the project's 20 countries."
    )

    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EUROPEAN SOVEREIGN DEFAULT COVERAGE AUDIT")
    print("=" * 78)

    source_inventory = inspect_sources()

    print("\nSOURCE INVENTORY")
    print(source_inventory.to_string(index=False))

    source_inventory.to_csv(
        PROCESSED_DIR / "europe_sovereign_default_source_coverage.csv",
        index=False
    )

    if not ESM_FILE.exists():
        raise FileNotFoundError(
            "Cannot perform event reconciliation: ESM source missing."
        )

    esm = load_esm()
    registry = load_registry()

    countries, universe_source = identify_european_countries(
        registry,
        args.countries
    )

    esm_europe, candidates = reconcile_events(
        esm,
        registry,
        countries
    )

    coverage = build_country_coverage(
        countries,
        esm_europe
    )

    candidates.to_csv(
        PROCESSED_DIR / "europe_sovereign_default_event_candidates.csv",
        index=False
    )

    coverage.to_csv(
        PROCESSED_DIR / "europe_sovereign_default_country_coverage.csv",
        index=False
    )

    total_records = len(esm_europe)
    valid_records = int(esm_europe["date_valid"].sum())

    overlap_count = int(
        (candidates["registry_match_status"]
         == "TEMPORAL_OVERLAP").sum()
    )

    report = [
        "=" * 78,
        "EUROPEAN SOVEREIGN DEFAULT COVERAGE AUDIT",
        "=" * 78,
        "",
        f"Project root: {PROJECT_ROOT}",
        f"European countries: {len(countries)}",
        f"Country universe source: {universe_source}",
        f"ESM global records: {len(esm)}",
        f"ESM European records: {total_records}",
        f"Valid ESM date records: {valid_records}",
        f"Invalid ESM date records: {total_records-valid_records}",
        f"Registry records: {len(registry)}",
        f"ESM records with temporal registry overlap: {overlap_count}",
        "",
        "SOURCE AVAILABILITY",
        source_inventory.to_string(index=False),
        "",
        "COUNTRY COVERAGE",
        coverage.to_string(index=False),
        "",
        "RECONCILIATION LIMITATIONS",
        "- Temporal overlap is a candidate match, not proof of identical events.",
        "- Same-country records without overlap require manual review.",
        "- ESM covers domestic-law instruments, not all sovereign defaults.",
        "- Missing source files indicate unavailable evidence, not zero defaults.",
        "- Registry row indices are technical references, not permanent event IDs.",
        "- Year-only dates are represented as January 1 and may distort overlap.",
        "- Instrument-level records must not automatically be counted as episodes.",
        "",
        "NO SOURCE FILES OR HISTORICAL REGISTRY RECORDS WERE MODIFIED.",
        "",
        "AUDIT COMPLETED."
    ]

    report_path = (
        REPORT_DIR / "europe_sovereign_default_coverage_audit.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\nSUMMARY")
    print(f"Countries assessed: {len(countries)}")
    print(f"European ESM records: {total_records}")
    print(f"Valid date records: {valid_records}")
    print(f"Temporal overlap candidates: {overlap_count}")
    print(f"\nReport: {report_path}")
    print("Audit completed.")


if __name__ == "__main__":
    main()
