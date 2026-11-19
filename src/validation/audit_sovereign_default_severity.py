
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

ESM_FILE = EXTERNAL / "esm_domestic_sovereign_defaults.xlsx"
REGISTRY_FILE = EXTERNAL / "europe_historical_stress_events.csv"

COUNTRIES = [
    "AUT", "BEL", "CHE", "CZE", "DEU",
    "DNK", "ESP", "FIN", "FRA", "GBR",
    "GRC", "HUN", "IRL", "ITA", "NLD",
    "NOR", "POL", "PRT", "ROU", "SWE",
]

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


def normalize(value):
    return (
        str(value).strip().lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def clean_identifier(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value or value.lower() in {
        "nan", "none", "null", "nat"
    }:
        return None

    return value


def find_col(df, aliases):
    lookup = {normalize(c): c for c in df.columns}

    for alias in aliases:
        key = normalize(alias)
        if key in lookup:
            return lookup[key]

    return None


def parse_date(value, is_end=False):
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, (int, float)):
        year = int(value)
        if 1800 <= year <= 2100:
            return pd.Timestamp(
                year=year,
                month=12 if is_end else 1,
                day=31 if is_end else 1,
            )

    text = str(value).strip()

    if text.isdigit() and len(text) == 4:
        year = int(text)
        if 1800 <= year <= 2100:
            return pd.Timestamp(
                year=year,
                month=12 if is_end else 1,
                day=31 if is_end else 1,
            )

    return pd.to_datetime(value, errors="coerce")


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load_esm():
    df = pd.read_excel(
        ESM_FILE,
        sheet_name="2022 version"
    )

    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "Country",
        "Start date",
        "End date",
        "Instrument",
        "Amount (mln US$)",
        "Face value reduction",
        "Maturity change",
        "Coupon change",
        "NPV loss (%)",
        "Post-default restructuring",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing ESM fields: {missing}")

    df["source_row"] = df.index
    df["country"] = df["Country"].astype(str).str.strip()
    df["country_key"] = df["country"].map(normalize)

    df["start_date"] = df["Start date"].apply(
        lambda x: parse_date(x, False)
    )
    df["end_date"] = df["End date"].apply(
        lambda x: parse_date(x, True)
    )

    df["amount_mln_usd"] = numeric(
        df["Amount (mln US$)"]
    )
    df["face_value_reduction"] = numeric(
        df["Face value reduction"]
    )
    df["maturity_change"] = numeric(
        df["Maturity change"]
    )
    df["coupon_change"] = numeric(
        df["Coupon change"]
    )
    df["npv_loss_raw"] = numeric(
        df["NPV loss (%)"]
    )
    df["post_default_restructuring"] = numeric(
        df["Post-default restructuring"]
    )

    df["date_valid"] = (
        df["start_date"].notna()
        & df["end_date"].notna()
        & (df["end_date"] >= df["start_date"])
    )

    df["severity_evidence_status"] = df.apply(
        lambda r: (
            "INVALID_DATE"
            if not r["date_valid"]
            else (
                "PARTIAL_SEVERITY_EVIDENCE"
                if pd.isna(r["npv_loss_raw"])
                else "NPV_EVIDENCE_AVAILABLE"
            )
        ),
        axis=1
    )

    return df


def load_registry():
    df = pd.read_csv(REGISTRY_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    event_type_col = find_col(
        df, ["event_type", "type", "crisis_type"]
    )

    if event_type_col is None:
        raise ValueError(
            "Registry requires an identifiable event_type field "
            "for semantic episode linkage."
        )

    df["registry_event_type"] = (
        df[event_type_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )

    country_col = find_col(
        df, ["country", "country_name", "nation"]
    )

    start_col = find_col(
        df,
        ["event_start_date", "start_date", "onset_year"]
    )

    end_col = find_col(
        df,
        ["event_end_date", "end_date", "end_year"]
    )

    onset_year_col = find_col(
        df, ["onset_year", "start_year", "event_start_year"]
    )
    end_year_col = find_col(
        df, ["end_year", "event_end_year"]
    )

    # Keep year-level evidence separate from exact-date evidence.
    df["registry_year_start"] = (
        df[onset_year_col].apply(lambda x: parse_date(x, False))
        if onset_year_col
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )

    df["registry_year_end"] = (
        df[end_year_col].apply(lambda x: parse_date(x, True))
        if end_year_col
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )

    # Canonical episode identifier is deliberately separate
    # from the individual registry event identifier.
    episode_col = find_col(
        df,
        [
            "canonical_episode_id",
            "episode_id",
            "episode_key",
            "episode_code",
        ]
    )

    event_id_col = find_col(
        df,
        [
            "event_id",
            "event_record_id",
            "record_id",
            "id",
        ]
    )

    if country_col is None or start_col is None:
        raise ValueError(
            "Registry requires identifiable country and start-date fields."
        )

    df["registry_row"] = df.index
    df["registry_country"] = df[country_col].astype(str)
    df["country_key"] = df["registry_country"].map(normalize)

    df["registry_start"] = df[start_col].apply(
        lambda x: parse_date(x, False)
    )

    df["registry_end"] = (
        df[end_col].apply(lambda x: parse_date(x, True))
        if end_col else df["registry_start"]
    )

    df["registry_episode_id"] = (
        df[episode_col].apply(clean_identifier)
        if episode_col else None
    )

    df["registry_event_id"] = (
        df[event_id_col].apply(clean_identifier)
        if event_id_col else None
    )

    print("\nREGISTRY IDENTIFIER INSPECTION")
    print("-" * 60)
    print(f"Canonical episode column: {episode_col}")
    print(f"Individual event column:  {event_id_col}")

    if episode_col is None:
        print(
            "WARNING: No canonical episode ID column found. "
            "Temporal matches will not be promoted to episode links."
        )

    return df


def get_registry_matches(row, registry):
    if not row["date_valid"]:
        return registry.iloc[0:0]

    candidates = registry[
        registry["country_key"] == row["country_key"]
    ]

    return candidates[
        candidates["registry_start"].notna()
        & candidates["registry_end"].notna()
        & (
            candidates["registry_start"]
            <= row["end_date"]
        )
        & (
            candidates["registry_end"]
            >= row["start_date"]
        )
    ]


def get_registry_year_candidates(row, registry):
    if not row["date_valid"]:
        return registry.iloc[0:0]

    candidates = registry[
        registry["country_key"] == row["country_key"]
    ].copy()

    # Year-level fallback applies only where exact dates are incomplete.
    missing_exact_dates = (
        candidates["registry_start"].isna()
        | candidates["registry_end"].isna()
    )

    return candidates[
        missing_exact_dates
        & candidates["registry_year_start"].notna()
        & candidates["registry_year_end"].notna()
        & (
            candidates["registry_year_start"]
            <= row["end_date"]
        )
        & (
            candidates["registry_year_end"]
            >= row["start_date"]
        )
    ]


def classify_instrument(row):
    findings = []

    if row["face_value_reduction"] == 1:
        findings.append("FACE_VALUE_REDUCTION_RECORDED")

    if row["maturity_change"] == 1:
        findings.append("MATURITY_CHANGED")

    if row["coupon_change"] == 1:
        findings.append("COUPON_CHANGED")

    if pd.notna(row["npv_loss_raw"]):
        findings.append("NPV_LOSS_RECORDED")
    else:
        findings.append("NPV_LOSS_MISSING")

    return ";".join(findings)


def resolve_episode_identity(matched, source_row):
    """
    Assign an episode key only when all matched registry rows
    identify exactly one explicit canonical episode.

    Event IDs and registry row numbers are provenance only.
    They are never converted into synthetic episode identities.
    """

    episode_ids = sorted({
        value
        for value in matched["registry_episode_id"].tolist()
        if value is not None
    })

    if len(episode_ids) == 1:
        return episode_ids[0], "CANONICAL_EPISODE_ID_MATCH"

    if len(episode_ids) > 1:
        return (
            f"UNLINKED_ESM_ROW_{source_row}",
            "AMBIGUOUS_MULTIPLE_CANONICAL_EPISODES",
        )

    if len(matched):
        return (
            f"UNLINKED_ESM_ROW_{source_row}",
            "TEMPORAL_MATCH_WITHOUT_CANONICAL_EPISODE_ID",
        )

    return (
        f"UNLINKED_ESM_ROW_{source_row}",
        "NO_TEMPORAL_MATCH",
    )


def reconcile(esm, registry, countries):
    resolved_countries = [
        ISO_TO_COUNTRY.get(str(c).strip().upper(), str(c).strip())
        for c in countries
    ]

    country_keys = (
        {normalize(c) for c in resolved_countries}
        | {normalize(c) for c in countries}
    )

    esm = esm[
        esm["country_key"].isin(country_keys)
    ].copy()

    matches = []
    episode_keys = []
    episode_link_statuses = []

    for _, row in esm.iterrows():
        matched = get_registry_matches(row, registry)
        year_candidates = get_registry_year_candidates(row, registry)

        year_compatible = year_candidates[
            year_candidates["registry_event_type"].eq(
                "sovereign_restructuring"
            )
        ]

        year_contextual = year_candidates[
            ~year_candidates["registry_event_type"].eq(
                "sovereign_restructuring"
            )
        ]

        compatible = matched[
            matched["registry_event_type"].eq(
                "sovereign_restructuring"
            )
        ].copy()

        contextual = matched[
            ~matched["registry_event_type"].eq(
                "sovereign_restructuring"
            )
        ].copy()

        # Original fields remain temporal-overlap provenance.
        event_ids = sorted({
            value
            for value in matched["registry_event_id"].tolist()
            if value is not None
        })

        canonical_episode_ids = sorted({
            value
            for value in matched["registry_episode_id"].tolist()
            if value is not None
        })

        registry_rows = sorted(
            matched["registry_row"].astype(str).tolist()
        )

        # Separate semantic-compatible provenance.
        compatible_event_ids = sorted({
            value
            for value in compatible["registry_event_id"].tolist()
            if value is not None
        })

        compatible_episode_ids = sorted({
            value
            for value in compatible["registry_episode_id"].tolist()
            if value is not None
        })

        compatible_registry_rows = sorted(
            compatible["registry_row"].astype(str).tolist()
        )

        # Only restructuring-compatible rows can establish episode identity.
        episode_key, episode_link_status = (
            resolve_episode_identity(
                compatible,
                row["source_row"]
            )
        )

        episode_keys.append(episode_key)
        episode_link_statuses.append(episode_link_status)

        matches.append({
            "source_row": row["source_row"],
            "country": row["country"],
            "episode_key": episode_key,
            "episode_link_status": episode_link_status,
            "registry_match_count": len(matched),
            "registry_episode_ids": ";".join(
                canonical_episode_ids
            ),
            "registry_event_ids": ";".join(event_ids),
            "registry_row_indices": ";".join(registry_rows),
            "compatible_registry_match_count": len(compatible),
            "contextual_registry_match_count": len(contextual),
            "compatible_registry_episode_ids": ";".join(
                compatible_episode_ids
            ),
            "compatible_registry_event_ids": ";".join(
                compatible_event_ids
            ),
            "compatible_registry_row_indices": ";".join(
                compatible_registry_rows
            ),
            "compatibility_status": (
                "RESTRUCTURING_COMPATIBLE_OVERLAP"
                if len(compatible)
                else "NO_RESTRUCTURING_COMPATIBLE_OVERLAP"
            ),
            "year_level_candidate_count": len(year_candidates),
            "year_level_compatible_count": len(year_compatible),
            "year_level_contextual_count": len(year_contextual),
            "year_level_candidate_event_ids": ";".join(sorted({
                value for value in year_candidates["registry_event_id"].tolist()
                if value is not None
            })),
            "year_level_candidate_event_types": ";".join(sorted({
                value for value in year_candidates["registry_event_type"].tolist()
                if value
            })),
            "year_level_candidate_episode_ids": ";".join(sorted({
                value for value in year_candidates["registry_episode_id"].tolist()
                if value is not None
            })),
            "year_level_candidate_row_indices": ";".join(
                sorted(year_candidates["registry_row"].astype(str).tolist())
            ),
            "year_level_status": (
                "YEAR_RANGE_OVERLAP_CANDIDATE_NOT_EXACT_DATE_MATCH"
                if len(year_candidates)
                else "NO_YEAR_LEVEL_CANDIDATE"
            ),
            "match_status": (
                "TEMPORAL_OVERLAP"
                if len(matched)
                else "NO_TEMPORAL_MATCH"
            ),
        })

    esm["episode_key"] = episode_keys
    esm["episode_link_status"] = episode_link_statuses

    match_df = pd.DataFrame(matches)

    return esm, match_df


def build_instrument_output(esm, matches):
    output = esm.merge(
        matches[
            [
                "source_row",
                "registry_match_count",
                "registry_episode_ids",
                "registry_event_ids",
                "registry_row_indices",
                "compatible_registry_match_count",
                "contextual_registry_match_count",
                "compatible_registry_episode_ids",
                "compatible_registry_event_ids",
                "compatible_registry_row_indices",
                "compatibility_status",
                "year_level_candidate_count",
                "year_level_compatible_count",
                "year_level_contextual_count",
                "year_level_candidate_event_ids",
                "year_level_candidate_event_types",
                "year_level_candidate_episode_ids",
                "year_level_candidate_row_indices",
                "year_level_status",
                "match_status",
            ]
        ],
        on="source_row",
        how="left"
    )

    output["instrument_severity_flags"] = output.apply(
        classify_instrument,
        axis=1
    )

    columns = [
        "source_row",
        "country",
        "episode_key",
        "episode_link_status",
        "start_date",
        "end_date",
        "Instrument",
        "amount_mln_usd",
        "face_value_reduction",
        "maturity_change",
        "coupon_change",
        "npv_loss_raw",
        "post_default_restructuring",
        "date_valid",
        "severity_evidence_status",
        "instrument_severity_flags",
        "registry_match_count",
        "registry_episode_ids",
        "registry_event_ids",
        "registry_row_indices",
        "compatible_registry_match_count",
        "contextual_registry_match_count",
        "compatible_registry_episode_ids",
        "compatible_registry_event_ids",
        "compatible_registry_row_indices",
        "compatibility_status",
        "year_level_candidate_count",
        "year_level_compatible_count",
        "year_level_contextual_count",
        "year_level_candidate_event_ids",
        "year_level_candidate_event_types",
        "year_level_candidate_episode_ids",
        "year_level_candidate_row_indices",
        "year_level_status",
        "match_status",
    ]

    return output[columns]


def build_episode_output(instruments):
    rows = []

    for episode_key, group in instruments.groupby(
        "episode_key", dropna=False
    ):
        npv = group["npv_loss_raw"]

        link_statuses = sorted(
            group["episode_link_status"].dropna().unique()
        )

        is_canonical = (
            len(link_statuses) == 1
            and link_statuses[0] == "CANONICAL_EPISODE_ID_MATCH"
        )

        registry_episode_ids = sorted({
            value
            for value in group["registry_episode_ids"].fillna("").str.split(";").explode()
            if value
        })

        registry_event_ids = sorted({
            value
            for value in group["registry_event_ids"].fillna("").str.split(";").explode()
            if value
        })

        compatible_registry_episode_ids = sorted({
            value
            for value in group[
                "compatible_registry_episode_ids"
            ].fillna("").str.split(";").explode()
            if value
        })

        compatible_registry_event_ids = sorted({
            value
            for value in group[
                "compatible_registry_event_ids"
            ].fillna("").str.split(";").explode()
            if value
        })

        rows.append({
            "episode_key": episode_key,
            "country": group["country"].iloc[0],
            "episode_link_status": ";".join(link_statuses),
            "canonical_episode_link": is_canonical,
            "instrument_count": len(group),
            "instruments": ";".join(
                sorted(
                    group["Instrument"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            ),
            "episode_start": group["start_date"].min(),
            "episode_end": group["end_date"].max(),
            "instrument_reported_amount_sum_mln_usd": (
                group["amount_mln_usd"].sum(min_count=1)
            ),
            "face_value_reduction_observations": int(
                group["face_value_reduction"].notna().sum()
            ),
            "maturity_change_observations": int(
                (group["maturity_change"] == 1).sum()
            ),
            "coupon_change_observations": int(
                (group["coupon_change"] == 1).sum()
            ),
            "npv_loss_observations": int(npv.notna().sum()),
            "npv_loss_missing_observations": int(npv.isna().sum()),
            "npv_loss_values_raw": ";".join(
                npv.dropna().astype(str).tolist()
            ),
            "registry_episode_ids": ";".join(
                registry_episode_ids
            ),
            "registry_event_ids": ";".join(
                registry_event_ids
            ),
            "compatible_registry_episode_ids": ";".join(
                compatible_registry_episode_ids
            ),
            "compatible_registry_event_ids": ";".join(
                compatible_registry_event_ids
            ),
            "registry_match_status": ";".join(
                sorted(group["match_status"].dropna().unique())
            ),
            "episode_classification": (
                "CANONICAL_REGISTRY_LINKED_CANDIDATE"
                if is_canonical
                else "UNLINKED_OR_AMBIGUOUS_EPISODE_CANDIDATE"
            ),
            "severity_assessment": (
                "INSUFFICIENT_NPV_EVIDENCE"
                if npv.isna().any()
                else "NPV_EVIDENCE_AVAILABLE_REVIEW_REQUIRED"
            ),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Sovereign default episode and severity audit."
    )

    parser.add_argument(
        "--countries",
        nargs="*",
        default=COUNTRIES
    )

    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not ESM_FILE.exists():
        raise FileNotFoundError(ESM_FILE)

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(REGISTRY_FILE)

    print("=" * 78)
    print("SOVEREIGN DEFAULT EPISODE CLASSIFICATION AND SEVERITY AUDIT")
    print("=" * 78)

    esm = load_esm()
    registry = load_registry()

    esm, matches = reconcile(
        esm,
        registry,
        args.countries
    )

    instruments = build_instrument_output(
        esm,
        matches
    )

    episodes = build_episode_output(instruments)

    instrument_path = (
        PROCESSED / "europe_sovereign_default_severity_instruments.csv"
    )

    episode_path = (
        PROCESSED / "europe_sovereign_default_severity_episodes.csv"
    )

    instruments.to_csv(instrument_path, index=False)
    episodes.to_csv(episode_path, index=False)

    total_instruments = len(instruments)
    countries_with_esm_records = int(instruments["country"].nunique())
    total_episodes = len(episodes)

    temporal_matches = int(
        instruments["match_status"].eq("TEMPORAL_OVERLAP").sum()
    )

    canonical_links = int(
        instruments["episode_link_status"]
        .eq("CANONICAL_EPISODE_ID_MATCH").sum()
    )

    ambiguous_links = int(
        instruments["episode_link_status"]
        .eq("AMBIGUOUS_MULTIPLE_CANONICAL_EPISODES").sum()
    )

    missing_canonical_links = int(
        instruments["episode_link_status"]
        .eq("TEMPORAL_MATCH_WITHOUT_CANONICAL_EPISODE_ID").sum()
    )

    npv_missing = int(
        instruments["npv_loss_raw"].isna().sum()
    )

    exact_restructuring_compatible_instruments = int(
        instruments["compatible_registry_match_count"]
        .fillna(0).astype(int).gt(0).sum()
    )

    year_level_candidate_pairings = int(
        instruments["year_level_candidate_count"]
        .fillna(0).astype(int).sum()
    )

    year_level_compatible_pairings = int(
        instruments["year_level_compatible_count"]
        .fillna(0).astype(int).sum()
    )

    year_level_contextual_pairings = int(
        instruments["year_level_contextual_count"]
        .fillna(0).astype(int).sum()
    )

    year_level_candidate_event_ids = sorted({
        event_id.strip()
        for value in instruments[
            "year_level_candidate_event_ids"
        ].dropna()
        for event_id in str(value).split(";")
        if event_id.strip()
    })

    year_level_distinct_events = len(
        year_level_candidate_event_ids
    )

    report = [
        "=" * 78,
        "SOVEREIGN DEFAULT EPISODE CLASSIFICATION AND SEVERITY AUDIT",
        "=" * 78,
        "",
        f"Countries in requested peer set: {len(args.countries)}",
        f"Countries with ESM instrument records: {countries_with_esm_records}",
        f"ESM instrument records: {total_instruments}",
        f"Candidate episode groups: {total_episodes}",
        f"Exact-date temporal instrument matches: {temporal_matches}",
        (
            "Exact-date instruments with restructuring-compatible overlap: "
            f"{exact_restructuring_compatible_instruments}"
        ),
        f"Canonical episode links: {canonical_links}",
        f"Temporal matches without canonical episode ID: {missing_canonical_links}",
        f"Ambiguous canonical episode links: {ambiguous_links}",
        f"Missing NPV observations: {npv_missing}",
        "",
        "YEAR-LEVEL CANDIDATE EVIDENCE (NOT EXACT-DATE MATCHES)",
        f"Year-range candidate instrument-event pairings: {year_level_candidate_pairings}",
        f"Distinct year-level candidate registry events: {year_level_distinct_events}",
        f"Year-level contextual candidate pairings: {year_level_contextual_pairings}",
        f"Year-level restructuring-compatible candidate pairings: {year_level_compatible_pairings}",
        "",
        "INSTRUMENT CLASSIFICATION",
        instruments[
            [
                "country",
                "Instrument",
                "episode_key",
                "episode_link_status",
                "instrument_severity_flags",
                "compatible_registry_match_count",
                "compatibility_status",
                "year_level_candidate_count",
                "year_level_contextual_count",
                "year_level_status",
                "match_status",
            ]
        ].to_string(index=False),
        "",
        "EPISODE CLASSIFICATION",
        episodes.to_string(index=False),
        "",
        "INTERPRETATION AND SAFEGUARDS",
        "- Canonical episode IDs are taken only from explicit registry episode fields.",
        "- Individual event IDs are retained separately as provenance.",
        "- Registry row numbers are provenance, not episode identifiers.",
        "- Exact-date temporal overlap is a candidate match, not proof of event identity.",
        "- Year-level candidates use year ranges where exact registry dates are incomplete.",
        "- Year-level candidates do not establish exact chronology or canonical episode identity.",
        "- Year-level candidate pairings are distinct from distinct registry event counts.",
        "- Contextual financial-assistance and banking-crisis records are not restructuring evidence.",
        "- Multiple canonical episode IDs produce an ambiguous, unlinked record.",
        "- Temporal matches without canonical episode IDs remain unlinked.",
        "- Unlinked instruments are kept separate rather than force-grouped.",
        "- Amount sums represent instrument-reported amounts, not verified unique debt.",
        "- NPV loss values are preserved in their original numeric scale.",
        "- Missing NPV observations are not zero creditor losses.",
        "- Face-value reduction, maturity changes and coupon changes are separate measures.",
        "- No composite severity score is assigned.",
        "- No existing registry records are modified.",
        "",
        "OUTPUT FILES",
        str(instrument_path),
        str(episode_path),
        "",
        "AUDIT COMPLETED."
    ]

    report_path = (
        REPORTS / "europe_sovereign_default_severity_audit.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\nSUMMARY")
    print(f"Instrument records: {total_instruments}")
    print(f"Candidate episode groups: {total_episodes}")
    print(f"Temporal matches: {temporal_matches}")
    print(f"Canonical episode links: {canonical_links}")
    print(f"Missing canonical episode links: {missing_canonical_links}")
    print(f"Ambiguous episode links: {ambiguous_links}")
    print(f"Missing NPV observations: {npv_missing}")
    print(f"\nReport: {report_path}")
    print("Audit completed successfully.")


if __name__ == "__main__":
    main()
