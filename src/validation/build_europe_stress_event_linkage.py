
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REGISTRY = (
    ROOT / "data" / "external"
    / "europe_historical_stress_events.csv"
)

OUT = ROOT / "data" / "processed"
REPORT = (
    ROOT / "outputs" / "reports"
    / "europe_stress_event_linkage_report.txt"
)

EVENT_OUTPUT = OUT / "europe_stress_event_linkage.csv"
EPISODE_OUTPUT = OUT / "europe_stress_episode_summary.csv"
RELATION_OUTPUT = (
    OUT / "europe_stress_event_relationship_candidates.csv"
)

REQUIRED_COLUMNS = [
    "event_id",
    "iso",
    "country",
    "event_type",
    "event_name",
    "onset_year",
    "end_year",
    "verification_status",
    "inclusion_decision",
]

# Candidate relationships are deliberately restricted to
# historically plausible event combinations.
RELATED_COUNTRIES = {"GRC", "IRL", "PRT", "ESP"}
MAX_YEAR_GAP = 3


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def year_value(value):
    if pd.isna(value) or clean(value) == "":
        return None
    return int(float(value))


def slug(value):
    value = re.sub(r"[^A-Z0-9]+", "_", str(value).upper())
    return value.strip("_")


def interval_gap(a_start, a_end, b_start, b_end):
    """Return 0 for overlapping intervals, otherwise years between."""
    if a_start <= b_end and b_start <= a_end:
        return 0
    if a_end < b_start:
        return b_start - a_end
    return a_start - b_end


if not REGISTRY.exists():
    raise FileNotFoundError(f"Registry not found: {REGISTRY}")

registry = pd.read_csv(REGISTRY)

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in registry.columns
]

if missing_columns:
    raise ValueError(
        f"Registry is missing required columns: {missing_columns}"
    )

if registry["event_id"].isna().any():
    raise ValueError("Registry contains missing event_id values.")

registry["event_id"] = registry["event_id"].astype(str).str.strip()

if registry["event_id"].duplicated().any():
    duplicates = registry.loc[
        registry["event_id"].duplicated(keep=False),
        "event_id"
    ].tolist()
    raise ValueError(f"Duplicate event IDs: {duplicates}")

# ---------------------------------------------------------
# 1. Assign episode IDs conservatively
# ---------------------------------------------------------

linkage = registry.copy()

if "episode_id" not in linkage.columns:
    linkage["episode_id"] = ""

linkage["episode_id"] = linkage["episode_id"].fillna("").astype(str).str.strip()

for index, row in linkage.iterrows():
    existing_episode = row["episode_id"]

    if existing_episode:
        # Preserve existing episode assignment, including the
        # two Greek PSI instrument tranches.
        continue

    event_id = row["event_id"]
    linkage.at[index, "episode_id"] = (
        f"EP_{slug(event_id)}"
    )

linkage["episode_assignment_method"] = "individual_event_default"

psi_mask = linkage["episode_id"].eq(
    "EU_SOV_GRC_PSI_2011_2012"
)

linkage.loc[
    psi_mask,
    "episode_assignment_method"
] = "shared_source_documented_restructuring_episode"

# Keep the source event record separate from the episode.
linkage["event_record_id"] = linkage["event_id"]

# ---------------------------------------------------------
# 2. Validate event and episode assignments
# ---------------------------------------------------------

if linkage["episode_id"].isna().any():
    raise ValueError("Some events have no episode assignment.")

if (linkage["episode_id"].str.strip() == "").any():
    raise ValueError("Some events have blank episode assignments.")

psi = linkage[psi_mask]

if len(psi) != 2:
    raise ValueError(
        "Expected exactly two Greek PSI tranche records "
        f"under the shared episode; found {len(psi)}."
    )

if psi["event_id"].nunique() != 2:
    raise ValueError("Greek PSI event IDs are not unique.")

if psi["episode_id"].nunique() != 1:
    raise ValueError("Greek PSI tranches do not share one episode.")

# ---------------------------------------------------------
# 3. Build episode-level summary
# ---------------------------------------------------------

episode_rows = []

for episode_id, group in linkage.groupby("episode_id", sort=True):

    onset_values = [
        year_value(v)
        for v in group["onset_year"]
    ]

    end_values = [
        year_value(v)
        for v in group["end_year"]
    ]

    onset_values = [v for v in onset_values if v is not None]
    end_values = [v for v in end_values if v is not None]

    countries = sorted(
        set(group["iso"].dropna().astype(str).str.strip())
    )

    event_types = sorted(
        set(group["event_type"].dropna().astype(str).str.strip())
    )

    episode_rows.append({
        "episode_id": episode_id,
        "country_count": len(countries),
        "countries": ";".join(countries),
        "event_record_count": len(group),
        "event_types": ";".join(event_types),
        "episode_start_year": min(onset_values) if onset_values else None,
        "episode_end_year": max(end_values) if end_values else None,
        "event_ids": ";".join(sorted(group["event_id"].tolist())),
        "verification_statuses": ";".join(
            sorted(set(group["verification_status"].fillna("").astype(str)))
        ),
        "inclusion_decisions": ";".join(
            sorted(set(group["inclusion_decision"].fillna("").astype(str)))
        ),
    })

episodes = pd.DataFrame(episode_rows)

# ---------------------------------------------------------
# 4. Generate candidate relationships for review
# ---------------------------------------------------------


# ---------------------------------------------------------
# 4. Generate episode-level relationship candidates
# ---------------------------------------------------------

relationships = []

for iso, country_group in linkage.groupby("iso"):

    if clean(iso) not in RELATED_COUNTRIES:
        continue

    # Each episode is represented once, even if it contains
    # multiple instrument-level event records.
    episode_groups = []

    for episode_id, episode_group in country_group.groupby(
        "episode_id", sort=True
    ):
        episode_groups.append({
            "episode_id": episode_id,
            "records": episode_group.to_dict("records"),
            "event_types": set(
                episode_group["event_type"].dropna().astype(str)
            ),
            "event_ids": sorted(
                episode_group["event_id"].astype(str).tolist()
            ),
        })

    for i in range(len(episode_groups)):
        for j in range(i + 1, len(episode_groups)):

            a = episode_groups[i]
            b = episode_groups[j]

            pair = a["event_types"] | b["event_types"]

            relationship_type = None
            rationale = None

            if (
                "systemic_banking_crisis" in a["event_types"]
                and "financial_assistance" in b["event_types"]
            ) or (
                "financial_assistance" in a["event_types"]
                and "systemic_banking_crisis" in b["event_types"]
            ):
                relationship_type = (
                    "banking_crisis_and_assistance_programme"
                )
                rationale = (
                    "Potential banking-sovereign stress relationship; "
                    "requires independent historical confirmation."
                )

            elif (
                clean(iso) == "GRC"
                and (
                    (
                        "financial_assistance" in a["event_types"]
                        and "sovereign_restructuring" in b["event_types"]
                    )
                    or (
                        "sovereign_restructuring" in a["event_types"]
                        and "financial_assistance" in b["event_types"]
                    )
                )
            ):
                relationship_type = (
                    "greek_assistance_and_psi_restructuring"
                )
                rationale = (
                    "Potential relationship between Greek assistance "
                    "programmes and the 2011-2012 PSI restructuring."
                )

            if relationship_type is None:
                continue

            def episode_bounds(episode):
                starts = [
                    year_value(r["onset_year"])
                    for r in episode["records"]
                ]
                ends = [
                    year_value(r["end_year"])
                    for r in episode["records"]
                ]

                starts = [v for v in starts if v is not None]
                ends = [v for v in ends if v is not None]

                if not starts or not ends:
                    return None, None

                return min(starts), max(ends)

            a_start, a_end = episode_bounds(a)
            b_start, b_end = episode_bounds(b)

            if None in (a_start, a_end, b_start, b_end):
                continue

            gap = interval_gap(
                a_start, a_end,
                b_start, b_end
            )

            if gap > MAX_YEAR_GAP:
                continue

            # Sort episode identifiers to make relationship IDs
            # independent of the order in which records are read.
            first, second = sorted([
                a["episode_id"],
                b["episode_id"]
            ])

            relationships.append({
                "relationship_id": (
                    f"REL_{slug(first)}_{slug(second)}"
                ),
                "iso": clean(iso),
                "country": clean(
                    a["records"][0]["country"]
                ),
                "episode_id_1": first,
                "event_ids_1": ";".join(
                    a["event_ids"]
                    if a["episode_id"] == first
                    else b["event_ids"]
                ),
                "episode_id_2": second,
                "event_ids_2": ";".join(
                    b["event_ids"]
                    if b["episode_id"] == second
                    else a["event_ids"]
                ),
                "interval_gap_years": gap,
                "relationship_type": relationship_type,
                "review_status": "candidate_not_confirmed",
                "rationale": rationale,
                "automatic_merge": False,
            })

relationship_df = pd.DataFrame(relationships)

if relationship_df.empty:
    relationship_df = pd.DataFrame(columns=[
        "relationship_id",
        "iso",
        "country",
        "episode_id_1",
        "event_ids_1",
        "episode_id_2",
        "event_ids_2",
        "interval_gap_years",
        "relationship_type",
        "review_status",
        "rationale",
        "automatic_merge",
    ])

if not relationship_df.empty:
    if relationship_df["relationship_id"].duplicated().any():
        raise ValueError("Duplicate episode-level relationship IDs.")


# ---------------------------------------------------------
# 5. Final validation
# ---------------------------------------------------------

if len(linkage) != len(registry):
    raise ValueError("Event linkage record count does not reconcile.")

if linkage["event_id"].nunique() != len(registry):
    raise ValueError("Event IDs are not unique in linkage output.")

if linkage["episode_id"].isna().any():
    raise ValueError("Missing episode IDs in linkage output.")

if episodes["episode_id"].duplicated().any():
    raise ValueError("Duplicate episode summary IDs.")

if len(episodes) != linkage["episode_id"].nunique():
    raise ValueError("Episode summary count does not reconcile.")

if len(relationship_df):
    if relationship_df["relationship_id"].duplicated().any():
        raise ValueError("Duplicate relationship IDs.")

# ---------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

linkage.to_csv(EVENT_OUTPUT, index=False, encoding="utf-8")
episodes.to_csv(EPISODE_OUTPUT, index=False, encoding="utf-8")
relationship_df.to_csv(
    RELATION_OUTPUT,
    index=False,
    encoding="utf-8"
)

# Re-read and validate saved outputs.
saved_linkage = pd.read_csv(EVENT_OUTPUT)
saved_episodes = pd.read_csv(EPISODE_OUTPUT)
saved_relationships = pd.read_csv(RELATION_OUTPUT)

if len(saved_linkage) != len(registry):
    raise ValueError("Saved event linkage count is incorrect.")

if len(saved_episodes) != len(episodes):
    raise ValueError("Saved episode summary count is incorrect.")

if len(saved_relationships) != len(relationship_df):
    raise ValueError("Saved relationship count is incorrect.")

# ---------------------------------------------------------
# 7. Report
# ---------------------------------------------------------

lines = [
    "=" * 76,
    "EUROPEAN HISTORICAL STRESS EVENT LINKAGE REPORT",
    "=" * 76,
    "",
    "INPUT",
    "-" * 76,
    f"Registry records: {len(registry)}",
    f"Countries: {registry['iso'].nunique()}",
    "",
    "EPISODE ASSIGNMENT",
    "-" * 76,
    f"Event records: {len(linkage)}",
    f"Unique episodes: {linkage['episode_id'].nunique()}",
    f"Records in shared Greek PSI episode: {len(psi)}",
    "",
    "EVENT TYPE COUNTS",
    "-" * 76,
]

for event_type, count in (
    linkage["event_type"].value_counts(dropna=False).items()
):
    lines.append(f"{event_type}: {count}")

lines.extend([
    "",
    "RELATIONSHIP CANDIDATES",
    "-" * 76,
    f"Candidate relationships: {len(relationship_df)}",
    "Automatically merged relationships: 0",
    "All candidate relationships require review.",
    "",
    "EPISODE SUMMARY",
    "-" * 76,
])

for _, row in episodes.iterrows():
    lines.append(
        f"{row['episode_id']} | "
        f"{row['countries']} | "
        f"events={row['event_record_count']} | "
        f"types={row['event_types']}"
    )

lines.extend([
    "",
    "OUTPUT FILES",
    "-" * 76,
    str(EVENT_OUTPUT),
    str(EPISODE_OUTPUT),
    str(RELATION_OUTPUT),
    "",
    "METHODOLOGICAL CONTROLS",
    "-" * 76,
    "- Original event registry was not modified.",
    "- Every source event remains individually identifiable.",
    "- Greek PSI tranches share one documented episode ID.",
    "- Other events receive separate episode IDs by default.",
    "- Potential relationships are not treated as proven causal links.",
    "- Financial assistance agreements are not automatically treated",
    "  as crisis onset dates.",
    "- Banking crises and sovereign defaults remain distinct event types.",
    "- Event counts and episode counts are reported separately.",
    "",
    "STATUS: VALIDATION PASSED",
])

REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
