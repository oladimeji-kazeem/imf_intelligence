
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "europe_growth_risk_input_panel.csv"
)

EPISODE_FILE = (
    ROOT / "data" / "processed"
    / "europe_growth_contraction_episodes.csv"
)

INDEX_FILE = (
    ROOT / "data" / "processed"
    / "europe_real_gdp_index_diagnostic.csv"
)

SUMMARY_FILE = (
    ROOT / "data" / "processed"
    / "europe_growth_recovery_summary.csv"
)

REPORT_DIR = ROOT / "outputs" / "reports"

REPORT_FILE = (
    REPORT_DIR / "europe_growth_recovery_report.txt"
)

RECOVERY_HORIZON = 5

print("=" * 78)
print("EUROPEAN GROWTH RECOVERY — EPISODE ANALYSIS")
print("=" * 78)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required = ["ISO", "country", "year", "growth_pct"]

if any(column not in df.columns for column in required):
    raise ValueError("Required input columns are missing.")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["growth_pct"] = pd.to_numeric(
    df["growth_pct"], errors="coerce"
)

if df[["ISO", "year"]].isna().any().any():
    raise ValueError("Invalid country or year keys.")

if df.duplicated(["ISO", "year"]).any():
    raise ValueError("Duplicate country-year observations.")

df["year"] = df["year"].astype(int)
df = df.sort_values(["ISO", "year"]).reset_index(drop=True)

index_frames = []
episode_rows = []

# ---------------------------------------------------------
# 1. Construct chained real GDP index
# ---------------------------------------------------------

for iso, country_data in df.groupby("ISO", sort=True):

    country_data = (
        country_data.sort_values("year").reset_index(drop=True)
    )

    country_name = country_data["country"].iloc[0]

    index_values = []
    chain_ids = []

    chain_number = 0
    previous_year = None
    previous_index = None

    for _, row in country_data.iterrows():

        year = int(row["year"])
        growth = row["growth_pct"]

        if pd.isna(growth):
            index_value = np.nan
            chain_id = np.nan
            previous_year = None
            previous_index = None

        elif (
            previous_year is None
            or year != previous_year + 1
        ):
            chain_number += 1
            chain_id = chain_number
            index_value = 100.0

        else:
            chain_id = chain_number
            index_value = (
                previous_index * (1 + growth / 100)
            )

        index_values.append(index_value)
        chain_ids.append(chain_id)

        if pd.notna(growth):
            previous_year = year
            previous_index = index_value

    country_data["real_gdp_index"] = index_values
    country_data["chain_id"] = chain_ids

    index_frames.append(country_data)

index_df = pd.concat(
    index_frames, ignore_index=True
)

# ---------------------------------------------------------
# 2. Identify consecutive contraction episodes
# ---------------------------------------------------------

for iso, country_data in index_df.groupby("ISO", sort=True):

    country_data = (
        country_data.sort_values("year").reset_index(drop=True)
    )

    country_name = country_data["country"].iloc[0]

    negative = (
        country_data["growth_pct"].notna()
        & country_data["growth_pct"].lt(0)
    )

    episode_number = 0
    i = 0

    while i < len(country_data):

        if not negative.iloc[i]:
            i += 1
            continue

        episode_number += 1

        start_position = i
        start_year = int(country_data.loc[i, "year"])

        episode_chain = country_data.loc[i, "chain_id"]

        while (
            i + 1 < len(country_data)
            and negative.iloc[i + 1]
            and country_data.loc[i + 1, "year"]
            == country_data.loc[i, "year"] + 1
            and country_data.loc[i + 1, "chain_id"]
            == episode_chain
        ):
            i += 1

        end_position = i
        end_year = int(country_data.loc[i, "year"])

        episode = country_data.iloc[
            start_position:end_position + 1
        ]

        baseline_year = start_year - 1

        baseline = country_data[
            country_data["year"].eq(baseline_year)
            & country_data["chain_id"].eq(episode_chain)
        ]

        if baseline.empty:
            baseline_index = np.nan
        else:
            baseline_index = float(
                baseline["real_gdp_index"].iloc[0]
            )

        episode_trough_index = float(
            episode["real_gdp_index"].min()
        )

        cumulative_contraction_pct = (
            episode_trough_index / baseline_index - 1
        ) * 100 if (
            pd.notna(baseline_index) and baseline_index > 0
        ) else np.nan

        recovery_candidates = country_data[
            country_data["year"].gt(end_year)
            & country_data["year"].le(
                end_year + RECOVERY_HORIZON
            )
            & country_data["chain_id"].eq(episode_chain)
        ].copy()

        if pd.isna(baseline_index):
            recovery_status = "unassessable"
            recovery_year = np.nan
            recovery_lag = np.nan

        else:
            recovered = recovery_candidates[
                recovery_candidates["real_gdp_index"]
                >= baseline_index
            ]

            if not recovered.empty:
                recovery_year = int(
                    recovered.iloc[0]["year"]
                )
                recovery_lag = recovery_year - end_year
                recovery_status = "recovered"

            elif (
                len(recovery_candidates) > 0
                and recovery_candidates["year"].max()
                >= end_year + RECOVERY_HORIZON
            ):
                recovery_year = np.nan
                recovery_lag = np.nan
                recovery_status = "not_recovered_within_5_years"

            else:
                recovery_year = np.nan
                recovery_lag = np.nan
                recovery_status = "follow_up_incomplete"

        episode_rows.append({
            "ISO": iso,
            "country": country_name,
            "episode_number": episode_number,
            "contraction_start_year": start_year,
            "contraction_end_year": end_year,
            "contraction_years": len(episode),
            "worst_annual_growth_pct": episode["growth_pct"].min(),
            "average_annual_growth_pct": episode["growth_pct"].mean(),
            "baseline_year": baseline_year,
            "baseline_real_gdp_index": baseline_index,
            "trough_real_gdp_index": episode_trough_index,
            "cumulative_output_change_to_trough_pct": (
                cumulative_contraction_pct
            ),
            "recovery_year": recovery_year,
            "years_to_output_level_recovery": recovery_lag,
            "recovery_status": recovery_status,
        })

        i += 1

# ---------------------------------------------------------
# 3. Create outputs
# ---------------------------------------------------------

episodes = pd.DataFrame(episode_rows)

if episodes.empty:
    raise ValueError("No contraction episodes identified.")

index_df.to_csv(INDEX_FILE, index=False)
episodes.to_csv(EPISODE_FILE, index=False)

summary = (
    episodes.groupby("ISO")
    .agg(
        contraction_episodes=("episode_number", "count"),
        recovered_episodes=(
            "recovery_status",
            lambda x: (x == "recovered").sum()
        ),
        not_recovered_within_5_years=(
            "recovery_status",
            lambda x: (
                x == "not_recovered_within_5_years"
            ).sum()
        ),
        unassessable_episodes=(
            "recovery_status",
            lambda x: (x == "unassessable").sum()
        ),
        incomplete_follow_up=(
            "recovery_status",
            lambda x: (x == "follow_up_incomplete").sum()
        ),
        average_recovery_years=(
            "years_to_output_level_recovery", "mean"
        ),
        average_output_loss_pct=(
            "cumulative_output_change_to_trough_pct", "mean"
        ),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_FILE, index=False)

# ---------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------

if episodes.duplicated(
    ["ISO", "episode_number"]
).any():
    raise ValueError("Duplicate contraction episode identifiers.")

if not episodes["contraction_years"].ge(1).all():
    raise ValueError("Invalid contraction episode duration.")

if not episodes["recovery_status"].isin([
    "recovered",
    "not_recovered_within_5_years",
    "unassessable",
    "follow_up_incomplete",
]).all():
    raise ValueError("Unexpected recovery classification.")

recovered = episodes[
    episodes["recovery_status"].eq("recovered")
]

if not (
    recovered["recovery_year"]
    > recovered["contraction_end_year"]
).all():
    raise ValueError("Recovery occurs before episode ends.")

if not (
    recovered["years_to_output_level_recovery"]
    <= RECOVERY_HORIZON
).all():
    raise ValueError("Recovery exceeds the specified horizon.")

# ---------------------------------------------------------
# 5. Report
# ---------------------------------------------------------

lines = []


def add(value=""):
    lines.append(str(value))


add("=" * 78)
add("EUROPEAN GROWTH RECOVERY DIAGNOSTICS REPORT")
add("=" * 78)

add(f"Countries: {episodes['ISO'].nunique()}")
add(f"Contraction episodes: {len(episodes)}")

add()
add("RECOVERY STATUS")
add("-" * 78)
add(
    episodes["recovery_status"]
    .value_counts(dropna=False)
    .to_string()
)

add()
add("COUNTRY RECOVERY SUMMARY")
add("-" * 78)
add(summary.round(3).to_string(index=False))

add()
add("CONTRACTION EPISODES")
add("-" * 78)
add(
    episodes.sort_values(
        ["ISO", "contraction_start_year"]
    ).round(3).to_string(index=False)
)

add()
add("METHODOLOGY")
add("-" * 78)
add(
    "Consecutive negative annual growth observations are "
    "grouped into contraction episodes."
)
add(
    "The real GDP index is chained from observed annual "
    "real GDP growth rates, using 100 as the starting value "
    "within each continuous data segment."
)
add(
    "Recovery occurs when the chained index returns to or "
    "exceeds its level immediately before the contraction episode."
)
add(
    "A missing observation or calendar-year gap breaks the "
    "index chain. Recovery cannot be established across that gap."
)
add(
    "Episodes without a preceding baseline are unassessable."
)
add(
    "This is an annual GDP-level recovery measure, not an "
    "official recession dating methodology."
)
add(
    "The chained index is a diagnostic reconstruction from "
    "annual growth rates, not a substitute for official real GDP "
    "volume levels."
)

add()
add("VALIDATION")
add("-" * 78)
add("Episode identifiers unique: PASS")
add("Episode durations valid: PASS")
add("Recovery dates after contraction: PASS")
add("Recovery horizon respected: PASS")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\n".join(lines))

print()
print("OUTPUTS CREATED")
print("-" * 78)
print(INDEX_FILE)
print(EPISODE_FILE)
print(SUMMARY_FILE)
print(REPORT_FILE)

print()
print("VALIDATION PASSED")
