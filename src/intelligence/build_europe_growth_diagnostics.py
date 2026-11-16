
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "europe_growth_risk_input_panel.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "europe_growth_diagnostics.csv"
)

ROLLING_FILE = (
    ROOT / "data" / "processed"
    / "europe_growth_rolling_diagnostics.csv"
)

EVENT_FILE = (
    ROOT / "data" / "processed"
    / "europe_growth_contraction_events.csv"
)

REPORT_DIR = ROOT / "outputs" / "reports"

REPORT_FILE = (
    REPORT_DIR / "europe_growth_diagnostics_report.txt"
)

WINDOWS = {
    "long_run_1980_2025": (1980, 2025),
    "pre_pandemic_2000_2019": (2000, 2019),
    "recent_pre_pandemic_2015_2019": (2015, 2019),
    "pandemic_post_pandemic_2020_2024": (2020, 2024),
}

ROLLING_WINDOW = 5
RECOVERY_HORIZON = 5

print("=" * 78)
print("EUROPEAN GROWTH RISK — DIAGNOSTIC ENGINE")
print("=" * 78)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required = [
    "ISO",
    "country",
    "year",
    "growth_pct",
    "observation_status",
]

missing_columns = [
    col for col in required if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["growth_pct"] = pd.to_numeric(
    df["growth_pct"], errors="coerce"
)

if df[["ISO", "year"]].isna().any().any():
    raise ValueError("Missing country or year keys.")

if df.duplicated(["ISO", "year"]).any():
    raise ValueError("Duplicate country-year observations found.")

if not df["observation_status"].eq("historical_actual").all():
    raise ValueError(
        "Non-historical observations detected in input panel."
    )

df["year"] = df["year"].astype(int)
df = df.sort_values(["ISO", "year"]).reset_index(drop=True)

countries = sorted(df["ISO"].unique())

# ---------------------------------------------------------
# 1. Diagnostic calculations
# ---------------------------------------------------------

def calculate_metrics(group, start_year, end_year):

    period = group[
        group["year"].between(start_year, end_year)
    ].copy()

    values = period["growth_pct"].dropna()

    n = len(values)
    records = len(period)

    if n == 0:
        return {
            "period_start": start_year,
            "period_end": end_year,
            "calendar_records": records,
            "numeric_observations": 0,
            "missing_observations": records,
            "mean_growth_pct": np.nan,
            "median_growth_pct": np.nan,
            "growth_std_pct": np.nan,
            "growth_mad_pct": np.nan,
            "growth_iqr_pct": np.nan,
            "contraction_count": 0,
            "contraction_frequency_pct": np.nan,
            "average_contraction_pct": np.nan,
            "worst_contraction_pct": np.nan,
            "best_growth_pct": np.nan,
            "downside_deviation_pct": np.nan,
        }

    contractions = values[values < 0]

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    median = values.median()

    downside = np.minimum(values.to_numpy(), 0)

    return {
        "period_start": start_year,
        "period_end": end_year,
        "calendar_records": records,
        "numeric_observations": n,
        "missing_observations": records - n,
        "mean_growth_pct": values.mean(),
        "median_growth_pct": median,
        "growth_std_pct": values.std(ddof=1),
        "growth_mad_pct": (
            np.median(np.abs(values.to_numpy() - median))
        ),
        "growth_iqr_pct": q3 - q1,
        "contraction_count": len(contractions),
        "contraction_frequency_pct": (
            len(contractions) / n * 100
        ),
        "average_contraction_pct": (
            contractions.mean()
            if len(contractions) else np.nan
        ),
        "worst_contraction_pct": (
            values.min()
        ),
        "best_growth_pct": (
            values.max()
        ),
        "downside_deviation_pct": (
            np.sqrt(np.mean(downside ** 2))
        ),
    }


# ---------------------------------------------------------
# 2. Country-period diagnostics
# ---------------------------------------------------------

diagnostic_rows = []

for iso in countries:

    country_data = df[df["ISO"].eq(iso)].copy()

    country_name = country_data["country"].iloc[0]

    for period_name, (start, end) in WINDOWS.items():

        metrics = calculate_metrics(
            country_data,
            start,
            end,
        )

        diagnostic_rows.append({
            "ISO": iso,
            "country": country_name,
            "period": period_name,
            **metrics,
        })

diagnostics = pd.DataFrame(diagnostic_rows)

# ---------------------------------------------------------
# 3. Rolling five-year growth diagnostics
# ---------------------------------------------------------

rolling_rows = []

for iso in countries:

    country_data = (
        df[df["ISO"].eq(iso)]
        .sort_values("year")
        .reset_index(drop=True)
    )

    country_name = country_data["country"].iloc[0]

    for i in range(ROLLING_WINDOW - 1, len(country_data)):

        window = country_data.iloc[
            i - ROLLING_WINDOW + 1:i + 1
        ].copy()

        years = window["year"].tolist()
        values = window["growth_pct"]

        # Require five consecutive calendar years.
        consecutive = (
            len(years) == ROLLING_WINDOW
            and all(
                years[j] - years[j - 1] == 1
                for j in range(1, len(years))
            )
        )

        complete = (
            consecutive
            and values.notna().all()
        )

        if not complete:
            continue

        numeric = values.astype(float)

        rolling_rows.append({
            "ISO": iso,
            "country": country_name,
            "window_start": years[0],
            "window_end": years[-1],
            "rolling_mean_growth_pct": numeric.mean(),
            "rolling_median_growth_pct": numeric.median(),
            "rolling_volatility_pct": numeric.std(ddof=1),
            "rolling_contraction_count": int(
                (numeric < 0).sum()
            ),
            "rolling_worst_growth_pct": numeric.min(),
        })

rolling = pd.DataFrame(rolling_rows)

# ---------------------------------------------------------
# 4. Contraction event and recovery analysis
# ---------------------------------------------------------

event_rows = []

for iso in countries:

    country_data = (
        df[df["ISO"].eq(iso)]
        .sort_values("year")
        .reset_index(drop=True)
    )

    country_name = country_data["country"].iloc[0]

    contractions = country_data[
        country_data["growth_pct"].notna()
        & country_data["growth_pct"].lt(0)
    ]

    for _, event in contractions.iterrows():

        event_year = int(event["year"])

        recovery_candidates = country_data[
            country_data["year"].gt(event_year)
            & country_data["year"].le(
                event_year + RECOVERY_HORIZON
            )
            & country_data["growth_pct"].gt(0)
        ]

        if recovery_candidates.empty:
            recovery_year = np.nan
            recovery_lag = np.nan
            recovered = False
        else:
            recovery_year = int(
                recovery_candidates.iloc[0]["year"]
            )
            recovery_lag = recovery_year - event_year
            recovered = True

        event_rows.append({
            "ISO": iso,
            "country": country_name,
            "contraction_year": event_year,
            "contraction_growth_pct": event["growth_pct"],
            "recovery_year": recovery_year,
            "years_to_positive_growth": recovery_lag,
            "positive_growth_within_5_years": recovered,
        })

events = pd.DataFrame(event_rows)

# ---------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------

expected_diagnostics = len(countries) * len(WINDOWS)

if len(diagnostics) != expected_diagnostics:
    raise ValueError(
        "Unexpected country-period diagnostic count."
    )

if diagnostics.duplicated(
    ["ISO", "period"]
).any():
    raise ValueError("Duplicate country-period diagnostics.")

if not rolling.empty and rolling.duplicated(
    ["ISO", "window_start", "window_end"]
).any():
    raise ValueError("Duplicate rolling diagnostic windows.")

if not events.empty and events.duplicated(
    ["ISO", "contraction_year"]
).any():
    raise ValueError("Duplicate contraction events.")

# ---------------------------------------------------------
# 6. Special observation checks
# ---------------------------------------------------------

ireland_2015 = df[
    df["ISO"].eq("IRL")
    & df["year"].eq(2015)
]

romania_1991 = df[
    df["ISO"].eq("ROU")
    & df["year"].eq(1991)
]

# ---------------------------------------------------------
# 7. Output files
# ---------------------------------------------------------

REPORT_DIR.mkdir(parents=True, exist_ok=True)

diagnostics.to_csv(OUTPUT_FILE, index=False)
rolling.to_csv(ROLLING_FILE, index=False)
events.to_csv(EVENT_FILE, index=False)

# ---------------------------------------------------------
# 8. Report generation
# ---------------------------------------------------------

lines = []


def add(value=""):
    lines.append(str(value))


add("=" * 78)
add("EUROPEAN GROWTH RISK DIAGNOSTICS REPORT")
add("=" * 78)

add(f"Countries: {len(countries)}")
add(f"Historical period: {df['year'].min()}-{df['year'].max()}")
add(f"Country-period diagnostics: {len(diagnostics)}")
add(f"Rolling five-year windows: {len(rolling)}")
add(f"Contraction events: {len(events)}")

add()
add("PERIOD DIAGNOSTIC COVERAGE")
add("-" * 78)

add(
    diagnostics.groupby("period")
    .agg(
        countries=("ISO", "nunique"),
        numeric_observations=("numeric_observations", "sum"),
        missing_observations=("missing_observations", "sum"),
    )
    .to_string()
)

for period_name in WINDOWS:

    subset = diagnostics[
        diagnostics["period"].eq(period_name)
    ].copy()

    add()
    add(f"PERIOD: {period_name}")
    add("-" * 78)

    columns = [
        "ISO",
        "mean_growth_pct",
        "median_growth_pct",
        "growth_std_pct",
        "contraction_frequency_pct",
        "worst_contraction_pct",
    ]

    add(
        subset[columns]
        .sort_values("mean_growth_pct")
        .round(3)
        .to_string(index=False)
    )

add()
add("ROLLING FIVE-YEAR GROWTH REGIME OBSERVATIONS")
add("-" * 78)

if rolling.empty:
    add("No complete rolling five-year windows.")
else:
    latest_rolling = (
        rolling.sort_values("window_end")
        .groupby("ISO")
        .tail(1)
        .sort_values("rolling_mean_growth_pct")
    )

    add("Latest available rolling window by country:")
    add(
        latest_rolling[
            [
                "ISO",
                "window_start",
                "window_end",
                "rolling_mean_growth_pct",
                "rolling_volatility_pct",
                "rolling_contraction_count",
            ]
        ].round(3).to_string(index=False)
    )

add()
add("CONTRACTION AND RECOVERY ANALYSIS")
add("-" * 78)

if events.empty:
    add("No contraction events identified.")
else:
    recovery_summary = (
        events.groupby("ISO")
        .agg(
            contraction_events=("contraction_year", "count"),
            recovered_within_5y=(
                "positive_growth_within_5_years", "sum"
            ),
            average_recovery_lag=(
                "years_to_positive_growth", "mean"
            ),
        )
        .reset_index()
    )

    add(recovery_summary.round(3).to_string(index=False))

add()
add("SPECIAL OBSERVATION CHECKS")
add("-" * 78)

if not ireland_2015.empty:
    add(
        "Ireland 2015 real GDP growth: "
        f"{ireland_2015['growth_pct'].iloc[0]:.3f}%"
    )

if not romania_1991.empty:
    add(
        "Romania 1991 real GDP growth: "
        f"{romania_1991['growth_pct'].iloc[0]:.3f}%"
    )

add()
add("METHODOLOGICAL LIMITATIONS")
add("-" * 78)
add(
    "Mean and median growth are descriptive historical measures, "
    "not forecasts."
)
add(
    "Standard deviation measures dispersion and is sensitive "
    "to extreme observations."
)
add(
    "Median absolute deviation and interquartile range provide "
    "robust dispersion measures."
)
add(
    "Contraction frequency is based on negative annual real "
    "GDP growth, not recession dating."
)
add(
    "Recovery means the first subsequent positive annual growth "
    "observation within five calendar years. It does not mean "
    "real GDP has regained its pre-contraction level."
)
add(
    "Rolling five-year statistics require five consecutive "
    "numeric annual observations."
)
add(
    "Ireland's exceptional 2015 growth is retained. It should "
    "not be interpreted as ordinary domestic economic growth."
)
add(
    "No arbitrary risk thresholds, composite scores, causal "
    "claims or forecasts are generated."
)

add()
add("VALIDATION")
add("-" * 78)
add("Country-period row count: PASS")
add("Country-period uniqueness: PASS")
add("Rolling window uniqueness: PASS")
add("Contraction event uniqueness: PASS")

REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\n".join(lines))

print()
print("OUTPUTS CREATED")
print("-" * 78)
print(OUTPUT_FILE)
print(ROLLING_FILE)
print(EVENT_FILE)
print(REPORT_FILE)

print()
print("VALIDATION PASSED")
