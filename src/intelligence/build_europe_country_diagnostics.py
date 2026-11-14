from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" /
    "europe_country_comparison.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# INDICATOR INTERPRETATION
# ---------------------------------------------------------

DIRECTIONAL_RULES = {
    "real_gdp_growth": {
        "direction": "higher",
        "dimension": "Growth sustainability",
        "description": "Real GDP growth"
    },
    "inflation": {
        "direction": "lower",
        "dimension": "Price stability",
        "description": "Consumer price inflation"
    },
    "government_debt": {
        "direction": "lower",
        "dimension": "Fiscal sustainability",
        "description": "General government gross debt"
    },
    "fiscal_balance": {
        "direction": "higher",
        "dimension": "Fiscal sustainability",
        "description": "General government fiscal balance"
    },
    "primary_balance": {
        "direction": "higher",
        "dimension": "Fiscal sustainability",
        "description": "General government primary balance"
    },
    "unemployment": {
        "direction": "lower",
        "dimension": "Labour market",
        "description": "Unemployment rate"
    }
}

CONTEXT_INDICATORS = {
    "current_account": "External position",
    "exports_volume_growth": "Trade performance",
    "imports_volume_growth": "Trade performance",
    "nominal_gdp_local": "Economic size",
    "nominal_gdp_usd": "Economic size"
}

MIN_COMPARISON_GROUP = 5


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "iso",
    "country",
    "indicator",
    "snapshot_year",
    "latest_value"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["latest_value"] = pd.to_numeric(
    df["latest_value"], errors="coerce"
)

df["snapshot_year"] = pd.to_numeric(
    df["snapshot_year"], errors="coerce"
)

if df.duplicated(["iso", "indicator"]).any():
    raise ValueError(
        "Duplicate country-indicator records detected."
    )

if df["latest_value"].isna().any():
    raise ValueError(
        "Missing latest values detected. "
        "Resolve these before diagnostics."
    )


# ---------------------------------------------------------
# DIRECTIONAL COMPARISON
# ---------------------------------------------------------

results = []

for indicator, rule in DIRECTIONAL_RULES.items():

    subset = df.loc[
        df["indicator"] == indicator
    ].copy()

    if subset.empty:
        print(f"WARNING: Indicator not found: {indicator}")
        continue

    for year, group in subset.groupby("snapshot_year"):

        group = group.copy()
        n = len(group)

        group["comparison_group_size"] = n
        group["favourable_percentile"] = np.nan
        group["relative_position"] = "insufficient_comparison"

        if n >= MIN_COMPARISON_GROUP:

            ascending = rule["direction"] == "higher"

            group["favourable_percentile"] = (
                group["latest_value"].rank(
                    pct=True,
                    ascending=ascending,
                    method="average"
                )
            )

            group.loc[
                group["favourable_percentile"] <= 1 / 3,
                "relative_position"
            ] = "relative_weakness"

            group.loc[
                group["favourable_percentile"] >= 2 / 3,
                "relative_position"
            ] = "relative_strength"

            middle = (
                (group["favourable_percentile"] > 1 / 3) &
                (group["favourable_percentile"] < 2 / 3)
            )

            group.loc[
                middle,
                "relative_position"
            ] = "middle_tercile"

        group["dimension"] = rule["dimension"]
        group["direction"] = rule["direction"]
        group["comparison_year"] = year

        results.append(group)

if not results:
    raise ValueError(
        "No directional indicators were available."
    )

directional = pd.concat(
    results, ignore_index=True
)


# ---------------------------------------------------------
# ADD CONTEXTUAL INDICATORS
# ---------------------------------------------------------

context_rows = df.loc[
    df["indicator"].isin(CONTEXT_INDICATORS)
].copy()

context_rows["dimension"] = (
    context_rows["indicator"].map(CONTEXT_INDICATORS)
)

context_rows["direction"] = "contextual"
context_rows["comparison_group_size"] = np.nan
context_rows["favourable_percentile"] = np.nan
context_rows["relative_position"] = "context_required"
context_rows["comparison_year"] = (
    context_rows["snapshot_year"]
)

# No automatic positive/negative interpretation is applied.
all_indicators = pd.concat(
    [directional, context_rows],
    ignore_index=True
)


# ---------------------------------------------------------
# HISTORICAL VOLATILITY SIGNALS
# ---------------------------------------------------------
# ---------------------------------------------------------
# HISTORICAL VOLATILITY SIGNALS
# ---------------------------------------------------------

VOLATILITY_INDICATORS = [
    "real_gdp_growth",
    "inflation",
    "government_debt",
    "fiscal_balance",
    "primary_balance",
    "unemployment",
    "current_account"
]

if "standard_deviation" not in df.columns:
    raise ValueError(
        "The comparison dataset does not contain "
        "'standard_deviation'."
    )

volatility = df.loc[
    df["indicator"].isin(VOLATILITY_INDICATORS),
    [
        "iso",
        "country",
        "indicator",
        "standard_deviation"
    ]
].copy()

volatility["standard_deviation"] = pd.to_numeric(
    volatility["standard_deviation"],
    errors="coerce"
)

volatility["volatility_percentile"] = np.nan
volatility["volatility_signal"] = "not_assessed"

for indicator, group in volatility.groupby("indicator"):

    valid = group["standard_deviation"].notna()
    valid_group = group.loc[valid].copy()

    if len(valid_group) < MIN_COMPARISON_GROUP:
        continue

    valid_group["volatility_percentile"] = (
        valid_group["standard_deviation"].rank(
            pct=True,
            ascending=True,
            method="average"
        )
    )

    valid_group["volatility_signal"] = (
        "moderate_dispersion"
    )

    valid_group.loc[
        valid_group["volatility_percentile"] >= 2 / 3,
        "volatility_signal"
    ] = "higher_historical_dispersion"

    valid_group.loc[
        valid_group["volatility_percentile"] <= 1 / 3,
        "volatility_signal"
    ] = "lower_historical_dispersion"

    volatility.loc[
        valid_group.index,
        "volatility_percentile"
    ] = valid_group["volatility_percentile"]

    volatility.loc[
        valid_group.index,
        "volatility_signal"
    ] = valid_group["volatility_signal"]

# ---------------------------------------------------------
# BUILD STRENGTHS AND WEAKNESSES
# ---------------------------------------------------------

strengths_weaknesses = directional[
    [
        "iso",
        "country",
        "indicator",
        "dimension",
        "snapshot_year",
        "latest_value",
        "unit",
        "favourable_percentile",
        "comparison_group_size",
        "relative_position"
    ]
].copy()

strengths_weaknesses = strengths_weaknesses.rename(
    columns={"snapshot_year": "reference_year"}
)


# ---------------------------------------------------------
# BUILD VULNERABILITY MATRIX
# ---------------------------------------------------------

vulnerability = directional.loc[
    directional["relative_position"] == "relative_weakness"
].copy()

vulnerability["vulnerability_type"] = (
    vulnerability["dimension"]
)

vulnerability["evidence"] = (
    "Relative weakness among countries with the same "
    "reference year."
)

vulnerability = vulnerability[
    [
        "iso",
        "country",
        "indicator",
        "dimension",
        "snapshot_year",
        "latest_value",
        "favourable_percentile",
        "vulnerability_type",
        "evidence"
    ]
]

# Add high historical dispersion as a separate signal.
# It is not treated as proof of economic vulnerability.

high_volatility = volatility.loc[
    volatility["volatility_signal"] ==
    "higher_historical_dispersion"
].copy()

if not high_volatility.empty:

    high_volatility["dimension"] = "Macroeconomic stability"
    high_volatility["snapshot_year"] = np.nan
    high_volatility["latest_value"] = np.nan
    high_volatility["favourable_percentile"] = np.nan
    high_volatility["vulnerability_type"] = (
        "Historical volatility"
    )
    high_volatility["evidence"] = (
        "Higher historical standard deviation relative "
        "to other countries. Historical sample windows "
        "may differ."
    )

    high_volatility = high_volatility[
        [
            "iso",
            "country",
            "indicator",
            "dimension",
            "snapshot_year",
            "latest_value",
            "favourable_percentile",
            "vulnerability_type",
            "evidence"
        ]
    ]

    vulnerability = pd.concat(
        [vulnerability, high_volatility],
        ignore_index=True
    )


# ---------------------------------------------------------
# COUNTRY-LEVEL DIAGNOSTICS
# ---------------------------------------------------------

summary_rows = []

for iso, group in directional.groupby("iso"):

    country = group["country"].iloc[0]

    strengths = int(
        (group["relative_position"] ==
         "relative_strength").sum()
    )

    weaknesses = int(
        (group["relative_position"] ==
         "relative_weakness").sum()
    )

    middle = int(
        (group["relative_position"] ==
         "middle_tercile").sum()
    )

    insufficient = int(
        (group["relative_position"] ==
         "insufficient_comparison").sum()
    )

    country_volatility = volatility.loc[
        volatility["iso"] == iso
    ]

    elevated_volatility = int(
        (country_volatility["volatility_signal"] ==
         "higher_historical_dispersion").sum()
    )

    summary_rows.append({
        "iso": iso,
        "country": country,
        "directional_indicators_assessed": len(group),
        "relative_strength_count": strengths,
        "relative_weakness_count": weaknesses,
        "middle_tercile_count": middle,
        "insufficient_comparison_count": insufficient,
        "higher_historical_dispersion_count": elevated_volatility,
        "diagnostic_note": (
            "Relative comparisons only; not a sovereign "
            "credit rating or absolute economic assessment."
        )
    })

country_diagnostics = pd.DataFrame(summary_rows)


# ---------------------------------------------------------
# GENERATE REPORT
# ---------------------------------------------------------

report_path = (
    REPORT_DIR /
    "europe_country_diagnostics_report.txt"
)

lines = []

lines.append(
    "EUROPEAN COUNTRY ECONOMIC DIAGNOSTICS"
)
lines.append("=" * 60)
lines.append("")
lines.append("DATASET SUMMARY")
lines.append("-" * 60)
lines.append(f"Countries: {df['iso'].nunique()}")
lines.append(f"Indicators: {df['indicator'].nunique()}")
lines.append(
    f"Directional diagnostic records: {len(directional)}"
)
lines.append(
    f"Strength/weakness records: {len(strengths_weaknesses)}"
)
lines.append(
    f"Vulnerability signals: {len(vulnerability)}"
)
lines.append("")

lines.append("COUNTRY DIAGNOSTIC COUNTS")
lines.append("-" * 60)

for _, row in country_diagnostics.iterrows():
    lines.append(
        f"{row['country']}: "
        f"{row['relative_strength_count']} strengths; "
        f"{row['relative_weakness_count']} weaknesses; "
        f"{row['insufficient_comparison_count']} "
        f"insufficient comparisons; "
        f"{row['higher_historical_dispersion_count']} "
        f"higher-dispersion signals"
    )

lines.append("")
lines.append("METHODOLOGICAL LIMITATIONS")
lines.append("-" * 60)
lines.append(
    "1. Relative positions are calculated within identical "
    "reference years."
)
lines.append(
    "2. Terciles are only calculated when a comparison "
    f"group contains at least {MIN_COMPARISON_GROUP} countries."
)
lines.append(
    "3. A relative weakness does not establish absolute "
    "economic distress."
)
lines.append(
    "4. Current account and trade growth are contextual "
    "indicators, not automatically good or bad."
)
lines.append(
    "5. Higher historical dispersion is not proof of "
    "economic vulnerability."
)
lines.append(
    "6. Historical sample windows differ between countries "
    "and indicators."
)
lines.append(
    "7. Latest historical observations are not real-time "
    "economic conditions."
)
lines.append(
    "8. No composite economic score or sovereign credit "
    "rating is calculated."
)

report_path.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

country_diagnostics.to_csv(
    OUTPUT_DIR / "europe_country_diagnostics.csv",
    index=False
)

strengths_weaknesses.to_csv(
    OUTPUT_DIR / "europe_country_strengths_weaknesses.csv",
    index=False
)

vulnerability.to_csv(
    OUTPUT_DIR / "europe_country_vulnerability_matrix.csv",
    index=False
)

all_indicators.to_csv(
    OUTPUT_DIR / "europe_country_diagnostic_indicators.csv",
    index=False
)

volatility.to_csv(
    OUTPUT_DIR / "europe_country_volatility_signals.csv",
    index=False
)


# ---------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------

print("\nEUROPEAN COUNTRY DIAGNOSTICS")
print("=" * 55)

print(f"Countries: {country_diagnostics['iso'].nunique()}")
print(f"Directional records: {len(directional)}")
print(f"Strength/weakness records: {len(strengths_weaknesses)}")
print(f"Vulnerability signals: {len(vulnerability)}")

print("\nOUTPUT FILES")
print("-" * 55)

print("data/processed/europe_country_diagnostics.csv")
print("data/processed/europe_country_strengths_weaknesses.csv")
print("data/processed/europe_country_vulnerability_matrix.csv")
print("data/processed/europe_country_diagnostic_indicators.csv")
print("data/processed/europe_country_volatility_signals.csv")
print("outputs/reports/europe_country_diagnostics_report.txt")

print("\nDIAGNOSTICS BUILD COMPLETED.")
