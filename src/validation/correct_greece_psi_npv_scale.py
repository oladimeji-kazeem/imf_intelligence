
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "external" / "esm_domestic_sovereign_defaults.xlsx"
REGISTRY = ROOT / "data" / "external" / "europe_historical_stress_events.csv"
REPORT = ROOT / "outputs" / "reports" / "europe_greece_psi_npv_correction_report.txt"

EPISODE_ID = "EU_SOV_GRC_PSI_2011_2012"

EXPECTED = {
    "BONDS_GUARANTEED_BONDS": 56.40,
    "BONDS_GOVERNMENT_BONDS": 55.85,
}

if not SOURCE.exists() or not REGISTRY.exists():
    raise FileNotFoundError("Source workbook or event registry is missing.")

source = pd.read_excel(SOURCE, sheet_name="2022 version")
registry = pd.read_csv(REGISTRY)

greek = source[
    source["Country"].astype(str).str.strip().str.casefold() == "greece"
].copy()

if len(greek) != 2:
    raise ValueError(f"Expected two Greek source records, found {len(greek)}")

updated = []

for _, row in greek.iterrows():
    instrument = str(row["Instrument"]).strip()
    slug = "".join(
        c if c.isalnum() else "_"
        for c in instrument.upper()
    )
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")

    event_id = f"{EPISODE_ID}_{slug}"

    if slug not in EXPECTED:
        raise ValueError(f"Unexpected instrument: {instrument}")

    mask = registry["event_id"].astype(str) == event_id

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one registry record for {event_id}"
        )

    raw_value = float(row["NPV loss (%)"])

    # ESM workbook stores these values as fractions.
    corrected_value = (
        raw_value * 100 if abs(raw_value) <= 1 else raw_value
    )

    expected_value = EXPECTED[slug]

    if abs(corrected_value - expected_value) > 0.02:
        raise ValueError(
            f"Unexpected NPV value for {event_id}: "
            f"raw={raw_value}, converted={corrected_value}, "
            f"expected={expected_value}"
        )

    registry.loc[mask, "npv_loss_pct"] = corrected_value

    updated.append({
        "event_id": event_id,
        "instrument": instrument,
        "raw_source_value": raw_value,
        "corrected_npv_loss_pct": corrected_value,
        "expected_npv_loss_pct": expected_value,
    })

# Validate both corrections before saving.
if len(updated) != 2:
    raise ValueError("Expected two corrected records.")

for item in updated:
    if abs(
        item["corrected_npv_loss_pct"]
        - item["expected_npv_loss_pct"]
    ) > 0.02:
        raise ValueError("NPV correction validation failed.")

# Save only after successful validation.
registry.to_csv(REGISTRY, index=False, encoding="utf-8")

# Re-read and verify persisted values.
saved = pd.read_csv(REGISTRY)

lines = [
    "=" * 72,
    "GREEK PSI NPV LOSS SCALE CORRECTION",
    "=" * 72,
    "",
]

for item in updated:
    saved_value = saved.loc[
        saved["event_id"] == item["event_id"],
        "npv_loss_pct"
    ].iloc[0]

    if abs(saved_value - item["expected_npv_loss_pct"]) > 0.02:
        raise ValueError("Saved NPV value does not reconcile.")

    lines.extend([
        f"Instrument: {item['instrument']}",
        f"Raw source value: {item['raw_source_value']}",
        f"Corrected NPV loss: {saved_value:.2f}%",
        "Validation: PASS",
        "",
    ])

lines.extend([
    f"Records corrected: {len(updated)}",
    "Unit conversion: decimal fraction to percentage",
    "STATUS: CORRECTION VALIDATED",
])

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
