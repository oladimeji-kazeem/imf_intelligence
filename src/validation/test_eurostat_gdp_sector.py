import requests

BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/gov_10dd_edpt1"
)

sectors = ["S1", "S13"]
countries = ["BE"]

for sector in sectors:
    params = {
        "freq": "A",
        "unit": "MIO_NAC",
        "sector": sector,
        "na_item": "B1GQ",
        "geo": countries[0],
        "sinceTimePeriod": "2020",
        "untilTimePeriod": "2024",
    }

    response = requests.get(BASE_URL, params=params, timeout=60)

    print("\n" + "=" * 65)
    print("SECTOR:", sector)
    print("HTTP STATUS:", response.status_code)
    print("REQUEST URL:", response.url)

    response.raise_for_status()
    data = response.json()

    values = data.get("value", {})

    print("OBSERVATIONS:", len(values))
    print("VALUES:", values)

    dimensions = data.get("id", [])
    sizes = data.get("size", [])

    for dimension, size in zip(dimensions, sizes):
        categories = (
            data.get("dimension", {})
            .get(dimension, {})
            .get("category", {})
            .get("index", {})
        )

        print(f"{dimension}: {list(categories.keys())}")