from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

AQI_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CITY = "Delhi"
TARGET = "PM2.5"


def load_city_day() -> pd.DataFrame:
    """city_day.csv dosyasını tarih sütunu datetime olacak şekilde yükler."""
    path = RAW_DIR / "city_day.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. Veri setini Kaggle'dan "
            "(Air Quality Data in India 2015-2020) indirip "
            "CSV dosyalarını data/raw/ klasörüne koyun."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    df["AQI_Bucket"] = pd.Categorical(df["AQI_Bucket"], categories=AQI_ORDER, ordered=True)
    return df.sort_values(["City", "Date"]).reset_index(drop=True)