import dagster as dg
import pandas as pd

NUMERIC_COLS = ["pm2.5", "pm10", "o3", "co", "so2", "no2", "aqi"]
REQUIRED_COLS = ["sitename", "county", "publishtime"] + NUMERIC_COLS


@dg.asset(group_name="clean")
def cleaned_air_quality(raw_air_quality: pd.DataFrame) -> pd.DataFrame:
    df = raw_air_quality.copy()

    # drop rows missing required fields
    df = df.dropna(subset=["sitename", "publishtime"])

    # coerce numeric columns; non-numeric values (e.g. "--") become NaN
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["publishtime"] = pd.to_datetime(df["publishtime"])
    df = df.sort_values("publishtime")

    return df
