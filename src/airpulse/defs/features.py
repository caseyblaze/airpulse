import numpy as np
import pandas as pd

TARGET = "pm25"
TARGET_LAGS = 3

# Exogenous numeric columns. Leakage rule: these may only enter the model as
# their lag-1 value, never the same-hour value.
EXO_COLS = [
    "pm10", "o3", "co", "so2", "no2", "nox", "no", "aqi",
    "pm25_avg", "pm10_avg", "o3_8hr", "co_8hr", "so2_avg", "wind_speed",
]
# Same-hour features that are known at prediction time (no leakage).
GEO_COLS = ["latitude", "longitude"]


def add_lag_features(df: pd.DataFrame, col: str, lags: int) -> pd.DataFrame:
    """Add `{col}_lag1..lags`, shifted within each station, time-ordered."""
    df = df.sort_values(["sitename", "publishtime"]).copy()
    for i in range(1, lags + 1):
        df[f"{col}_lag{i}"] = df.groupby("sitename")[col].shift(i)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = "pm25", window: int = 3) -> pd.DataFrame:
    """Mean/std over the lagged values only (leakage-safe). Requires
    `{col}_lag1..window` to already exist."""
    df = df.copy()
    lag_cols = [f"{col}_lag{i}" for i in range(1, window + 1)]
    df[f"{col}_roll{window}_mean"] = df[lag_cols].mean(axis=1, skipna=False)
    df[f"{col}_roll{window}_std"] = df[lag_cols].std(axis=1, skipna=False)
    return df
