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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour/day-of-week/month encodings from publishtime (same-hour,
    known at prediction time)."""
    df = df.copy()
    t = pd.to_datetime(df["publishtime"])
    for name, values, period in [
        ("hour", t.dt.hour, 24),
        ("dow", t.dt.dayofweek, 7),
        ("month", t.dt.month, 12),
    ]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    return df


def add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the lagged wind direction (degrees) as sin/cos. Requires
    `wind_direc_lag1` to already exist."""
    df = df.copy()
    rad = np.deg2rad(df["wind_direc_lag1"])
    df["wind_dir_sin_lag1"] = np.sin(rad)
    df["wind_dir_cos_lag1"] = np.cos(rad)
    return df


def add_county_onehot(df: pd.DataFrame):
    """One-hot encode county (~22 levels). Returns (df, new_column_names)."""
    df = df.copy()
    dummies = pd.get_dummies(df["county"], prefix="county")
    df = pd.concat([df, dummies], axis=1)
    return df, list(dummies.columns)


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Global timestamp cutoff: the latest `test_frac` of distinct timestamps
    is the test set; train is strictly earlier. Returns (train_df, test_df)."""
    times = np.sort(df["publishtime"].unique())
    n_test = max(1, int(round(len(times) * test_frac)))
    cutoff = times[-n_test]
    train = df[df["publishtime"] < cutoff]
    test = df[df["publishtime"] >= cutoff]
    return train, test


def build_feature_matrix(history: pd.DataFrame):
    """Assemble the leakage-safe feature matrix from accumulated history.

    Returns (feat_df, feature_cols). Exogenous pollutants/meteorology enter
    only as lag-1; only time and geo features use the same-hour value. Rows
    without complete features (cold start) are dropped."""
    df = history.copy()
    df["publishtime"] = pd.to_datetime(df["publishtime"])
    df = df.sort_values(["sitename", "publishtime"])

    df = add_lag_features(df, TARGET, TARGET_LAGS)
    for col in EXO_COLS:
        if col in df.columns:
            df = add_lag_features(df, col, 1)
    if "wind_direc" in df.columns:
        df = add_lag_features(df, "wind_direc", 1)
        df = add_wind_direction_features(df)
    df = add_rolling_features(df, TARGET, 3)
    df = add_time_features(df)
    df, county_cols = add_county_onehot(df)

    feature_cols = (
        [f"{TARGET}_lag{i}" for i in range(1, TARGET_LAGS + 1)]
        + [f"{TARGET}_roll3_mean", f"{TARGET}_roll3_std"]
        + [f"{c}_lag1" for c in EXO_COLS if c in history.columns]
        + (["wind_dir_sin_lag1", "wind_dir_cos_lag1"] if "wind_direc" in history.columns else [])
        + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        + [c for c in GEO_COLS if c in df.columns]
        + county_cols
    )

    df = df.dropna(subset=[TARGET, *feature_cols])
    return df, feature_cols
