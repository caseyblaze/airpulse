import numpy as np
import pandas as pd

from airpulse.defs.cleaning import clean_frame, NUMERIC_COLS


def test_clean_frame_coerces_all_numeric_and_drops_missing_keys():
    raw = pd.DataFrame(
        {
            "sitename": ["A", "B", None],
            "county": ["X", "Y", "Z"],
            "publishtime": ["2026/06/05 22:00:00", "2026/06/05 22:00:00", "2026/06/05 22:00:00"],
            "pm2.5": ["3", "--", "9"],
            "wind_speed": ["1.2", "2.4", "3.0"],
            "latitude": ["25.0", "24.9", "24.8"],
        }
    )
    out = clean_frame(raw)

    # row with missing sitename dropped
    assert len(out) == 2
    # "--" coerced to NaN, numerics are float
    assert out["pm2.5"].isna().sum() == 1
    assert str(out["wind_speed"].dtype).startswith("float")
    assert str(out["latitude"].dtype).startswith("float")
    # new meteorology/geo fields are registered as numeric
    assert "wind_speed" in NUMERIC_COLS
    assert "latitude" in NUMERIC_COLS


from airpulse.defs.history import COL_MAP, ALL_COLS


def test_history_stores_widened_normalized_columns():
    # dotted API names normalized to valid SQL identifiers
    assert COL_MAP["pm2.5"] == "pm25"
    assert COL_MAP["pm2.5_avg"] == "pm25_avg"
    # new fields are mapped for storage
    for api_field in ["wind_speed", "wind_direc", "latitude", "longitude", "no", "nox"]:
        assert api_field in COL_MAP
    # identity + stored numeric names are all present in ALL_COLS
    for stored in ["sitename", "siteid", "county", "publishtime", "pm25", "wind_speed", "latitude"]:
        assert stored in ALL_COLS


from airpulse.defs.features import add_lag_features


def _two_site_series():
    times = pd.date_range("2026-06-01", periods=5, freq="h")
    rows = []
    for s in ["A", "B"]:
        for i, t in enumerate(times):
            rows.append({"sitename": s, "publishtime": t, "pm25": float(i)})
    return pd.DataFrame(rows)


def test_add_lag_features_shifts_within_site():
    df = add_lag_features(_two_site_series(), "pm25", 2)
    a = df[df.sitename == "A"].sort_values("publishtime").reset_index(drop=True)
    assert pd.isna(a.loc[0, "pm25_lag1"])
    assert pd.isna(a.loc[1, "pm25_lag2"])
    assert a.loc[2, "pm25_lag1"] == 1.0
    assert a.loc[2, "pm25_lag2"] == 0.0
    assert a["pm25_lag1"].dropna().tolist() == [0.0, 1.0, 2.0, 3.0]


from airpulse.defs.features import add_rolling_features


def test_add_rolling_features_uses_only_lags():
    df = add_lag_features(_two_site_series(), "pm25", 3)
    df = add_rolling_features(df, "pm25", 3)
    a = df[df.sitename == "A"].sort_values("publishtime").reset_index(drop=True)
    assert a.loc[3, "pm25_roll3_mean"] == 1.0
    assert pd.isna(a.loc[2, "pm25_roll3_mean"])
    assert "pm25_roll3_std" in a.columns


from airpulse.defs.features import add_time_features


def test_add_time_features_are_cyclical_and_bounded():
    df = pd.DataFrame({"publishtime": pd.to_datetime(["2026-06-06 00:00:00", "2026-06-06 12:00:00"])})
    out = add_time_features(df)
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert c in out.columns
        assert out[c].between(-1.0, 1.0).all()
    assert abs(out.loc[0, "hour_sin"] - 0.0) < 1e-9
    assert abs(out.loc[0, "hour_cos"] - 1.0) < 1e-9
