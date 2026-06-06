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
