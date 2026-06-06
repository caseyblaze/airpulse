import dagster as dg
import numpy as np
import pandas as pd
from sqlalchemy import text

from airpulse.defs.governance import DATA_OWNER, PUBLIC_TAGS
from airpulse.defs.postgres import PostgresResource

# API field name -> stored column name (dots removed for valid SQL identifiers)
COL_MAP = {
    "pm2.5": "pm25", "pm10": "pm10", "o3": "o3", "co": "co",
    "so2": "so2", "no2": "no2", "aqi": "aqi",
    "pm2.5_avg": "pm25_avg", "pm10_avg": "pm10_avg", "o3_8hr": "o3_8hr",
    "co_8hr": "co_8hr", "so2_avg": "so2_avg", "no": "no", "nox": "nox",
    "wind_speed": "wind_speed", "wind_direc": "wind_direc",
    "latitude": "latitude", "longitude": "longitude",
}
NUM_COLS = list(COL_MAP.values())
ID_COLS = ["sitename", "siteid", "county", "publishtime"]
ALL_COLS = ID_COLS + NUM_COLS

DDL = """
CREATE TABLE IF NOT EXISTS air_quality_history (
    sitename     TEXT NOT NULL,
    siteid       TEXT,
    county       TEXT,
    publishtime  TIMESTAMP NOT NULL,
    pm25 FLOAT, pm10 FLOAT, o3 FLOAT, co FLOAT, so2 FLOAT, no2 FLOAT, aqi FLOAT,
    pm25_avg FLOAT, pm10_avg FLOAT, o3_8hr FLOAT, co_8hr FLOAT, so2_avg FLOAT,
    no FLOAT, nox FLOAT, wind_speed FLOAT, wind_direc FLOAT,
    latitude FLOAT, longitude FLOAT,
    PRIMARY KEY (sitename, publishtime)
)
"""

# Idempotent migration for tables created before the widening.
MIGRATIONS = [
    f"ALTER TABLE air_quality_history ADD COLUMN IF NOT EXISTS {c} {t}"
    for c, t in [
        ("siteid", "TEXT"), ("pm25_avg", "FLOAT"), ("pm10_avg", "FLOAT"),
        ("o3_8hr", "FLOAT"), ("co_8hr", "FLOAT"), ("so2_avg", "FLOAT"),
        ("no", "FLOAT"), ("nox", "FLOAT"), ("wind_speed", "FLOAT"),
        ("wind_direc", "FLOAT"), ("latitude", "FLOAT"), ("longitude", "FLOAT"),
    ]
]


@dg.asset(
    group_name="clean",
    description="Durable accumulated time series; each snapshot upserted to postgres.",
    kinds={"pandas", "postgres"},
    owners=[DATA_OWNER],
    tags=PUBLIC_TAGS,
)
def air_quality_history(
    context: dg.AssetExecutionContext,
    cleaned_air_quality: pd.DataFrame,
    postgres: PostgresResource,
) -> pd.DataFrame:
    """Append the current snapshot to a durable history table and return the
    full accumulated time series, which downstream modeling reads from."""
    engine = postgres.get_engine()

    snap = cleaned_air_quality.rename(columns=COL_MAP)
    snap = snap[[c for c in ALL_COLS if c in snap.columns]].copy()
    snap = snap.dropna(subset=["sitename", "publishtime"])
    snap["publishtime"] = pd.to_datetime(snap["publishtime"]).dt.to_pydatetime()
    snap = snap.replace({np.nan: None})  # NaN -> SQL NULL
    rows = snap.to_dict("records")
    insert_cols = [c for c in ALL_COLS if c in snap.columns]

    with engine.begin() as conn:
        conn.execute(text(DDL))
        for stmt in MIGRATIONS:
            conn.execute(text(stmt))
        if rows:
            cols = ", ".join(insert_cols)
            placeholders = ", ".join(f":{c}" for c in insert_cols)
            conn.execute(
                text(
                    f"INSERT INTO air_quality_history ({cols}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT (sitename, publishtime) DO NOTHING"
                ),
                rows,
            )
        history = pd.read_sql("SELECT * FROM air_quality_history", conn)

    context.add_output_metadata(
        {
            "snapshot_rows": len(snap),
            "total_history_rows": len(history),
            "distinct_timestamps": (
                int(history["publishtime"].nunique()) if len(history) else 0
            ),
        }
    )
    return history
