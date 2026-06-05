import dagster as dg
import numpy as np
import pandas as pd
from sqlalchemy import text

from airpulse.defs.governance import DATA_OWNER, PUBLIC_TAGS
from airpulse.defs.postgres import PostgresResource

# pm2.5 -> pm25 so every stored column is a valid SQL identifier
COL_MAP = {
    "pm2.5": "pm25",
    "pm10": "pm10",
    "o3": "o3",
    "co": "co",
    "so2": "so2",
    "no2": "no2",
    "aqi": "aqi",
}
NUM_COLS = list(COL_MAP.values())
ALL_COLS = ["sitename", "county", "publishtime"] + NUM_COLS

DDL = """
CREATE TABLE IF NOT EXISTS air_quality_history (
    sitename     TEXT NOT NULL,
    county       TEXT,
    publishtime  TIMESTAMP NOT NULL,
    pm25 FLOAT, pm10 FLOAT, o3 FLOAT, co FLOAT, so2 FLOAT, no2 FLOAT, aqi FLOAT,
    PRIMARY KEY (sitename, publishtime)
)
"""


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

    with engine.begin() as conn:
        conn.execute(text(DDL))
        if rows:
            cols = ", ".join(ALL_COLS)
            placeholders = ", ".join(f":{c}" for c in ALL_COLS)
            conn.execute(
                text(
                    f"INSERT INTO air_quality_history ({cols}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT (sitename, publishtime) DO NOTHING"
                ),
                rows,
            )
        history = pd.read_sql(
            "SELECT sitename, county, publishtime, "
            "pm25, pm10, o3, co, so2, no2, aqi FROM air_quality_history",
            conn,
        )

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
