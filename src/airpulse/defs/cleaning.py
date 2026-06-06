import dagster as dg
import pandas as pd

from airpulse.defs.governance import DATA_OWNER, PUBLIC_TAGS

# Every numeric field we ingest and want stored as a real number.
NUMERIC_COLS = [
    "pm2.5", "pm10", "o3", "co", "so2", "no2", "aqi",
    "pm2.5_avg", "pm10_avg", "o3_8hr", "co_8hr", "so2_avg",
    "no", "nox", "wind_speed", "wind_direc", "latitude", "longitude",
]
REQUIRED_COLS = ["sitename", "county", "publishtime"] + NUMERIC_COLS

COLUMN_SCHEMA = dg.TableSchema(
    columns=[
        dg.TableColumn("sitename", "string", description="Monitoring station name"),
        dg.TableColumn("county", "string", description="Administrative county"),
        dg.TableColumn("publishtime", "datetime", description="Reading timestamp"),
        *[
            dg.TableColumn(c, "float", description="Pollutant / meteorology / geo measurement")
            for c in NUMERIC_COLS
        ],
    ]
)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + type-coerce a raw EPA snapshot. Pure (no Dagster context)."""
    df = df.copy()

    # drop rows missing required identity fields
    df = df.dropna(subset=["sitename", "publishtime"])

    # coerce numeric columns; non-numeric values (e.g. "--") become NaN
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["publishtime"] = pd.to_datetime(df["publishtime"])
    df = df.sort_values("publishtime")
    return df


@dg.asset(
    group_name="clean",
    description="Validated, type-coerced air-quality readings, sorted by time.",
    kinds={"pandas"},
    owners=[DATA_OWNER],
    tags=PUBLIC_TAGS,
)
def cleaned_air_quality(
    context: dg.AssetExecutionContext, raw_air_quality: pd.DataFrame
) -> pd.DataFrame:
    df = clean_frame(raw_air_quality)
    context.add_output_metadata(
        {
            "row_count": len(df),
            "distinct_sites": int(df["sitename"].nunique()),
            "dagster/column_schema": COLUMN_SCHEMA,
        }
    )
    return df
