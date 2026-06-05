import dagster as dg
import pandas as pd

from airpulse.defs.governance import DATA_OWNER, PUBLIC_TAGS

NUMERIC_COLS = ["pm2.5", "pm10", "o3", "co", "so2", "no2", "aqi"]
REQUIRED_COLS = ["sitename", "county", "publishtime"] + NUMERIC_COLS

COLUMN_SCHEMA = dg.TableSchema(
    columns=[
        dg.TableColumn("sitename", "string", description="Monitoring station name"),
        dg.TableColumn("county", "string", description="Administrative county"),
        dg.TableColumn("publishtime", "datetime", description="Reading timestamp"),
        *[
            dg.TableColumn(c, "float", description="Pollutant / index measurement")
            for c in NUMERIC_COLS
        ],
    ]
)


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
    df = raw_air_quality.copy()

    # drop rows missing required fields
    df = df.dropna(subset=["sitename", "publishtime"])

    # coerce numeric columns; non-numeric values (e.g. "--") become NaN
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["publishtime"] = pd.to_datetime(df["publishtime"])
    df = df.sort_values("publishtime")

    context.add_output_metadata(
        {
            "row_count": len(df),
            "distinct_sites": int(df["sitename"].nunique()),
            "dagster/column_schema": COLUMN_SCHEMA,
        }
    )
    return df
