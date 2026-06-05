import os

import dagster as dg
import pandas as pd
import requests

from airpulse.defs.governance import DATA_OWNER, PUBLIC_TAGS

EPA_API_URL = "https://data.moenv.gov.tw/api/v2/aqx_p_432"


@dg.asset(
    group_name="raw",
    description="Hourly snapshot of Taiwan EPA air-quality readings (aqx_p_432).",
    kinds={"python", "api"},
    owners=[DATA_OWNER],
    tags=PUBLIC_TAGS,
    # The EPA endpoint is an external dependency prone to transient timeouts /
    # 5xx; back off and retry rather than failing the whole run.
    retry_policy=dg.RetryPolicy(
        max_retries=3,
        delay=10,  # seconds, grows exponentially: ~10s, 20s, 40s
        backoff=dg.Backoff.EXPONENTIAL,
        jitter=dg.Jitter.PLUS_MINUS,
    ),
)
def raw_air_quality(context: dg.AssetExecutionContext) -> pd.DataFrame:
    resp = requests.get(
        EPA_API_URL,
        params={"api_key": os.getenv("EPA_API_KEY"), "limit": 1000, "offset": 0},
        timeout=30,
    )
    resp.raise_for_status()
    # The MOENV v2 API returns a bare JSON list of station records; older/other
    # endpoints wrap them under a "records" key, so handle both shapes.
    payload = resp.json()
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    df = pd.DataFrame(records)
    context.add_output_metadata(
        {"row_count": len(df), "source_url": EPA_API_URL}
    )
    return df
