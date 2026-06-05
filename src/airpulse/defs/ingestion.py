import os

import dagster as dg
import pandas as pd
import requests

EPA_API_URL = "https://data.moenv.gov.tw/api/v2/aqx_p_432"


@dg.asset(group_name="raw")
def raw_air_quality() -> pd.DataFrame:
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
    return pd.DataFrame(records)
