"""Data-quality asset checks — governance gates that surface as pass/fail
markers in the Dagster catalog and block downstream work on hard failures."""

from datetime import datetime, timedelta, timezone

import dagster as dg
import pandas as pd

FRESHNESS_HOURS = 3
AQI_MIN, AQI_MAX = 0, 500  # EPA AQI scale bounds


@dg.asset_check(asset="raw_air_quality", description="Ingestion returned rows.")
def raw_not_empty(raw_air_quality: pd.DataFrame) -> dg.AssetCheckResult:
    n = len(raw_air_quality)
    return dg.AssetCheckResult(
        passed=n > 0,
        severity=dg.AssetCheckSeverity.ERROR,
        metadata={"row_count": n},
    )


@dg.asset_check(
    asset="cleaned_air_quality",
    description="Required identity fields (sitename, publishtime) are present.",
)
def no_missing_keys(cleaned_air_quality: pd.DataFrame) -> dg.AssetCheckResult:
    missing = int(
        cleaned_air_quality[["sitename", "publishtime"]].isna().any(axis=1).sum()
    )
    return dg.AssetCheckResult(
        passed=missing == 0,
        severity=dg.AssetCheckSeverity.ERROR,
        metadata={"rows_missing_keys": missing},
    )


@dg.asset_check(
    asset="cleaned_air_quality",
    description="pm2.5 is non-negative (negative concentration is impossible).",
)
def pm25_non_negative(cleaned_air_quality: pd.DataFrame) -> dg.AssetCheckResult:
    col = cleaned_air_quality.get("pm2.5")
    bad = 0 if col is None else int((col < 0).sum())
    return dg.AssetCheckResult(
        passed=bad == 0,
        severity=dg.AssetCheckSeverity.ERROR,
        metadata={"negative_pm25_rows": bad},
    )


@dg.asset_check(
    asset="cleaned_air_quality",
    description=f"AQI values fall within the valid {AQI_MIN}-{AQI_MAX} scale.",
)
def aqi_in_range(cleaned_air_quality: pd.DataFrame) -> dg.AssetCheckResult:
    col = cleaned_air_quality.get("aqi")
    if col is None:
        out_of_range = 0
    else:
        valid = col.dropna()
        out_of_range = int(((valid < AQI_MIN) | (valid > AQI_MAX)).sum())
    return dg.AssetCheckResult(
        passed=out_of_range == 0,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"out_of_range_rows": out_of_range},
    )


@dg.asset_check(
    asset="cleaned_air_quality",
    description=f"Latest reading is within the last {FRESHNESS_HOURS}h.",
)
def data_is_fresh(cleaned_air_quality: pd.DataFrame) -> dg.AssetCheckResult:
    times = pd.to_datetime(cleaned_air_quality["publishtime"], errors="coerce")
    if times.notna().any():
        latest = times.max()
        if latest.tzinfo is None:
            latest = latest.tz_localize("Asia/Taipei")
        age_hours = (
            datetime.now(timezone.utc) - latest.tz_convert("UTC")
        ) / timedelta(hours=1)
    else:
        latest, age_hours = None, None
    return dg.AssetCheckResult(
        passed=age_hours is not None and age_hours <= FRESHNESS_HOURS,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={
            "latest_reading": str(latest),
            "age_hours": round(age_hours, 2) if age_hours is not None else "n/a",
        },
    )
