# airpulse Pipeline Architecture

> **Language:** English | [繁體中文](PIPELINE.zh-TW.md)

Taipei / Taiwan air quality ML pipeline built on **Dagster+ (Serverless)**. Every hour it
fetches air quality observations from the Ministry of Environment (MOENV) open-data API,
cleans and accumulates them, trains a model, and writes model metrics to PostgreSQL (Neon).
This document covers three core areas:

1. [Hourly Ingestion Job Structure](#1-hourly-ingestion-job-structure)
2. [Data Cleaning](#2-data-cleaning)
3. [Schema Validation Layer Structure](#3-schema-validation-layer-structure)
4. [ML Modeling Layer](#4-ml-modeling-layer)

---

## 1. Hourly Ingestion Job Structure

### 1.1 Job and Schedule

| Item | Details |
| --- | --- |
| Job name | `hourly_air_quality_pipeline` |
| Schedule | `ScheduleDefinition`, cron `0 * * * *` (top of every hour, aligned with EPA API update frequency) |
| Entry point | `src/airpulse/defs/pipeline_defs.py` |
| Resource | `postgres` (`PostgresResource`, connecting to Neon) |
| Runtime | Dagster+ Serverless (PEX fast deployment) |

> Both the job and schedule run **once per hour**, consistent with the data source's hourly update frequency.

### 1.2 Asset Lineage (DAG)

The pipeline consists of 5 assets with the following dependency chain:

```
raw_air_quality            (group: raw)   ← EPA API fetch
        │
        ▼
cleaned_air_quality        (group: clean) ← cleaning + type coercion
        │
        ▼
air_quality_history        (group: clean) ← upsert-accumulate into PostgreSQL, return full history
        │
        ▼
model_predictions          (group: ml)    ← read history, leak-free features + temporal split, RF predict pm2.5
        │
        ▼
model_metrics              (group: ml)    ← overall + per-site metrics (MAE/R²/relative error) and drift
```

### 1.3 Asset Responsibilities

| Asset | Kind badge | Responsibility summary |
| --- | --- | --- |
| `raw_air_quality` | api, python | GET `aqx_p_432` (limit 1000), return raw DataFrame |
| `cleaned_air_quality` | pandas | Drop missing keys, coerce numeric columns, sort by time |
| `air_quality_history` | pandas, postgres | Upsert snapshot into (widened) history table, return accumulated time series |
| `model_predictions` | sklearn | Read accumulated history, build leak-free features, temporal split, RF predict pm2.5, output overall + per-site metrics |
| `model_metrics` | postgres | Persist overall metrics (including `relative_mae`) + per-site table, flag drift via relative error |

### 1.4 Ingestion Details (`raw_air_quality`)

- Data source: `https://data.moenv.gov.tw/api/v2/aqx_p_432` (MOENV real-time air quality).
- **Response structure tolerance**: The MOENV v2 API returns a "bare JSON list" rather than a
  `{"records": [...]}` wrapper; the code handles both structures:

  ```python
  payload = resp.json()
  records = payload.get("records", []) if isinstance(payload, dict) else payload
  ```

- **External dependency retry (RetryPolicy)**: When the API experiences occasional timeouts or
  5xx errors, exponential backoff retries prevent a single transient failure from failing the
  entire run:

  ```python
  retry_policy=dg.RetryPolicy(
      max_retries=3,
      delay=10,                    # ~10s -> 20s -> 40s
      backoff=dg.Backoff.EXPONENTIAL,
      jitter=dg.Jitter.PLUS_MINUS,
  )
  ```

- Output metadata: `row_count`, `source_url`.

> **Data source characteristics (important)**: This API is a "real-time snapshot" — each call
> returns only one record per station for the current hour (approximately 84 stations, a single
> timestamp). A single fetch therefore cannot form a time series; `air_quality_history` must
> accumulate records across runs. This is the fundamental reason the pipeline is designed around
> "accumulated history + time-series forecasting."

---

## 2. Data Cleaning

Cleaning is split across two assets: **`cleaned_air_quality` (single-snapshot cleaning)** and
**`air_quality_history` (cross-run accumulation and normalization)**.

### 2.1 Single-Snapshot Cleaning (`cleaned_air_quality`)

Takes the raw DataFrame from `raw_air_quality` and applies the following steps in order:

1. **Drop missing keys**: `dropna(subset=["sitename", "publishtime"])` — rows missing a station
   name or timestamp cannot serve as valid observations and are discarded immediately.
2. **Coerce numeric columns**: Apply `pd.to_numeric(errors="coerce")` to the columns below,
   converting non-numeric values (e.g. `"--"` used by the EPA to indicate no data) to `NaN`:

   ```python
   NUMERIC_COLS = [
       "pm2.5", "pm10", "o3", "co", "so2", "no2", "aqi",
       "pm2.5_avg", "pm10_avg", "o3_8hr", "co_8hr", "so2_avg",
       "no", "nox", "wind_speed", "wind_direc", "latitude", "longitude",
   ]
   ```

3. **Parse timestamp column**: `pd.to_datetime(df["publishtime"])` (raw format e.g.
   `2026/06/05 22:00:00`).
4. **Sort**: Sort by `publishtime` to ensure chronological consistency.
5. **Output metadata**: `row_count`, `distinct_sites`, and the column-level
   `dagster/column_schema` (see §3.3).

> Design trade-off: this layer handles only "structuring and type normalization" — **rows with
> missing numeric values are not dropped** (numeric gaps become `NaN` and are retained). The
> decision of "whether a value is reasonable" is delegated to the downstream validation layer,
> keeping cleaning and validation responsibilities separate.

### 2.2 Accumulation and Normalization (`air_quality_history`)

This asset is the pipeline's **persistent state layer** and single source of truth:

1. **Column name normalization**: API column names containing dots are renamed to valid SQL
   identifiers (`pm2.5 → pm25`, `pm2.5_avg → pm25_avg`). The history table stores **all**
   raw columns (pollutants, moving averages / 8-hour values, precursors, meteorology,
   geography) as raw material for downstream feature engineering:

   ```python
   COL_MAP = {"pm2.5": "pm25", "pm2.5_avg": "pm25_avg",
              "wind_speed": "wind_speed", "wind_direc": "wind_direc",
              "latitude": "latitude", "longitude": "longitude", ...}  # 18 numeric columns total
   ```

2. **NaN → SQL NULL**: `snap.replace({np.nan: None})` to avoid type issues on write.
3. **Idempotent upsert**: Uses `(sitename, publishtime)` as the primary key with
   `INSERT ... ON CONFLICT (sitename, publishtime) DO NOTHING` —
   **re-runs never insert duplicate rows**, which is the foundation that makes safe
   "resume from failure / step retry" possible.
4. **Return full accumulated series**: `SELECT ... FROM air_quality_history` returns the
   entire history table for downstream modeling.
5. **Output metadata**: `snapshot_rows` (added this run), `total_history_rows` (cumulative
   total), `distinct_timestamps` (number of distinct timestamps accumulated so far).

#### History Table Schema (`air_quality_history`)

```sql
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
```

> When widening the table, `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` is used for idempotent
> migration — previously accumulated history is never lost.

---

## 3. Schema Validation Layer Structure

The validation layer is implemented as Dagster **Asset Checks** (`src/airpulse/defs/checks.py`),
executed automatically on every materialization and displayed as pass / fail badges in the
Dagster catalog.

### 3.1 Design Principles

- **Severity tiers**:
  - `ERROR`: Indicates "untrustworthy bad data / model." Combined with `blocking=True`,
    **a failure immediately fails the run**, prevents errors from propagating downstream,
    and triggers the Slack alert sensor.
  - `WARN`: Indicates "anomalous but not fatal" — annotates without interrupting the pipeline.
- **Validation and cleaning are separate**: Cleaning handles structuring; validation judges
  "whether values are reasonable / fresh / whether the model is drifting."

### 3.2 Validation Checks Overview

| Check | Target asset | Severity | Blocking | Rule |
| --- | --- | --- | --- | --- |
| `raw_not_empty` | `raw_air_quality` | ERROR | ✅ | Fetched row count > 0 |
| `no_missing_keys` | `cleaned_air_quality` | ERROR | ✅ | `sitename` and `publishtime` are both non-null |
| `pm25_non_negative` | `cleaned_air_quality` | ERROR | ✅ | `pm2.5` ≥ 0 (negative concentration is impossible) |
| `aqi_in_range` | `cleaned_air_quality` | WARN | ✗ | `aqi` falls within 0–500 (EPA AQI range) |
| `data_is_fresh` | `cleaned_air_quality` | WARN | ✗ | Latest reading is ≤ 3 hours old |
| `model_not_drifting` | `model_metrics` | ERROR | ✅ | Latest relative error has not degraded beyond the threshold relative to the median of the last 3 runs (see §4.4) |

Key constants: `FRESHNESS_HOURS = 3`, `AQI_MIN, AQI_MAX = 0, 500`, drift threshold 15%
(`DRIFT_THRESHOLD`, defined in `metrics.py`).

### 3.3 Column-Level Schema (Column Schema)

`cleaned_air_quality` publishes column definitions via `dagster/column_schema` metadata,
enabling the Dagster catalog to display complete column documentation:

| Column | Type | Description |
| --- | --- | --- |
| `sitename` | string | Station name |
| `county` | string | Administrative district (county/city) |
| `publishtime` | datetime | Observation timestamp |
| `pm2.5` / `pm10` / `o3` / `co` / `so2` / `no2` / `aqi` | float | Pollutant concentrations / index |
| `pm2.5_avg` / `pm10_avg` / `o3_8hr` / `co_8hr` / `so2_avg` | float | Moving averages / 8-hour values |
| `no` / `nox` / `wind_speed` / `wind_direc` | float | Precursors and meteorology |
| `latitude` / `longitude` | float | Station geographic coordinates |

### 3.4 Defensive Error Handling

- Potentially absent columns: numeric checks use `df.get("column")` to retrieve values, so
  missing columns do not raise errors.
- Cold start: `model_not_drifting` returns `passed=True` (WARN-level info) when no trained
  model exists yet (`status != "ok"`), avoiding false drift alarms.

### 3.5 Validation → Alert Closed Loop

```
Data / model anomaly
   → ERROR-level blocking asset check fails
   → Entire run is marked failed
   → run_failure_sensor (slack_on_run_failure) reads SLACK_WEBHOOK_URL
   → POST to Slack Incoming Webhook
```

`model_not_drifting` elevates the `drift_flag` — previously written silently to the DB — into
an alertable governance gate, completing the "detect → record → alert" monitoring loop.

---

## 4. ML Modeling Layer

`model_predictions` (`modeling.py`) and `model_metrics` (`metrics.py`) form the modeling and
monitoring layer. Full design rationale is in [`MODELING_PLAN.md`](./MODELING_PLAN.md); what
follows is the actual implementation.

### 4.1 Prediction Target and Data Split

- **Target**: Predict each station's pm2.5 value for the **next hour**.
- **Temporal split**: All distinct `publishtime` values are sorted; the latest 20% of
  timestamps form the test set and the earlier timestamps form the training set
  (`features.temporal_split`). All stations appear in both splits — the difference is that the
  test set contains later timestamps — so the evaluation measures true future performance with
  **no temporal leakage**.
- **Cold-start threshold**: `MIN_TIMESTAMPS = 12`; if data is insufficient, the asset returns
  `status="insufficient_data"` rather than raising an error. v1 trains on full history (data
  is still sparse; a rolling training window is planned for v2).

### 4.2 Feature Engineering (`features.py`, leak-free)

> **Strict rule**: Exogenous variables (other pollutants, meteorology) always use only their
> **lagged** versions (current-period values are unavailable at prediction time); only
> **temporal and geographic** features — known at prediction time — may use same-period values.

| Group | Features |
| --- | --- |
| Autoregressive | `pm25_lag1/2/3` |
| Rolling statistics | `pm25_roll3_mean`, `pm25_roll3_std` (computed from lags, `skipna=False`) |
| Concurrent pollutants (lag1) | `_lag1` versions of `pm10`, `o3`, `co`, `so2`, `no2`, `nox`, `no`, `aqi` |
| Averages / 8-hour (lag1) | `_lag1` versions of `pm25_avg`, `pm10_avg`, `o3_8hr`, `co_8hr`, `so2_avg` |
| Meteorology (lag1) | `wind_speed_lag1`, wind direction `wind_dir_sin/cos_lag1` (circular encoding) |
| Temporal (same period) | sin/cos circular encoding of `hour`, `dow`, `month` |
| Geographic (same period) | `latitude`, `longitude` |
| Station (same period) | `county` one-hot |

- **Core requirement + imputation**: Only the core signal (target + pm25 lags) is required to
  be non-null; sparse exogenous columns retain NaN and are imputed with the **training-set
  median** after splitting (`impute_features`, falling back to 0 for fully-missing columns).
  This prevents a single all-null EPA column from wiping out the entire batch — a bug that was
  encountered in practice.

### 4.3 Model and Evaluation Metrics

- **Model**: `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`, a single
  shared model across all 84 stations (cross-station learning).
- **Metrics** (written to `model_metrics`):
  - `mae`, `r2`
  - `relative_mae = mae / mean(actual pm2.5 during test period)` — comparable across periods
    of varying difficulty
  - `n_train`, `n_test`, `n_timestamps`, `n_features`
- **Per-site performance**: `groupby("sitename")` on the test set to compute per-station
  `mae` / `relative_mae` / `n_test`, written to the `model_metrics_by_site` table.

### 4.4 Model Drift Detection

```
relative_mae = mae / mean(測試期 pm2.5)
baseline     = median(最近 3 次的 relative_mae)
drift_flag   = (relative_mae - baseline) / baseline > 0.15   # DRIFT_THRESHOLD
```

- **Relative error** rather than raw MAE is used, avoiding misidentifying a genuinely
  difficult period as model degradation.
- The **median of the last 3 runs** is used as the baseline to smooth out single-run noise;
  a fixed baseline is not used (defining "stable" precisely is deferred to v2).
- `drift_flag` is elevated by the `model_not_drifting` check in §3.2 (ERROR + blocking) into
  an alertable governance gate.

### 4.5 ML Table Schema

```sql
-- Overall metrics (one row per run)
model_metrics(run_at, status, mae, r2, relative_mae,
              n_train, n_test, n_timestamps, drift_flag)

-- Per-site metrics (one row per station)
model_metrics_by_site(run_at, sitename, n_test, mae, relative_mae)
```

---

## Related Documents

- ML modeling design blueprint and trade-offs: [`MODELING_PLAN.md`](./MODELING_PLAN.md)
- Data governance (classification, quality gates, lineage, ownership): [`DATA_GOVERNANCE.md`](./DATA_GOVERNANCE.md)
- Code location: `src/airpulse/defs/` (`ingestion` / `cleaning` / `history` /
  `modeling` / `metrics` / `features` / `evaluation` / `checks` / `sensors` /
  `governance` / `postgres`)
