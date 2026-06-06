# ML Time-Series Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the accidental spatial train/test split with a true temporal split, expand the feature set to all available pollutant/meteorology/geo/time signals (leakage-safe via lagging), and add per-site metrics plus a relative-error / median-of-3 drift detector.

**Architecture:** Pure, unit-testable helper modules (`features.py`, `evaluation.py`) hold all feature engineering, the temporal split, and metric math. The Dagster assets (`modeling.py`, `metrics.py`) orchestrate them and do the DB I/O. The history table is widened to store every raw field so lag features can be built; `model_metrics` gains `relative_mae`; a new `model_metrics_by_site` table stores per-station results.

**Tech Stack:** Python 3.10+, Dagster (`dg`), pandas, scikit-learn (RandomForestRegressor), SQLAlchemy + PostgreSQL (Neon), uv, pytest.

---

## Design decisions (locked)

- **Split:** global timestamp cutoff — last 20% of distinct `publishtime` is the test set; train is strictly earlier. Gate: `MIN_TIMESTAMPS = 12`.
- **Training:** full accumulated history (rolling window deferred to v2).
- **Leakage rule:** exogenous pollutants + meteorology enter the model **only as `_lag1`** (and target as `_lag1..3` + rolling). Only **time** and **geo** features use the same-hour value.
- **county:** one-hot (~22 cols). `siteid` target-encoding deferred to v2.
- **Drift:** `relative_mae = mae / mean(test pm2.5)`; baseline = median of last 3 `relative_mae`; flag if `> 15%` worse. No fixed baseline.
- **Per-site metrics:** new table `model_metrics_by_site`.

## File Structure

- `src/airpulse/defs/features.py` — **new.** Pure feature engineering + temporal split.
- `src/airpulse/defs/evaluation.py` — **new.** Pure metric helpers (relative MAE, per-site, drift).
- `src/airpulse/defs/cleaning.py` — modify. Factor pure `clean_frame()`, widen `NUMERIC_COLS`.
- `src/airpulse/defs/history.py` — modify. Widen `COL_MAP`/`ALL_COLS`/DDL + migration.
- `src/airpulse/defs/modeling.py` — rewrite `model_predictions` to use the new helpers.
- `src/airpulse/defs/metrics.py` — rewrite `model_metrics` for `relative_mae`, per-site table, new drift.
- `tests/test_features.py` — **new.** Unit tests for features + split.
- `tests/test_evaluation.py` — **new.** Unit tests for metric helpers.
- `pyproject.toml` — modify. Add `pytest` dev dependency.

---

### Task 1: Add pytest dev dependency

**Files:**
- Modify: `pyproject.toml` (dependency-groups.dev)

- [ ] **Step 1: Add pytest**

Run: `uv add --dev pytest`
Expected: `pytest` appears under `[dependency-groups] dev` in `pyproject.toml` and `uv.lock` updates.

- [ ] **Step 2: Verify pytest runs**

Run: `uv run pytest --version`
Expected: prints `pytest 8.x.x` (no ModuleNotFoundError).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pytest dev dependency"
```

---

### Task 2: Factor pure clean_frame() and widen numeric columns

The cleaning asset must coerce every numeric field we will store (meteorology, geo, averages), not just the original seven. Extracting `clean_frame()` makes the logic unit-testable.

**Files:**
- Modify: `src/airpulse/defs/cleaning.py`
- Test: `tests/test_features.py` (shared test module; created here)

- [ ] **Step 1: Write the failing test**

Create `tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_clean_frame_coerces_all_numeric_and_drops_missing_keys -v`
Expected: FAIL with `ImportError: cannot import name 'clean_frame'`.

- [ ] **Step 3: Rewrite cleaning.py**

Replace the entire contents of `src/airpulse/defs/cleaning.py` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_clean_frame_coerces_all_numeric_and_drops_missing_keys -v`
Expected: PASS.

- [ ] **Step 5: Validate Dagster still loads**

Run: `uv run dg check defs`
Expected: `All definitions loaded successfully.`

- [ ] **Step 6: Commit**

```bash
git add src/airpulse/defs/cleaning.py tests/test_features.py
git commit -m "feat(cleaning): factor pure clean_frame and widen numeric columns"
```

---

### Task 3: Widen the air_quality_history table

Store every raw field (with name normalization, e.g. `pm2.5 -> pm25`) so lag features can be built later. Use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` so existing rows migrate without loss.

**Files:**
- Modify: `src/airpulse/defs/history.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_history_stores_widened_normalized_columns -v`
Expected: FAIL with `KeyError: 'wind_speed'` (or assertion error on `siteid`).

- [ ] **Step 3: Rewrite history.py**

Replace the entire contents of `src/airpulse/defs/history.py` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_history_stores_widened_normalized_columns -v`
Expected: PASS.

- [ ] **Step 5: Validate Dagster still loads**

Run: `uv run dg check defs`
Expected: `All definitions loaded successfully.`

- [ ] **Step 6: Commit**

```bash
git add src/airpulse/defs/history.py tests/test_features.py
git commit -m "feat(history): widen table to store all raw fields with migration"
```

---

### Task 4: Lag features

**Files:**
- Create: `src/airpulse/defs/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
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
    # first row has no lag1/lag2
    assert pd.isna(a.loc[0, "pm25_lag1"])
    assert pd.isna(a.loc[1, "pm25_lag2"])
    # third row: lag1 is previous value, lag2 is two back
    assert a.loc[2, "pm25_lag1"] == 1.0
    assert a.loc[2, "pm25_lag2"] == 0.0
    # site B never leaks into site A: A's lags only use A's values
    assert a["pm25_lag1"].dropna().tolist() == [0.0, 1.0, 2.0, 3.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_add_lag_features_shifts_within_site -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'airpulse.defs.features'`.

- [ ] **Step 3: Create features.py with add_lag_features**

Create `src/airpulse/defs/features.py`:

```python
import numpy as np
import pandas as pd

TARGET = "pm25"
TARGET_LAGS = 3

# Exogenous numeric columns. Leakage rule: these may only enter the model as
# their lag-1 value, never the same-hour value.
EXO_COLS = [
    "pm10", "o3", "co", "so2", "no2", "nox", "no", "aqi",
    "pm25_avg", "pm10_avg", "o3_8hr", "co_8hr", "so2_avg", "wind_speed",
]
# Same-hour features that are known at prediction time (no leakage).
GEO_COLS = ["latitude", "longitude"]


def add_lag_features(df: pd.DataFrame, col: str, lags: int) -> pd.DataFrame:
    """Add `{col}_lag1..lags`, shifted within each station, time-ordered."""
    df = df.sort_values(["sitename", "publishtime"]).copy()
    for i in range(1, lags + 1):
        df[f"{col}_lag{i}"] = df.groupby("sitename")[col].shift(i)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_add_lag_features_shifts_within_site -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/airpulse/defs/features.py tests/test_features.py
git commit -m "feat(features): add per-site lag feature builder"
```

---

### Task 5: Rolling features

**Files:**
- Modify: `src/airpulse/defs/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
from airpulse.defs.features import add_rolling_features


def test_add_rolling_features_uses_only_lags():
    df = add_lag_features(_two_site_series(), "pm25", 3)
    df = add_rolling_features(df, "pm25", 3)
    a = df[df.sitename == "A"].sort_values("publishtime").reset_index(drop=True)
    # row index 3 has lag1=2, lag2=1, lag3=0 -> mean = 1.0
    assert a.loc[3, "pm25_roll3_mean"] == 1.0
    # earlier rows lack full lags -> NaN (will be dropped at matrix build)
    assert pd.isna(a.loc[2, "pm25_roll3_mean"])
    assert "pm25_roll3_std" in a.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_add_rolling_features_uses_only_lags -v`
Expected: FAIL with `ImportError: cannot import name 'add_rolling_features'`.

- [ ] **Step 3: Add add_rolling_features**

Append to `src/airpulse/defs/features.py`:

```python
def add_rolling_features(df: pd.DataFrame, col: str = "pm25", window: int = 3) -> pd.DataFrame:
    """Mean/std over the lagged values only (leakage-safe). Requires
    `{col}_lag1..window` to already exist."""
    df = df.copy()
    lag_cols = [f"{col}_lag{i}" for i in range(1, window + 1)]
    df[f"{col}_roll{window}_mean"] = df[lag_cols].mean(axis=1)
    df[f"{col}_roll{window}_std"] = df[lag_cols].std(axis=1)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_add_rolling_features_uses_only_lags -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/airpulse/defs/features.py tests/test_features.py
git commit -m "feat(features): add leakage-safe rolling mean/std"
```

---

### Task 6: Time features (cyclical)

**Files:**
- Modify: `src/airpulse/defs/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
from airpulse.defs.features import add_time_features


def test_add_time_features_are_cyclical_and_bounded():
    df = pd.DataFrame({"publishtime": pd.to_datetime(["2026-06-06 00:00:00", "2026-06-06 12:00:00"])})
    out = add_time_features(df)
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert c in out.columns
        assert out[c].between(-1.0, 1.0).all()
    # hour 0 -> sin 0, cos 1
    assert abs(out.loc[0, "hour_sin"] - 0.0) < 1e-9
    assert abs(out.loc[0, "hour_cos"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_add_time_features_are_cyclical_and_bounded -v`
Expected: FAIL with `ImportError: cannot import name 'add_time_features'`.

- [ ] **Step 3: Add add_time_features**

Append to `src/airpulse/defs/features.py`:

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour/day-of-week/month encodings from publishtime (same-hour,
    known at prediction time)."""
    df = df.copy()
    t = pd.to_datetime(df["publishtime"])
    for name, values, period in [
        ("hour", t.dt.hour, 24),
        ("dow", t.dt.dayofweek, 7),
        ("month", t.dt.month, 12),
    ]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_add_time_features_are_cyclical_and_bounded -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/airpulse/defs/features.py tests/test_features.py
git commit -m "feat(features): add cyclical time features"
```

---

### Task 7: Wind direction circular encoding (on lag)

**Files:**
- Modify: `src/airpulse/defs/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
from airpulse.defs.features import add_wind_direction_features


def test_wind_direction_encoded_from_lag():
    # wind_direc_lag1 must already exist; 90 degrees -> sin 1, cos 0
    df = pd.DataFrame({"wind_direc_lag1": [0.0, 90.0]})
    out = add_wind_direction_features(df)
    assert abs(out.loc[0, "wind_dir_sin_lag1"] - 0.0) < 1e-9
    assert abs(out.loc[0, "wind_dir_cos_lag1"] - 1.0) < 1e-9
    assert abs(out.loc[1, "wind_dir_sin_lag1"] - 1.0) < 1e-9
    assert abs(out.loc[1, "wind_dir_cos_lag1"] - 0.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_wind_direction_encoded_from_lag -v`
Expected: FAIL with `ImportError: cannot import name 'add_wind_direction_features'`.

- [ ] **Step 3: Add add_wind_direction_features**

Append to `src/airpulse/defs/features.py`:

```python
def add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the lagged wind direction (degrees) as sin/cos. Requires
    `wind_direc_lag1` to already exist."""
    df = df.copy()
    rad = np.deg2rad(df["wind_direc_lag1"])
    df["wind_dir_sin_lag1"] = np.sin(rad)
    df["wind_dir_cos_lag1"] = np.cos(rad)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_wind_direction_encoded_from_lag -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/airpulse/defs/features.py tests/test_features.py
git commit -m "feat(features): add circular wind-direction encoding"
```

---

### Task 8: County one-hot

**Files:**
- Modify: `src/airpulse/defs/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
from airpulse.defs.features import add_county_onehot


def test_county_onehot_returns_columns():
    df = pd.DataFrame({"county": ["X", "Y", "X"]})
    out, cols = add_county_onehot(df)
    assert set(cols) == {"county_X", "county_Y"}
    assert out["county_X"].tolist() == [1, 1, 0] or out["county_X"].tolist() == [True, True, False]
    assert out.loc[1, "county_Y"] in (1, True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_county_onehot_returns_columns -v`
Expected: FAIL with `ImportError: cannot import name 'add_county_onehot'`.

- [ ] **Step 3: Add add_county_onehot**

Append to `src/airpulse/defs/features.py`:

```python
def add_county_onehot(df: pd.DataFrame):
    """One-hot encode county (~22 levels). Returns (df, new_column_names)."""
    df = df.copy()
    dummies = pd.get_dummies(df["county"], prefix="county")
    df = pd.concat([df, dummies], axis=1)
    return df, list(dummies.columns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_county_onehot_returns_columns -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/airpulse/defs/features.py tests/test_features.py
git commit -m "feat(features): add county one-hot encoding"
```

---

### Task 9: Temporal split

**Files:**
- Modify: `src/airpulse/defs/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
from airpulse.defs.features import temporal_split


def test_temporal_split_holds_out_latest_times_without_leakage():
    times = pd.date_range("2026-06-01", periods=10, freq="h")
    rows = [{"sitename": s, "publishtime": t, "pm25": 1.0} for s in ["A", "B"] for t in times]
    df = pd.DataFrame(rows)
    train, test = temporal_split(df, test_frac=0.2)
    # last 20% of 10 distinct timestamps -> 2 timestamps in test
    assert test["publishtime"].nunique() == 2
    assert train["publishtime"].nunique() == 8
    # no leakage: every train timestamp is strictly before every test timestamp
    assert train["publishtime"].max() < test["publishtime"].min()
    # both stations appear in both splits
    assert set(train["sitename"]) == {"A", "B"}
    assert set(test["sitename"]) == {"A", "B"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_temporal_split_holds_out_latest_times_without_leakage -v`
Expected: FAIL with `ImportError: cannot import name 'temporal_split'`.

- [ ] **Step 3: Add temporal_split**

Append to `src/airpulse/defs/features.py`:

```python
def temporal_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Global timestamp cutoff: the latest `test_frac` of distinct timestamps
    is the test set; train is strictly earlier. Returns (train_df, test_df)."""
    times = np.sort(df["publishtime"].unique())
    n_test = max(1, int(round(len(times) * test_frac)))
    cutoff = times[-n_test]
    train = df[df["publishtime"] < cutoff]
    test = df[df["publishtime"] >= cutoff]
    return train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_temporal_split_holds_out_latest_times_without_leakage -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/airpulse/defs/features.py tests/test_features.py
git commit -m "feat(features): add global temporal train/test split"
```

---

### Task 10: build_feature_matrix orchestration (with leakage guard)

**Files:**
- Modify: `src/airpulse/defs/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
from airpulse.defs.features import build_feature_matrix, EXO_COLS, TARGET


def _wide_history(n_times=8):
    times = pd.date_range("2026-06-01", periods=n_times, freq="h")
    rows = []
    for si, s in enumerate(["A", "B"]):
        for i, t in enumerate(times):
            row = {
                "sitename": s, "county": "X", "publishtime": t,
                "pm25": float(i + si), "latitude": 25.0, "longitude": 121.0,
                "wind_direc": float((i * 30) % 360),
            }
            for c in EXO_COLS:
                row[c] = float(i + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_feature_matrix_has_no_same_hour_exogenous_leakage():
    feat, feature_cols = build_feature_matrix(_wide_history())

    # target itself is never a feature
    assert TARGET not in feature_cols
    # no same-hour exogenous column (only their _lag1 variants are allowed)
    for c in EXO_COLS:
        assert c not in feature_cols
    # raw same-hour wind direction is excluded; only encoded lag is present
    assert "wind_direc" not in feature_cols
    assert "wind_dir_sin_lag1" in feature_cols
    # expected feature families present
    assert "pm25_lag1" in feature_cols and "pm25_lag3" in feature_cols
    assert "pm25_roll3_mean" in feature_cols
    assert "hour_sin" in feature_cols
    assert "latitude" in feature_cols
    assert any(c.startswith("county_") for c in feature_cols)
    # no NaNs remain in the returned feature matrix
    assert feat[feature_cols].isna().sum().sum() == 0
    assert len(feat) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py::test_build_feature_matrix_has_no_same_hour_exogenous_leakage -v`
Expected: FAIL with `ImportError: cannot import name 'build_feature_matrix'`.

- [ ] **Step 3: Add build_feature_matrix**

Append to `src/airpulse/defs/features.py`:

```python
def build_feature_matrix(history: pd.DataFrame):
    """Assemble the leakage-safe feature matrix from accumulated history.

    Returns (feat_df, feature_cols). Exogenous pollutants/meteorology enter
    only as lag-1; only time and geo features use the same-hour value. Rows
    without complete features (cold start) are dropped."""
    df = history.copy()
    df["publishtime"] = pd.to_datetime(df["publishtime"])
    df = df.sort_values(["sitename", "publishtime"])

    df = add_lag_features(df, TARGET, TARGET_LAGS)
    for col in EXO_COLS:
        if col in df.columns:
            df = add_lag_features(df, col, 1)
    if "wind_direc" in df.columns:
        df = add_lag_features(df, "wind_direc", 1)
        df = add_wind_direction_features(df)
    df = add_rolling_features(df, TARGET, 3)
    df = add_time_features(df)
    df, county_cols = add_county_onehot(df)

    feature_cols = (
        [f"{TARGET}_lag{i}" for i in range(1, TARGET_LAGS + 1)]
        + [f"{TARGET}_roll3_mean", f"{TARGET}_roll3_std"]
        + [f"{c}_lag1" for c in EXO_COLS if c in history.columns]
        + (["wind_dir_sin_lag1", "wind_dir_cos_lag1"] if "wind_direc" in history.columns else [])
        + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        + [c for c in GEO_COLS if c in df.columns]
        + county_cols
    )

    df = df.dropna(subset=[TARGET, *feature_cols])
    return df, feature_cols
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py::test_build_feature_matrix_has_no_same_hour_exogenous_leakage -v`
Expected: PASS.

- [ ] **Step 5: Run the whole features test file**

Run: `uv run pytest tests/test_features.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/airpulse/defs/features.py tests/test_features.py
git commit -m "feat(features): assemble leakage-safe feature matrix"
```

---

### Task 11: Evaluation helpers (relative MAE, per-site, drift)

**Files:**
- Create: `src/airpulse/defs/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation.py`:

```python
import pandas as pd

from airpulse.defs.evaluation import relative_mae, per_site_metrics, compute_drift


def test_relative_mae_divides_by_mean_actual():
    # mae = 1.0, mean(y_true) = 10 -> 0.1
    assert abs(relative_mae([10, 10], [9, 11]) - 0.1) < 1e-9
    # zero mean -> None (avoid div by zero)
    assert relative_mae([0, 0], [1, -1]) is None


def test_per_site_metrics_groups_by_site():
    sites = ["A", "A", "B"]
    y_true = [10.0, 10.0, 20.0]
    y_pred = [9.0, 11.0, 18.0]
    out = {r["sitename"]: r for r in per_site_metrics(sites, y_true, y_pred)}
    assert out["A"]["n_test"] == 2
    assert abs(out["A"]["mae"] - 1.0) < 1e-9
    assert abs(out["A"]["relative_mae"] - 0.1) < 1e-9
    assert out["B"]["n_test"] == 1


def test_compute_drift_uses_median_of_recent():
    # baseline = median([0.10, 0.10, 0.20]) = 0.10; current 0.12 -> +20% > 15% -> drift
    assert compute_drift(0.12, [0.10, 0.10, 0.20], threshold=0.15) is True
    # current 0.11 -> +10% < 15% -> no drift
    assert compute_drift(0.11, [0.10, 0.10, 0.20], threshold=0.15) is False
    # no history -> no drift
    assert compute_drift(0.50, [], threshold=0.15) is False
    # None current -> no drift
    assert compute_drift(None, [0.10], threshold=0.15) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'airpulse.defs.evaluation'`.

- [ ] **Step 3: Create evaluation.py**

Create `src/airpulse/defs/evaluation.py`:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def relative_mae(y_true, y_pred):
    """MAE normalized by the mean of actuals; comparable across test windows
    of differing difficulty. None if the mean is zero."""
    mae = mean_absolute_error(y_true, y_pred)
    denom = float(np.mean(y_true))
    return float(mae / denom) if denom else None


def per_site_metrics(sitenames, y_true, y_pred):
    """Per-station MAE / relative MAE / test count. Returns a list of dicts."""
    df = pd.DataFrame(
        {"sitename": list(sitenames), "y_true": list(y_true), "y_pred": list(y_pred)}
    )
    out = []
    for site, g in df.groupby("sitename"):
        mae = float(mean_absolute_error(g["y_true"], g["y_pred"]))
        denom = float(g["y_true"].mean())
        out.append(
            {
                "sitename": site,
                "n_test": int(len(g)),
                "mae": mae,
                "relative_mae": (mae / denom) if denom else None,
            }
        )
    return out


def compute_drift(current_relative_mae, recent_relative_maes, threshold=0.15):
    """Drift if current relative MAE exceeds the median of recent runs by more
    than `threshold`. Safe (False) when data is missing."""
    vals = [v for v in recent_relative_maes if v is not None]
    if current_relative_mae is None or not vals:
        return False
    baseline = float(np.median(vals))
    if baseline == 0:
        return False
    return (current_relative_mae - baseline) / baseline > threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/airpulse/defs/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): relative MAE, per-site metrics, median drift"
```

---

### Task 12: Rewrite model_predictions to use temporal split + full features

This asset reads the DB, so it is validated via `dg check defs` and a real run rather than a local unit test (no DB/SSL locally). The pure logic it depends on is already covered by Tasks 4–11.

**Files:**
- Modify: `src/airpulse/defs/modeling.py`

- [ ] **Step 1: Rewrite modeling.py**

Replace the entire contents of `src/airpulse/defs/modeling.py` with:

```python
import dagster as dg
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from airpulse.defs.evaluation import per_site_metrics, relative_mae
from airpulse.defs.features import build_feature_matrix, temporal_split
from airpulse.defs.governance import DATA_OWNER, PUBLIC_TAGS
from airpulse.defs.postgres import PostgresResource

# Need enough distinct hourly timestamps to carve a temporal holdout.
MIN_TIMESTAMPS = 12
TEST_FRAC = 0.2


def _insufficient(context, n_timestamps, reason):
    result = {
        "status": "insufficient_data",
        "reason": reason,
        "n_timestamps": int(n_timestamps),
        "mae": None,
        "r2": None,
        "relative_mae": None,
        "n_train": 0,
        "n_test": 0,
        "by_site": [],
    }
    context.add_output_metadata(
        {k: (v if v is not None else "null") for k, v in result.items() if k != "by_site"}
    )
    return result


@dg.asset(
    group_name="ml",
    deps=["air_quality_history"],
    description="Random-forest pm2.5 forecast with leakage-safe features and a temporal split.",
    kinds={"sklearn"},
    owners=[DATA_OWNER],
    tags=PUBLIC_TAGS,
)
def model_predictions(
    context: dg.AssetExecutionContext,
    postgres: PostgresResource,
) -> dict:
    """Forecast next-hour pm2.5 per station, trained on accumulated history.

    Reads the durable history table directly (serverless IO is ephemeral).
    Returns status='insufficient_data' until enough hourly snapshots exist."""
    engine = postgres.get_engine()
    history = pd.read_sql("SELECT * FROM air_quality_history", engine)

    n_timestamps = int(history["publishtime"].nunique()) if len(history) else 0
    if n_timestamps < MIN_TIMESTAMPS:
        return _insufficient(context, n_timestamps, "not_enough_timestamps")

    feat, feature_cols = build_feature_matrix(history)
    train, test = temporal_split(feat, TEST_FRAC)
    if len(train) == 0 or len(test) == 0:
        return _insufficient(context, n_timestamps, "empty_split")

    X_train, y_train = train[feature_cols], train["pm25"]
    X_test, y_test = test[feature_cols], test["pm25"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {
        "status": "ok",
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "relative_mae": relative_mae(y_test, y_pred),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_timestamps": n_timestamps,
        "n_features": len(feature_cols),
        "by_site": per_site_metrics(test["sitename"], y_test, y_pred),
    }
    context.add_output_metadata(
        {k: v for k, v in result.items() if k != "by_site"}
    )
    return result
```

- [ ] **Step 2: Validate Dagster loads**

Run: `uv run dg check defs`
Expected: `All definitions loaded successfully.`

- [ ] **Step 3: Commit**

```bash
git add src/airpulse/defs/modeling.py
git commit -m "feat(modeling): temporal split, full feature set, per-site metrics"
```

---

### Task 13: Rewrite model_metrics for relative MAE, per-site table, median drift

**Files:**
- Modify: `src/airpulse/defs/metrics.py`

- [ ] **Step 1: Rewrite metrics.py**

Replace the entire contents of `src/airpulse/defs/metrics.py` with:

```python
from datetime import datetime, timezone

import dagster as dg
from sqlalchemy import text

from airpulse.defs.evaluation import compute_drift
from airpulse.defs.governance import DATA_OWNER, PUBLIC_TAGS
from airpulse.defs.postgres import PostgresResource

DRIFT_THRESHOLD = 0.15  # flag if relative MAE worsens >15% vs median of last 3

DDL_METRICS = """
CREATE TABLE IF NOT EXISTS model_metrics (
    id           SERIAL PRIMARY KEY,
    run_at       TIMESTAMPTZ NOT NULL,
    status       TEXT,
    mae          FLOAT,
    r2           FLOAT,
    relative_mae FLOAT,
    n_train      INT,
    n_test       INT,
    n_timestamps INT,
    drift_flag   BOOLEAN
)
"""
# Migration for tables created before relative_mae existed.
MIGRATION_METRICS = (
    "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS relative_mae FLOAT"
)
DDL_BY_SITE = """
CREATE TABLE IF NOT EXISTS model_metrics_by_site (
    id           SERIAL PRIMARY KEY,
    run_at       TIMESTAMPTZ NOT NULL,
    sitename     TEXT NOT NULL,
    n_test       INT,
    mae          FLOAT,
    relative_mae FLOAT
)
"""


@dg.asset(
    group_name="ml",
    description="Persist overall + per-site metrics and flag relative-MAE drift.",
    kinds={"postgres"},
    owners=[DATA_OWNER],
    tags=PUBLIC_TAGS,
)
def model_metrics(
    context: dg.AssetExecutionContext,
    model_predictions: dict,
    postgres: PostgresResource,
) -> None:
    engine = postgres.get_engine()
    run_at = datetime.now(timezone.utc)
    rel = model_predictions.get("relative_mae")

    with engine.begin() as conn:
        conn.execute(text(DDL_METRICS))
        conn.execute(text(MIGRATION_METRICS))
        conn.execute(text(DDL_BY_SITE))

        # drift vs median of the last 3 trained relative MAEs
        recent = [
            r[0]
            for r in conn.execute(
                text(
                    "SELECT relative_mae FROM model_metrics "
                    "WHERE relative_mae IS NOT NULL ORDER BY run_at DESC LIMIT 3"
                )
            ).fetchall()
        ]
        drift_flag = compute_drift(rel, recent, DRIFT_THRESHOLD)

        conn.execute(
            text(
                """
                INSERT INTO model_metrics
                    (run_at, status, mae, r2, relative_mae,
                     n_train, n_test, n_timestamps, drift_flag)
                VALUES
                    (:run_at, :status, :mae, :r2, :relative_mae,
                     :n_train, :n_test, :n_timestamps, :drift_flag)
                """
            ),
            {
                "run_at": run_at,
                "status": model_predictions.get("status", "unknown"),
                "mae": model_predictions.get("mae"),
                "r2": model_predictions.get("r2"),
                "relative_mae": rel,
                "n_train": model_predictions.get("n_train", 0),
                "n_test": model_predictions.get("n_test", 0),
                "n_timestamps": model_predictions.get("n_timestamps", 0),
                "drift_flag": drift_flag,
            },
        )

        by_site = model_predictions.get("by_site", [])
        if by_site:
            conn.execute(
                text(
                    """
                    INSERT INTO model_metrics_by_site
                        (run_at, sitename, n_test, mae, relative_mae)
                    VALUES (:run_at, :sitename, :n_test, :mae, :relative_mae)
                    """
                ),
                [{"run_at": run_at, **row} for row in by_site],
            )

    context.add_output_metadata(
        {
            "status": model_predictions.get("status", "unknown"),
            "relative_mae": rel if rel is not None else "null",
            "drift_flag": drift_flag,
            "sites_scored": len(by_site),
        }
    )
```

- [ ] **Step 2: Validate Dagster loads**

Run: `uv run dg check defs`
Expected: `All definitions loaded successfully.`

- [ ] **Step 3: Commit**

```bash
git add src/airpulse/defs/metrics.py
git commit -m "feat(metrics): relative MAE, per-site table, median-of-3 drift"
```

---

### Task 14: Full validation, deploy, and verify

**Files:** none (verification + deploy)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests in `tests/test_features.py` and `tests/test_evaluation.py` PASS.

- [ ] **Step 2: Final defs check and definition inventory**

Run: `uv run dg check defs && uv run dg list defs`
Expected: `All definitions loaded successfully.` and the asset list still shows
`raw_air_quality, cleaned_air_quality, air_quality_history, model_predictions, model_metrics` plus all asset checks.

- [ ] **Step 3: Push and watch CI**

```bash
git push origin main
RUN_ID=$(gh run list --repo caseyblaze/airpulse --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo caseyblaze/airpulse --exit-status --interval 15
```
Expected: workflow completes with `Deploy to Dagster Cloud ✓` and exit 0.

- [ ] **Step 4: Manual verification in Dagster Cloud**

Materialize `hourly_air_quality_pipeline` once. Expected:
- All 5 assets succeed; asset checks run.
- Until ~12 distinct hourly timestamps have accumulated, `model_predictions`
  reports `status=insufficient_data` (this is expected, not a failure).
- After enough history: `model_metrics` row has a non-null `mae`, `r2`,
  `relative_mae`; `model_metrics_by_site` gains one row per station.

- [ ] **Step 5: Update memory**

Update `/Users/kc/.claude/projects/-Users-kc-get-the-job/memory/airpulse-known-bugs.md`
to record: temporal split replaces the old spatial split; `MIN_TIMESTAMPS=12`;
full leakage-safe feature set; `relative_mae` + median-of-3 drift; new
`model_metrics_by_site` table; widened `air_quality_history` schema.

---

## Self-Review

**Spec coverage (vs MODELING_PLAN.md):**
- §1 temporal split → Tasks 9, 12 ✓ (incl. `MIN_TIMESTAMPS=12`)
- §2 feature set + leakage rule → Tasks 4–10 ✓ (lag/roll/time/wind/county/geo; leakage guard test in Task 10)
- §3 full-history training, RF → Task 12 ✓
- §4 metrics incl. relative_mae + per-site → Tasks 11, 12, 13 ✓
- §5 drift: relative + median of 3, no fixed baseline → Tasks 11, 13 ✓
- §6 schema: history widening, model_metrics.relative_mae, model_metrics_by_site → Tasks 3, 13 ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test step shows full test bodies and exact commands. ✓

**Type/name consistency:** `clean_frame`, `NUMERIC_COLS`, `COL_MAP`, `ALL_COLS`, `add_lag_features`, `add_rolling_features`, `add_time_features`, `add_wind_direction_features`, `add_county_onehot`, `temporal_split`, `build_feature_matrix`, `EXO_COLS`, `TARGET`, `relative_mae`, `per_site_metrics`, `compute_drift`, `MIN_TIMESTAMPS` are used identically across defining and consuming tasks. The `model_predictions` dict keys (`status`, `mae`, `r2`, `relative_mae`, `n_train`, `n_test`, `n_timestamps`, `by_site`) match exactly what `model_metrics` reads. ✓

**Note:** `checks.py::model_not_drifting` reads the latest `model_metrics.drift_flag` and needs no change — the drift definition changed but the stored column name is unchanged.
