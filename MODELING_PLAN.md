# airpulse ML Modeling Design (v1, implemented)

> **Language:** English | [繁體中文](MODELING_PLAN.zh-TW.md)

> Status: **Implemented and deployed.** This document records the v1 modeling design and its
> rationale; the v2 backlog at the end is still future work.
> Prediction target: pm2.5 **one hour ahead** for each monitoring station.
> Code: `src/airpulse/defs/features.py`, `evaluation.py`, `modeling.py`, `metrics.py`,
> `history.py`. Unit tests: `tests/test_features.py`, `tests/test_evaluation.py`.

## 0. Design Goals and Constraints

- **Goal**: Produce a one-hour-ahead pm2.5 forecast for each station and be able to evaluate its **future** performance.
- **Constraint**: The data source is a real-time snapshot (one row per station per hour); `air_quality_history` accumulates a time series across runs.
- **v1 Principle**: Prioritize **correct evaluation + complete features** first; hyperparameter tuning, feature pruning, and rolling training windows are deferred to v2.

---

## 1. Data Split: True Temporal Split

Implemented as `features.temporal_split`, replacing the original `train_test_split(shuffle=False)` — which, applied to data sorted by `["sitename","publishtime"]`, actually produced a **spatial split** (the last few stations alphabetically), not a temporal one.

**Approach (global time cutoff):**

1. Sort all unique `publishtime` values and choose the cutoff `T` at the last 20% (`TEST_FRAC = 0.2`).
2. `train` = rows where `publishtime < T`; `test` = rows where `publishtime >= T`.
3. **All stations appear in both train and test** — the difference is that test contains later timestamps → the evaluation targets truly "future" data, and every station can be predicted.

`temporal_split` returns empty/empty on empty input (defensive against the cold-start path).

**Cold-start threshold**: Temporal splitting requires a minimum number of timestamps. `model_predictions` enforces `MIN_TIMESTAMPS = 12` (approximately 10 train / 2 test); when this is not met — or if the feature matrix ends up empty — it returns `status="insufficient_data"` (reasons `not_enough_timestamps` / `no_feature_rows` / `empty_split`) and skips training instead of crashing.

---

## 2. Feature Engineering

Implemented in `features.py`; `build_feature_matrix(history)` assembles the matrix and returns `(feat_df, feature_cols)`.

### 2.1 Iron Rule (Preventing Data Leakage)

> Exogenous variables (other pollutants, weather) **always use lagged versions only**; only **time and geographic** features — those that are inherently known at prediction time — may use contemporaneous values. The history table still stores all current raw values (as the source material for computing lags), but **contemporaneous exogenous columns are excluded from the feature matrix X**. This invariant is guarded by `test_build_feature_matrix_has_no_same_hour_exogenous_leakage`.

### 2.2 Full Feature Set (v1, to be pruned by feature importance later)

| Group | Features | Contemporaneous / Lag | Notes |
| --- | --- | --- | --- |
| Autoregressive | `pm25_lag1/2/3` | lag | Primary signal |
| Rolling statistics | `pm25_roll3_mean`, `pm25_roll3_std` | lag (computed from lag1..3, `skipna=False`) | Recent trend / volatility |
| Concurrent pollutants | `pm10_lag1`, `o3_lag1`, `co_lag1`, `so2_lag1`, `no2_lag1`, `nox_lag1`, `no_lag1`, `aqi_lag1` | lag | |
| Averages / 8 hr | `pm25_avg_lag1`, `pm10_avg_lag1`, `o3_8hr_lag1`, `co_8hr_lag1`, `so2_avg_lag1` | lag | Prone to collinearity |
| Weather | `wind_speed_lag1`, `wind_dir_sin_lag1`, `wind_dir_cos_lag1` | lag | Wind direction uses sin/cos circular encoding |
| Time | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` | **contemporaneous** | Diurnal / seasonal cycles, known at prediction time |
| Geography | `latitude`, `longitude` | **contemporaneous** | Allows the model to learn spatial distribution |
| Station | `county` (**one-hot**, ~22 dimensions) | **contemporaneous** | If `siteid` (84 stations) is added later, switch to target encoding |

**Dropped**: `pollutant` and `status` describe the current air quality state and are therefore equivalent to leakage; not used in v1.

### 2.3 Missing-value handling (train-median imputation)

`build_feature_matrix` requires only the **core autoregressive signal** (target + `pm25_lag1/2/3`) to be non-null; sparse exogenous/weather columns keep their `NaN`. The actual values are filled **after the split** by `impute_features(train, test, feature_cols)` using **train-set medians** (0.0 fallback for an entirely-missing column), so there is no test-set leakage.

> This came from a real bug: the first implementation dropped rows on *all* feature columns, so a single entirely-missing EPA field (e.g. `nox`) NaN'd out every row → empty matrix → `temporal_split` crashed with `IndexError`. Requiring only the core signal + imputing the rest fixed both the crash and the data loss.

---

## 3. Model

- **`RandomForestRegressor`** (baseline), `n_estimators=100`, `random_state=42`, `n_jobs=-1` (parallel training).
- **Single model shared across all stations** (cross-station learning).
- **v1 trains on full history** (data volume is still small; evaluate effect first).
  - v2 todo: rolling training window (use only the most recent K days) or recent-sample weighting so forecasts are more sensitive to rapidly changing air quality. **Note**: this is separate from the drift window below — the drift window affects only alerting, while training data recency determines whether the model reflects current conditions.

---

## 4. Evaluation Metrics

`model_predictions` outputs (helpers in `evaluation.py`):

| Metric | Definition |
| --- | --- |
| `mae` | `mean_absolute_error(y_test, y_pred)` |
| `r2` | `r2_score(y_test, y_pred)` |
| `relative_mae` | `mae / mean(actual pm2.5 over test period)` — comparable across periods of differing difficulty; `None` if the mean is not strictly positive |
| `n_train` / `n_test` / `n_timestamps` / `n_features` | Sample, timestamp, and feature counts; monitors data growth |

**Per-station performance**: On the test set, `per_site_metrics` computes `mae` / `relative_mae` / `n_test` via `groupby("sitename")`; `model_metrics` writes these to the `model_metrics_by_site` table.

---

## 5. Drift Detection

Implemented as `evaluation.compute_drift`, called from `model_metrics`:

```
relative_mae = mae / mean(actual pm2.5 over test period)
baseline     = median(relative_mae of the last 3 runs)   # fall back to available runs if fewer than 3
drift_flag   = (relative_mae - baseline) / baseline > 0.15   # DRIFT_THRESHOLD
```

**Design rationale**:
- **Relative error**: After temporal splitting, each test window covers a different future period with varying difficulty. Using relative error enables fair comparison and avoids mistaking "an inherently hard period" for model degradation.
- **Median of last 3 runs**: Smooths single-run noise (one spike does not immediately become the baseline) while still reacting within 3 hours — appropriate for the rapid-change characteristics of air quality.
- **No fixed baseline**: A fixed baseline is best for catching slow degradation, but requires precisely defining "stable" and a reset policy; without those it is counterproductive. Deferred to v2 alongside a stability specification.

**Alert loop**: `drift_flag` → `model_not_drifting` (ERROR + blocking) asset check fails → run fails → `slack_on_run_failure` sensor → Slack.

---

## 6. Schema (as implemented)

### 6.1 Widened `air_quality_history`

All raw fields are stored (column names normalized: `pm2.5→pm25`, `pm2.5_avg→pm25_avg`). Existing tables are upgraded idempotently with `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so accumulated history is never lost.

```sql
-- Original: sitename, county, publishtime, pm25, pm10, o3, co, so2, no2, aqi
-- Added:
siteid       TEXT,
latitude     FLOAT,
longitude    FLOAT,
pm25_avg     FLOAT,
pm10_avg     FLOAT,
o3_8hr       FLOAT,
co_8hr       FLOAT,
so2_avg      FLOAT,
no           FLOAT,
nox          FLOAT,
wind_speed   FLOAT,
wind_direc   FLOAT
-- PK remains (sitename, publishtime)
```

`ingestion` and `cleaning` retain these columns (`NUMERIC_COLS` widened to 18 numeric fields).

### 6.2 `model_metrics` (added column)

```sql
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS relative_mae FLOAT;
```

Full columns: `run_at, status, mae, r2, relative_mae, n_train, n_test, n_timestamps, drift_flag`.

### 6.3 `model_metrics_by_site` (new table)

```sql
CREATE TABLE IF NOT EXISTS model_metrics_by_site (
    id           SERIAL PRIMARY KEY,
    run_at       TIMESTAMPTZ NOT NULL,
    sitename     TEXT NOT NULL,
    n_test       INT,
    mae          FLOAT,
    relative_mae FLOAT
)
```

---

## 7. Delivery (completed)

Delivered incrementally with TDD, each step unit-tested and deployed:

1. ✅ Widened `ingestion` / `cleaning` / `air_quality_history` columns (+ idempotent migration).
2. ✅ Feature engineering functions (lag / roll / time / circular encoding / one-hot) + temporal split.
3. ✅ Evaluation metrics (added `relative_mae`) + per-station metrics to `model_metrics_by_site`.
4. ✅ Drift via relative error + median of last 3 runs.
5. ✅ Unit tests (no leakage in split, cold start, sparse-exogenous survival, imputation, drift logic) → deployed.

Post-delivery fix: train-median imputation (§2.3) after a sparse-column bug; review polish (`n_jobs=-1`, `relative_mae` guards `denom > 0`).

---

## 8. v2+ Backlog

- Rolling training window / recent-sample weighting.
- Hyperparameter tuning; feature pruning based on feature importance.
- Fixed baseline + stability specification (catching slow seasonal degradation).
- Multi-step forecasting (t+2, t+3).

---

## Related Documents

- Pipeline architecture: [`PIPELINE.md`](./PIPELINE.md)
- Data governance: [`DATA_GOVERNANCE.md`](./DATA_GOVERNANCE.md)
