# airpulse ML Modeling Plan (v1 Blueprint)

> **Language:** English | [繁體中文](MODELING_PLAN.zh-TW.md)

> Status: **Planning — not yet implemented**. This document is an implementation blueprint for incremental delivery.
> Prediction target: pm2.5 **one hour ahead** for each monitoring station.

## 0. Design Goals and Constraints

- **Goal**: Produce a one-hour-ahead pm2.5 forecast for each station and be able to evaluate its **future** performance.
- **Constraint**: The data source is a real-time snapshot (one row per station per hour); `air_quality_history` must accumulate a time series across runs.
- **v1 Principle**: Prioritize **correct evaluation + complete features** first; hyperparameter tuning, feature pruning, and rolling training windows are deferred to v2.

---

## 1. Data Split: True Temporal Split

Replace the current `train_test_split(shuffle=False)` — when applied to data sorted by `["sitename","publishtime"]`, it actually produces a **spatial split** (the last few stations alphabetically), not a temporal one.

**New approach (global time cutoff):**

1. Sort all unique `publishtime` values and choose the cutoff `T` at the last 20%.
2. `train` = rows where `publishtime < T`; `test` = rows where `publishtime >= T`.
3. **All stations appear in both train and test** — the difference is that test contains later timestamps → the evaluation targets truly "future" data, and every station can be predicted.

**Cold-start threshold**: Temporal splitting requires a minimum number of timestamps. Set `MIN_TIMESTAMPS = 12` (approximately 10 train / 2 test). When this is not met, `model_predictions` returns `status="insufficient_data"` and skips training.

---

## 2. Feature Engineering

### 2.1 Iron Rule (Preventing Data Leakage)

> Exogenous variables (other pollutants, weather) **must always use lagged versions**; only **time and geographic** features — those that are inherently known at prediction time — may use contemporaneous values. The history table still stores all current raw values (as the source material for computing lags), but **contemporaneous exogenous columns are excluded from the feature matrix X**.

### 2.2 Recommended Full Feature Set (v1, to be pruned by feature importance later)

| Group | Features | Contemporaneous / Lag | Notes |
| --- | --- | --- | --- |
| Autoregressive | `pm25_lag1/2/3` | lag | Primary signal |
| Rolling statistics | `pm25_roll3_mean`, `pm25_roll3_std` | lag (computed from lag1..3) | Recent trend / volatility |
| Concurrent pollutants | `pm10_lag1`, `o3_lag1`, `co_lag1`, `so2_lag1`, `no2_lag1`, `nox_lag1`, `no_lag1`, `aqi_lag1` | lag | |
| Averages / 8 hr | `pm25_avg_lag1`, `pm10_avg_lag1`, `o3_8hr_lag1`, `co_8hr_lag1`, `so2_avg_lag1` | lag | Optional; prone to collinearity |
| Weather | `wind_speed_lag1`, `wind_dir_sin_lag1`, `wind_dir_cos_lag1` | lag | Wind direction uses sin/cos circular encoding |
| Time | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` | **contemporaneous** | Diurnal / seasonal cycles, known at prediction time |
| Geography | `latitude`, `longitude` | **contemporaneous** | Allows the model to learn spatial distribution |
| Station | `county` (**one-hot**, ~22 dimensions) | **contemporaneous** | If `siteid` (84 stations) is added later, switch to target encoding |

**Drop / handle with care**: `pollutant` and `status` describe the current air quality state and are therefore equivalent to leakage; they would need to be lagged before use — drop in v1.

---

## 3. Model

- **RandomForestRegressor** (baseline), `n_estimators=100`, `random_state=42`.
- **Single model shared across all stations** (cross-station learning).
- **v1 trains on full history** (data volume is still small; evaluate effect first).
  - v2 todo: rolling training window (use only the most recent K days) or recent-sample weighting so forecasts are more sensitive to rapidly changing air quality. **Note**: this is separate from the drift window below — the drift window affects only alerting, while training data recency determines whether the model reflects current conditions.

---

## 4. Evaluation Metrics

`model_predictions` outputs:

| Metric | Definition |
| --- | --- |
| `mae` | `mean_absolute_error(y_test, y_pred)` |
| `r2` | `r2_score(y_test, y_pred)` |
| `relative_mae` | `mae / mean(actual pm2.5 over test period)` — comparable across periods of differing difficulty |
| `n_train` / `n_test` / `n_timestamps` | Sample and timestamp counts; monitors data growth |

**Per-station performance**: On the test set, compute `mae` / `relative_mae` / `n_test` via `groupby("sitename")` and write results to the new table `model_metrics_by_site`.

---

## 5. Drift Detection (Revised)

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

## 6. Schema Changes

### 6.1 Widening `air_quality_history`

New columns (for future feature construction; column names normalized: `pm2.5→pm25`, `pm2.5_avg→pm25_avg`):

```sql
-- Existing: sitename, county, publishtime, pm25, pm10, o3, co, so2, no2, aqi
-- New:
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

Accordingly, `ingestion` and `cleaning` must retain these columns (currently discarded).

### 6.2 Adding a Column to `model_metrics`

```sql
ALTER TABLE model_metrics ADD COLUMN relative_mae FLOAT;
```

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

## 7. Phased Implementation Order

1. Widen `ingestion` / `cleaning` / `air_quality_history` columns.
2. Feature engineering functions (lag / roll / time / circular encoding / one-hot) + temporal split.
3. Evaluation metrics (add `relative_mae`) + write per-station metrics to `model_metrics_by_site`.
4. Convert drift to relative error + median of last 3 runs.
5. Unit tests (no leakage in split, cold start, drift logic) → deploy.

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
