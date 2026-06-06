# airpulse

> An hourly Taiwan air-quality **ML pipeline** on **Dagster+ (Serverless)** that turns a
> stateless real-time API into a self-accumulating, self-forecasting, self-monitoring system.

`Dagster+ Serverless` · `PostgreSQL (Neon)` · `scikit-learn` · `pandas` · `pytest`

> **Language:** English | [繁體中文](README.zh-TW.md)

---

## The core insight

Taiwan's environment ministry (MOENV) publishes air quality as a **real-time snapshot**: each
call returns *one current reading per station* (~84 stations, a single timestamp). There is **no
history endpoint** — one call can never be a time series.

Every design decision in airpulse follows from that constraint:

- To forecast, the pipeline must **accumulate its own history** across runs → an idempotent
  PostgreSQL state layer (`air_quality_history`) is the single source of truth.
- To evaluate *future* performance honestly, the model needs a **true temporal split**, not a
  random one.
- Because each hourly run depends on the last, **safe replays and reliability gates** aren't
  optional polish — they're load-bearing.

So this isn't a toy "fetch → train" script. It's a small production system designed around the
data it actually has.

## Architecture at a glance

One hourly job (`hourly_air_quality_pipeline`, cron `0 * * * *`, aligned to the API's refresh
cadence) materializes a five-asset DAG:

```
raw_air_quality        (api, python)      ← GET MOENV aqx_p_432
        │
        ▼
cleaned_air_quality    (pandas)           ← drop missing keys, coerce numerics, sort by time
        │
        ▼
air_quality_history    (pandas, postgres) ← idempotent upsert; returns the full accumulated series
        │
        ▼
model_predictions      (sklearn)          ← leakage-safe features + temporal split → RF forecast of pm2.5
        │
        ▼
model_metrics          (postgres)         ← overall + per-site MAE/R²/relative-error + drift flag
```

Cleaning and validation are deliberately **separated**: cleaning only structures and type-coerces
(missing numbers become `NaN`, never dropped); whether a *value* is plausible, fresh, or drifting
is the validation layer's job. Full asset responsibilities and the history-table schema are in
[`PIPELINE.md`](./PIPELINE.md).

## Reliability & fault tolerance

Because every run builds on the last, the pipeline is built to survive the messiness of a public
API and to make re-runs safe:

- **External-dependency retries** — the ingestion asset uses a Dagster `RetryPolicy` with
  exponential backoff + jitter (≈10s → 20s → 40s), so a transient timeout or 5xx doesn't fail the
  whole run.
- **Payload-shape tolerance** — MOENV's v2 API returns a *bare JSON list*, not the documented
  `{"records": [...]}` wrapper. The parser handles both shapes instead of trusting the schema.
- **Idempotent upserts → safe replays** — history is written with
  `INSERT ... ON CONFLICT (sitename, publishtime) DO NOTHING`. Re-running a step or backfilling
  never duplicates rows, which is *what makes retry-from-failure safe* rather than dangerous.
- **Idempotent migrations** — widening the history table uses `ADD COLUMN IF NOT EXISTS`, so schema
  changes never drop already-accumulated history.
- **Blocking quality gates → alerting closed loop** — `ERROR`-severity asset checks are
  `blocking=True`: bad data or a drifting model fails the run, which a `run_failure_sensor` turns
  into a **Slack** alert. Detect → record → alert, with no silent corruption flowing downstream.

## Data governance

Governance is treated as a framework, not an afterthought — designed so the *same* pipeline extends
to regulated data by changing **classification, not architecture** (full detail in
[`DATA_GOVERNANCE.md`](./DATA_GOVERNANCE.md)):

- **Classification** — every asset carries a `data_classification` tag
  (`public` → `internal` → `confidential` → `restricted`). The current EPA data is `public` and
  PII-free; a `restricted` field would be required to pass a single audited masking path before
  storage.
- **Quality as asset checks** — six checks run on every materialization and surface as pass/fail
  markers in the catalog, split by severity: `ERROR` (blocking, e.g. `raw_not_empty`,
  `pm25_non_negative`, `model_not_drifting`) vs `WARN` (surface-only, e.g. `aqi_in_range`,
  `data_is_fresh`).
- **Lineage, ownership & metadata** — the assets form an explicit DAG; each declares
  `owners=["team:data-engineering"]`; `cleaned_air_quality` publishes a column-level schema; and
  every run emits operational metadata (row counts, distinct sites, source URL, model metrics) for
  catalog observability.

## ML done carefully

The modeling layer forecasts **next-hour pm2.5 per station**, and most of the work is in *not
fooling yourself* (design rationale in [`MODELING_PLAN.md`](./MODELING_PLAN.md)):

- **True temporal split** — the test set is the latest 20% of *timestamps*; every station appears
  in both train and test, differing only by time. This replaced an earlier `shuffle=False` split
  that, on station-then-time-sorted data, was silently a **spatial** split (last few stations
  alphabetically) rather than a future-performance test.
- **Leakage-safe features (the iron rule)** — exogenous signals (other pollutants, weather) enter
  **only as lags**, because their current-hour values aren't known at prediction time. Only
  *already-known* features (time-of-day, day-of-week, season via cyclical sin/cos encoding;
  geography; county one-hot) use the concurrent value. Includes autoregressive lags and lag-derived
  rolling mean/std.
- **Robust imputation** — only core signals (target + pm2.5 lags) are required; sparse exogenous
  columns keep `NaN` and are filled with the **train-set median after the split**. (This came from a
  real bug: a single all-missing EPA column would otherwise wipe the whole feature matrix.)
- **Comparable metrics** — alongside MAE/R², a `relative_mae = mae / mean(actual pm2.5 in test)`
  makes runs comparable across easy and hard periods, plus a **per-site** metrics table.
- **Drift as a governance gate** — `drift_flag` fires when relative error worsens by >15% versus the
  **median of the last 3 runs** (relative, so a genuinely hard period isn't mistaken for decay;
  median, to smooth single-run noise). It's promoted from a quiet DB column into the blocking
  `model_not_drifting` check that can page Slack.
- **Honest cold-start** — below `MIN_TIMESTAMPS = 12` accumulated timepoints the model returns
  `status="insufficient_data"` instead of training on noise, and drift checks pass rather than
  false-alarm.

## What this demonstrates

| Capability | Concrete evidence | Deep dive |
| --- | --- | --- |
| **Reading the data, not the brochure** | Recognized a snapshot API has no history → designed accumulation + forecasting around it | [PIPELINE §1.4](./PIPELINE.md) |
| **Production reliability** | RetryPolicy, payload-shape tolerance, idempotent upserts & migrations, safe replays | [PIPELINE §1–2](./PIPELINE.md) |
| **Monitoring & alerting** | Blocking asset checks → run-failure sensor → Slack closed loop | [PIPELINE §3.5](./PIPELINE.md) |
| **Data governance** | Classification framework, quality gates, lineage/ownership/metadata | [DATA_GOVERNANCE.md](./DATA_GOVERNANCE.md) |
| **ML evaluation rigor** | Leakage-safe features, true temporal split, relative MAE, per-site metrics, drift detection | [MODELING_PLAN.md](./MODELING_PLAN.md) |
| **Engineering judgment** | Caught & fixed real bugs (spatial-split, feature-matrix wipe); documented v1/v2 trade-offs; unit-tested split/cold-start/drift | git history · `tests/` |

## Tech stack

- **Orchestration:** Dagster+ Serverless (PEX deploys), assets + asset checks + schedules + sensors
- **Storage:** PostgreSQL on Neon (idempotent upsert as source of truth)
- **ML:** scikit-learn `RandomForestRegressor` (single cross-station model), pandas feature engineering
- **Alerting:** Slack incoming webhook via run-failure sensor
- **Tooling:** uv / pip, pytest

## Getting started

**Install** (uv recommended):

```bash
uv sync
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

<details>
<summary>pip alternative</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```
</details>

**Run Dagster locally** — then open http://localhost:3000:

```bash
dg dev
```

**Run the tests:**

```bash
pytest
```

Configuration (Postgres connection, `SLACK_WEBHOOK_URL`) is via environment variables; see
[`.env.example`](./.env.example). Secrets live only in Dagster+ prod scope, never in the repo.

## Documentation map

| Document | What's inside |
| --- | --- |
| [`PIPELINE.md`](./PIPELINE.md) | Pipeline architecture: ingestion, cleaning, validation layer, ML layer |
| [`DATA_GOVERNANCE.md`](./DATA_GOVERNANCE.md) | Classification, quality gates, lineage, ownership, roadmap |
| [`MODELING_PLAN.md`](./MODELING_PLAN.md) | ML design blueprint, trade-offs, and v2 backlog |

## Roadmap

- **Modeling v2** — rolling training window / recency-weighting (so forecasts track fast-changing
  air), hyperparameter tuning, feature pruning by importance, multi-step (t+2, t+3) forecasts, and a
  fixed-baseline drift spec to catch slow seasonal decay.
- **Platform (Phase 2, GCP)** — history-table retention/partitioning, DB role separation, and
  secret rotation.
