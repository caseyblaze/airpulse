# Data Governance

How the **airpulse** pipeline governs the data it ingests, transforms, and
serves. The current dataset (Taiwan EPA open air-quality data) is **public and
PII-free**; the framework below is designed so the same pipeline extends
cleanly to regulated data (health, financial) by changing classification, not
architecture.

## 1. Data classification

Every asset is tagged with a `data_classification` tag (visible in the Dagster
catalog). Levels, in rising sensitivity:

| Level | Meaning | Handling |
| --- | --- | --- |
| `public` | Open data, no restrictions | None — **all current airpulse assets** |
| `internal` | Business-internal, non-sensitive | Access-controlled storage |
| `confidential` | Commercially sensitive | Encryption at rest, restricted roles |
| `restricted` | PII / regulated (PDPA, HIPAA-like, financial) | Masking + audit + least-privilege |

Defined in `src/airpulse/defs/governance.py`. A `restricted` field is required
to pass through the single audited masking path (`mask_pii`) before storage —
deterministic pseudonymization so joins still work while the original is
irrecoverable.

## 2. Data quality (asset checks)

Quality gates run on every materialization and surface as pass/fail markers in
the catalog (`src/airpulse/defs/checks.py`):

| Check | Asset | Severity | Rule |
| --- | --- | --- | --- |
| `raw_not_empty` | raw_air_quality | ERROR | Ingestion returned rows |
| `no_missing_keys` | cleaned_air_quality | ERROR | sitename + publishtime present |
| `pm25_non_negative` | cleaned_air_quality | ERROR | pm2.5 ≥ 0 |
| `aqi_in_range` | cleaned_air_quality | WARN | AQI within 0–500 |
| `data_is_fresh` | cleaned_air_quality | WARN | Latest reading ≤ 3h old |

ERROR-severity failures signal bad data that should block downstream trust;
WARN surfaces anomalies without halting the pipeline.

## 3. Metadata, lineage & ownership

- **Lineage**: assets form an explicit DAG —
  `raw_air_quality → cleaned_air_quality → air_quality_history → model_predictions → model_metrics`.
- **Ownership**: every asset has `owners=["team:data-engineering"]`.
- **Schema**: `cleaned_air_quality` publishes a column-level `TableSchema`.
- **Operational metadata**: row counts, distinct sites, source URL, and model
  metrics (MAE/R²/drift) are emitted per run for catalog observability.
- **Kinds**: assets are badged by technology (api, pandas, sklearn, postgres).

## 4. Retention & access (roadmap)

Out of scope for the current phase; planned for the GCP migration (Phase 2):
history-table retention/partitioning, DB role separation, and secret rotation
(secrets currently live only in Dagster+ prod scope, never in the repo).
