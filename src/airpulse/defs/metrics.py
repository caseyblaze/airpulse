from datetime import datetime, timezone

import dagster as dg
from sqlalchemy import text

from airpulse.defs.evaluation import compute_drift
from airpulse.defs.governance import DATA_OWNER, PUBLIC_TAGS
from airpulse.defs.postgres import PostgresResource

# Flag a run when relative MAE worsens >30% vs the median of a wider baseline
# window. A wider window + looser threshold ride out normal hour-to-hour
# variance in the shifting temporal holdout; sustained drift is caught by the
# streak logic in the model_not_drifting check rather than a single twitchy run.
DRIFT_THRESHOLD = 0.30
DRIFT_BASELINE_WINDOW = 12  # trained runs (hours) the baseline median spans

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

        # drift vs the median of the last DRIFT_BASELINE_WINDOW trained relative MAEs
        recent = [
            r[0]
            for r in conn.execute(
                text(
                    "SELECT relative_mae FROM model_metrics "
                    "WHERE relative_mae IS NOT NULL "
                    "ORDER BY run_at DESC LIMIT :window"
                ),
                {"window": DRIFT_BASELINE_WINDOW},
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
