from datetime import datetime, timezone

import dagster as dg
from sqlalchemy import text

from airpulse.defs.postgres import PostgresResource

DRIFT_THRESHOLD = 0.15  # flag if MAE degrades more than 15% vs last trained run

DDL = """
CREATE TABLE IF NOT EXISTS model_metrics (
    id           SERIAL PRIMARY KEY,
    run_at       TIMESTAMPTZ NOT NULL,
    status       TEXT,
    mae          FLOAT,
    r2           FLOAT,
    n_train      INT,
    n_test       INT,
    n_timestamps INT,
    drift_flag   BOOLEAN
)
"""


@dg.asset(group_name="ml")
def model_metrics(
    context: dg.AssetExecutionContext,
    model_predictions: dict,
    postgres: PostgresResource,
) -> None:
    engine = postgres.get_engine()
    row = {
        "run_at": datetime.now(timezone.utc),
        "status": model_predictions.get("status", "unknown"),
        "mae": model_predictions.get("mae"),
        "r2": model_predictions.get("r2"),
        "n_train": model_predictions.get("n_train", 0),
        "n_test": model_predictions.get("n_test", 0),
        "n_timestamps": model_predictions.get("n_timestamps", 0),
        "drift_flag": False,
    }

    with engine.begin() as conn:
        conn.execute(text(DDL))

        # drift only makes sense once we actually have a trained MAE to compare
        if row["mae"] is not None:
            result = conn.execute(
                text(
                    "SELECT mae FROM model_metrics "
                    "WHERE mae IS NOT NULL ORDER BY run_at DESC LIMIT 1"
                )
            ).fetchone()
            if result and result[0]:
                baseline_mae = result[0]
                if (row["mae"] - baseline_mae) / baseline_mae > DRIFT_THRESHOLD:
                    row["drift_flag"] = True

        conn.execute(
            text(
                """
                INSERT INTO model_metrics
                    (run_at, status, mae, r2, n_train, n_test,
                     n_timestamps, drift_flag)
                VALUES
                    (:run_at, :status, :mae, :r2, :n_train, :n_test,
                     :n_timestamps, :drift_flag)
                """
            ),
            row,
        )

    context.add_output_metadata(
        {k: (v if v is not None else "null") for k, v in row.items() if k != "run_at"}
    )
