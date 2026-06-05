from datetime import datetime, timezone

import dagster as dg
from sqlalchemy import text

from airpulse.defs.postgres import PostgresResource

DRIFT_THRESHOLD = 0.15  # flag if MAE degrades more than 15% vs baseline


@dg.asset(group_name="ml")
def model_metrics(model_predictions: dict, postgres: PostgresResource) -> None:
    engine = postgres.get_engine()
    row = {
        "run_at": datetime.now(timezone.utc),
        "mae": model_predictions["mae"],
        "r2": model_predictions["r2"],
        "n_train": model_predictions["n_train"],
        "n_test": model_predictions["n_test"],
        "drift_flag": False,
    }

    with engine.begin() as conn:
        # create table if first run
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    run_at TIMESTAMPTZ NOT NULL,
                    mae FLOAT,
                    r2 FLOAT,
                    n_train INT,
                    n_test INT,
                    drift_flag BOOLEAN
                )
                """
            )
        )

        # compare with last run to detect drift
        result = conn.execute(
            text("SELECT mae FROM model_metrics ORDER BY run_at DESC LIMIT 1")
        ).fetchone()
        if result and result[0]:
            baseline_mae = result[0]
            if (row["mae"] - baseline_mae) / baseline_mae > DRIFT_THRESHOLD:
                row["drift_flag"] = True

        conn.execute(
            text(
                """
                INSERT INTO model_metrics
                    (run_at, mae, r2, n_train, n_test, drift_flag)
                VALUES (:run_at, :mae, :r2, :n_train, :n_test, :drift_flag)
                """
            ),
            row,
        )
