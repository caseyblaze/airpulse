import dagster as dg
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from airpulse.defs.evaluation import per_site_metrics, relative_mae
from airpulse.defs.features import build_feature_matrix, impute_features, temporal_split
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
    if len(feat) == 0:
        return _insufficient(context, n_timestamps, "no_feature_rows")

    train, test = temporal_split(feat, TEST_FRAC)
    if len(train) == 0 or len(test) == 0:
        return _insufficient(context, n_timestamps, "empty_split")

    X_train, X_test = impute_features(train, test, feature_cols)
    y_train, y_test = train["pm25"], test["pm25"]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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
