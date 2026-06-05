import dagster as dg
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from airpulse.defs.postgres import PostgresResource

LAGS = 3
# Each site needs LAGS+1 consecutive hourly readings to yield one training row,
# so the model only trains once enough history has accumulated.
MIN_FEATURE_ROWS = 30


def make_lag_features(df: pd.DataFrame, target: str, lags: int = LAGS) -> pd.DataFrame:
    df = df.sort_values(["sitename", "publishtime"])
    lag_cols = []
    for i in range(1, lags + 1):
        col = f"{target}_lag{i}"
        df[col] = df.groupby("sitename")[target].shift(i)
        lag_cols.append(col)
    return df.dropna(subset=[target, *lag_cols])


@dg.asset(group_name="ml", deps=["air_quality_history"])
def model_predictions(
    context: dg.AssetExecutionContext,
    postgres: PostgresResource,
) -> dict:
    """Forecast pm2.5 from its own recent lags, trained on accumulated history.

    Reads the durable history table directly (the source of truth for the
    cross-run accumulated series) rather than receiving it through the IO
    manager, which is ephemeral on serverless. Returns
    status='insufficient_data' instead of crashing until enough hourly
    snapshots have been collected to build lag features."""
    engine = postgres.get_engine()
    history = pd.read_sql(
        "SELECT sitename, publishtime, pm25 FROM air_quality_history", engine
    )

    n_timestamps = (
        int(history["publishtime"].nunique()) if len(history) else 0
    )

    feat = make_lag_features(history.copy(), target="pm25")

    if len(feat) < MIN_FEATURE_ROWS:
        result = {
            "status": "insufficient_data",
            "n_feature_rows": int(len(feat)),
            "n_timestamps": n_timestamps,
            "mae": None,
            "r2": None,
            "n_train": 0,
            "n_test": 0,
        }
        context.add_output_metadata(result)
        return result

    feature_cols = [f"pm25_lag{i}" for i in range(1, LAGS + 1)]
    X = feat[feature_cols]
    y = feat["pm25"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {
        "status": "ok",
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_timestamps": n_timestamps,
    }
    context.add_output_metadata(result)
    return result
