import dagster as dg
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def make_lag_features(df: pd.DataFrame, target: str, lags: int = 3) -> pd.DataFrame:
    for i in range(1, lags + 1):
        df[f"{target}_lag{i}"] = df.groupby("sitename")[target].shift(i)
    return df.dropna()


@dg.asset(group_name="ml")
def model_predictions(cleaned_air_quality: pd.DataFrame) -> dict:
    df = make_lag_features(cleaned_air_quality.copy(), target="pm2.5", lags=3)

    feature_cols = ["pm2.5_lag1", "pm2.5_lag2", "pm2.5_lag3"]
    X = df[feature_cols]
    y = df["pm2.5"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
