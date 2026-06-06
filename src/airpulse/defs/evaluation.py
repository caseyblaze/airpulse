import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def relative_mae(y_true, y_pred):
    """MAE normalized by the mean of actuals; comparable across test windows
    of differing difficulty. None if the mean is not strictly positive."""
    mae = mean_absolute_error(y_true, y_pred)
    denom = float(np.mean(y_true))
    return float(mae / denom) if denom > 0 else None


def per_site_metrics(sitenames, y_true, y_pred):
    """Per-station MAE / relative MAE / test count. Returns a list of dicts."""
    df = pd.DataFrame(
        {"sitename": list(sitenames), "y_true": list(y_true), "y_pred": list(y_pred)}
    )
    out = []
    for site, g in df.groupby("sitename"):
        mae = float(mean_absolute_error(g["y_true"], g["y_pred"]))
        denom = float(g["y_true"].mean())
        out.append(
            {
                "sitename": site,
                "n_test": int(len(g)),
                "mae": mae,
                "relative_mae": float(mae / denom) if denom > 0 else None,
            }
        )
    return out


def compute_drift(current_relative_mae, recent_relative_maes, threshold=0.15):
    """Drift if current relative MAE exceeds the median of recent runs by more
    than `threshold`. Safe (False) when data is missing."""
    vals = [v for v in recent_relative_maes if v is not None]
    if current_relative_mae is None or not vals:
        return False
    baseline = float(np.median(vals))
    if baseline == 0:
        return False
    return (current_relative_mae - baseline) / baseline > threshold


def sustained_drift(drift_flags, streak):
    """True only when the most recent `streak` runs are all flagged for drift.

    `drift_flags` is newest-first. Requiring a full streak filters transient
    single-run spikes (normal hour-to-hour variance) so only persistent drift
    raises an alert. False when there isn't enough history to fill the streak.
    """
    if streak <= 0 or len(drift_flags) < streak:
        return False
    return all(bool(f) for f in drift_flags[:streak])
