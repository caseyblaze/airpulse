import pandas as pd

from airpulse.defs.evaluation import relative_mae, per_site_metrics, compute_drift


def test_relative_mae_divides_by_mean_actual():
    # mae = 1.0, mean(y_true) = 10 -> 0.1
    assert abs(relative_mae([10, 10], [9, 11]) - 0.1) < 1e-9
    # zero mean -> None (avoid div by zero)
    assert relative_mae([0, 0], [1, -1]) is None


def test_per_site_metrics_groups_by_site():
    sites = ["A", "A", "B"]
    y_true = [10.0, 10.0, 20.0]
    y_pred = [9.0, 11.0, 18.0]
    out = {r["sitename"]: r for r in per_site_metrics(sites, y_true, y_pred)}
    assert out["A"]["n_test"] == 2
    assert abs(out["A"]["mae"] - 1.0) < 1e-9
    assert abs(out["A"]["relative_mae"] - 0.1) < 1e-9
    assert out["B"]["n_test"] == 1


def test_compute_drift_uses_median_of_recent():
    # baseline = median([0.10, 0.10, 0.20]) = 0.10; current 0.12 -> +20% > 15% -> drift
    assert compute_drift(0.12, [0.10, 0.10, 0.20], threshold=0.15) is True
    # current 0.11 -> +10% < 15% -> no drift
    assert compute_drift(0.11, [0.10, 0.10, 0.20], threshold=0.15) is False
    # no history -> no drift
    assert compute_drift(0.50, [], threshold=0.15) is False
    # None current -> no drift
    assert compute_drift(None, [0.10], threshold=0.15) is False
