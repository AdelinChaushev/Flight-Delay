"""Unit tests for src/metrics.py — the regression scoring helpers."""
import numpy as np
import pytest

from src.metrics import rmse, regression_scores, evaluate


def test_rmse_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])  # squared errors 0, 0, 4 -> MSE 4/3
    assert rmse(y_true, y_pred) == pytest.approx((4 / 3) ** 0.5)


def test_rmse_zero_when_perfect():
    y = np.array([3.0, -1.0, 7.5])
    assert rmse(y, y) == pytest.approx(0.0)


def test_regression_scores_keys_and_mae():
    y_true = np.array([0.0, 10.0, 20.0, 30.0])
    y_pred = np.array([1.0, 9.0, 22.0, 27.0])  # abs errors 1, 1, 2, 3 -> MAE 1.75
    scores = regression_scores(y_true, y_pred)
    assert set(scores) == {"MAE", "RMSE", "R2"}
    assert scores["MAE"] == pytest.approx(1.75)
    assert scores["R2"] <= 1.0


def test_regression_scores_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    scores = regression_scores(y, y)
    assert scores["MAE"] == pytest.approx(0.0)
    assert scores["RMSE"] == pytest.approx(0.0)
    assert scores["R2"] == pytest.approx(1.0)


def test_evaluate_appends_row_and_returns_scores():
    results = []
    y = np.array([0.0, 2.0, 4.0])
    out = evaluate("perfect", y, y, results)
    assert len(results) == 1
    assert results[0]["model"] == "perfect"
    assert results[0]["MAE"] == pytest.approx(0.0)
    assert out["MAE"] == pytest.approx(0.0)  # evaluate also returns the score dict


def test_evaluate_without_results_list_is_safe():
    # results defaults to None -> must not raise, still returns the scores
    scores = evaluate("noappend", np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    assert scores["MAE"] == pytest.approx(1.0)
