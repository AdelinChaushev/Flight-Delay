"""Regression metrics for the flight-delay models.

Thin, reusable wrappers around scikit-learn so every model in the Modeling
notebook is scored the same way — MAE, RMSE, and R^2, always against the
*real* (uncapped) target. MAE is the primary metric; R^2 is reported for
context (it measures variance explained, which the heavy tail dominates, so
it reads pessimistically low here).
"""
from __future__ import annotations

from sklearn.metrics import mean_absolute_error, r2_score

try:  
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:  # fall back for older versions
    from sklearn.metrics import mean_squared_error

    def rmse(y_true, y_pred) -> float:
        """Root mean squared error."""
        return mean_squared_error(y_true, y_pred) ** 0.5


def regression_scores(y_true, y_pred) -> dict:
    """Return ``{'MAE': ..., 'RMSE': ..., 'R2': ...}`` for a set of predictions."""
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def evaluate(name, y_true, y_pred, results=None) -> dict:
    """Score predictions, print a one-line summary, and optionally append a
    row to a running ``results`` list for a later comparison table.

    Parameters
    ----------
    name : str
        Label for the model / baseline.
    y_true, y_pred : array-like
        Ground truth and predictions (score against the real target).
    results : list, optional
        If given, ``{'model': name, 'MAE': ..., 'RMSE': ..., 'R2': ...}`` is appended.
    """
    scores = regression_scores(y_true, y_pred)
    print(f"{name:<26} MAE {scores['MAE']:6.2f}   RMSE {scores['RMSE']:6.2f}   "
          f"R2 {scores['R2']:6.3f}")
    if results is not None:
        results.append({"model": name, **scores})
    return scores
