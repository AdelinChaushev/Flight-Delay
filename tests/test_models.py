"""Unit tests for src/models.py — preprocessing / pipeline / tuning-objective builders."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.models import make_preprocessor, make_pipe, make_lgbm_l1_objective, as_categoricals


@pytest.fixture
def toy():
    """Small mixed categorical/numeric frame with a continuous target."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "carrier": rng.choice(["AA", "DL", "UA"], n),
        "origin":  rng.choice(["ORD", "ATL", "JFK"], n),
        "dist":    rng.normal(800, 200, n),
        "hour":    rng.integers(0, 24, n),
    })
    y = df["dist"] * 0.01 + (df["carrier"] == "AA") * 5 + rng.normal(0, 1, n)
    return df, y.to_numpy(), ["dist", "hour"], ["carrier", "origin"]


def test_make_preprocessor_is_column_transformer(toy):
    _, _, num, cat = toy
    pre = make_preprocessor(num, cat)
    assert isinstance(pre, ColumnTransformer)
    assert {t[0] for t in pre.transformers} == {"cat", "num"}


def test_make_preprocessor_collapses_categoricals(toy):
    df, y, num, cat = toy
    Xt = make_preprocessor(num, cat).fit_transform(df, y)
    assert Xt.shape == (len(df), len(cat) + len(num))
    assert np.isfinite(Xt).all()


def test_make_pipe_structure_fits_and_predicts(toy):
    df, y, num, cat = toy
    pipe = make_pipe(Ridge(alpha=1.0), num, cat)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["pre", "model"]
    pipe.fit(df, y)
    pred = pipe.predict(df)
    assert pred.shape == (len(df),)
    assert np.isfinite(pred).all()


def test_as_categoricals_casts_and_copies(toy):
    df, y, num, cat = toy
    out = as_categoricals(df, cat)
    assert all(str(out[c].dtype) == "category" for c in cat)   # requested cols cast
    assert all(str(out[c].dtype) != "category" for c in num)   # numerics untouched
    assert all(str(df[c].dtype) != "category" for c in cat)    # original not mutated (a copy)


def test_make_lgbm_l1_objective_returns_valid_mae(toy):
    pytest.importorskip("lightgbm")
    optuna = pytest.importorskip("optuna")
    df, y, num, cat = toy
    Xf = df.copy()
    for c in cat:
        Xf[c] = Xf[c].astype("category")
    half = len(df) // 2
    objective = make_lgbm_l1_objective(
        Xf.iloc[:half], y[:half], Xf.iloc[half:], y[half:], y[half:])
    assert callable(objective)

    trial = optuna.trial.FixedTrial({
        "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20,
        "feature_fraction": 1.0, "subsample": 1.0,
    })
    mae = objective(trial)
    assert np.isfinite(mae)
    assert mae >= 0.0
