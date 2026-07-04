"""Model / preprocessing builders for the flight-delay linear models."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder


def make_preprocessor(num_cols, cat_cols):
    """Target-encode categoricals + standardize numerics.

    Fit on train only — the target encoder learns from the train targets.
    Kept separate so a hyperparameter sweep can fit it once and reuse it.
    """
    return ColumnTransformer([
        ("cat", TargetEncoder(target_type="continuous"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def make_pipe(model, num_cols, cat_cols):
    """Preprocessing (see :func:`make_preprocessor`) followed by ``model``,
    as a single scikit-learn ``Pipeline`` fit end-to-end on train."""
    return Pipeline([("pre", make_preprocessor(num_cols, cat_cols)),
                     ("model", model)])
