"""Model / preprocessing builders and the shared LightGBM tuning objective."""
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


def make_lgbm_l1_objective(X_fit, y_fit, X_val, y_val_cap, y_val_true):
    """Build an Optuna objective that tunes an L1 (MAE) LightGBM.

    The returned ``objective(trial)`` fits a gradient-boosted model on
    ``(X_fit, y_fit)`` with early stopping against the **capped** validation
    target, then scores its predictions against the **true** (uncapped)
    validation target — the metric we actually care about. Shared by the
    Stage 2 and after-departure searches so both explore an identical space.

    Categorical columns are auto-detected from ``category`` dtype, so pass the
    feature frames with those columns already cast.
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    def objective(trial):
        params = dict(
            objective="regression_l1", n_estimators=2000, n_jobs=-1, verbose=-1,
            subsample_freq=1,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 500),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
        )
        model = lgb.LGBMRegressor(**params).fit(
            X_fit, y_fit, eval_set=[(X_val, y_val_cap)], eval_metric="l1",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        return mean_absolute_error(y_val_true, model.predict(X_val))

    return objective
