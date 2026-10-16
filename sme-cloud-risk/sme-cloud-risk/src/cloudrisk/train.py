"""
Model benchmarking, nested hyperparameter tuning and selection.

Purpose : fit three candidate classifiers inside identical preprocessing
          pipelines, tune each with randomised search on inner folds, score
          the tuned pipelines with stratified outer k-fold cross-validation
          on the training set only, select the best by ROC-AUC, and refit the
          winner on the full training set. Because tuning happens inside each
          outer fold (nested CV), the reported CV scores are not optimistic
          about the hyperparameters, and the hold-out set in evaluate.py is
          never touched during selection.
Inputs  : X_train, y_train DataFrames from features.split_xy.
Outputs : tidy DataFrame of nested-CV results, dict of tuned pipelines, name
          of the selected model.
Depends : scikit-learn, pandas, numpy, scipy.

Candidates
----------
logistic_regression : L2-regularised; C tuned. Interpretable baseline whose
                      coefficients give the odds ratios in explain.py. No class
                      re-weighting: at 25-30% prevalence it is unnecessary and
                      would distort probabilities used by the dashboard.
random_forest       : bagged trees; depth and leaf size tuned.
gradient_boosting   : HistGradientBoosting; learning rate, leaves and
                      regularisation tuned, with early stopping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import N_FOLDS, PRIMARY_METRIC, SEED
from .features import make_preprocessor

# Metrics reported for every candidate. ROC-AUC drives selection; average
# precision is included because incidents are the minority class, and Brier
# score because calibrated probabilities matter for a risk dashboard.
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "brier": "neg_brier_score",
    "f1": "f1",
}

# Inner search budget. Kept modest so the full nested run finishes in about a
# minute on a laptop; raise N_ITER for a final production fit.
N_ITER = 8
INNER_FOLDS = 3


def build_candidates(seed: int = SEED) -> dict[str, Pipeline]:
    """Return the untuned candidate pipelines keyed by short name."""
    return {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                             class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", make_preprocessor(scale_numeric=False)),
            ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400,
                                                     max_leaf_nodes=15, l2_regularization=1.0,
                                                     early_stopping=True, validation_fraction=0.15,
                                                     random_state=seed)),
        ]),
    }


def search_spaces() -> dict[str, dict]:
    """Randomised-search distributions for each candidate's model step."""
    return {
        "logistic_regression": {
            "model__C": loguniform(1e-2, 1e2),
        },
        "random_forest": {
            "model__max_depth": randint(4, 16),
            "model__min_samples_leaf": randint(2, 20),
            "model__max_features": uniform(0.3, 0.6),
        },
        "gradient_boosting": {
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_leaf_nodes": randint(7, 40),
            "model__l2_regularization": loguniform(1e-2, 10),
            "model__min_samples_leaf": randint(10, 60),
        },
    }


def build_searches(seed: int = SEED, n_iter: int = N_ITER) -> dict[str, RandomizedSearchCV]:
    """Wrap each candidate in a RandomizedSearchCV over its search space."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    spaces = search_spaces()
    return {
        name: RandomizedSearchCV(pipe, spaces[name], n_iter=n_iter, scoring=PRIMARY_METRIC,
                                 cv=inner, random_state=seed, n_jobs=-1, refit=True)
        for name, pipe in build_candidates(seed).items()
    }


def benchmark(X_train: pd.DataFrame, y_train: pd.Series, n_folds: int = N_FOLDS,
              seed: int = SEED, tune: bool = True,
              n_iter: int = N_ITER) -> tuple[pd.DataFrame, dict]:
    """
    Nested cross-validation of every candidate on the training set.

    Parameters
    ----------
    tune : if True, each outer fold runs an inner randomised search, so the
           outer scores are honest about hyperparameter selection. If False,
           the fixed defaults in build_candidates are scored (faster; used by
           the tests).

    Returns
    -------
    results    : one row per candidate with mean and std of each outer metric.
    estimators : the (unfitted) search objects or plain pipelines, for refit.
    """
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    estimators = build_searches(seed, n_iter) if tune else build_candidates(seed)
    rows = []
    for name, est in estimators.items():
        scores = cross_validate(est, X_train, y_train, cv=outer, scoring=SCORING, n_jobs=1)
        row = {"model": name, "tuned": tune}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            if metric == "brier":           # undo sklearn's sign convention
                vals = -vals
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        rows.append(row)
    results = (pd.DataFrame(rows)
               .sort_values(f"{PRIMARY_METRIC}_mean", ascending=False)
               .reset_index(drop=True))
    return results, estimators


def _unwrap(est) -> Pipeline:
    """Return the fitted Pipeline from either a search object or a pipeline."""
    return est.best_estimator_ if isinstance(est, RandomizedSearchCV) else est


def fit_all(estimators: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    """Fit every candidate on the full training set and return plain pipelines."""
    return {name: _unwrap(est.fit(X_train, y_train)) for name, est in estimators.items()}


def select_and_fit(results: pd.DataFrame, estimators: dict,
                   X_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, Pipeline]:
    """Refit the top-ranked candidate on the full training set."""
    best_name = results.iloc[0]["model"]
    best = _unwrap(estimators[best_name].fit(X_train, y_train))
    return best_name, best


def chosen_hyperparameters(fitted: dict[str, Pipeline]) -> pd.DataFrame:
    """Tidy table of the model-step parameters that the searches selected."""
    rows = []
    for name, pipe in fitted.items():
        params = pipe.named_steps["model"].get_params()
        keys = [k.split("__")[-1] for k in search_spaces()[name]]
        rows.append({"model": name, **{k: params[k] for k in keys}})
    return pd.DataFrame(rows)
