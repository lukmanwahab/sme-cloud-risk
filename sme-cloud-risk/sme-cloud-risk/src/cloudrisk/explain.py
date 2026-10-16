"""
Model interpretation: SHAP values and logistic odds ratios.

Purpose : turn a fitted classifier into a ranked list of security control
          gaps. Two complementary views are produced:
          1. SHAP mean |value| per feature from the selected model, which
             captures non-linear and interaction effects.
          2. Odds ratios with 95% confidence intervals from the logistic
             baseline, which stakeholders without ML background can read
             directly ("firms without MFA have X times the odds").
          The control-gap table combines both with each control's adoption
          rate so that the recommendation reflects impact and prevalence.
Inputs  : fitted pipelines, hold-out X and y, feature names.
Outputs : DataFrames (shap_importance, odds_ratios, control_gaps) and PNGs.
Depends : shap, scikit-learn, pandas, numpy, matplotlib, scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.pipeline import Pipeline

from .config import CONTROLS, N_SHAP_BACKGROUND, SEED
from .features import feature_names


def _transformed(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Apply the fitted preprocessor and return a dense matrix with names."""
    prep = pipe.named_steps["prep"]
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return np.asarray(Xt, dtype=float), feature_names(prep)


def shap_values(pipe: Pipeline, X: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute SHAP values for the model step of a fitted pipeline.

    Tree models use the exact TreeExplainer. Linear models use the
    LinearExplainer. Anything else falls back to the model-agnostic
    permutation explainer on a background sample.
    """
    Xt, names = _transformed(pipe, X)
    model = pipe.named_steps["model"]
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(Xt), size=min(N_SHAP_BACKGROUND, len(Xt)), replace=False)
    background = Xt[bg_idx]

    model_type = type(model).__name__
    if model_type in {"RandomForestClassifier", "HistGradientBoostingClassifier"}:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        values = explainer.shap_values(Xt)
        # Binary tree explainers return either (n, p, 2) or a list of two arrays.
        if isinstance(values, list):
            values = values[1]
        elif values.ndim == 3:
            values = values[:, :, 1]
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(background, max_samples=len(background)))
        values = explainer.shap_values(Xt)
    else:
        explainer = shap.Explainer(model.predict_proba, background, seed=seed)
        values = explainer(Xt).values[:, :, 1]
    return np.asarray(values), Xt, names


def shap_importance(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, descending."""
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def odds_ratios(logit_pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Odds ratios with Wald 95% confidence intervals from the logistic pipeline.

    Standard errors come from the inverse observed information matrix of the
    unpenalised likelihood evaluated at the fitted (lightly penalised)
    coefficients. With C=1 and n in the thousands the penalty is negligible
    for interval width, and this avoids a second fitting library.
    """
    Xt, names = _transformed(logit_pipe, X_train)
    model = logit_pipe.named_steps["model"]
    beta = model.coef_.ravel()
    Xd = np.column_stack([np.ones(len(Xt)), Xt])        # add intercept column
    p = 1.0 / (1.0 + np.exp(-(Xd @ np.concatenate([model.intercept_, beta]))))
    W = p * (1 - p)
    info = (Xd * W[:, None]).T @ Xd                       # observed information
    cov = np.linalg.pinv(info)
    # Clip guards against tiny negative variances from pseudo-inverse rounding
    # when a feature is nearly collinear (e.g. an interaction with its parent).
    se = np.sqrt(np.clip(np.diag(cov), 0, None))[1:]
    z = stats.norm.ppf(0.975)
    out = pd.DataFrame({
        "feature": names,
        "coef": beta,
        "odds_ratio": np.exp(beta),
        "ci_low": np.exp(beta - z * se),
        "ci_high": np.exp(beta + z * se),
        "p_value": 2 * (1 - stats.norm.cdf(np.abs(np.divide(beta, se, out=np.full_like(beta, np.inf),
                                                               where=se > 0)))),
    })
    return out.sort_values("odds_ratio").reset_index(drop=True)


def control_gap_table(shap_imp: pd.DataFrame, ors: pd.DataFrame,
                      X: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Rank security controls by combined evidence of impact.

    For each surveyed control the table reports its adoption rate, SHAP
    importance from the selected model, and the logistic odds ratio. The
    "priority" score is SHAP importance scaled by the share of firms that
    lack the control, so a highly protective control that nearly everyone
    already has ranks below one that is protective and widely missing.
    """
    adoption = X[CONTROLS].mean().rename("adoption_rate")
    tbl = (shap_imp[shap_imp["feature"].isin(CONTROLS)]
           .merge(ors[["feature", "odds_ratio", "ci_low", "ci_high"]], on="feature", how="left")
           .merge(adoption, left_on="feature", right_index=True, how="left"))
    tbl["gap_rate"] = 1.0 - tbl["adoption_rate"]
    tbl["priority"] = tbl["mean_abs_shap"] * tbl["gap_rate"]
    tbl = tbl.sort_values("priority", ascending=False).reset_index(drop=True)
    tbl.insert(0, "rank", np.arange(1, len(tbl) + 1))
    return tbl.head(top_n) if top_n else tbl
