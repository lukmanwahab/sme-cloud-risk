"""
Feature engineering: raw survey responses to a model-ready matrix.

Purpose : convert the raw survey frame into numeric features with explicit,
          leakage-safe handling of missing control responses, derived
          exposure measures and a one-hot sector encoding. Everything is
          expressed as a scikit-learn ColumnTransformer so the exact same
          transformation is fitted on the training folds and applied to the
          hold-out set, which is what prevents preprocessing leakage.
Inputs  : raw DataFrame from simulate.py (or the real survey with the same
          schema).
Outputs : (X, y) split, plus a fitted preprocessing pipeline factory.
Depends : pandas, numpy, scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONTROLS, TARGET

# Column groups. Keeping them explicit makes the ColumnTransformer auditable.
NUMERIC_RAW = ["employees", "remote_share", "it_staff", "cloud_maturity",
               "n_cloud_services"]
BINARY_RAW = ["byod_allowed", "prior_incident"]
CATEGORICAL = ["sector"]

# Derived features added before the ColumnTransformer runs.
DERIVED = ["log_employees", "it_staff_per_100", "controls_in_place",
           "controls_missing", "exposure_index", "mfa_x_byod"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features that encode domain knowledge.

    - log_employees      : size enters risk roughly log-linearly.
    - it_staff_per_100   : staffing intensity rather than raw headcount.
    - controls_in_place  : count of adopted controls (treats missing as 0 for
                           the count; the missing count is carried separately).
    - controls_missing   : number of unanswered control questions. Item
                           non-response is informative in security surveys,
                           so it is kept as a feature rather than hidden.
    - exposure_index     : remote share scaled by cloud service sprawl, a
                           simple proxy for attack surface.
    - mfa_x_byod         : interaction flagged by the domain literature; MFA
                           matters most where personal devices are allowed.
    """
    out = df.copy()
    out["log_employees"] = np.log(out["employees"])
    out["it_staff_per_100"] = 100.0 * out["it_staff"] / out["employees"]
    out["controls_in_place"] = out[CONTROLS].fillna(0).sum(axis=1)
    out["controls_missing"] = out[CONTROLS].isna().sum(axis=1)
    out["exposure_index"] = out["remote_share"] * out["n_cloud_services"]
    out["mfa_x_byod"] = out["mfa_enforced"].fillna(0) * out["byod_allowed"]
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target, dropping identifiers."""
    df = add_derived_features(df)
    X = df.drop(columns=[TARGET, "firm_id"])
    y = df[TARGET].astype(int)
    return X, y


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer.

    Parameters
    ----------
    scale_numeric : standardise numeric columns. Needed for logistic
                    regression; harmless for tree ensembles.

    Notes
    -----
    Control columns are imputed with the most frequent value inside each
    training fold. The number of missing responses was already captured in
    `controls_missing`, so imputation does not throw that signal away.
    """
    numeric_cols = NUMERIC_RAW + DERIVED
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_RAW + CONTROLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names from a fitted preprocessor."""
    return list(preprocessor.get_feature_names_out())
