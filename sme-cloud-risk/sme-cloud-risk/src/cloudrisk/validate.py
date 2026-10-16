"""
Input schema validation for survey data supplied with --data.

Purpose : fail fast with a readable message when a real survey file does not
          match the schema the pipeline expects, instead of producing a
          cryptic scikit-learn error three stages later. Checks column
          presence, types, value ranges, the binary coding of controls, and
          that the target is present and non-degenerate.
Inputs  : a DataFrame.
Outputs : the same DataFrame (returned for chaining) or a SchemaError.
Depends : pandas, numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONTROLS, TARGET
from .simulate import COLUMN_DESCRIPTIONS, SECTORS


class SchemaError(ValueError):
    """Raised when the survey frame violates the documented schema."""


def validate_survey(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    problems: list[str] = []

    expected = [c for c in COLUMN_DESCRIPTIONS if require_target or c != TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        raise SchemaError("; ".join(problems))     # nothing else is checkable

    if df["firm_id"].duplicated().any():
        problems.append("firm_id contains duplicates")

    bad_sector = set(df["sector"].dropna().unique()) - set(SECTORS)
    if bad_sector:
        problems.append(f"unknown sector values: {sorted(bad_sector)}")

    if not df["employees"].between(1, 249).all():
        problems.append("employees must be in [1, 249] (UK SME definition)")
    if not df["remote_share"].between(0, 1).all():
        problems.append("remote_share must be in [0, 1]")
    for col in ["it_staff", "cloud_maturity", "n_cloud_services"]:
        if (df[col] < 0).any():
            problems.append(f"{col} must be non-negative")

    binary_cols = ["byod_allowed", "prior_incident"] + CONTROLS + ([TARGET] if require_target else [])
    for col in binary_cols:
        vals = set(df[col].dropna().unique())
        if not vals <= {0, 1, 0.0, 1.0}:
            problems.append(f"{col} must be coded 0/1, found {sorted(vals)[:5]}")

    non_control = [c for c in expected if c not in CONTROLS]
    if df[non_control].isna().any().any():
        cols = df[non_control].columns[df[non_control].isna().any()].tolist()
        problems.append(f"missing values allowed only in control columns, found in {cols}")

    if require_target:
        rate = df[TARGET].mean()
        if not 0.02 < rate < 0.98:
            problems.append(f"target prevalence {rate:.3f} is degenerate")

    if problems:
        raise SchemaError("survey schema violations: " + "; ".join(problems))
    return df
