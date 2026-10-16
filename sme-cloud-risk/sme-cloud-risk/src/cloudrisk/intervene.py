"""
Counterfactual intervention analysis: what should each firm fix first?

Purpose : a ranked list of control gaps at the population level tells a
          policy audience where to focus. An individual SME needs a
          different answer: given its own characteristics and the controls
          it already has, which single missing control would reduce its
          predicted incident probability the most? This module answers that
          by scoring counterfactual copies of each firm with one absent
          control switched on, using the selected model. It also aggregates
          the firm-level results into expected incidents averted per control
          across the portfolio, which is the number a funder cares about.
Inputs  : fitted pipeline, feature frame for the firms to analyse.
Outputs : firm-level recommendation table; portfolio-level impact table; PNG.
Depends : pandas, numpy, matplotlib, scikit-learn.

Caveat (stated in the model card as well): these are predictive
counterfactuals from an observational model, not causal estimates. Firms
that adopt a control differ from firms that do not, and the simulator's
confounding is only partly captured by the covariates. The ranking is a
prioritisation aid, not an effect size.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import CONTROLS, FIGURE_DIR
from .features import add_derived_features


def _score(pipe: Pipeline, X_raw: pd.DataFrame) -> np.ndarray:
    """Predict incident probability from the raw (pre-derived) feature frame."""
    return pipe.predict_proba(add_derived_features(X_raw))[:, 1]


def counterfactual_gains(pipe: Pipeline, X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    For every firm and every control it lacks, predicted risk reduction if
    that one control were adopted, holding everything else fixed.

    Returns a long table: firm_id, control, baseline_risk, counterfactual_risk,
    risk_reduction.
    """
    base = _score(pipe, X_raw)
    rows = []
    for control in CONTROLS:
        lacking = X_raw[control].fillna(0) == 0        # treat non-response as absent
        if not lacking.any():
            continue
        cf = X_raw.loc[lacking].copy()
        cf[control] = 1.0
        cf_risk = _score(pipe, cf)
        rows.append(pd.DataFrame({
            "firm_id": X_raw.loc[lacking, "firm_id"].values,
            "control": control,
            "baseline_risk": base[lacking.values],
            "counterfactual_risk": cf_risk,
        }))
    out = pd.concat(rows, ignore_index=True)
    out["risk_reduction"] = out["baseline_risk"] - out["counterfactual_risk"]
    return out


def firm_recommendations(gains: pd.DataFrame) -> pd.DataFrame:
    """Best single next control per firm, with the expected risk reduction."""
    idx = gains.groupby("firm_id")["risk_reduction"].idxmax()
    rec = gains.loc[idx, ["firm_id", "baseline_risk", "control", "counterfactual_risk", "risk_reduction"]]
    rec = rec.rename(columns={"control": "recommended_control"})
    return rec.sort_values("risk_reduction", ascending=False).reset_index(drop=True)


def portfolio_impact(gains: pd.DataFrame) -> pd.DataFrame:
    """
    Expected incidents averted per control if every firm lacking it adopted it.

    Summing predicted risk reductions over firms gives expected incidents
    averted; dividing by the number of firms gives the mean per-firm gain,
    which separates "big effect, few firms" from "small effect, many firms".
    """
    agg = (gains.groupby("control")
           .agg(firms_lacking=("firm_id", "nunique"),
                mean_risk_reduction=("risk_reduction", "mean"),
                expected_incidents_averted=("risk_reduction", "sum"))
           .sort_values("expected_incidents_averted", ascending=False)
           .reset_index())
    agg.insert(0, "rank", np.arange(1, len(agg) + 1))
    return agg


def plot_portfolio_impact(impact: pd.DataFrame) -> None:
    """Bar chart of expected incidents averted per control."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = impact.sort_values("expected_incidents_averted")
    ax.barh(d["control"], d["expected_incidents_averted"], color="#333333")
    for yk, (v, n) in enumerate(zip(d["expected_incidents_averted"], d["firms_lacking"])):
        ax.text(v + 0.2, yk, f"{n} firms", va="center", fontsize=8, color="#555555")
    ax.set(xlabel="Expected incidents averted if every lacking firm adopted the control",
           title="Counterfactual portfolio impact (hold-out firms)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "portfolio_impact.png", dpi=150)
    plt.close(fig)
