"""
Decision analysis: thresholds, net benefit and subgroup performance.

Purpose : a probability is only useful once someone decides what to do with
          it. This module (1) chooses an operating threshold from an explicit
          cost ratio between missed incidents and unnecessary interventions,
          (2) runs a decision-curve analysis showing the net benefit of acting
          on the model across the full range of risk thresholds versus the
          "treat everyone" and "treat no one" policies, and (3) audits
          discrimination and calibration within sectors and firm-size bands
          so that a model which works on average but fails for micro firms
          or one sector is caught before deployment.
Inputs  : hold-out labels, predicted probabilities, feature frame.
Outputs : threshold table, decision-curve DataFrame, subgroup DataFrame, PNG.
Depends : numpy, pandas, scikit-learn, matplotlib.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

from .config import FIGURE_DIR


def cost_sensitive_threshold(y: np.ndarray, p: np.ndarray,
                             cost_fn: float = 5.0, cost_fp: float = 1.0) -> pd.DataFrame:
    """
    Sweep thresholds and return expected cost per firm at each.

    cost_fn : cost of a missed incident (false negative), e.g. breach cost.
    cost_fp : cost of an unnecessary intervention (false positive), e.g. a
              security review. The ratio is what matters; 5:1 is a
              conservative placeholder a client would replace with its own.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "flagged_share": yhat.mean(),
            "expected_cost_per_firm": (cost_fn * fn + cost_fp * fp) / len(y),
        })
    tbl = pd.DataFrame(rows)
    tbl["optimal"] = tbl["expected_cost_per_firm"] == tbl["expected_cost_per_firm"].min()
    return tbl


def decision_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """
    Net benefit of the model versus treat-all and treat-none policies.

    Net benefit at threshold t (Vickers and Elkin, 2006):
        NB = TP/n - FP/n * t/(1-t)
    The threshold is interpreted as the risk level at which a decision-maker
    is indifferent between intervening and not, which encodes the cost ratio.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    prevalence = y.mean()
    rows = []
    for t in np.round(np.arange(0.05, 0.80, 0.01), 2):
        yhat = p >= t
        tp = np.sum(yhat & (y == 1))
        fp = np.sum(yhat & (y == 0))
        w = t / (1 - t)
        rows.append({
            "threshold": t,
            "net_benefit_model": tp / n - fp / n * w,
            "net_benefit_treat_all": prevalence - (1 - prevalence) * w,
            "net_benefit_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def plot_decision_curve(dc: pd.DataFrame, ct: pd.DataFrame) -> None:
    """Two panels: decision curve, and expected cost with the optimum marked."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.plot(dc["threshold"], dc["net_benefit_model"], color="#222222", label="model")
    ax1.plot(dc["threshold"], dc["net_benefit_treat_all"], color="#888888", ls="--", label="intervene everywhere")
    ax1.plot(dc["threshold"], dc["net_benefit_treat_none"], color="#888888", ls=":", label="intervene nowhere")
    ax1.set(xlabel="Risk threshold for intervention", ylabel="Net benefit",
            title="Decision curve (hold-out)", ylim=(-0.05, None))
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    opt = ct.loc[ct["optimal"]].iloc[0]
    ax2.plot(ct["threshold"], ct["expected_cost_per_firm"], color="#222222")
    ax2.axvline(opt["threshold"], color="#888888", ls="--")
    ax2.annotate(f"optimum {opt['threshold']:.2f}\nrecall {opt['recall']:.2f}, flags {opt['flagged_share']:.0%}",
                 xy=(opt["threshold"], opt["expected_cost_per_firm"]),
                 xytext=(opt["threshold"] + 0.15, opt["expected_cost_per_firm"] * 1.15), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#555555"))
    ax2.set(xlabel="Threshold", ylabel="Expected cost per firm (FN=5, FP=1)",
            title="Cost-sensitive threshold")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "decision_curve.png", dpi=150)
    plt.close(fig)


def subgroup_performance(X: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """
    AUC, Brier and calibration-in-the-large by sector and firm-size band.

    calibration_gap = mean predicted risk minus observed rate. A large gap in
    one subgroup means the dashboard would systematically over- or
    under-state that group's risk even if the overall calibration is fine.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    size_band = pd.cut(X["employees"], [0, 9, 49, 249], labels=["micro (1-9)", "small (10-49)", "medium (50-249)"])
    groups = {"sector": X["sector"].values, "size_band": size_band.astype(str).values}
    rows = []
    for dim, labels in groups.items():
        for g in pd.unique(labels):
            m = labels == g
            if m.sum() < 30 or y[m].min() == y[m].max():
                continue
            rows.append({
                "dimension": dim, "group": g, "n": int(m.sum()),
                "incident_rate": y[m].mean(),
                "roc_auc": roc_auc_score(y[m], p[m]),
                "brier": brier_score_loss(y[m], p[m]),
                "calibration_gap": p[m].mean() - y[m].mean(),
            })
    return pd.DataFrame(rows).sort_values(["dimension", "roc_auc"]).reset_index(drop=True)
