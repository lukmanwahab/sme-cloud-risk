"""
Hold-out evaluation, calibration and dashboard exports.

Purpose : score the selected model once on the untouched hold-out set,
          produce the standard discrimination and calibration diagnostics,
          and write the tidy CSVs that feed a Power BI / Tableau dashboard
          (firm-level risk scores, model comparison, control-gap ranking).
Inputs  : fitted pipelines, hold-out X and y, CV results, explanation tables.
Outputs : metrics dict, PNG figures under reports/figures, CSVs under reports.
Depends : scikit-learn, matplotlib, pandas, numpy.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")                          # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline

from .config import FIGURE_DIR, REPORT_DIR


def holdout_metrics(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                    threshold: float = 0.5) -> dict:
    """Discrimination, calibration and confusion metrics at one threshold."""
    p = pipe.predict_proba(X_test)[:, 1]
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_test, p)),
        "average_precision": float(average_precision_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "threshold": threshold,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "n_test": int(len(y_test)),
        "incident_rate_test": float(y_test.mean()),
    }


def plot_roc_pr(fitted: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """ROC and precision-recall curves for every candidate on the hold-out set."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, pipe in fitted.items():
        p = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
        ax2.plot(rec, prec, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (hold-out)")
    ax2.axhline(y_test.mean(), color="k", ls="--", lw=0.8, label="baseline")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (hold-out)")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_calibration(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> None:
    """Reliability diagram for the selected model."""
    p = pipe.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed incident rate",
           title="Calibration (hold-out, decile bins)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration.png", dpi=150)
    plt.close(fig)


def plot_shap_summary(values: np.ndarray, Xt: np.ndarray, names: list[str], top: int = 15) -> None:
    """Horizontal bar chart of mean |SHAP| for the top features."""
    imp = np.abs(values).mean(axis=0)
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order][::-1], imp[order][::-1], color="#444444")
    ax.set(xlabel="Mean |SHAP value|", title="Feature importance (selected model)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)


def plot_control_gaps(gaps: pd.DataFrame) -> None:
    """Odds ratios with CIs for the ranked control gaps."""
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(gaps))[::-1]
    ax.errorbar(gaps["odds_ratio"], y,
                xerr=[gaps["odds_ratio"] - gaps["ci_low"], gaps["ci_high"] - gaps["odds_ratio"]],
                fmt="o", color="#222222", capsize=3)
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f}  ({a:.0%} adopted)" for f, a in zip(gaps["feature"], gaps["adoption_rate"])])
    ax.set(xlabel="Odds ratio of incident when control is in place (95% CI)",
           title="Priority control gaps")
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "control_gaps.png", dpi=150)
    plt.close(fig)


def export_dashboard_tables(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                            firm_ids: pd.Series, cv_results: pd.DataFrame,
                            gaps: pd.DataFrame, metrics: dict, best_name: str) -> None:
    """Write tidy CSV/JSON outputs that a BI tool can consume directly."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = pipe.predict_proba(X_test)[:, 1]
    scores = pd.DataFrame({
        "firm_id": firm_ids.values,
        "sector": X_test["sector"].values,
        "employees": X_test["employees"].values,
        "risk_score": p.round(4),
        "risk_band": pd.cut(p, [0, 0.2, 0.5, 1.0], labels=["low", "medium", "high"], include_lowest=True),
        "actual_incident": y_test.values,
    })
    scores.to_csv(REPORT_DIR / "firm_risk_scores.csv", index=False)
    cv_results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    gaps.to_csv(REPORT_DIR / "control_gap_ranking.csv", index=False)
    with open(REPORT_DIR / "holdout_metrics.json", "w") as fh:
        json.dump({"selected_model": best_name, **metrics}, fh, indent=2)
