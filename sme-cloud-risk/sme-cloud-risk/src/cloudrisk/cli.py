"""
Command-line entry point for the cloudrisk pipeline.

Purpose : run the full workflow end to end with one command, or score new
          firms with a saved model.
              cloudrisk run        simulate/load -> tune -> explain -> decide -> evaluate
              cloudrisk simulate   write the synthetic survey only
              cloudrisk score      batch-score a CSV with the saved model and
                                   attach per-firm control recommendations
          Each stage is a plain function so notebooks and tests can call them
          individually.
Inputs  : CLI flags (number of firms, seed, data path, tuning switch).
Outputs : models/, reports/ and reports/figures/ populated; summary printed.
Depends : the other cloudrisk modules, joblib.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from . import (config, decision, evaluate, explain, features, intervene, simulate, train,
               uncertainty, validate)


def _fmt(df: pd.DataFrame, cols=None) -> str:
    return (df if cols is None else df[cols]).to_string(index=False, float_format=lambda v: f"{v:.3f}")


def run_pipeline(n_firms: int = 3000, seed: int = config.SEED, data_path: Path | None = None,
                 tune: bool = True, n_boot: int = 1000, verbose: bool = True) -> dict:
    """Execute every stage and return a summary dictionary."""
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    # ------------------------------------------------------------ 1. Data --
    if data_path is None:
        log(f"[1/7] Simulating survey for {n_firms:,} SMEs (seed={seed})")
        df = simulate.write_dataset(n_firms=n_firms, seed=seed)
    else:
        log(f"[1/7] Loading and validating survey from {data_path}")
        df = validate.validate_survey(pd.read_csv(data_path))
    log(f"      incident rate = {df[config.TARGET].mean():.3f}, "
        f"missing control responses = {df[config.CONTROLS].isna().mean().mean():.1%}")

    # --------------------------------------------------------- 2. Split ----
    X, y = features.split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE,
                                              stratify=y, random_state=seed)
    raw_te = df.loc[X_te.index]
    log(f"[2/7] Stratified split: train={len(X_tr):,}  hold-out={len(X_te):,}")

    # ----------------------------------------------------- 3. Benchmark ----
    log(f"[3/7] {'Nested ' if tune else ''}{config.N_FOLDS}-fold stratified CV on training set"
        + (f" (inner {train.INNER_FOLDS}-fold randomised search, {train.N_ITER} draws)" if tune else ""))
    cv_results, estimators = train.benchmark(X_tr, y_tr, seed=seed, tune=tune)
    log(_fmt(cv_results, ["model", "roc_auc_mean", "roc_auc_std", "average_precision_mean", "brier_mean"]))
    fitted = train.fit_all(estimators, X_tr, y_tr)
    best_name = cv_results.iloc[0]["model"]
    best = fitted[best_name]
    hparams = train.chosen_hyperparameters(fitted)
    log(f"      selected: {best_name}")

    # ------------------------------------------------------- 4. Explain ----
    log("[4/7] SHAP values, odds ratios and control-gap ranking")
    sv, Xt, names = explain.shap_values(best, X_te, seed=seed)
    shap_imp = explain.shap_importance(sv, names)
    ors = explain.odds_ratios(fitted["logistic_regression"], X_tr, y_tr)
    gaps = explain.control_gap_table(shap_imp, ors, X, top_n=5)
    log(_fmt(gaps, ["rank", "feature", "adoption_rate", "odds_ratio", "ci_low", "ci_high"]))

    # --------------------------------------------- 5. Hold-out + bootstrap --
    log(f"[5/7] Hold-out evaluation with {n_boot} bootstrap resamples")
    probs = {name: pipe.predict_proba(X_te)[:, 1] for name, pipe in fitted.items()}
    metrics = evaluate.holdout_metrics(best, X_te, y_te)
    boot = uncertainty.bootstrap_metrics(y_te, probs[best_name], n_boot=n_boot, seed=seed)
    paired = uncertainty.paired_auc_difference(
        y_te, probs[best_name], {k: v for k, v in probs.items() if k != best_name}, n_boot=n_boot, seed=seed)
    log(_fmt(boot))
    log(_fmt(paired))

    # ------------------------------------------------------- 6. Decide ----
    log("[6/7] Threshold selection, decision curve, subgroup audit")
    ct = decision.cost_sensitive_threshold(y_te, probs[best_name])
    dc = decision.decision_curve(y_te, probs[best_name])
    sub = decision.subgroup_performance(X_te, y_te, probs[best_name])
    opt = ct.loc[ct["optimal"]].iloc[0]
    log(f"      cost-optimal threshold {opt['threshold']:.2f}: recall {opt['recall']:.2f}, "
        f"precision {opt['precision']:.2f}, flags {opt['flagged_share']:.0%} of firms")
    log(_fmt(sub, ["dimension", "group", "n", "roc_auc", "brier", "calibration_gap"]))

    # --------------------------------------------- 7. Counterfactuals ------
    log("[7/7] Counterfactual interventions and exports")
    gains = intervene.counterfactual_gains(best, raw_te)
    recs = intervene.firm_recommendations(gains)
    impact = intervene.portfolio_impact(gains)
    log(_fmt(impact.head(5)))

    # --------------------------------------------------------- Exports -----
    evaluate.plot_roc_pr(fitted, X_te, y_te)
    evaluate.plot_calibration(best, X_te, y_te, best_name)
    evaluate.plot_shap_summary(sv, Xt, names)
    evaluate.plot_control_gaps(gaps)
    decision.plot_decision_curve(dc, ct)
    intervene.plot_portfolio_impact(impact)
    metrics["cost_optimal_threshold"] = float(opt["threshold"])
    evaluate.export_dashboard_tables(best, X_te, y_te, raw_te["firm_id"], cv_results, gaps, metrics, best_name)
    boot.to_csv(config.REPORT_DIR / "holdout_bootstrap_ci.csv", index=False)
    paired.to_csv(config.REPORT_DIR / "paired_auc_differences.csv", index=False)
    ct.to_csv(config.REPORT_DIR / "threshold_analysis.csv", index=False)
    sub.to_csv(config.REPORT_DIR / "subgroup_performance.csv", index=False)
    recs.to_csv(config.REPORT_DIR / "firm_recommendations.csv", index=False)
    impact.to_csv(config.REPORT_DIR / "portfolio_impact.csv", index=False)
    hparams.to_csv(config.REPORT_DIR / "chosen_hyperparameters.csv", index=False)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, config.MODEL_DIR / "selected_model.joblib")
    with open(config.MODEL_DIR / "selected_model.json", "w") as fh:
        json.dump({"model": best_name, "seed": seed, "threshold": float(opt["threshold"])}, fh, indent=2)
    log(f"      hold-out ROC-AUC={metrics['roc_auc']:.3f}  AP={metrics['average_precision']:.3f}  "
        f"Brier={metrics['brier']:.3f}  ({time.time() - t0:.0f}s)")

    return {"selected_model": best_name, "cv_results": cv_results, "holdout": metrics,
            "bootstrap": boot, "control_gaps": gaps, "portfolio_impact": impact}


def score_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Score new firms with the saved model and attach the best next control."""
    model = joblib.load(config.MODEL_DIR / "selected_model.joblib")
    with open(config.MODEL_DIR / "selected_model.json") as fh:
        meta = json.load(fh)
    df = validate.validate_survey(pd.read_csv(input_path), require_target=False)
    gains = intervene.counterfactual_gains(model, df)
    recs = intervene.firm_recommendations(gains).set_index("firm_id")
    out = df[["firm_id", "sector", "employees"]].copy()
    out["risk_score"] = model.predict_proba(features.add_derived_features(df))[:, 1].round(4)
    out["flag"] = (out["risk_score"] >= meta["threshold"]).astype(int)
    out = out.join(recs[["recommended_control", "risk_reduction"]], on="firm_id")
    out.to_csv(output_path, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cloudrisk",
                                     description="Interpretable cloud-security risk model for SMEs")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline")
    run.add_argument("--n-firms", type=int, default=3000)
    run.add_argument("--seed", type=int, default=config.SEED)
    run.add_argument("--data", type=str, default=None, help="CSV with the documented schema")
    run.add_argument("--no-tune", action="store_true", help="skip nested hyperparameter search")
    run.add_argument("--n-boot", type=int, default=1000)

    sim = sub.add_parser("simulate", help="write the synthetic survey only")
    sim.add_argument("--n-firms", type=int, default=3000)
    sim.add_argument("--seed", type=int, default=config.SEED)

    sc = sub.add_parser("score", help="batch-score a CSV with the saved model")
    sc.add_argument("--input", required=True)
    sc.add_argument("--output", default="scored_firms.csv")

    args = parser.parse_args(argv)
    if args.command == "simulate":
        df = simulate.write_dataset(n_firms=args.n_firms, seed=args.seed)
        print(f"Wrote {len(df):,} rows to {config.RAW_DATA_PATH}")
    elif args.command == "score":
        out = score_file(Path(args.input), Path(args.output))
        print(f"Scored {len(out):,} firms -> {args.output}; {out['flag'].mean():.0%} flagged")
    else:
        run_pipeline(args.n_firms, args.seed, Path(args.data) if args.data else None,
                     tune=not args.no_tune, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
