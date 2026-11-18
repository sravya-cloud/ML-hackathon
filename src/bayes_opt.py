"""Bayesian hyperparameter optimisation (scikit-optimize, Gaussian-process
surrogate with Expected Improvement) for the selected model, then a like-for-like
comparison of untuned vs tuned.

Search
    * model            : HistGradientBoosting (selected from the model comparison)
    * objective        : mean Macro-F1 over trip-grouped CV folds (maximised)
    * CV               : GroupKFold by trip_id on the TRAIN partition (80 trips)
    * search sample    : observation-level subsample (--bo-rows) to keep trials cheap
Search space
    learning_rate      log-uniform  [0.02, 0.30]
    max_iter           integer      [100, 600]
    max_leaf_nodes     integer      [15, 127]
    min_samples_leaf   log-uniform  [20, 400]
    l2_regularization  log-uniform  [1e-3, 10]
    max_features       uniform      [0.5, 1.0]

Comparison (same sample, same folds, same metrics for both models)
    CV Macro-F1, student-test Top-1 / Macro-F1 / Log Loss, inference time.

Outputs
    results/bayes_trials.csv, results/best_params.json,
    results/bayes_comparison.csv, results/plots/bayes_convergence.png,
    results/plots/bayes_tuned_vs_untuned.png

Usage (from src/):  python bayes_opt.py --data-root "../data/Student folder"
The evaluator-test partition is never read.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from skopt import gp_minimize
from skopt.space import Integer, Real

from data_loader import list_trips, load_trip
from evaluate import observation_metrics, route_continuity
from features import FEATURE_COLUMNS
from models import model_zoo
from train_compare import cv_scores, subsample_observations

SPACE = [
    Real(0.02, 0.30, prior="log-uniform", name="learning_rate"),
    Integer(100, 600, name="max_iter"),
    Integer(15, 127, name="max_leaf_nodes"),
    Integer(20, 400, name="min_samples_leaf"),
    Real(1e-3, 10.0, prior="log-uniform", name="l2_regularization"),
    Real(0.5, 1.0, name="max_features"),
]
NAMES = [d.name for d in SPACE]


def make_est(params, seed):
    p = dict(params)
    return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **p)


def to_params(x):
    out = {}
    for n, v in zip(NAMES, x):
        out[n] = int(v) if n in ("max_iter", "max_leaf_nodes", "min_samples_leaf") else float(v)
    return out


def group_cv_f1(params, df, folds, seed):
    gkf = GroupKFold(n_splits=folds)
    X, y, groups = df[FEATURE_COLUMNS].values, df["label"].values, df["trip_id"].values
    scores = []
    for tr, va in gkf.split(X, groups=groups):
        est = make_est(params, seed)
        est.fit(X[tr], y[tr])
        dv = df.iloc[va][["g_obs", "label", "labeled_obs"]].copy()
        dv["proba"] = est.predict_proba(X[va])[:, 1]
        scores.append(observation_metrics(dv)["macro_f1"])
    return float(np.mean(scores))


def student_test_eval(est, comp, test_lab, st_trips):
    t0 = time.time()
    est.fit(comp[FEATURE_COLUMNS].values, comp["label"].values)
    fit_s = time.time() - t0
    t0 = time.time()
    p = est.predict_proba(test_lab[FEATURE_COLUMNS].values)[:, 1]
    infer_s = time.time() - t0
    dv = test_lab[["g_obs", "trip_id", "obs_idx", "arc_id", "label", "labeled_obs"]].copy()
    dv["proba"] = p
    m = observation_metrics(dv[["g_obs", "label", "proba", "labeled_obs"]])
    rcs = []
    for tid, g in dv.groupby("trip_id"):
        sel = g.loc[g.groupby("obs_idx")["proba"].idxmax()].sort_values("obs_idx")
        rcs.append(route_continuity(sel["arc_id"].values, st_trips[tid]))
    m.update(route_continuity=float(np.nanmean(rcs)), fit_s=fit_s,
             infer_ms_per_obs=1000 * infer_s / max(m["n_obs"], 1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("PATHSNAP_DATA"))
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--bo-rows", type=int, default=120_000)
    ap.add_argument("--bo-folds", type=int, default=3)
    ap.add_argument("--n-calls", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=260_000,
                    help="comparison sample size (same as train_compare.py)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["training"]["random_seed"]
    n_calls = args.n_calls or cfg["bayes_opt"]["n_calls"]
    data_root = args.data_root or cfg["data"]["root"]

    train = pd.read_parquet(os.path.join(args.results, "dataset_train.parquet"))
    train = train[train["labeled_obs"] == 1].reset_index(drop=True)
    search = subsample_observations(train, args.bo_rows, seed + 1).reset_index(drop=True)
    print(f"search sample: {len(search)} rows / {search['trip_id'].nunique()} trips, "
          f"{args.bo_folds}-fold group CV, {n_calls} trials", flush=True)

    trials = []

    def objective(x):
        params = to_params(x)
        t0 = time.time()
        f1 = group_cv_f1(params, search, args.bo_folds, seed)
        trials.append(dict(trial=len(trials) + 1, cv_macro_f1=f1,
                           seconds=round(time.time() - t0, 1), **params))
        best = max(t["cv_macro_f1"] for t in trials)
        print(f"trial {len(trials):2d}  cv_macro_f1={f1:.4f}  best={best:.4f}  "
              f"({trials[-1]['seconds']}s)", flush=True)
        return -f1

    t0 = time.time()
    res = gp_minimize(objective, SPACE, n_calls=n_calls,
                      n_initial_points=min(10, n_calls), acq_func="EI",
                      random_state=seed)
    search_s = time.time() - t0
    best = to_params(res.x)
    print("\nbest params:", best, f"\nbest search CV macro-F1: {-res.fun:.4f}  "
          f"({search_s / 60:.1f} min)", flush=True)

    tdf = pd.DataFrame(trials)
    tdf.to_csv(os.path.join(args.results, "bayes_trials.csv"), index=False)
    json.dump(dict(best, early_stopping=False),
              open(os.path.join(args.results, "best_params.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(tdf["trial"], tdf["cv_macro_f1"], "o", alpha=0.6, label="trial")
    ax.plot(tdf["trial"], tdf["cv_macro_f1"].cummax(), "-", color="#d62728", label="best so far")
    ax.axvline(min(10, n_calls) + 0.5, color="grey", ls=":", label="random start | GP-guided")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Macro-F1 (search sample)")
    ax.set_title("Bayesian optimisation convergence"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "plots", "bayes_convergence.png"), dpi=160)
    plt.close(fig)

    # ---- untuned vs tuned on the SAME comparison sample, folds and metrics ----
    comp = subsample_observations(train, args.max_rows, seed).reset_index(drop=True)
    test = pd.read_parquet(os.path.join(args.results, "dataset_student_test.parquet"))
    test_lab = test[test["labeled_obs"] == 1].reset_index(drop=True)
    st_trips = {os.path.basename(d): load_trip(d, "student_test")
                for d in list_trips(data_root, cfg["data"]["student_test_dir"])}
    base = model_zoo(seed)["HistGradientBoosting"]["estimator"]
    tuned = make_est(best, seed)

    rows = []
    for name, est in [("untuned (default)", base), ("tuned (Bayesian)", tuned)]:
        cv = cv_scores(name, dict(estimator=est, subsample=None), comp, FEATURE_COLUMNS,
                       cfg["training"]["cv_folds"], seed)
        m = student_test_eval(clone(est), comp, test_lab, st_trips)
        rows.append(dict(model=name, cv_macro_f1=cv["macro_f1"].mean(),
                         cv_macro_f1_std=cv["macro_f1"].std(),
                         cv_log_loss=cv["log_loss"].mean(),
                         st_top1_acc=m["top1_accuracy"], st_macro_f1=m["macro_f1"],
                         st_log_loss=m["log_loss"], route_continuity=m["route_continuity"],
                         fit_seconds=m["fit_s"], infer_ms_per_obs=m["infer_ms_per_obs"]))
        print(rows[-1], flush=True)
    comp_df = pd.DataFrame(rows)
    comp_df["search_seconds"] = [0.0, round(search_s, 1)]
    comp_df.to_csv(os.path.join(args.results, "bayes_comparison.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, col, ttl in zip(axes, ["cv_macro_f1", "st_top1_acc", "st_log_loss"],
                            ["CV Macro-F1", "Student-test Top-1", "Student-test Log Loss"]):
        ax.bar(comp_df["model"], comp_df[col], color=["#9db8d9", "#2a6fbb"])
        for i, v in enumerate(comp_df[col]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        lo, hi = comp_df[col].min(), comp_df[col].max()
        ax.set_ylim(lo - (hi - lo) * 2 - 0.002, hi + (hi - lo) * 1.5 + 0.002)
        ax.set_title(ttl); ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "plots", "bayes_tuned_vs_untuned.png"), dpi=160)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
