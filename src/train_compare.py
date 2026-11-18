"""Train and compare the model zoo (hackathon requires >=5 models).

Protocol:
  * Load labeled (obs,candidate) rows from the train partition.
  * For a fair, fast comparison, fit on a group-aware OBSERVATION subsample
    (keeps all candidates of each sampled observation; keeps all 80 trips so
    GroupKFold is valid). The final selected model is later refit on 100% of
    the data (bayes_opt.py / export).
  * 5-fold GroupKFold by trip_id -> CV Macro-F1 (primary selection metric) and
    CV Log Loss, computed at the observation level.
  * Refit on the whole comparison sample, evaluate on the 10 student-test trips:
    Top-1 Accuracy, Macro-F1, Log Loss, Route Continuity, inference time.

Outputs: results/model_comparison.csv and results/cv_fold_scores.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from features import FEATURE_COLUMNS
from models import model_zoo
from evaluate import observation_metrics, route_continuity
from geo_utils import init_transformers
from data_loader import list_trips, load_trip


def subsample_observations(df, max_rows, seed):
    """Keep all candidate rows of a random subset of observations until ~max_rows."""
    if max_rows is None or len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    obs = df["g_obs"].drop_duplicates().values
    rng.shuffle(obs)
    sizes = df.groupby("g_obs").size()
    keep, total = [], 0
    for o in obs:
        keep.append(o); total += int(sizes[o])
        if total >= max_rows:
            break
    return df[df["g_obs"].isin(set(keep))].copy()


def fit_subsample(Xg, max_fit, seed):
    """Group-aware row subsample index for models that don't scale to full data."""
    if max_fit is None or len(Xg) <= max_fit:
        return np.arange(len(Xg))
    rng = np.random.default_rng(seed)
    return rng.choice(len(Xg), size=max_fit, replace=False)


def proba_of(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    s = est.decision_function(X)
    return 1.0 / (1.0 + np.exp(-s))


def cv_scores(name, spec, df, feat, folds, seed):
    gkf = GroupKFold(n_splits=folds)
    groups = df["trip_id"].values
    X = df[feat].values
    rows = []
    for k, (tr, va) in enumerate(gkf.split(X, groups=groups)):
        est = spec["estimator"]
        from sklearn.base import clone
        est = clone(est)
        tr_idx = tr
        if spec["subsample"]:
            sub = fit_subsample(tr, spec["subsample"], seed + k)
            tr_idx = tr[sub]
        t0 = time.time()
        est.fit(X[tr_idx], df["label"].values[tr_idx])
        p = proba_of(est, X[va])
        dv = df.iloc[va][["g_obs", "label", "labeled_obs"]].copy()
        dv["proba"] = p
        m = observation_metrics(dv)
        m.update(fold=k, model=name, fit_s=time.time() - t0)
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("PATHSNAP_DATA"))
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--max-rows", type=int, default=260_000,
                    help="comparison-stage observation subsample size")
    ap.add_argument("--models", default="",
                    help="comma-separated model names to run (default: all)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files, e.g. 'mohit'")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["training"]["random_seed"]
    folds = cfg["training"]["cv_folds"]
    data_root = args.data_root or cfg["data"]["root"]
    init_transformers()

    feat = FEATURE_COLUMNS
    train = pd.read_parquet(os.path.join(args.results, "dataset_train.parquet"))
    train = train[train["labeled_obs"] == 1].reset_index(drop=True)
    comp = subsample_observations(train, args.max_rows, seed).reset_index(drop=True)
    print(f"comparison sample: {len(comp)} rows / {comp['g_obs'].nunique()} obs / "
          f"{comp['trip_id'].nunique()} trips")

    test = pd.read_parquet(os.path.join(args.results, "dataset_student_test.parquet"))
    test_lab = test[test["labeled_obs"] == 1].reset_index(drop=True)

    # Student-test trips for route-continuity (needs network connectivity).
    st_trips = {os.path.basename(d): load_trip(d, "student_test")
                for d in list_trips(data_root, cfg["data"]["student_test_dir"])}

    zoo = model_zoo(seed)
    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in want if m not in zoo]
        if unknown:
            raise SystemExit(f"unknown models {unknown}; choose from {list(zoo)}")
        zoo = {m: zoo[m] for m in want}
    suffix = f"_{args.tag}" if args.tag else ""
    from sklearn.base import clone
    all_cv, summary = [], []
    for name, spec in zoo.items():
        try:
            print(f"\n=== {name} ({spec['family']}) ===", flush=True)
            cv = cv_scores(name, spec, comp, feat, folds, seed)
            all_cv.append(cv)
            cv_f1, cv_f1_std = cv["macro_f1"].mean(), cv["macro_f1"].std()
            cv_ll = cv["log_loss"].mean()
            print(f"  CV macro-F1={cv_f1:.4f}+/-{cv_f1_std:.4f}  CV logloss={cv_ll:.4f}")

            # Refit on full comparison sample, evaluate on student-test.
            est = clone(spec["estimator"])
            Xc = comp[feat].values
            idx = (fit_subsample(np.arange(len(comp)), spec["subsample"], seed)
                   if spec["subsample"] else np.arange(len(comp)))
            est.fit(Xc[idx], comp["label"].values[idx])

            t0 = time.time()
            p = proba_of(est, test_lab[feat].values)
            infer_s = time.time() - t0
            dv = test_lab[["g_obs", "trip_id", "obs_idx", "arc_id", "label",
                           "labeled_obs"]].copy()
            dv["proba"] = p
            m = observation_metrics(dv[["g_obs", "label", "proba", "labeled_obs"]])

            # Route continuity: top-1 arc per observation, in time order, per trip.
            rcs = []
            for tid, g in dv.groupby("trip_id"):
                sel = g.loc[g.groupby("obs_idx")["proba"].idxmax()]
                sel = sel.sort_values("obs_idx")
                rcs.append(route_continuity(sel["arc_id"].values, st_trips[tid]))
            rc = float(np.nanmean(rcs))

            summary.append(dict(
                model=name, family=spec["family"], config=spec["config"],
                cv_macro_f1=cv_f1, cv_macro_f1_std=cv_f1_std, cv_log_loss=cv_ll,
                st_top1_acc=m["top1_accuracy"], st_macro_f1=m["macro_f1"],
                st_log_loss=m["log_loss"], route_continuity=rc,
                candidate_miss_rate=m["candidate_miss_rate"],
                infer_s=infer_s, infer_ms_per_obs=1000 * infer_s / max(m["n_obs"], 1),
                subsample=spec["subsample"] or 0))
            print(f"  ST top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                  f"logloss={m['log_loss']:.4f} routeCont={rc:.4f} "
                  f"infer={1000*infer_s/max(m['n_obs'],1):.3f} ms/obs", flush=True)
        except Exception as e:
            print(f"  !! {name} failed: {e}", flush=True)
            summary.append(dict(model=name, family=spec.get("family"), error=str(e)))

    pd.concat(all_cv, ignore_index=True).to_csv(
        os.path.join(args.results, f"cv_fold_scores{suffix}.csv"), index=False)
    sm = pd.DataFrame(summary).sort_values("cv_macro_f1", ascending=False)
    sm.to_csv(os.path.join(args.results, f"model_comparison{suffix}.csv"), index=False)
    print("\n=== SUMMARY (sorted by CV macro-F1) ===")
    cols = ["model", "family", "cv_macro_f1", "st_top1_acc", "st_macro_f1",
            "st_log_loss", "route_continuity", "infer_ms_per_obs"]
    print(sm[[c for c in cols if c in sm.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
