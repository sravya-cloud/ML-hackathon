"""Train the chosen model on ALL labeled training rows and export artifacts.

Writes to artifacts/:
  model.joblib           fitted estimator (scaler included where the model has one)
  feature_columns.json   exact feature order used at training time
  model_card.json        model name, config, seed, data sizes, student-test scores

Usage (from src/):
  python export_model.py --model HistGradientBoosting
  python export_model.py --model HistGradientBoosting --params-json ../results/best_params.json

`--params-json` applies tuned hyperparameters (e.g. from the Bayesian search) on
top of the default configuration. Keys may be plain estimator parameter names;
for pipeline models use the `clf__` prefix.

Training uses the train partition only. The evaluator-test partition is never
read here.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

from features import FEATURE_COLUMNS
from models import model_zoo
from evaluate import observation_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HistGradientBoosting")
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--artifacts", default="../artifacts")
    ap.add_argument("--params-json", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["training"]["random_seed"]
    zoo = model_zoo(seed)
    if args.model not in zoo:
        raise SystemExit(f"unknown model {args.model}; choose from {list(zoo)}")
    spec = zoo[args.model]
    est = clone(spec["estimator"])
    params = {}
    if args.params_json:
        params = json.load(open(args.params_json))
        est.set_params(**params)

    train = pd.read_parquet(os.path.join(args.results, "dataset_train.parquet"))
    train = train[train["labeled_obs"] == 1]
    X, y = train[FEATURE_COLUMNS].values, train["label"].values
    print(f"fitting {args.model} on {len(train)} rows / {train['g_obs'].nunique()} obs "
          f"/ {train['trip_id'].nunique()} trips")
    t0 = time.time()
    est.fit(X, y)
    fit_s = time.time() - t0
    print(f"fit done in {fit_s:.1f}s")

    # Student-test scores of the exported model (development partition only).
    test = pd.read_parquet(os.path.join(args.results, "dataset_student_test.parquet"))
    test = test[test["labeled_obs"] == 1].copy()
    t0 = time.time()
    test["proba"] = est.predict_proba(test[FEATURE_COLUMNS].values)[:, 1]
    infer_s = time.time() - t0
    m = observation_metrics(test[["g_obs", "label", "proba", "labeled_obs"]])
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    os.makedirs(args.artifacts, exist_ok=True)
    joblib.dump(est, os.path.join(args.artifacts, "model.joblib"), compress=3)
    json.dump(FEATURE_COLUMNS, open(os.path.join(args.artifacts, "feature_columns.json"), "w"))
    card = {
        "model": args.model, "family": spec["family"], "default_config": spec["config"],
        "tuned_params": params, "random_seed": seed,
        "train_rows": int(len(train)), "train_trips": int(train["trip_id"].nunique()),
        "fit_seconds": round(fit_s, 1),
        "student_test": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()},
        "student_test_infer_ms_per_obs": round(1000 * infer_s / max(m["n_obs"], 1), 4),
        "exported_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    json.dump(card, open(os.path.join(args.artifacts, "model_card.json"), "w"), indent=2)
    print("saved:", os.listdir(args.artifacts))


if __name__ == "__main__":
    main()
