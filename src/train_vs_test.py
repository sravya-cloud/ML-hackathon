"""Training vs student-test performance of the exported model (over/underfitting
evidence for the report) and the boosting loss history.

Reads artifacts/model.joblib, scores the TRAIN and STUDENT-TEST partitions with
the same observation-level metrics, and plots the per-iteration training and
validation loss that HistGradientBoosting records while fitting.

Outputs:
  results/train_vs_test.csv
  results/plots/hgb_loss_curve.png   (only for models with a boosting history)
  results/plots/train_vs_test.png

Usage (from src/):  python train_vs_test.py
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from evaluate import observation_metrics


def score(model, path):
    df = pd.read_parquet(path)
    df = df[df["labeled_obs"] == 1].copy()
    df["proba"] = model.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]
    return observation_metrics(df[["g_obs", "label", "proba", "labeled_obs"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="../artifacts")
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    plots = os.path.join(args.results, "plots")
    os.makedirs(plots, exist_ok=True)

    model = joblib.load(os.path.join(args.artifacts, "model.joblib"))
    card = json.load(open(os.path.join(args.artifacts, "model_card.json")))
    name = card.get("model", "model")

    rows = []
    for part in ["train", "student_test"]:
        m = score(model, os.path.join(args.results, f"dataset_{part}.parquet"))
        rows.append(dict(partition=part, **m))
        print(f"{part:13s} top1={m['top1_accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
              f"log_loss={m['log_loss']:.4f}  n_obs={m['n_obs']}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results, "train_vs_test.csv"), index=False)

    gap = out.iloc[0]["top1_accuracy"] - out.iloc[1]["top1_accuracy"]
    print(f"train - student_test Top-1 gap: {100 * gap:.2f} percentage points")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2); w = 0.38
    ax.bar(x - w / 2, out["top1_accuracy"], w, label="Top-1 accuracy", color="#2a6fbb")
    ax.bar(x + w / 2, out["macro_f1"], w, label="Macro-F1", color="#e08a2c")
    ax.set_xticks(x); ax.set_xticklabels(["train (80 trips)", "student-test (10 trips)"])
    ax.set_ylim(0.6, 1.0); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{name}: training vs student-test")
    fig.tight_layout(); fig.savefig(os.path.join(plots, "train_vs_test.png"), dpi=160)
    plt.close(fig)

    if hasattr(model, "train_score_"):
        tr = -np.asarray(model.train_score_)            # scores are negative loss
        va = getattr(model, "validation_score_", None)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(tr[1:], label="training loss")
        if va is not None and len(va) > 1:
            ax.plot(-np.asarray(va)[1:], label="validation loss")
        ax.set_xlabel("boosting iteration"); ax.set_ylabel("log loss")
        ax.set_title(f"{name}: loss per boosting iteration")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(plots, "hgb_loss_curve.png"), dpi=160)
        plt.close(fig)
        print(f"boosting iterations used: {model.n_iter_} "
              f"(final train loss {tr[-1]:.4f}"
              + (f", final validation loss {-np.asarray(va)[-1]:.4f})" if va is not None else ")"))
    else:
        print("model has no per-iteration history; loss curve skipped")


if __name__ == "__main__":
    main()
