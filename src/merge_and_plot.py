"""Merge per-person model results into one comparison table and make plots.

Reads every results/model_comparison_<tag>.csv (one per team member), keeps one
row per model, writes:
  results/model_comparison_all.csv     consolidated table (report 3.1)
  results/model_comparison_all.md      same table as markdown
  results/plots/cv_macro_f1.png        CV Macro-F1 with fold std (error bars)
  results/plots/student_test_top1_f1.png
  results/plots/log_loss.png
  results/plots/route_continuity.png
  results/plots/inference_time.png
  results/plots/cv_fold_boxplot.png    per-fold CV Macro-F1 spread

Usage (from src/):
  python merge_and_plot.py --results ../results
If two files contain the same model, the file that sorts last wins.
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "model_comparison_*.csv")))
    files = [f for f in files if not f.endswith("_all.csv")]
    if not files:
        raise SystemExit("no model_comparison_<tag>.csv files found")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = os.path.basename(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        bad = df[df["error"].notna()]
        if len(bad):
            print("WARNING: failed models ignored:", bad["model"].tolist())
        df = df[df["error"].isna()]
    df = df.drop_duplicates(subset="model", keep="last")
    return df.sort_values("cv_macro_f1", ascending=False).reset_index(drop=True), files


def load_folds(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "cv_fold_scores_*.csv")))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d.drop_duplicates(subset=["model", "fold"], keep="last")


def bar(df, col, title, ylabel, path, err=None, lower_better=False, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    colors = ["#2a6fbb" if i == (df[col].idxmin() if lower_better else df[col].idxmax())
              else "#9db8d9" for i in df.index]
    ax.bar(x, df[col], yerr=err, color=colors, capsize=4)
    for xi, v in zip(x, df[col]):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    lo, hi = df[col].min(), df[col].max()
    if not lower_better and lo > 0.3:
        ax.set_ylim(max(0, lo - 0.05), min(1.0, hi + 0.03))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def grouped_top1_f1(df, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df["st_top1_acc"], w, label="Top-1 accuracy", color="#2a6fbb")
    ax.bar(x + w / 2, df["st_macro_f1"], w, label="Macro-F1", color="#e08a2c")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right")
    ax.set_ylim(max(0, min(df["st_top1_acc"].min(), df["st_macro_f1"].min()) - 0.05), 1.0)
    ax.set_title("Student-test (10 trips): Top-1 accuracy and Macro-F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fold_boxplot(folds, order, path):
    data = [folds[folds["model"] == m]["macro_f1"].values for m in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel("Macro-F1 per CV fold")
    ax.set_title("Stability across the 5 trip-grouped CV folds")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    plots = os.path.join(args.results, "plots")
    os.makedirs(plots, exist_ok=True)

    df, files = load_all(args.results)
    print("merged files:", [os.path.basename(f) for f in files])

    cols = ["model", "family", "config", "cv_macro_f1", "cv_macro_f1_std", "cv_log_loss",
            "st_top1_acc", "st_macro_f1", "st_log_loss", "route_continuity",
            "candidate_miss_rate", "infer_ms_per_obs"]
    out = df[[c for c in cols if c in df.columns]]
    out.to_csv(os.path.join(args.results, "model_comparison_all.csv"), index=False)
    with open(os.path.join(args.results, "model_comparison_all.md"), "w") as fh:
        fh.write(out.round(4).to_markdown(index=False))
    print(out.round(4).to_string(index=False))

    bar(df, "cv_macro_f1", "5-fold trip-grouped CV: Macro-F1 (mean, error = std over folds)",
        "Macro-F1", os.path.join(plots, "cv_macro_f1.png"), err=df["cv_macro_f1_std"])
    grouped_top1_f1(df, os.path.join(plots, "student_test_top1_f1.png"))
    bar(df, "st_log_loss", "Student-test Log Loss (lower is better)", "Log Loss",
        os.path.join(plots, "log_loss.png"), lower_better=True)
    bar(df, "route_continuity", "Student-test route continuity (classifier only)",
        "Fraction of connected consecutive arcs", os.path.join(plots, "route_continuity.png"))
    bar(df, "infer_ms_per_obs", "Inference time per GPS point (lower is better)",
        "ms per observation", os.path.join(plots, "inference_time.png"),
        lower_better=True, fmt="{:.3f}")

    folds = load_folds(args.results)
    if folds is not None:
        order = [m for m in df["model"] if m in set(folds["model"])]
        fold_boxplot(folds, order, os.path.join(plots, "cv_fold_boxplot.png"))
    print("plots written to", plots)


if __name__ == "__main__":
    main()
