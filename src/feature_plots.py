"""Feature-extraction plots (report 2.2): what the features look like and why
they were chosen. Uses the TRAIN partition only.

  results/plots/feature_distributions.png   true arc vs wrong arcs, key features
  results/plots/feature_correlation.png     correlation between all features
  results/plots/candidates_per_obs.png      number of candidate arcs per GPS point
  results/feature_summary.csv               mean/std per feature, by class

Usage (from src/):  python feature_plots.py --results ../results
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS

KEY = [("dist_m", "Distance GPS point to road (m)"),
       ("heading_diff", "Heading difference, vehicle vs road (deg)"),
       ("dist_minus_best", "Distance minus the best candidate's (m)"),
       ("speed_mps", "GPS speed (m/s)"),
       ("rank", "Rank of candidate by distance"),
       ("arc_len_m", "Road segment length (m, clipped at 300)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    plots = os.path.join(args.results, "plots")
    os.makedirs(plots, exist_ok=True)

    df = pd.read_parquet(os.path.join(args.results, "dataset_train.parquet"))
    df = df[df["labeled_obs"] == 1]
    sample = df.sample(n=min(200_000, len(df)), random_state=42)

    # 1. distributions: true arc vs wrong arcs
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, title) in zip(axes.ravel(), KEY):
        v = sample[col].clip(upper=300) if col == "arc_len_m" else sample[col]
        lo, hi = np.percentile(v, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40) if col != "rank" else np.arange(-0.5, 12.5, 1)
        for lab, name, c in [(0, "wrong road", "#9aa5b1"), (1, "true road", "#d62728")]:
            ax.hist(v[sample["label"] == lab], bins=bins, alpha=0.65, density=True,
                    label=name, color=c)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("density")
    axes[0, 0].legend()
    fig.suptitle("Feature distributions: true road vs wrong candidate roads (train, 200k-row sample)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "feature_distributions.png"), dpi=160)
    plt.close(fig)

    # 2. correlation heatmap
    corr = sample[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS))); ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=8)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation (train sample)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "feature_correlation.png"), dpi=160)
    plt.close(fig)

    # 3. candidates per observation
    n = df.groupby("g_obs")["arc_id"].size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(n, bins=np.arange(0.5, 13.5, 1), color="#2a6fbb")
    ax.set_xlabel("candidate roads per GPS point")
    ax.set_ylabel("GPS points")
    ax.set_title("Candidate set size (radius 45 m, capped at 12)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "candidates_per_obs.png"), dpi=160)
    plt.close(fig)

    # 4. summary table by class
    rows = []
    for c in FEATURE_COLUMNS:
        t, w = sample[sample["label"] == 1][c], sample[sample["label"] == 0][c]
        rows.append(dict(feature=c, true_mean=t.mean(), true_std=t.std(),
                         wrong_mean=w.mean(), wrong_std=w.std()))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(args.results, "feature_summary.csv"),
                                       index=False)

    hc = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
    hc = hc[hc.abs() > 0.95].sort_values(key=abs, ascending=False)
    print("saved 3 plots + feature_summary.csv")
    print("feature pairs with |corr| > 0.95 (redundant):")
    print(hc.round(3).to_string() if len(hc) else "none")


if __name__ == "__main__":
    main()
