"""Diagnostic plots for the model comparison (report 3.3).

  * confusion matrix per model (student-test, one chosen candidate per GPS point)
  * MLP training/validation loss curve (the only model with an epoch history)
  * permutation feature importance for one model

Everything is computed from real fits on the same comparison sample used in
train_compare.py. Usage (from src/):
  python diagnostics.py --models LogisticRegression,RandomForest,HistGradientBoosting
  python diagnostics.py --models MLP --mlp-curve
  python diagnostics.py --models HistGradientBoosting --importance
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

from features import FEATURE_COLUMNS
from models import model_zoo
from train_compare import subsample_observations, fit_subsample, proba_of


def top1_binary(df):
    """Binary prediction vector: 1 for the single top-scored candidate per obs."""
    idx = df.groupby("g_obs")["proba"].idxmax()
    pred = pd.Series(0, index=df.index)
    pred.loc[idx] = 1
    return pred.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--max-rows", type=int, default=260_000)
    ap.add_argument("--mlp-curve", action="store_true")
    ap.add_argument("--importance", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["training"]["random_seed"]
    plots = os.path.join(args.results, "plots")
    os.makedirs(plots, exist_ok=True)

    train = pd.read_parquet(os.path.join(args.results, "dataset_train.parquet"))
    train = train[train["labeled_obs"] == 1].reset_index(drop=True)
    comp = subsample_observations(train, args.max_rows, seed).reset_index(drop=True)
    test = pd.read_parquet(os.path.join(args.results, "dataset_student_test.parquet"))
    test = test[test["labeled_obs"] == 1].reset_index(drop=True)

    zoo = model_zoo(seed)
    for name in [m.strip() for m in args.models.split(",")]:
        spec = zoo[name]
        est = clone(spec["estimator"])
        idx = (fit_subsample(np.arange(len(comp)), spec["subsample"], seed)
               if spec["subsample"] else np.arange(len(comp)))
        est.fit(comp[FEATURE_COLUMNS].values[idx], comp["label"].values[idx])
        test["proba"] = proba_of(est, test[FEATURE_COLUMNS].values)
        pred = top1_binary(test)

        cm = confusion_matrix(test["label"].values, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["not chosen", "chosen"]); ax.set_yticklabels(["wrong arc", "true arc"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name}: student-test")
        fig.tight_layout()
        fig.savefig(os.path.join(plots, f"confusion_{name}.png"), dpi=160)
        plt.close(fig)
        print(f"{name}: confusion matrix saved  TN,FP,FN,TP = {cm.ravel().tolist()}")

        if args.mlp_curve and name == "MLP":
            clf = est.named_steps["clf"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(clf.loss_curve_, label="training loss")
            ax2 = ax.twinx()
            ax2.plot(clf.validation_scores_, color="#e08a2c", label="validation accuracy")
            ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
            ax2.set_ylabel("validation accuracy")
            ax.set_title("MLP training history (early stopping)")
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="center right")
            fig.tight_layout()
            fig.savefig(os.path.join(plots, "mlp_loss_curve.png"), dpi=160)
            plt.close(fig)
            print(f"MLP curve saved ({len(clf.loss_curve_)} epochs)")

        if args.importance:
            sub = test.sample(n=min(30_000, len(test)), random_state=seed)
            r = permutation_importance(
                est, sub[FEATURE_COLUMNS].values, sub["label"].values,
                scoring="neg_log_loss", n_repeats=3, random_state=seed, n_jobs=1)
            imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": r.importances_mean,
                                "std": r.importances_std}).sort_values("importance")
            imp.to_csv(os.path.join(args.results, f"feature_importance_{name}.csv"), index=False)
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#2a6fbb")
            ax.set_xlabel("increase in log loss when the feature is shuffled")
            ax.set_title(f"{name}: permutation importance (student-test sample)")
            fig.tight_layout()
            fig.savefig(os.path.join(plots, f"feature_importance_{name}.png"), dpi=160)
            plt.close(fig)
            print("importance saved; top 5:", imp.tail(5)["feature"].tolist()[::-1])


if __name__ == "__main__":
    main()
