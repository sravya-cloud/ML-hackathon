"""Evaluation metrics for PathSnap.

The task is candidate ranking, so metrics are computed at the observation level
(not the raw (obs,candidate) row level) wherever the route objective matters.

  * observation-level Top-1 Accuracy : per observation pick the highest-scoring
    candidate and compare to the reference arc.
  * Macro F1-score : binary candidate classification (true arc vs not), using the
    Top-1 decision rule (exactly one predicted-positive per observation).
  * Log Loss : over all (obs,candidate) rows using predicted probabilities.
  * Candidate Miss Rate : labeled observations whose reference arc is not even in
    the candidate set (a ceiling on achievable Top-1 accuracy).
  * Route Continuity : fraction of consecutive predicted arcs that are connected
    in the road network (a route-realism measure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def _select_top1(df):
    """Index of the highest-proba row within each observation group."""
    return df.loc[df.groupby("g_obs")["proba"].idxmax()]


def observation_metrics(df):
    """df columns: g_obs, label, proba, labeled_obs. Returns a metrics dict.

    Metrics are computed over labeled observations only (those with a known
    reference arc). Observations whose true arc missed the candidate set still
    count against Top-1 accuracy.
    """
    lab = df[df["labeled_obs"] == 1].copy()
    if lab.empty:
        return {}
    n_obs = lab["g_obs"].nunique()

    top1 = _select_top1(lab)
    top1_acc = top1["label"].mean()  # selected row is the true arc?

    # Macro-F1 under the top-1 decision rule.
    lab = lab.sort_values(["g_obs"])
    pred = np.zeros(len(lab), dtype=int)
    sel_pos = lab.index.get_indexer(top1.index)
    pred_series = pd.Series(0, index=lab.index)
    pred_series.loc[top1.index] = 1
    macro_f1 = f1_score(lab["label"].values, pred_series.values, average="macro",
                        zero_division=0)

    # Log loss over all candidate rows of labeled observations.
    p = np.clip(lab["proba"].values, 1e-7, 1 - 1e-7)
    try:
        ll = log_loss(lab["label"].values, p, labels=[0, 1])
    except ValueError:
        ll = float("nan")

    # Candidate miss rate: labeled obs with no positive candidate row at all.
    has_true = lab.groupby("g_obs")["label"].max()
    miss_rate = float((has_true == 0).mean())

    return {
        "top1_accuracy": float(top1_acc),
        "macro_f1": float(macro_f1),
        "log_loss": float(ll),
        "candidate_miss_rate": miss_rate,
        "n_obs": int(n_obs),
    }


def route_continuity(pred_arc_seq, trip):
    """Fraction of consecutive (distinct) predicted arcs that are connected in
    the road network: to_node(a_i) == from_node(a_{i+1}) (a shared junction),
    or the same arc repeated. Returns NaN if fewer than 2 transitions.
    """
    seq = np.asarray(pred_arc_seq)
    seq = seq[seq >= 0]
    if seq.size < 2:
        return float("nan")
    arcs = trip.arcs
    ok = 0; tot = 0
    for a, b in zip(seq[:-1], seq[1:]):
        if a == b:
            continue  # staying on the same arc is trivially continuous
        tot += 1
        if a < len(arcs) and b < len(arcs) and arcs[a, 1] == arcs[b, 0]:
            ok += 1
    if tot == 0:
        return float("nan")
    return ok / tot
