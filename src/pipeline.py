"""End-to-end prediction pipeline for one trip.

    raw trip -> candidate arcs -> features -> classifier scores
             -> (optional) HMM/Viterbi decoding -> one arc per GPS point

The prediction functions never see the reference route: `predict_trip` works on
a copy of the trip whose `route` is empty. Evaluation against the reference is a
separate step (`evaluate_prediction`) that only the server runs.

HMM hook
--------
If a module `hmm_viterbi.py` exists in src/ and defines

    decode(trip, cand_df, cfg) -> np.ndarray of arc_ids, length trip.n_obs

where `cand_df` has columns obs_idx, arc_id, proba (classifier score per
candidate row) and cfg is the `hmm:` block of config.yaml, the pipeline uses it.
Otherwise it falls back to classifier-only (highest score per GPS point).
"""
from __future__ import annotations

import copy
import importlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from candidates import generate_candidates
from features import build_features
from labeling import label_observations
from evaluate import route_continuity


class Artifacts:
    """Trained model plus everything needed to reproduce inference."""

    def __init__(self, artifacts_dir, config_path):
        self.model = joblib.load(os.path.join(artifacts_dir, "model.joblib"))
        with open(os.path.join(artifacts_dir, "feature_columns.json")) as fh:
            self.feature_columns = json.load(fh)
        card_path = os.path.join(artifacts_dir, "model_card.json")
        self.card = json.load(open(card_path)) if os.path.exists(card_path) else {}
        self.cfg = yaml.safe_load(open(config_path))
        try:
            self.hmm = importlib.import_module("hmm_viterbi")
            if not hasattr(self.hmm, "decode"):
                self.hmm = None
        except ImportError:
            self.hmm = None


def _score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-s))


def blind_copy(trip):
    """Copy of the trip with the reference route removed (prediction input)."""
    t = copy.copy(trip)
    t.route = np.empty((0,), dtype=int)
    return t


def predict_trip(trip, art: Artifacts, use_hmm=True):
    """Return dict with per-observation arcs and timing. No reference access."""
    t0 = time.time()
    trip = blind_copy(trip)
    ccfg = art.cfg["candidates"]
    cand = generate_candidates(trip, radius_m=ccfg["radius_m"],
                               max_per_obs=ccfg["max_per_obs"])
    df = build_features(trip, cand, labels=None,
                        heading_min_move_m=art.cfg["features"]["heading_min_move_m"])
    df["proba"] = _score(art.model, df[art.feature_columns].values)

    # Classifier-only: highest score per observation.
    top = df.loc[df.groupby("obs_idx")["proba"].idxmax()].sort_values("obs_idx")
    clf_arcs = np.full(trip.n_obs, -1, dtype=int)
    clf_arcs[top["obs_idx"].values] = top["arc_id"].values

    final_arcs, used_hmm = clf_arcs, False
    if use_hmm and art.hmm is not None:
        final_arcs = np.asarray(art.hmm.decode(trip, df[["obs_idx", "arc_id", "proba"]],
                                               art.cfg["hmm"]), dtype=int)
        used_hmm = True
    return {
        "cand_df": df,
        "classifier_arcs": clf_arcs,
        "final_arcs": final_arcs,
        "used_hmm": used_hmm,
        "elapsed_s": time.time() - t0,
    }


def route_from_arcs(arcs):
    """Collapse per-observation arcs into the ordered list of distinct arcs."""
    arcs = np.asarray(arcs)
    arcs = arcs[arcs >= 0]
    if arcs.size == 0:
        return []
    keep = np.concatenate([[True], arcs[1:] != arcs[:-1]])
    return arcs[keep].astype(int).tolist()


def arcs_to_lonlat(trip, arc_ids):
    """Polyline (lon, lat) following a list of arc ids, for display."""
    pts = []
    for a in arc_ids:
        for n in trip.arcs[a]:
            lon, lat = trip.nodes_lonlat[n]
            if not pts or pts[-1] != [float(lon), float(lat)]:
                pts.append([float(lon), float(lat)])
    return pts


def evaluate_prediction(trip, pred, max_route_dist_m=60.0):
    """Server-side scoring against the reference route (never sent to the UI)."""
    labels, _, _ = label_observations(trip, max_route_dist_m=max_route_dist_m)
    lab = labels >= 0
    out = {"n_obs": int(trip.n_obs), "n_labeled_obs": int(lab.sum())}
    if not lab.any():
        return out

    def acc(arcs):
        return float((arcs[lab] == labels[lab]).mean())

    df = pred["cand_df"]
    final = pred["final_arcs"]
    # Macro-F1 under the "one chosen candidate per observation" rule.
    chosen = (df["arc_id"].values == final[df["obs_idx"].values]).astype(int)
    truth = (df["arc_id"].values == labels[df["obs_idx"].values]).astype(int)
    keep = lab[df["obs_idx"].values]
    macro_f1 = float(f1_score(truth[keep], chosen[keep], average="macro", zero_division=0))

    cont_clf = route_continuity(pred["classifier_arcs"], trip)
    cont_fin = route_continuity(final, trip)
    out.update({
        "top1_accuracy": acc(final),
        "macro_f1": macro_f1,
        "classifier_only_top1_accuracy": acc(pred["classifier_arcs"]),
        "route_continuity": None if np.isnan(cont_fin) else float(cont_fin),
        "classifier_only_route_continuity": None if np.isnan(cont_clf) else float(cont_clf),
        "used_hmm": pred["used_hmm"],
    })
    return out
