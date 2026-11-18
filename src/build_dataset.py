"""Build the modelling datasets for all trips in a partition.

Produces one row per (observation, candidate-arc) pair with features, the
binary label (candidate == reference arc), and group keys (trip_id, obs global
index) needed for group-aware cross-validation and observation-level metrics.

Usage:
  python build_dataset.py --data-root "<path>" --config ../config.yaml --out ../results
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import yaml

from geo_utils import init_transformers
from data_loader import list_trips, load_trip
from candidates import generate_candidates
from labeling import label_observations
from features import build_features


def build_partition(data_root, sub_dir, partition, cfg):
    dirs = list_trips(data_root, sub_dir)
    frames = []
    miss_rows = []
    for d in dirs:
        tid = os.path.basename(d)
        trip = load_trip(d, partition)
        cand = generate_candidates(trip,
                                   radius_m=cfg["candidates"]["radius_m"],
                                   max_per_obs=cfg["candidates"]["max_per_obs"])
        labels, rdist, _ = label_observations(
            trip, max_route_dist_m=cfg["labeling"]["max_route_dist_m"])
        df = build_features(trip, cand, labels,
                            heading_min_move_m=cfg["features"]["heading_min_move_m"])
        df.insert(0, "trip_id", tid)
        # Global observation id unique across the partition for group metrics.
        df["g_obs"] = tid + "_" + df["obs_idx"].astype(str)
        frames.append(df)

        # Candidate miss rate bookkeeping (labeled obs whose true arc not a candidate).
        lab_obs = np.where(labels >= 0)[0]
        hit = df[df["label"] == 1]["obs_idx"].unique()
        miss = np.setdiff1d(lab_obs, hit)
        miss_rows.append((tid, len(lab_obs), len(miss)))
    full = pd.concat(frames, ignore_index=True)
    miss = pd.DataFrame(miss_rows, columns=["trip_id", "labeled_obs", "missed_obs"])
    return full, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("PATHSNAP_DATA"))
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_root = args.data_root or cfg["data"]["root"]
    init_transformers(cfg["geo"]["crs_geographic"], cfg["geo"]["crs_metric"])
    os.makedirs(args.out, exist_ok=True)

    for sub, part in [(cfg["data"]["train_dir"], "train"),
                      (cfg["data"]["student_test_dir"], "student_test")]:
        t0 = time.time()
        full, miss = build_partition(data_root, sub, part, cfg)
        full.to_parquet(os.path.join(args.out, f"dataset_{part}.parquet"))
        miss.to_csv(os.path.join(args.out, f"candidate_miss_{part}.csv"), index=False)
        n_obs = full["g_obs"].nunique()
        n_lab = full[full.get("labeled_obs", 1) == 1]["g_obs"].nunique()
        cmr = miss["missed_obs"].sum() / max(miss["labeled_obs"].sum(), 1)
        print(f"[{part}] trips={full['trip_id'].nunique()} rows={len(full)} "
              f"obs={n_obs} labeled_obs={n_lab} candidate_miss_rate={cmr:.4f} "
              f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
