"""Feature engineering for (observation, candidate-arc) pairs.

Each row is one candidate arc for one GPS observation. Features combine:
  * geometric fit      - perpendicular distance, projection position, endpoint gaps
  * directional fit    - GPS heading vs (directed) arc bearing
  * kinematics         - GPS speed, time gaps, local movement
  * relative/context   - rank within the observation, distance relative to the
                         best candidate, number of competing candidates

The label is 1 if the candidate arc equals the reference arc for the
observation, else 0. The classifier therefore scores how well each candidate
explains the observation; those scores become HMM emission probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from geo_utils import bearing, angle_diff

FEATURE_COLUMNS = [
    "dist_m", "dist_m_sq", "t_proj", "t_interior", "arc_len_m",
    "heading_diff", "heading_cos", "dist_to_from_node", "dist_to_to_node",
    "speed_mps", "dt_prev", "dt_next", "accel",
    "rank", "n_cand", "dist_rel_best", "dist_minus_best", "dist_z",
    "foot_frac_len", "is_nearest",
]


def _gps_kinematics(trip, heading_min_move_m=2.0):
    """Per-observation speed, time gaps, acceleration and a robust heading."""
    xy = trip.track_xy
    t = trip.track_t
    T = trip.n_obs
    dt_prev = np.ones(T); dt_next = np.ones(T)
    dt_prev[1:] = np.clip(t[1:] - t[:-1], 1e-3, None)
    dt_next[:-1] = np.clip(t[1:] - t[:-1], 1e-3, None)
    step_prev = np.zeros(T); step_next = np.zeros(T)
    step_prev[1:] = np.hypot(xy[1:, 0] - xy[:-1, 0], xy[1:, 1] - xy[:-1, 1])
    step_next[:-1] = step_prev[1:]
    speed = np.zeros(T)
    speed[1:-1] = (step_prev[1:-1] + step_next[1:-1]) / (dt_prev[1:-1] + dt_next[1:-1])
    if T > 1:
        speed[0] = step_next[0] / dt_next[0]
        speed[-1] = step_prev[-1] / dt_prev[-1]
    accel = np.zeros(T)
    accel[1:] = (speed[1:] - speed[:-1]) / dt_prev[1:]

    # Robust heading: direction from previous to next moving point.
    heading = np.full(T, np.nan)
    for i in range(T):
        lo, hi = i, i
        while lo > 0 and np.hypot(xy[i, 0] - xy[lo, 0], xy[i, 1] - xy[lo, 1]) < heading_min_move_m:
            lo -= 1
        while hi < T - 1 and np.hypot(xy[hi, 0] - xy[i, 0], xy[hi, 1] - xy[i, 1]) < heading_min_move_m:
            hi += 1
        if hi > lo:
            heading[i] = bearing(xy[lo, 0], xy[lo, 1], xy[hi, 0], xy[hi, 1])
    # Forward/backward fill NaNs
    valid = ~np.isnan(heading)
    if valid.any():
        idx = np.where(valid, np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        heading = heading[idx]
        heading[np.isnan(heading)] = heading[valid][0]
    else:
        heading = np.zeros(T)
    return speed, dt_prev, dt_next, accel, heading


def build_features(trip, cand, labels=None, heading_min_move_m=2.0):
    """Return a DataFrame of features for all (obs,candidate) pairs of a trip."""
    obs = cand["obs_idx"]
    arc = cand["arc_id"]
    dist = cand["dist_m"]
    df = pd.DataFrame({
        "obs_idx": obs,
        "arc_id": arc,
        "dist_m": dist,
        "t_proj": cand["t_proj"],
        "arc_len_m": cand["arc_len_m"],
    })
    df["dist_m_sq"] = dist ** 2
    df["t_interior"] = ((df["t_proj"] > 0.02) & (df["t_proj"] < 0.98)).astype(float)
    df["foot_frac_len"] = df["t_proj"]

    # Directional fit against the directed arc bearing.
    speed, dt_prev, dt_next, accel, heading = _gps_kinematics(trip, heading_min_move_m)
    hd = angle_diff(heading[obs], cand["arc_bearing"])
    df["heading_diff"] = hd
    df["heading_cos"] = np.cos(np.radians(hd))
    # Clip kinematics to sane ranges: huge GPS gaps/jumps (data anomalies noted
    # in the dataset doc) otherwise dominate scaled/linear models. The raw gap
    # is still used by the HMM transition model separately.
    df["speed_mps"] = np.clip(speed[obs], 0.0, 60.0)
    df["dt_prev"] = np.clip(dt_prev[obs], 1.0, 300.0)
    df["dt_next"] = np.clip(dt_next[obs], 1.0, 300.0)
    df["accel"] = np.clip(accel[obs], -20.0, 20.0)

    # Endpoint gaps (distance from GPS point to arc's from/to nodes).
    ax = trip.arc_ax[arc]; ay = trip.arc_ay[arc]
    bx = trip.arc_bx[arc]; by = trip.arc_by[arc]
    px = trip.track_xy[obs, 0]; py = trip.track_xy[obs, 1]
    df["dist_to_from_node"] = np.hypot(px - ax, py - ay)
    df["dist_to_to_node"] = np.hypot(px - bx, py - by)

    # Relative / context features computed within each observation.
    g = df.groupby("obs_idx")["dist_m"]
    best = g.transform("min")
    spread = g.transform("std").fillna(0.0) + 1e-6
    df["n_cand"] = df.groupby("obs_idx")["arc_id"].transform("size")
    df["dist_rel_best"] = np.clip(dist / (best + 1e-6), 0.0, 100.0)
    df["dist_minus_best"] = dist - best
    df["dist_z"] = (dist - g.transform("mean")) / spread
    df["rank"] = df.groupby("obs_idx")["dist_m"].rank(method="first").astype(int) - 1
    df["is_nearest"] = (df["rank"] == 0).astype(float)

    if labels is not None:
        df["label"] = (df["arc_id"].values == labels[obs]).astype(int)
        # Mark rows whose observation has no valid label (route arc unknown).
        df["labeled_obs"] = (labels[obs] >= 0).astype(int)
    return df
