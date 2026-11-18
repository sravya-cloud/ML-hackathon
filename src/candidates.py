"""Candidate road-arc generation for each GPS observation.

For every GPS point we retrieve the set of nearby directed arcs (within
`radius_m`) using an STRtree spatial index over arc segments, then compute exact
point-to-segment geometry. These candidates are the hidden-state universe for
both the classifier (rank candidates) and the HMM (states per observation).
"""
from __future__ import annotations

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point

from geo_utils import point_segment_distance, bearing


def build_arc_index(trip):
    """STRtree over the trip's arc segments (projected metres)."""
    trip.build_arc_geometry()
    segs = [LineString([(trip.arc_ax[i], trip.arc_ay[i]),
                        (trip.arc_bx[i], trip.arc_by[i])])
            for i in range(trip.n_arcs)]
    tree = STRtree(segs)
    return tree


def generate_candidates(trip, radius_m=45.0, max_per_obs=12):
    """Return a dict of parallel arrays describing (obs, candidate) pairs.

    Keys: obs_idx, arc_id, dist_m, t_proj, foot_x, foot_y, arc_bearing, arc_len_m
    """
    tree = build_arc_index(trip)
    arc_len = np.hypot(trip.arc_bx - trip.arc_ax, trip.arc_by - trip.arc_ay)
    arc_brg = bearing(trip.arc_ax, trip.arc_ay, trip.arc_bx, trip.arc_by)

    obs_idx_all, arc_id_all = [], []
    dist_all, t_all, fx_all, fy_all = [], [], [], []

    pts = [Point(trip.track_xy[i, 0], trip.track_xy[i, 1])
           for i in range(trip.n_obs)]

    for i, pt in enumerate(pts):
        cand = tree.query(pt, predicate="dwithin", distance=radius_m)
        if len(cand) == 0:
            # Fallback: nearest single arc so no observation is left stateless.
            cand = np.array([tree.nearest(pt)])
        cand = np.unique(np.asarray(cand, dtype=int))
        px = np.full(cand.shape, trip.track_xy[i, 0])
        py = np.full(cand.shape, trip.track_xy[i, 1])
        dist, t, fx, fy = point_segment_distance(
            px, py,
            trip.arc_ax[cand], trip.arc_ay[cand],
            trip.arc_bx[cand], trip.arc_by[cand])
        # Keep within radius (plus the forced nearest fallback), nearest first.
        order = np.argsort(dist)
        cand, dist, t, fx, fy = cand[order], dist[order], t[order], fx[order], fy[order]
        keep = dist <= radius_m
        if not keep.any():
            keep[0] = True  # always keep nearest fallback
        cand, dist, t, fx, fy = cand[keep], dist[keep], t[keep], fx[keep], fy[keep]
        if len(cand) > max_per_obs:
            cand, dist, t, fx, fy = (cand[:max_per_obs], dist[:max_per_obs],
                                     t[:max_per_obs], fx[:max_per_obs], fy[:max_per_obs])
        n = len(cand)
        obs_idx_all.append(np.full(n, i))
        arc_id_all.append(cand)
        dist_all.append(dist); t_all.append(t); fx_all.append(fx); fy_all.append(fy)

    obs_idx = np.concatenate(obs_idx_all)
    arc_id = np.concatenate(arc_id_all)
    return {
        "obs_idx": obs_idx,
        "arc_id": arc_id,
        "dist_m": np.concatenate(dist_all),
        "t_proj": np.concatenate(t_all),
        "foot_x": np.concatenate(fx_all),
        "foot_y": np.concatenate(fy_all),
        "arc_bearing": arc_brg[arc_id],
        "arc_len_m": arc_len[arc_id],
    }
