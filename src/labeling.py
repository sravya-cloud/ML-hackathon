"""Per-observation ground-truth labels from the reference route.

The `.route` file is a chronological sequence of arc_ids. To train an
observation-level candidate classifier we need, for each GPS observation, the
reference arc it corresponds to. We build the route polyline (concatenated arc
segments), project every observation onto it, and assign the nearest route arc
(subject to a max perpendicular distance). This mirrors the standard
map-matching labelling procedure.

Note (per the Dataset doc): the supplied route is *reference* information, not
independently verified physical ground truth, so observations that fall far
from the route are treated as unlabeled rather than forced onto it.
"""
from __future__ import annotations

import numpy as np

from geo_utils import point_segment_distance


def label_observations(trip, max_route_dist_m=60.0):
    """Return (label_arc_id, route_dist_m, along_s) arrays, length n_obs.

    label_arc_id = -1 where no route is supplied or the observation is too far
    from the reference route to be confidently labeled.
    """
    T = trip.n_obs
    labels = np.full(T, -1, dtype=int)
    rdist = np.full(T, np.nan)
    along = np.full(T, np.nan)
    if trip.route.size == 0:
        return labels, rdist, along

    r = trip.route
    ax = trip.arc_ax[r]; ay = trip.arc_ay[r]
    bx = trip.arc_bx[r]; by = trip.arc_by[r]
    seg_len = np.hypot(bx - ax, by - ay)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]  # start distance of each seg

    prev_s = -np.inf
    for i in range(T):
        px = np.full(r.shape, trip.track_xy[i, 0])
        py = np.full(r.shape, trip.track_xy[i, 1])
        d, t, _, _ = point_segment_distance(px, py, ax, ay, bx, by)
        s = cum + t * seg_len
        # Prefer the nearest segment, but discourage large backward jumps along
        # the route (keeps labels temporally coherent through self-crossings).
        penalty = np.where(s < prev_s - 5.0, 0.5 * (prev_s - s), 0.0)
        j = int(np.argmin(d + penalty))
        rdist[i] = d[j]
        along[i] = s[j]
        if d[j] <= max_route_dist_m:
            labels[i] = int(r[j])
            prev_s = s[j]
    return labels, rdist, along
