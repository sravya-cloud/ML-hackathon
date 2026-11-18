"""Geographical utilities for PathSnap.

The dataset uses WGS84 angular coordinates (lon, lat). Euclidean distance in
degrees is meaningless, so we project to a local metric system and do all
distance/bearing maths in metres. All point-to-segment geometry is vectorised
with NumPy.

IMPORTANT: trips are NOT all in the same city (e.g. most are around Heidelberg,
Germany at ~8.6 E / 49.8 N, but at least one trip sits near Quito, Ecuador at
~-78 W / 0.3 N). A single fixed UTM zone would produce NaN/garbage for
far-away trips, so we build a *per-trip* Azimuthal Equidistant (AEQD)
projection centred on that trip's own coordinates. AEQD preserves distances
from its centre, which is exactly right for local map-matching geometry.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

_GEOG = "EPSG:4326"


class LocalProjector:
    """Per-trip metric projection centred on (lon0, lat0) using AEQD."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        proj = (f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} "
                f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
        self._fwd = Transformer.from_crs(_GEOG, proj, always_xy=True)
        self._inv = Transformer.from_crs(proj, _GEOG, always_xy=True)

    def to_metric(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_geographic(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


# Backwards-compatible module-level helper (uses a default Heidelberg centre).
# Prefer a per-trip LocalProjector in real pipelines.
_DEFAULT = None


def init_transformers(*_args, **_kwargs) -> None:
    global _DEFAULT
    _DEFAULT = LocalProjector(8.6, 49.8)


def to_metric(lon, lat):
    if _DEFAULT is None:
        init_transformers()
    return _DEFAULT.to_metric(lon, lat)


def to_geographic(x, y):
    if _DEFAULT is None:
        init_transformers()
    return _DEFAULT.to_geographic(x, y)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Perpendicular distance from points P to segments A->B, plus the
    projection parameter t (clamped to [0,1]) and the foot-of-perpendicular.

    All inputs are 1-D arrays, broadcast element-wise. Returns (dist, t, fx, fy).
    """
    px = np.asarray(px, float); py = np.asarray(py, float)
    ax = np.asarray(ax, float); ay = np.asarray(ay, float)
    bx = np.asarray(bx, float); by = np.asarray(by, float)
    dx = bx - ax
    dy = by - ay
    seg_len2 = dx * dx + dy * dy
    # Avoid division by zero for degenerate (zero-length) segments.
    safe = seg_len2 > 1e-12
    t = np.zeros_like(px)
    t[safe] = ((px[safe] - ax[safe]) * dx[safe] +
               (py[safe] - ay[safe]) * dy[safe]) / seg_len2[safe]
    t = np.clip(t, 0.0, 1.0)
    fx = ax + t * dx
    fy = ay + t * dy
    dist = np.hypot(px - fx, py - fy)
    return dist, t, fx, fy


def bearing(ax, ay, bx, by):
    """Compass bearing (degrees, 0=up/north-ish in projected space) of A->B.

    Uses projected axes; good enough for *differences* between GPS heading and
    arc direction at this spatial scale.
    """
    return np.degrees(np.arctan2(np.asarray(bx) - np.asarray(ax),
                                 np.asarray(by) - np.asarray(ay))) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings in degrees [0,180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)
