"""Loading and representing a single PathSnap trip.

Raw files per trip (whitespace separated, no headers):
  .track : lon  lat  timestamp_seconds        (GPS observations, ~1 Hz)
  .nodes : lon  lat                            (row index = node_id, 0-based)
  .arcs  : from_node_id  to_node_id            (row index = arc_id, 0-based; directed)
  .route : arc_id                              (chronological ground-truth, where supplied)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from geo_utils import LocalProjector


@dataclass
class Trip:
    trip_id: str
    partition: str
    # GPS track
    track_lonlat: np.ndarray          # (T,2) lon,lat
    track_t: np.ndarray               # (T,) timestamps (s)
    track_xy: np.ndarray              # (T,2) projected metres
    # Road network
    nodes_lonlat: np.ndarray          # (N,2)
    nodes_xy: np.ndarray              # (N,2)
    arcs: np.ndarray                  # (A,2) int from,to
    # Reference route (may be empty)
    route: np.ndarray                 # (R,) arc_ids
    # Derived arc geometry (projected endpoints), filled lazily
    arc_ax: np.ndarray = field(default=None)
    arc_ay: np.ndarray = field(default=None)
    arc_bx: np.ndarray = field(default=None)
    arc_by: np.ndarray = field(default=None)
    projector: object = field(default=None)   # LocalProjector for this trip
    n_bad_nodes: int = 0                       # nodes with invalid coordinates

    @property
    def n_obs(self) -> int:
        return self.track_lonlat.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.nodes_lonlat.shape[0]

    @property
    def n_arcs(self) -> int:
        return self.arcs.shape[0]

    def build_arc_geometry(self):
        """Projected endpoint coordinates for every arc (vectorised)."""
        if self.arc_ax is not None:
            return
        fr = self.arcs[:, 0]
        to = self.arcs[:, 1]
        self.arc_ax = self.nodes_xy[fr, 0]
        self.arc_ay = self.nodes_xy[fr, 1]
        self.arc_bx = self.nodes_xy[to, 0]
        self.arc_by = self.nodes_xy[to, 1]


def _load_txt(path, dtype=float):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.empty((0,), dtype=dtype)
    return np.loadtxt(path, dtype=dtype)


def load_trip(trip_dir: str, partition: str) -> Trip:
    trip_id = os.path.basename(trip_dir.rstrip("/"))
    track = _load_txt(os.path.join(trip_dir, f"{trip_id}.track"), float)
    nodes = _load_txt(os.path.join(trip_dir, f"{trip_id}.nodes"), float)
    arcs = _load_txt(os.path.join(trip_dir, f"{trip_id}.arcs"), int)
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    route = _load_txt(route_path, int)

    track = np.atleast_2d(track)
    nodes = np.atleast_2d(nodes)
    arcs = np.atleast_2d(arcs).astype(int)
    route = np.atleast_1d(route).astype(int) if route.size else np.empty((0,), int)

    track_lonlat = track[:, :2]
    track_t = track[:, 2]

    # Per-trip projection centred on the trip's own coordinates (handles trips
    # in different cities/countries). Centre on the median of valid node coords.
    valid = (np.abs(nodes[:, 0]) <= 180) & (np.abs(nodes[:, 1]) <= 90)
    centre_src = nodes[valid] if valid.any() else track_lonlat
    lon0 = float(np.median(centre_src[:, 0]))
    lat0 = float(np.median(centre_src[:, 1]))
    projector = LocalProjector(lon0, lat0)

    tx, ty = projector.to_metric(track_lonlat[:, 0], track_lonlat[:, 1])
    track_xy = np.column_stack([tx, ty])
    nx, ny = projector.to_metric(nodes[:, 0], nodes[:, 1])
    nodes_xy = np.column_stack([nx, ny])

    # Any node that still fails to project (truly invalid) is pushed far away so
    # it never becomes a candidate, and counted as a data-quality issue.
    bad = ~(np.isfinite(nodes_xy[:, 0]) & np.isfinite(nodes_xy[:, 1]))
    n_bad = int(bad.sum())
    if n_bad:
        nodes_xy[bad] = 1e12

    trip = Trip(
        trip_id=trip_id, partition=partition,
        track_lonlat=track_lonlat, track_t=track_t, track_xy=track_xy,
        nodes_lonlat=nodes, nodes_xy=nodes_xy, arcs=arcs, route=route,
        projector=projector, n_bad_nodes=n_bad,
    )
    trip.build_arc_geometry()
    return trip


def list_trips(data_root: str, sub_dir: str):
    base = os.path.join(data_root, sub_dir)
    if not os.path.isdir(base):
        return []
    return sorted(
        os.path.join(base, d) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    )


def load_partition(data_root: str, sub_dir: str, partition_name: str):
    return [load_trip(d, partition_name) for d in list_trips(data_root, sub_dir)]
