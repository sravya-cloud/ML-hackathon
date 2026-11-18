"""Folium evaluation maps: reference path vs predicted path before and after the HMM.

One HTML map per student-test trip (results/maps/trip_<id>.html) plus index.html.
Layers (toggle with the layer control):
  raw GPS points (blue), road network near the trip (grey), reference route (green),
  classifier-only prediction (orange), classifier + HMM prediction (red).

Only the student-test partition (which ships with its reference routes) is mapped.
Evaluator-test routes are never drawn or written to disk.

Usage (from src/):
  python folium_maps.py --data-root "<path to Student folder>"
"""
import argparse, os
import numpy as np, folium

from data_loader import list_trips, load_trip
from pipeline import (Artifacts, predict_trip, evaluate_prediction, route_from_arcs,
                      arcs_to_lonlat)


def ll(line):
    return [[la, lo] for lo, la in line]


def build_map(trip, pred, m):
    gps = [[float(la), float(lo)] for lo, la in trip.track_lonlat]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", attr="Tiles &copy; Esri")

    # road network near the trip: arcs touching a node close to any GPS point (capped)
    lons = [p[1] for p in gps]; lats = [p[0] for p in gps]
    pad = 0.003
    net = folium.FeatureGroup(name="Road network (grey)", show=False)
    n = 0
    for a in range(trip.n_arcs):
        pts = [trip.nodes_lonlat[k] for k in trip.arcs[a]]
        la = [p[1] for p in pts]; lo = [p[0] for p in pts]
        if (min(lo) > max(lons) + pad or max(lo) < min(lons) - pad or
                min(la) > max(lats) + pad or max(la) < min(lats) - pad):
            continue
        folium.PolyLine([[p[1], p[0]] for p in pts], color="#999999", weight=1.5,
                        opacity=0.6).add_to(net)
        n += 1
        if n >= 2000:
            break
    net.add_to(fmap)

    g = folium.FeatureGroup(name="Raw GPS points (blue)")
    step = max(1, len(gps) // 1500)
    for p in gps[::step]:
        folium.CircleMarker(p, radius=2, color="#1f5fbf", fill=True, fill_opacity=0.7,
                            weight=0).add_to(g)
    g.add_to(fmap)

    layers = [("Reference route (green)", "#2ca02c", 6, 0.8, list(map(int, trip.route)), True),
              ("Classifier only (orange)", "#ff8c00", 4, 0.9,
               route_from_arcs(pred["classifier_arcs"]), True),
              ("Classifier + HMM (red)", "#d62728", 4, 0.9,
               route_from_arcs(pred["final_arcs"]), True)]
    for name, color, w, op, arcs, show in layers:
        fg = folium.FeatureGroup(name=name, show=show)
        folium.PolyLine(ll(arcs_to_lonlat(trip, arcs)), color=color, weight=w, opacity=op).add_to(fg)
        fg.add_to(fmap)
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    box = (f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:#fff;'
           f'padding:8px 12px;border:1px solid #bbb;border-radius:6px;font:12px sans-serif">'
           f'<b>Trip {trip.trip_id}</b><br>'
           f'Top-1: classifier {m["classifier_only_top1_accuracy"]:.3f} &rarr; HMM {m["top1_accuracy"]:.3f}<br>'
           f'Route F1: {m["classifier_only_route_f1"]:.3f} &rarr; {m["route_f1"]:.3f}<br>'
           f'Green = reference, orange = before HMM, red = after HMM</div>')
    fmap.get_root().html.add_child(folium.Element(box))
    return fmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("PATHSNAP_DATA"))
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--artifacts", default="../artifacts")
    ap.add_argument("--out", default="../results/maps")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    art = Artifacts(args.artifacts, args.config)
    items = []
    for d in list_trips(args.data_root, "student_test"):
        trip = load_trip(d, "student_test")
        pred = predict_trip(trip, art, use_hmm=True)
        m = evaluate_prediction(trip, pred)
        build_map(trip, pred, m).save(os.path.join(args.out, f"trip_{trip.trip_id}.html"))
        items.append((trip.trip_id, m))
        print("map", trip.trip_id, flush=True)
    li = "".join(f'<li><a href="trip_{t}.html">Trip {t}</a> &mdash; Top-1 {m["classifier_only_top1_accuracy"]:.3f} '
                 f'&rarr; {m["top1_accuracy"]:.3f} with HMM</li>' for t, m in items)
    open(os.path.join(args.out, "index.html"), "w").write(
        f"<html><body style='font-family:sans-serif'><h2>PathSnap evaluation maps (student-test trips)</h2>"
        f"<p>Green: reference route. Orange: classifier only. Red: classifier + HMM.</p><ul>{li}</ul></body></html>")


if __name__ == "__main__":
    main()
