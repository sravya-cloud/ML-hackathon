"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
---------
GET  /health                  liveness + which mode is running
GET  /trips                   ids of the ten evaluation trips
POST /predict  {"trip_id": "...", "include_map": false}
                              runs the full pipeline on that trip and returns the
                              predicted route plus server-side accuracy metrics

The reference routes stay on the server. They are read only by the scoring step
after prediction, and the response never contains them.

Run (from the repo root):
  uvicorn api.main:app --host 0.0.0.0 --port 8000

Modes (environment variables)
-----------------------------
  PATHSNAP_STUB=1        fake data, no model needed (lets the UI be built first)
  PATHSNAP_TRIPS_DIR     folder containing the trip folders to serve
                         (default: <data root>/evaluator_test if it exists,
                          otherwise <data root>/student_test for development)
  PATHSNAP_DATA          data root (default: from config.yaml)
  PATHSNAP_ARTIFACTS     artifacts folder (default: ./artifacts)
"""
from __future__ import annotations

import datetime
import logging
import os
import sys
import time

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

CONFIG_PATH = os.path.join(ROOT, "config.yaml")
CFG = yaml.safe_load(open(CONFIG_PATH))
STUB = os.environ.get("PATHSNAP_STUB", "0") == "1"
ARTIFACTS = os.environ.get("PATHSNAP_ARTIFACTS", os.path.join(ROOT, "artifacts"))

# ---- logging: every request is printed on the API laptop's console -------------
log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
for h in (logging.StreamHandler(sys.stdout),
          logging.FileHandler(os.path.join(ROOT, "api_requests.log"))):
    h.setFormatter(_fmt)
    log.addHandler(h)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str
    include_map: bool = False


# ---- trip discovery --------------------------------------------------------------
def _trips_dir():
    env = os.environ.get("PATHSNAP_TRIPS_DIR")
    if env:
        return env
    data_root = os.environ.get("PATHSNAP_DATA") or os.path.join(ROOT, CFG["data"]["root"])
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(ROOT, data_root))
    ev = os.path.join(data_root, CFG["data"]["evaluator_test_dir"])
    return ev if os.path.isdir(ev) else os.path.join(data_root, CFG["data"]["student_test_dir"])


def _list_trip_ids():
    if STUB:
        return [f"{i:08d}" for i in range(10)]
    d = _trips_dir()
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


# ---- real pipeline (loaded once) -------------------------------------------------
_ART = None


def _artifacts():
    global _ART
    if _ART is None:
        from pipeline import Artifacts
        _ART = Artifacts(ARTIFACTS, CONFIG_PATH)
        log.info(f"model loaded: {_ART.card.get('model', '?')} | "
                 f"HMM decoder: {'yes' if _ART.hmm is not None else 'no (classifier only)'}")
    return _ART


def _make_map(trip, gps, pred_line):
    import folium
    lat0 = sum(p[1] for p in gps) / len(gps)
    lon0 = sum(p[0] for p in gps) / len(gps)
    m = folium.Map(location=[lat0, lon0], zoom_start=14, tiles="OpenStreetMap")
    g1 = folium.FeatureGroup(name="Raw GPS points (blue)")
    step = max(1, len(gps) // 1500)
    for lon, lat in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="#1f5fbf", fill=True,
                            fill_opacity=0.7, weight=0).add_to(g1)
    g1.add_to(m)
    g2 = folium.FeatureGroup(name="Predicted route (red)")
    if pred_line:
        folium.PolyLine([[la, lo] for lo, la in pred_line], color="#d62728",
                        weight=4, opacity=0.9).add_to(g2)
    g2.add_to(m)
    folium.LayerControl().add_to(m)
    return m.get_root().render()


def _predict_real(trip_id, include_map):
    from data_loader import load_trip
    from pipeline import (predict_trip, route_from_arcs, arcs_to_lonlat,
                          evaluate_prediction)
    tdir = os.path.join(_trips_dir(), trip_id)
    if not os.path.isdir(tdir):
        raise HTTPException(status_code=404, detail=f"unknown trip_id {trip_id}")
    art = _artifacts()
    trip = load_trip(tdir, "evaluation")
    pred = predict_trip(trip, art)                      # never sees the reference
    metrics = evaluate_prediction(trip, pred,           # server-side scoring only
                                  CFG["labeling"]["max_route_dist_m"])
    route = route_from_arcs(pred["final_arcs"])
    gps = trip.track_lonlat.tolist()
    line = arcs_to_lonlat(trip, route)
    metrics["inference_seconds"] = round(pred["elapsed_s"], 3)
    metrics["model"] = art.card.get("model", "unknown")
    return {
        "trip_id": trip_id,
        "n_gps_points": int(trip.n_obs),
        "predicted_route_arc_ids": route,
        "predicted_route_lonlat": line,
        "gps_points_lonlat": gps,
        "metrics": metrics,
        "map_html": _make_map(trip, gps, line) if include_map else None,
    }


def _predict_stub(trip_id):
    import math
    base_lon, base_lat = 8.69, 49.40
    gps = [[base_lon + 0.0004 * i, base_lat + 0.0002 * math.sin(i / 6)] for i in range(60)]
    line = [[p[0], p[1] + 0.00005] for p in gps[::5]]
    return {
        "trip_id": trip_id, "n_gps_points": len(gps),
        "predicted_route_arc_ids": list(range(12)),
        "predicted_route_lonlat": line, "gps_points_lonlat": gps,
        "metrics": {"top1_accuracy": 0.0, "macro_f1": 0.0, "route_continuity": 0.0,
                    "inference_seconds": 0.0, "model": "STUB (fake data)"},
        "map_html": None,
    }


# ---- routes ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "mode": "stub" if STUB else "real"}


@app.get("/trips")
def trips():
    ids = _list_trip_ids()
    log.info(f"GET /trips -> {len(ids)} trips")
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    client = request.client.host if request.client else "?"
    log.info(f"POST /predict  trip_id={req.trip_id}  from={client}")
    if req.trip_id not in _list_trip_ids():
        log.info(f"  -> 404 unknown trip {req.trip_id}")
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id}")
    t0 = time.time()
    out = _predict_stub(req.trip_id) if STUB else _predict_real(req.trip_id, req.include_map)
    m = out["metrics"]
    log.info(f"  -> done in {time.time() - t0:.2f}s | gps_points={out['n_gps_points']} "
             f"top1={m.get('top1_accuracy')} macro_f1={m.get('macro_f1')} "
             f"continuity={m.get('route_continuity')}")
    return out
