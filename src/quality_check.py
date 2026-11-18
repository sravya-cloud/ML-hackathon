"""Data-quality check over every trip. Usage (from src/):
  python quality_check.py "../data/Student folder" ../results/data_quality_summary.csv
"""
import os, sys, numpy as np, pandas as pd
from data_loader import list_trips, load_trip
ROOT = sys.argv[1] if len(sys.argv) > 1 else "../data/Student folder"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../results/data_quality_summary.csv"
rows = []
for part in ["train", "student_test"]:
    for d in list_trips(ROOT, part):
        tid = os.path.basename(d)
        raw = np.loadtxt(os.path.join(d, tid + ".track"))
        t = raw[:, 2]
        dt = np.diff(t)
        tr = load_trip(d, part)
        step = np.hypot(*np.diff(tr.track_xy, axis=0).T)
        used = np.unique(tr.arcs.ravel())
        isolated = tr.n_nodes - len(used)
        lab_route_ok = tr.route.size > 0 and tr.route.max() < tr.n_arcs
        rows.append(dict(
            partition=part, trip=tid, n_obs=len(raw),
            nan_track=int(np.isnan(raw).sum()),
            bad_lonlat=int(((np.abs(raw[:,0])>180)|(np.abs(raw[:,1])>90)).sum()),
            non_monotonic_t=int((dt <= 0).sum()),
            dup_points=int((step == 0).sum()),
            gap_gt_1s=int((dt > 1.5).sum()),
            gap_gt_10s=int((dt > 10).sum()),
            max_gap_s=float(dt.max()) if len(dt) else 0,
            max_jump_m=float(step.max()) if len(step) else 0,
            nodes=tr.n_nodes, arcs=tr.n_arcs, isolated_nodes=int(isolated),
            route_len=int(tr.route.size), route_ids_valid=bool(lab_route_ok),
            lon=float(np.median(raw[:,0])), lat=float(np.median(raw[:,1]))))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("trips:", df.groupby("partition").size().to_dict())
print("obs per trip: min/median/max", df.n_obs.min(), int(df.n_obs.median()), df.n_obs.max())
print("total NaN in tracks:", df.nan_track.sum(), "| bad lon/lat:", df.bad_lonlat.sum())
print("trips with non-monotonic time:", int((df.non_monotonic_t>0).sum()), "rows:", df.non_monotonic_t.sum())
print("trips with duplicate points:", int((df.dup_points>0).sum()), "rows:", df.dup_points.sum())
print("gaps >1.5s total:", df.gap_gt_1s.sum(), "| gaps >10s total:", df.gap_gt_10s.sum(), "| trips with gap>10s:", int((df.gap_gt_10s>0).sum()))
print("largest gaps (s):", df.nlargest(3,"max_gap_s")[["trip","max_gap_s"]].values.tolist())
print("largest jumps (m):", df.nlargest(3,"max_jump_m")[["trip","max_jump_m"]].values.tolist())
print("trips with isolated nodes:", int((df.isolated_nodes>0).sum()), "total isolated nodes:", df.isolated_nodes.sum(), "of", df.nodes.sum())
print("trips with invalid route ids:", int((~df.route_ids_valid).sum()), "| trips with no route:", int((df.route_len==0).sum()))
far = df[(df.lon<5)|(df.lon>12)|(df.lat<45)|(df.lat>52)]
print("trips outside Germany area:", far[["partition","trip","lon","lat"]].values.tolist())
