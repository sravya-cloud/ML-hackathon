"""Classifier-only vs classifier + HMM/Viterbi on the student-test trips.

Writes results/hmm_vs_classifier.csv (per trip + overall) and
results/plots/hmm_vs_classifier.png. Uses artifacts/ (the tuned, exported model)
and the hmm block of config.yaml. Student-test partition only.

Usage (from src/):
  python evaluate_hmm.py --data-root "<path to Student folder>"
"""
import argparse, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import list_trips, load_trip
from pipeline import Artifacts, predict_trip, evaluate_prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("PATHSNAP_DATA"))
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--artifacts", default="../artifacts")
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    art = Artifacts(args.artifacts, args.config)
    rows = []
    for d in list_trips(args.data_root, "student_test"):
        trip = load_trip(d, "student_test")
        t0 = time.time()
        pred = predict_trip(trip, art, use_hmm=True)
        m = evaluate_prediction(trip, pred)
        rows.append({
            "trip_id": trip.trip_id, "n_obs": m["n_obs"], "n_labeled": m["n_labeled_obs"],
            "top1_classifier": m["classifier_only_top1_accuracy"], "top1_hmm": m["top1_accuracy"],
            "continuity_classifier": m["classifier_only_route_continuity"],
            "continuity_hmm": m["route_continuity"],
            "route_f1_classifier": m["classifier_only_route_f1"], "route_f1_hmm": m["route_f1"],
            "macro_f1_hmm": m["macro_f1"], "seconds_total": round(time.time() - t0, 2)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    w = df["n_labeled"]
    tot = {"trip_id": "ALL (obs-weighted)", "n_obs": df.n_obs.sum(), "n_labeled": w.sum()}
    for c in df.columns[3:-1]:
        v = df[c]; ok = v.notna()
        tot[c] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
    tot["seconds_total"] = df.seconds_total.sum()
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    df.round(4).to_csv(os.path.join(args.results, "hmm_vs_classifier.csv"), index=False)
    print(df.round(4).to_string(index=False))

    body = df.iloc[:-1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(body)); wd = 0.4
    for a, (k, title) in zip(ax, [("top1", "Top-1 accuracy"), ("continuity", "Route continuity"),
                                  ("route_f1", "Route F1")]):
        a.bar(x - wd / 2, body[f"{k}_classifier"], wd, label="Classifier only", color="#e08a1e")
        a.bar(x + wd / 2, body[f"{k}_hmm"], wd, label="+ HMM/Viterbi", color="#d62728")
        a.set_xticks(x); a.set_xticklabels(body.trip_id, rotation=60, fontsize=7)
        a.set_title(title); a.set_ylim(0, 1.02)
    ax[0].legend()
    plt.tight_layout()
    os.makedirs(os.path.join(args.results, "plots"), exist_ok=True)
    plt.savefig(os.path.join(args.results, "plots", "hmm_vs_classifier.png"), dpi=140)


if __name__ == "__main__":
    main()
