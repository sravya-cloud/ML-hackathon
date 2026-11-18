"""Tune the HMM decoder (transition beta, emission temperature) on held-out TRAIN trips.

A tuned classifier is fitted on 64 train trips; the other 16 train trips (random,
<= 4000 observations each) are decoded for every (beta, temperature) pair and the
pair with the best observation-level Top-1 is written to config.yaml and
results/hmm_best_params.json. Student-test and evaluator-test trips are not used.

Usage (from src/):
  python tune_hmm.py --data-root "<path to Student folder>"
"""
import argparse, json, os, time
import numpy as np, pandas as pd, yaml
from sklearn.base import clone

from data_loader import load_trip
from features import FEATURE_COLUMNS
from models import model_zoo
from labeling import label_observations
import hmm_viterbi as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("PATHSNAP_DATA"))
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--params-json", default="../results/best_params.json")
    ap.add_argument("--n-holdout", type=int, default=16)
    ap.add_argument("--max-obs", type=int, default=4000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = cfg["training"]["random_seed"]

    df = pd.read_parquet(os.path.join(args.results, "dataset_train.parquet"))
    n_obs = df.groupby("trip_id")["obs_idx"].max() + 1
    pool = sorted(n_obs[n_obs <= args.max_obs].index)
    rng = np.random.RandomState(seed)
    hold = sorted(rng.choice(pool, size=min(args.n_holdout, len(pool)), replace=False))
    print("held-out trips:", hold)

    est = clone(model_zoo(seed)["HistGradientBoosting"]["estimator"])
    if os.path.exists(args.params_json):
        est.set_params(**json.load(open(args.params_json)))
        print("using tuned params")
    tr = df[(~df.trip_id.isin(hold)) & (df.labeled_obs == 1)]
    t0 = time.time()
    est.fit(tr[FEATURE_COLUMNS].values, tr["label"].values)
    print(f"fit {len(tr)} rows in {time.time()-t0:.0f}s")

    betas = [10, 20, 40, 80, 160]
    temps = [0.5, 1.0, 2.0]
    hc = dict(cfg["hmm"])
    score = {(b, t): [0, 0] for b in betas for t in temps}   # correct, total
    base = [0, 0]
    for tid in hold:
        trip = load_trip(os.path.join(args.data_root, "train", tid), "train")
        g = df[df.trip_id == tid].copy()
        g["proba"] = est.predict_proba(g[FEATURE_COLUMNS].values)[:, 1]
        labels, _, _ = label_observations(trip, max_route_dist_m=cfg["labeling"]["max_route_dist_m"])
        lab = labels >= 0
        top = g.loc[g.groupby("obs_idx")["proba"].idxmax()]
        clf = np.full(trip.n_obs, -1); clf[top.obs_idx.values] = top.arc_id.values
        base[0] += int((clf[lab] == labels[lab]).sum()); base[1] += int(lab.sum())
        prep = H.prepare(trip, g[["obs_idx", "arc_id", "proba", "t_proj", "arc_len_m"]], hc)
        for b in betas:
            for t in temps:
                arcs = np.asarray(H.viterbi(prep, beta=b, temperature=t))
                score[(b, t)][0] += int((arcs[lab] == labels[lab]).sum())
                score[(b, t)][1] += int(lab.sum())
        print(tid, "done", flush=True)

    rows = [{"beta_m": b, "temperature": t, "top1": c / n} for (b, t), (c, n) in score.items()]
    out = pd.DataFrame(rows).sort_values("top1", ascending=False)
    out.to_csv(os.path.join(args.results, "hmm_tuning_grid.csv"), index=False)
    best = out.iloc[0]
    print(out.round(4).to_string(index=False))
    print(f"classifier-only top1 on held-out: {base[0]/base[1]:.4f}; best HMM {best.top1:.4f}")
    res = {"transition_beta_m": float(best.beta_m), "emission_temperature": float(best.temperature),
           "heldout_top1_hmm": float(best.top1), "heldout_top1_classifier": base[0] / base[1],
           "heldout_trips": list(hold)}
    json.dump(res, open(os.path.join(args.results, "hmm_best_params.json"), "w"), indent=2)

    # write chosen values into config.yaml (text edit keeps the comments)
    txt = open(args.config).read().splitlines()
    for i, ln in enumerate(txt):
        if ln.strip().startswith("emission_temperature:"):
            txt[i] = f"  emission_temperature: {res['emission_temperature']}"
        if ln.strip().startswith("transition_beta_m:"):
            txt[i] = f"  transition_beta_m: {res['transition_beta_m']}"
    open(args.config, "w").write("\n".join(txt) + "\n")
    print("config.yaml updated")


if __name__ == "__main__":
    main()
