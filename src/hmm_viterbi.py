"""HMM + Viterbi route reconstruction.

Hidden states
    For every GPS observation i the hidden state is one of its candidate road
    arcs (the same candidate set the classifier scored). State k of observation
    i keeps its arc_id, so the decoded path maps straight back to road segments.

Emission probability
    The classifier's score p(arc | observation) for that candidate, tempered and
    normalised over the observation's candidates:
        log e_i(k) = (log p_ik) / T  -  logsumexp_j (log p_ij) / T
    T = emission_temperature (T > 1 flattens, T < 1 sharpens the classifier).

Transition probability
    Between consecutive observations, from candidate a (obs i-1) to candidate b
    (obs i), we compare the distance the vehicle must have driven along the
    network with the straight-line distance the GPS moved:
        d_route(a,b) = remaining length on a  +  shortest path(to_node(a) ->
                       from_node(b))  +  position along b           (a != b)
        d_route(a,a) = (t_b - t_a) * length(a)                        (same arc)
        log P(a -> b) = - | d_route - d_gps | / beta
    beta = transition_beta_m (metres). Connected routes of about the right
    length score high; detours and impossible jumps score low.

Missing / disconnected cases
    * No shortest path within the search limit, or a route longer than the
      vehicle could drive at max_speed_mps: transition gets the small fallback
      probability `broken_transition_prob` (the chain is allowed to continue but
      is strongly penalised).
    * Time gap larger than gap_reset_s: the observations are treated as
      independent (all transitions equal), so the decoder restarts from the
      classifier scores alone.
    * Every observation always has at least one candidate (the candidate
      generator falls back to the nearest arc), so no observation is stateless.

Decoding
    Viterbi in log space: delta_i(k) = log e_i(k) + max_j [delta_{i-1}(j) +
    log P(j -> k)], with back-pointers, then a backward pass.

Interface used by pipeline.py:
    decode(trip, cand_df, cfg) -> np.ndarray of arc_ids, length trip.n_obs
    cand_df needs columns obs_idx, arc_id, proba, t_proj, arc_len_m.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DEFAULTS = {
    "emission_temperature": 1.0,
    "transition_beta_m": 50.0,
    "broken_transition_prob": 1e-6,
    "max_path_search_m": 2000.0,
    "max_speed_mps": 70.0,
    "gap_reset_s": 30.0,
    "search_slack_m": 150.0,
}


def _cfg(cfg):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return out


class Network:
    """Directed road graph with arc lengths and a cached bounded Dijkstra."""

    def __init__(self, trip):
        length = np.hypot(trip.arc_bx - trip.arc_ax, trip.arc_by - trip.arc_ay)
        w = np.maximum(length, 1e-3)
        df = pd.DataFrame({"u": trip.arcs[:, 0], "v": trip.arcs[:, 1], "w": w})
        df = df.groupby(["u", "v"], as_index=False)["w"].min()   # parallel arcs: keep shortest
        n = trip.n_nodes
        g = csr_matrix((df["w"].values, (df["u"].values, df["v"].values)), shape=(n, n))
        self.indptr, self.indices, self.data = g.indptr, g.indices, g.data
        self.length = length
        self.from_node = trip.arcs[:, 0]
        self.to_node = trip.arcs[:, 1]
        self._cache = {}                                          # node -> (limit, {node: dist})

    def reach(self, src, limit):
        """Distances from `src` to every node within `limit` metres."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        cap = max(limit, 400.0)
        dist = {src: 0.0}
        heap = [(0.0, src)]
        ip, ix, dt = self.indptr, self.indices, self.data
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for k in range(ip[u], ip[u + 1]):
                v = ix[k]
                nd = d + dt[k]
                if nd <= cap and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (cap, dist)
        return dist


def prepare(trip, cand_df, cfg):
    """Per-trip quantities that do not depend on beta / temperature."""
    c = _cfg(cfg)
    net = Network(trip)
    d = cand_df.sort_values(["obs_idx"], kind="stable")
    obs_ids = d["obs_idx"].values
    bounds = np.flatnonzero(np.diff(obs_ids)) + 1
    groups = np.split(np.arange(len(d)), bounds)
    arc_col = d["arc_id"].values
    p_col = np.clip(d["proba"].values, 1e-6, 1.0)
    t_col = d["t_proj"].values
    l_col = d["arc_len_m"].values

    steps = []                       # one dict per observation that has candidates
    for g in groups:
        steps.append({
            "obs": int(obs_ids[g[0]]),
            "arc": arc_col[g].astype(int),
            "logp": np.log(p_col[g]),
            "t": t_col[g],
            "len": l_col[g],
        })

    xy, ts = trip.track_xy, trip.track_t
    D = [None]                       # D[i]: route-distance matrix step i-1 -> i (None = reset)
    for i in range(1, len(steps)):
        a, b = steps[i - 1], steps[i]
        oa, ob = a["obs"], b["obs"]
        dt = ts[ob] - ts[oa]
        if dt > c["gap_reset_s"]:
            D.append(None)
            continue
        d_gps = float(np.hypot(*(xy[ob] - xy[oa])))
        limit = min(c["max_path_search_m"], 3.0 * d_gps + c["search_slack_m"])
        Ka, Kb = len(a["arc"]), len(b["arc"])
        M = np.full((Ka, Kb), np.inf)
        for ia in range(Ka):
            arc_a = a["arc"][ia]
            reach = net.reach(int(net.to_node[arc_a]), limit)
            rem_a = (1.0 - a["t"][ia]) * a["len"][ia]
            for ib in range(Kb):
                arc_b = b["arc"][ib]
                if arc_a == arc_b and b["t"][ib] >= a["t"][ia] - 1e-9:
                    M[ia, ib] = (b["t"][ib] - a["t"][ia]) * a["len"][ia]
                    continue
                sp = reach.get(int(net.from_node[arc_b]))
                if sp is not None:
                    M[ia, ib] = rem_a + sp + b["t"][ib] * b["len"][ib]
        D.append((M, d_gps, dt))
    return {"steps": steps, "D": D, "n_obs": trip.n_obs, "cfg": c}


def viterbi(prep, beta=None, temperature=None, broken=None):
    """Decode with the given parameters (defaults come from the config)."""
    c = prep["cfg"]
    beta = c["transition_beta_m"] if beta is None else beta
    temp = c["emission_temperature"] if temperature is None else temperature
    broken = c["broken_transition_prob"] if broken is None else broken
    log_broken = np.log(broken)
    steps, D = prep["steps"], prep["D"]

    def log_emission(s):
        z = s["logp"] / temp
        m = z.max()
        return z - (m + np.log(np.exp(z - m).sum()))

    delta = log_emission(steps[0])
    back = [None]
    for i in range(1, len(steps)):
        e = log_emission(steps[i])
        item = D[i]
        if item is None:                                   # large gap: restart from emissions
            trans = np.zeros((len(steps[i - 1]["arc"]), len(steps[i]["arc"])))
        else:
            M, d_gps, dt = item
            ok = np.isfinite(M) & (M <= c["max_speed_mps"] * max(dt, 1.0) + c["search_slack_m"])
            trans = np.where(ok, -np.abs(M - d_gps) / beta, log_broken)
        scores = delta[:, None] + trans
        bp = np.argmax(scores, axis=0)
        delta = scores[bp, np.arange(scores.shape[1])] + e
        back.append(bp)

    k = int(np.argmax(delta))
    path = [k]
    for i in range(len(steps) - 1, 0, -1):
        k = int(back[i][k])
        path.append(k)
    path.reverse()

    out = np.full(prep["n_obs"], -1, dtype=int)
    for s, k in zip(steps, path):
        out[s["obs"]] = s["arc"][k]
    # observations without any candidate row inherit the previous decoded arc
    last = -1
    for i in range(out.size):
        if out[i] < 0:
            out[i] = last
        else:
            last = out[i]
    return out


def decode(trip, cand_df, cfg):
    """Entry point used by the pipeline."""
    return viterbi(prepare(trip, cand_df, cfg))
