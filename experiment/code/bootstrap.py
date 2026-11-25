"""Outbreak-level bootstrap of the main analysis (rebuilt 2026-11-25).

Scheme (pre-specified): B = 2000 replicates, seed 20261125; replicate b uses its own generator
numpy.random.default_rng([seed, b]) so that a resumed or parallel run equals an uninterrupted
one. Within each group x common-support cell the group's outbreaks are resampled with
replacement (same number); cells are fixed; pooled-case weights are recomputed per replicate.
A resampled cell without a recorded rate keeps worst-case bounds at every delta; the delta = 0
direction is then taken from the midpoint and delta* = 0 if the ordering is not guaranteed at
delta = 0 (amendment 1). Replicates are streamed to a JSONL file; only compact per-replicate
summaries are kept in memory. Quantiles: numpy method 'inverted_cdf' (no interpolation;
infinite delta* allowed, reported as null).

Usage (stand-alone):
  python bootstrap.py --derived data/derived --out rebuild/bootstrap_block.json \
      [--B 2000] [--workers 2] [--jsonl logs/bootstrap_replicates.jsonl]
      [--order group_cell|cell_group] [--draw integers|choice]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

import bounds as BD
import strata as S

_STATE = {}


def _prepare(main_df, t_main):
    """Per group x cell arrays of outbreak-level counts in cohort-file order."""
    groups, cells = t_main.groups, t_main.cells
    arrs = {}
    for g in groups:
        for c in cells:
            d = main_df[(main_df["group"] == g) & (main_df["cell"] == c)]
            arrs[(g, c)] = np.stack([d[k].to_numpy(dtype=float) for k in ("C", "H", "U", "H_rec", "K_rec")])
    return groups, cells, arrs


def _init(groups, cells, arrs, seed, order, draw, main_pairs):
    _STATE.update(groups=groups, cells=cells, arrs=arrs, seed=seed, order=order, draw=draw, main_pairs=main_pairs)


def _replicate(b: int):
    st = _STATE
    groups, cells, arrs = st["groups"], st["cells"], st["arrs"]
    rng = np.random.default_rng([st["seed"], b])
    G, K = len(groups), len(cells)
    sums = np.zeros((5, G, K))
    n = np.zeros((G, K))
    if st["order"] == "group_cell":
        seq = [(i, j) for i in range(G) for j in range(K)]
    elif st["order"] == "cell_group":
        seq = [(i, j) for j in range(K) for i in range(G)]
    else:  # "sorted": alphabetical (group, cell) keys, as pandas groupby would iterate
        seq = sorted(((i, j) for i in range(G) for j in range(K)), key=lambda ij: (groups[ij[0]], cells[ij[1]]))
    for i, j in seq:
        a = arrs[(groups[i], cells[j])]
        m = a.shape[1]
        if st["draw"] == "integers":
            idx = rng.integers(0, m, m)
        else:
            idx = rng.choice(m, size=m, replace=True)
        sums[:, i, j] = a[:, idx].sum(axis=1)
        n[i, j] = m
    t = S.Table(groups=list(groups), cells=list(cells), C=sums[0], H=sums[1], U=sums[2], H_rec=sums[3],
                K_rec=sums[4], n=n)
    t.w = S.weights(t, "pooled_cases")
    rec = {"b": b, "L": [], "U": [], "theta0": [], "L5": [], "U5": [], "undefined_strata": int(np.isnan(t.p_obs).sum())}
    for i in range(G):
        Lw, Uw = BD.worst_bounds(t, i)
        L0, U0 = BD.point_delta0(t, i)
        L5, U5 = BD.group_bounds(t, i, 0.05)
        rec["L"].append(Lw); rec["U"].append(Uw); rec["theta0"].append((L0 + U0) / 2.0)
        rec["L5"].append(L5); rec["U5"].append(U5)
    pairs = []
    for i in range(G):
        for j in range(i + 1, G):
            a, bb = groups[i], groups[j]
            wc = BD.verdict(rec["L"][i], rec["U"][i], rec["L"][j], rec["U"][j], a, bb)
            d5 = BD.verdict(rec["L5"][i], rec["U5"][i], rec["L5"][j], rec["U5"][j], a, bb)
            h, l = BD.direction_delta0(t, i, j)
            ex, _ = BD.delta_star(t, h, l, bisect=False)
            pairs.append({"pair": f"{a}|{bb}", "worst": wc, "delta5": d5, "higher0": groups[h],
                          "delta_star_pp": None if ex is None else ex * 100.0})
    rec["pairs"] = pairs
    return rec


def _q(x, q):
    return float(np.quantile(np.asarray(x, dtype=float), q, method="inverted_cdf"))


def _qn(x, q):
    v = _q(x, q)
    return None if not np.isfinite(v) else v


def run(main_df, t_main, main_res, B: int = 2000, seed: int = 20261125, workers: int = 2, jsonl=None,
        order: str = "group_cell", draw: str = "integers"):
    groups, cells, arrs = _prepare(main_df, t_main)
    main_pairs = main_res["pairs"]
    G = len(groups)
    pkeys = list(main_pairs.keys())
    L = np.zeros((B, G)); U = np.zeros((B, G)); T0 = np.zeros((B, G))
    worst = np.empty((B, len(pkeys)), dtype=object); d5 = np.empty_like(worst); hi0 = np.empty_like(worst)
    ds = np.zeros((B, len(pkeys)))
    n_undef = 0
    fh = open(jsonl, "w", encoding="utf-8") if jsonl else None
    init = (groups, cells, arrs, seed, order, draw, main_pairs)
    if workers and workers > 1:
        method = "fork" if "fork" in get_all_start_methods() else "spawn"
        ctx = get_context(method)
        pool = ctx.Pool(min(workers, 2), initializer=_init, initargs=init)
        it = pool.imap(_replicate, range(B), chunksize=50)
    else:
        _init(*init)
        pool = None
        it = map(_replicate, range(B))
    for rec in it:
        b = rec["b"]
        L[b], U[b], T0[b] = rec["L"], rec["U"], rec["theta0"]
        n_undef += rec["undefined_strata"] > 0
        for k, p in enumerate(rec["pairs"]):
            worst[b, k], d5[b, k], hi0[b, k] = p["worst"], p["delta5"], p["higher0"]
            ds[b, k] = np.inf if p["delta_star_pp"] is None else p["delta_star_pp"]
        if fh:
            fh.write(json.dumps(rec, allow_nan=False) + "\n")
    if pool:
        pool.close(); pool.join()
    if fh:
        fh.close()

    per_group = OrderedDict()
    for i, g in enumerate(groups):
        per_group[g] = OrderedDict(
            L_p2_5=_q(L[:, i], 0.025), L_p50=_q(L[:, i], 0.5), L_p97_5=_q(L[:, i], 0.975),
            U_p2_5=_q(U[:, i], 0.025), U_p50=_q(U[:, i], 0.5), U_p97_5=_q(U[:, i], 0.975),
            theta_delta0_p2_5=_q(T0[:, i], 0.025), theta_delta0_p97_5=_q(T0[:, i], 0.975))
    pairs = OrderedDict()
    for k, key in enumerate(pkeys):
        mp = main_pairs[key]
        a, bb = mp["A"], mp["B"]
        w, v5 = worst[:, k], d5[:, k]
        pairs[key] = OrderedDict(
            share_worst_A=float(np.mean(w == a)), share_worst_B=float(np.mean(w == bb)),
            share_worst_undetermined=float(np.mean(w == BD.UNDET)),
            share_worst_main_verdict=float(np.mean(w == mp["worst_case"])),
            share_delta5_A=float(np.mean(v5 == a)), share_delta5_B=float(np.mean(v5 == bb)),
            share_delta5_undetermined=float(np.mean(v5 == BD.UNDET)),
            share_delta5_main_verdict=float(np.mean(v5 == mp["verdict_by_delta_pp"]["5"])),
            share_same_delta0_direction=float(np.mean(hi0[:, k] == mp["higher_at_delta0"])),
            share_holds_without_assumption=float(np.mean(w != BD.UNDET)),
            delta_star_pp_p2_5=_qn(ds[:, k], 0.025), delta_star_pp_p50=_qn(ds[:, k], 0.5),
            delta_star_pp_p97_5=_qn(ds[:, k], 0.975))

    def dist(mat):
        cnt = np.sum(mat != BD.UNDET, axis=1)
        vals, c = np.unique(cnt, return_counts=True)
        return OrderedDict((str(int(v)), float(x / len(cnt))) for v, x in zip(vals, c))

    return OrderedDict(
        B=B, B_planned=2000, seed=seed,
        scheme="resample outbreaks with replacement within group x common-support cell; cells fixed; pooled-case "
               "weights recomputed per replicate",
        quantile_method="inverted_cdf (no interpolation; infinite delta* allowed)",
        per_group=per_group, pairs=pairs,
        n_determined_worst_distribution=dist(worst), n_determined_delta5_distribution=dist(d5),
        jsonl="logs/bootstrap_replicates.jsonl",
    )


def main(argv=None):
    import pandas as pd
    import analyze as A
    import nors_rules as R
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--order", default="group_cell", choices=["group_cell", "cell_group"])
    ap.add_argument("--draw", default="integers", choices=["integers", "choice"])
    args = ap.parse_args(argv)
    cohort = pd.read_csv(args.derived / "cohort_outbreaks.csv", dtype={"H_raw": str, "K_raw": str, "C_raw": str,
                                                                       "month": str, "pilot_group": str},
                         keep_default_na=False, na_values=[""])
    for c in ("H_raw", "K_raw", "C_raw", "pilot_group"):
        cohort[c] = cohort[c].fillna("")
    main_df = A.cohorts(cohort)["keep_main"]
    t = S.build(main_df, R.GROUPS, "setting_period", threshold=A.THRESHOLD)
    res = A.analyze_table(t)
    blk = run(main_df, t, res, B=args.B, workers=args.workers, jsonl=args.jsonl, order=args.order, draw=args.draw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blk, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
