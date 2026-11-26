"""Main, sensitivity and robustness analyses -> results JSON (rebuilt 2026-11-25).

Usage:
  python analyze.py --derived data/derived --out rebuild_check/results_rebuilt.json \
      [--bootstrap-B 2000] [--workers 2] [--bootstrap-jsonl logs/bootstrap_replicates.jsonl]
      [--skip-bootstrap]

Reads cohort_outbreaks.csv, eligibility_flow.json and record_audit_summary.json written by
prepare_data.py. Deterministic except the bootstrap, which is seeded (seed 20261125,
per-replicate generator numpy.random.default_rng([seed, b])).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

import bounds as B
import inference_voi as IV
import nors_rules as R
import strata as S

SEED = 20261125
DELTA_GRID_PP = [0, 1, 2, 5, 10]
DELTA_FINE_PP = [round(i * 0.05, 2) for i in range(201)]
TWO_D_PP = [round(i * 0.25, 2) for i in range(81)]
THRESHOLD = 3


def labels_for(groups):
    return {g: (R.GROUP_LABELS.get(g) or R.COMBINED_NOV_LABEL) for g in groups}


def pair_list(groups):
    return [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]


def fnone(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def analyze_table(t: S.Table, detail: bool = False) -> OrderedDict:
    groups = t.groups
    lab = labels_for(groups)
    per_group = OrderedDict()
    for i, g in enumerate(groups):
        Lw, Uw = B.worst_bounds(t, i)
        L0, U0 = B.point_delta0(t, i)
        bd = OrderedDict()
        for d in DELTA_GRID_PP:
            L, U = B.group_bounds(t, i, d / 100.0)
            bd[str(d)] = OrderedDict(L=L, U=U)
        cases, unk = float(t.C[i].sum()), float(t.U[i].sum())
        per_group[g] = OrderedDict(
            label=lab[g], L_worst=Lw, U_worst=Uw, width_worst=Uw - Lw, theta_delta0=(L0 + U0) / 2.0,
            delta0_interval=[L0, U0], strata_without_recorded_rate=int(np.isnan(t.p_obs[i]).sum()),
            bounds_by_delta_pp=bd, cases=cases, unknown_cases=unk, hospitalized_known=float(t.H[i].sum()),
            unknown_share=unk / cases, outbreaks=int(t.n[i].sum()),
        )
    pairs = OrderedDict()
    max_bis = 0.0
    for i, j in pair_list(groups):
        a, b = groups[i], groups[j]
        Lw_a, Uw_a = B.worst_bounds(t, i)
        Lw_b, Uw_b = B.worst_bounds(t, j)
        wc = B.verdict(Lw_a, Uw_a, Lw_b, Uw_b, a, b)
        h, l = B.direction_delta0(t, i, j)
        ex, bis = B.delta_star(t, h, l, bisect=True)
        if ex is not None and bis is not None:
            max_bis = max(max_bis, abs(ex - bis) * 100.0)
        vb = OrderedDict()
        for d in DELTA_GRID_PP:
            La, Ua = B.group_bounds(t, i, d / 100.0)
            Lb, Ub = B.group_bounds(t, j, d / 100.0)
            vb[str(d)] = B.verdict(La, Ua, Lb, Ub, a, b)
        pairs[f"{a}|{b}"] = OrderedDict(
            A=a, B=b, worst_case=wc, higher_at_delta0=groups[h], lower_at_delta0=groups[l],
            holds_without_assumption=wc != B.UNDET,
            delta_star_pp=None if ex is None else ex * 100.0,
            delta_star_bisect_pp=None if bis is None else bis * 100.0,
            verdict_by_delta_pp=vb,
            # overlap = U_l - L_h: how far the lower group's upper bound reaches above the higher group's
            # lower bound (h, l = higher / lower group at delta = 0)
            worst_case_overlap=(B.worst_bounds(t, l)[1] - B.worst_bounds(t, h)[0]) if wc == B.UNDET else None,
        )
    n_det = OrderedDict(worst_case=sum(p["worst_case"] != B.UNDET for p in pairs.values()))
    for d in DELTA_GRID_PP:
        n_det[str(d)] = sum(p["verdict_by_delta_pp"][str(d)] != B.UNDET for p in pairs.values())
    order = sorted(groups, key=lambda g: -per_group[g]["theta_delta0"])
    out = OrderedDict(
        groups=list(groups), n_pairs=len(pairs),
        cells=[OrderedDict(cell=c, weight=float(w)) for c, w in zip(t.cells, t.w)], n_cells=len(t.cells),
        per_group=per_group, pairs=pairs, n_determined=n_det, order_by_delta0=order,
    )
    if detail:
        sd = []
        p_obs = t.p_obs
        for i, g in enumerate(groups):
            for s, c in enumerate(t.cells):
                sd.append(OrderedDict(group=g, cell=c, weight=float(t.w[s]), C=float(t.C[i, s]),
                                      H=float(t.H[i, s]), U=float(t.U[i, s]), p_obs=fnone(p_obs[i, s]),
                                      L=float(t.H[i, s] / t.C[i, s]),
                                      U_bound=float((t.H[i, s] + t.U[i, s]) / t.C[i, s])))
        out["strata_detail"] = sd
        two_d = OrderedDict()
        for i, j in pair_list(groups):
            h, l = B.direction_delta0(t, i, j)
            bnd = []
            for d in TWO_D_PP:
                v = B.lo_boundary(t, h, l, d / 100.0)
                bnd.append(None if v is None else v * 100.0)
            two_d[f"{groups[i]}|{groups[j]}"] = OrderedDict(hi=groups[h], lo=groups[l], d_hi_pp=list(TWO_D_PP),
                                                            d_lo_boundary_pp=bnd)
        out["two_d"] = two_d
        d1 = 0.0
        for i in range(len(groups)):
            Lw, Uw = B.worst_bounds(t, i)
            L1, U1 = B.group_bounds(t, i, 1.0)
            d1 = max(d1, abs(L1 - Lw), abs(U1 - Uw))
        out["checks"] = OrderedDict(
            max_abs_diff_exact_vs_bisection_pp=max_bis, weights_sum=float(np.sum(t.w)),
            max_abs_diff_delta1_vs_worst_case=d1, strata_with_undefined_p_obs=int(np.isnan(t.p_obs).sum()),
        )
    return out


def delta_fine(t: S.Table) -> OrderedDict:
    det = OrderedDict()
    for i, j in pair_list(t.groups):
        row = []
        for d in DELTA_FINE_PP:
            La, Ua = B.group_bounds(t, i, d / 100.0)
            Lb, Ub = B.group_bounds(t, j, d / 100.0)
            row.append(B.verdict(La, Ua, Lb, Ub, "A", "B") != B.UNDET)
        det[f"{t.groups[i]}|{t.groups[j]}"] = row
    return OrderedDict(grid_pp=list(DELTA_FINE_PP), determined=det)


# --------------------------------------------------------------------------- cohorts
def cohorts(cohort: pd.DataFrame, suspected: pd.DataFrame | None = None) -> dict:
    kept = cohort[cohort["kept"]]
    main = kept[kept["in_main"]].copy()
    out = {
        "keep_main": main,
        "keep_single_etiology": main[main["single_etiology"]].copy(),
        "keep_complete_only": main[main["in_complete_only"]].copy(),
        "keep_pilot_rule": kept[kept["in_pilot_rule"]].assign(group=lambda d: d["pilot_group"]).copy(),
        "keep_excl_healthcare": main[~main["is_healthcare"].astype(bool)].copy(),
    }
    if suspected is not None and len(suspected) > 0:
        sus_kept = suspected[suspected["kept"]].copy()
        out["keep_include_suspected"] = pd.concat([main, sus_kept], ignore_index=True)
    else:
        out["keep_include_suspected"] = main.copy()
    return out


ROBUSTNESS = OrderedDict([
    ("R1a_crude_all", ("Crude: no standardization, all eligible outbreaks", {"kind": "crude", "cohort": "keep_main"})),
    ("R1b_crude_support", ("Crude within the common support (no standardization)",
                           {"kind": "crude_support", "cohort": "keep_main"})),
    ("R2_period_only", ("Period-only standardization (3 strata, all settings)", {"kind": "period", "cohort": "keep_main"})),
    ("R3a_single_etiology", ("Cohort: single-etiology confirmed only", {"cohort": "keep_single_etiology"})),
    ("R3b_complete_only", ("Cohort: records with any blank H or K dropped", {"cohort": "keep_complete_only"})),
    ("R3c_pilot_rule", ("Cohort: pilot rule (exact strings, 3,703 outbreaks)", {"cohort": "keep_pilot_rule"})),
    ("R4a_threshold_2", ("Common support ≥ 2 outbreaks per group per cell", {"threshold": 2})),
    ("R4b_threshold_5", ("Common support ≥ 5 outbreaks per group per cell", {"threshold": 5})),
    ("R5_named_settings", ("Drop Unknown and Multiple setting cells", {"drop_settings": ["Unknown", "Multiple"]})),
    ("R6a_w_group_balanced", ("Weights: group-balanced case mix", {"weights": "group_balanced"})),
    ("R6b_w_pooled_outbreaks", ("Weights: pooled outbreak mix", {"weights": "pooled_outbreaks"})),
    ("R6c_w_uniform", ("Weights: uniform over cells", {"weights": "uniform"})),
    ("R7_norovirus_combined", ("Norovirus GI+GII combined (5 groups, 10 pairs)", {"combined": True})),
    ("R8_include_suspected", ("Cohort: include suspected study-group etiologies", {"cohort": "keep_include_suspected"})),
    ("R9_excl_healthcare", ("Cohort: exclude healthcare/LTCF outbreaks", {"cohort": "keep_excl_healthcare"})),
    ("R10_outbreak_weighted", ("Within-cell aggregation: equal weight per outbreak", {"kind": "outbreak_weighted", "cohort": "keep_main"})),
])


def variant_table(spec: dict, co: dict, main_cells: list, main_weights: np.ndarray | None = None) -> tuple[S.Table, int]:
    cohort = co[spec.get("cohort", "keep_main")]
    kind = spec.get("kind")
    if kind == "crude":
        return S.build(cohort, R.GROUPS, "all", threshold=0), len(cohort)
    if kind == "crude_support":
        sub = cohort[cohort["cell"].isin(main_cells)]
        return S.build(sub, R.GROUPS, "all", threshold=0), len(cohort)
    if kind == "period":
        return S.build(cohort, R.GROUPS, "period", threshold=THRESHOLD), len(cohort)
    if kind == "outbreak_weighted":
        sub = cohort[cohort["cell"].isin(main_cells)].copy()
        sub["H"] = sub["H"] / sub["C"]
        sub["U"] = sub["U"] / sub["C"]
        sub["C"] = 1.0
        t_ow = S.build(sub, R.GROUPS, "setting_period", fixed_cells=main_cells)
        if main_weights is not None:
            t_ow.w = main_weights.copy()
        return t_ow, len(cohort)
    if spec.get("combined"):
        c2 = cohort.assign(group=cohort["group"].replace({"NOV_GI": "NOV", "NOV_GII": "NOV"}))
        groups = ["SAL", "STEC", "CAMP", "CPERF", "NOV"]
        return S.build(c2, groups, "setting_period", threshold=THRESHOLD), len(cohort)
    return S.build(cohort, R.GROUPS, "setting_period", threshold=spec.get("threshold", THRESHOLD),
                   weight_scheme=spec.get("weights", "pooled_cases"),
                   drop_settings=tuple(spec.get("drop_settings", ()))), len(cohort)


def changes_vs_main(main_res, var_res):
    rows = []
    for k, pm in main_res["pairs"].items():
        pv = var_res["pairs"][k]
        rows.append(OrderedDict(
            pair=k, worst_case_main=pm["worst_case"], worst_case_variant=pv["worst_case"],
            delta5_main=pm["verdict_by_delta_pp"]["5"], delta5_variant=pv["verdict_by_delta_pp"]["5"],
            delta_star_main_pp=pm["delta_star_pp"], delta_star_variant_pp=pv["delta_star_pp"],
            direction_main=pm["higher_at_delta0"], direction_variant=pv["higher_at_delta0"]))
    return rows


def robustness(co: dict, main_res, main_cells, main_weights: np.ndarray | None = None) -> OrderedDict:
    out = OrderedDict()
    for key, (label, spec) in ROBUSTNESS.items():
        t, n_coh = variant_table(spec, co, main_cells, main_weights=main_weights)
        res = analyze_table(t)
        res["n_outbreaks_in_cohort"] = int(n_coh)
        res["label"] = label
        res["spec"] = spec
        if not spec.get("combined"):
            ch = changes_vs_main(main_res, res)
            res["changes_vs_main"] = ch
            res["n_worst_case_verdicts_changed"] = sum(c["worst_case_main"] != c["worst_case_variant"] for c in ch)
            res["n_delta5_verdicts_changed"] = sum(c["delta5_main"] != c["delta5_variant"] for c in ch)
            res["n_delta0_directions_changed"] = sum(c["direction_main"] != c["direction_variant"] for c in ch)
        out[key] = res
    return out


# --------------------------------------------------------------------------- retention, width
def crude_block(d: pd.DataFrame) -> OrderedDict:
    C, H, U = d["C"].sum(), d["H"].sum(), d["U"].sum()
    p = d["H_rec"].sum() / d["K_rec"].sum()
    return OrderedDict(L=float(H / C), U=float((H + U) / C), theta_delta0=float((H + U * p) / C),
                       recorded_proportion=float(p), unknown_share=float(U / C))


def retention(main: pd.DataFrame, cells: list, strata_csv: Path | None) -> OrderedDict:
    per = OrderedDict()
    sup = main[main["cell"].isin(cells)]
    for g in R.GROUPS:
        e, s = main[main["group"] == g], sup[sup["group"] == g]
        dropped = e[~e["cell"].isin(cells)]
        bys = dropped.groupby("setting_group")["C"].sum()
        byp = dropped.groupby("period")["C"].sum()
        bys = bys[bys > 0].sort_values(ascending=False, kind="mergesort")
        byp = byp[byp > 0].sort_values(ascending=False, kind="mergesort")
        per[g] = OrderedDict(
            label=R.GROUP_LABELS[g], eligible_outbreaks=int(len(e)), eligible_cases=float(e["C"].sum()),
            support_outbreaks=int(len(s)), support_cases=float(s["C"].sum()),
            support_outbreaks_pct=100.0 * len(s) / len(e), support_cases_pct=100.0 * s["C"].sum() / e["C"].sum(),
            dropped_cases=float(dropped["C"].sum()),
            dropped_cases_by_setting=OrderedDict((k, float(v)) for k, v in bys.items()),
            dropped_cases_by_period=OrderedDict((k, float(v)) for k, v in byp.items()),
            crude_before=crude_block(e), crude_after=crude_block(s),
        )
    total = OrderedDict(eligible_outbreaks=int(len(main)), eligible_cases=float(main["C"].sum()),
                        support_outbreaks=int(len(sup)), support_cases=float(sup["C"].sum()))
    maxdiff = None
    if strata_csv is not None and strata_csv.exists():
        sc = pd.read_csv(strata_csv)
        diffs = []
        for g in R.GROUPS:
            x = sc[sc["group"] == g]
            diffs.append(abs(x["C"].sum() - per[g]["eligible_cases"]))
            diffs.append(abs(x[x["in_common_support"]]["C"].sum() - per[g]["support_cases"]))
            diffs.append(abs(x[x["in_common_support"]]["outbreaks"].sum() - per[g]["support_outbreaks"]))
        maxdiff = float(max(diffs))
    return OrderedDict(per_group=per, total=total, max_abs_diff_vs_step1_file=maxdiff)


def width_decomposition(t: S.Table, main_res) -> OrderedDict:
    contrib = {}
    per = OrderedDict()
    for i, g in enumerate(t.groups):
        c = t.w * t.U[i] / t.C[i]
        contrib[g] = c
        width = float(np.sum(c))
        order = np.argsort(-c, kind="mergesort")[:3]
        per[g] = OrderedDict(width=width, top_cells=[OrderedDict(
            cell=t.cells[s], contribution=float(c[s]), share_of_width=float(c[s] / width),
            unknown_cases=float(t.U[i, s])) for s in order])
    und = OrderedDict()
    for k, p in main_res["pairs"].items():
        if p["worst_case"] != B.UNDET:
            continue
        items = []
        for g in (p["A"], p["B"]):
            i = t.groups.index(g)
            for s in range(len(t.cells)):
                items.append((float(contrib[g][s]), g, s))
        items.sort(key=lambda x: -x[0])
        und[k] = OrderedDict(overlap=p["worst_case_overlap"], top_group_cells=[OrderedDict(
            group=g, cell=t.cells[s], contribution=v, unknown_cases=float(t.U[t.groups.index(g), s]))
            for v, g, s in items[:3]])
    return OrderedDict(per_group=per, undetermined_pairs=und)


def pilot_grid(co: dict) -> list:
    rows = []
    for name, key in (("main", "keep_main"), ("pilot_rule", "keep_pilot_rule")):
        for thr in range(1, 7):
            t = S.build(co[key], R.GROUPS, "setting_period", threshold=thr)
            res = analyze_table(t)
            rows.append(OrderedDict(
                cohort=name, threshold=thr, n_cells=res["n_cells"],
                determined_worst=res["n_determined"]["worst_case"], determined_delta5=res["n_determined"]["5"],
                determined_pairs_worst=[k for k, p in res["pairs"].items() if p["worst_case"] != B.UNDET]))
    return rows


PILOT_TARGETS = OrderedDict(eligible=3799, analysed=3703, excluded=96, common_cells=10, case_retention_pct="54-80",
                            determined_worst_case=6, determined_delta5=12, n_pairs=15)

TEXT = OrderedDict(
    task="Partial-identification ranking of six foodborne pathogen groups by the standardized hospitalization "
         "proportion among reported outbreak cases (CDC NORS 2009-2023)",
    task_slug="nors-hosp-ranking-partial-id",
    dataset="CDC National Outbreak Reporting System (NORS) public dataset, data.cdc.gov id 5xkq-dg7x, snapshot "
            "2026-11-25 (portal rows updated 2024-12-20)",
    metrics=["L_worst_case", "U_worst_case", "theta_delta0", "L_delta5", "U_delta5"],
    metric_definitions=OrderedDict([
        ("L_worst_case", "standardized lower bound, all unrecorded-status cases not hospitalized"),
        ("U_worst_case", "standardized upper bound, all unrecorded-status cases hospitalized"),
        ("theta_delta0", "standardized proportion if unrecorded-status cases had the recorded rate (delta = 0)"),
        ("L_delta5 / U_delta5", "bounds if the unrecorded-status rate is within 5 percentage points of the "
                                "recorded rate"),
    ]),
    statistics=OrderedDict(std_ddof=0, note="Sample bounds are deterministic identification intervals; sampling "
                           "uncertainty is quantified by stratified outbreak-level bootstrap, state-clustered "
                           "M-estimation sandwich covariances, Imbens-Manski 95% CIs, and Max-T FWER tests."),
    notes="All numbers are recomputed from the 2026-11-25 snapshot. 'undetermined' means the recorded data do not "
          "fix the direction; it does not mean equal. delta is a scenario parameter, not an estimate.",
    pilot_note="Pilot figures come from earlier exploratory checks and are retained to document cohort convergence.",
)


def summary_block(main_res) -> OrderedDict:
    out = OrderedDict()
    for g, pg in main_res["per_group"].items():
        m = lambda v: OrderedDict(mean=v, std=None, n_seeds=1)
        out[pg["label"]] = OrderedDict(
            L_worst_case=m(pg["L_worst"]), U_worst_case=m(pg["U_worst"]), theta_delta0=m(pg["theta_delta0"]),
            L_delta5=m(pg["bounds_by_delta_pp"]["5"]["L"]), U_delta5=m(pg["bounds_by_delta_pp"]["5"]["U"]))
    return out


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bootstrap-B", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bootstrap-jsonl", type=Path, default=None)
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--bootstrap-json", type=Path, default=None,
                    help="reuse a bootstrap block written by bootstrap.py instead of recomputing")
    args = ap.parse_args(argv)
    t0 = time.time()

    cohort = pd.read_csv(args.derived / "cohort_outbreaks.csv", dtype={"H_raw": str, "K_raw": str, "C_raw": str,
                                                                       "month": str, "pilot_group": str},
                         keep_default_na=False, na_values=[""])
    for c in ("H_raw", "K_raw", "C_raw", "pilot_group"):
        cohort[c] = cohort[c].fillna("")
    sus_path = args.derived / "suspected_cohort_outbreaks.csv"
    suspected = pd.read_csv(sus_path, dtype={"H_raw": str, "K_raw": str, "C_raw": str, "month": str},
                            keep_default_na=False, na_values=[""]) if sus_path.exists() else None
    co = cohorts(cohort, suspected=suspected)
    main_df = co["keep_main"]
    t_main = S.build(main_df, R.GROUPS, "setting_period", threshold=THRESHOLD)
    main_res = analyze_table(t_main, detail=True)
    print(f"main: {main_res['n_cells']} cells, determined {dict(main_res['n_determined'])}", flush=True)

    res = OrderedDict()
    res.update((k, TEXT[k]) for k in ("task", "task_slug", "dataset", "metrics", "metric_definitions"))
    res["seeds"] = [SEED]
    res["statistics"] = TEXT["statistics"]
    res["simulated"] = False
    raw_meta = json.load(open(args.derived / "data_manifest.json"))
    res["provenance"] = OrderedDict(
        mode="measured", simulated=False,
        source="measured: code/analyze.py on the audited cohort built by code/prepare_data.py from the CDC NORS snapshot",
        data_manifest="data/derived/data_manifest.json",
        raw_snapshot_sha256=raw_meta["raw_sha256"], cohort_file="data/derived/cohort_outbreaks.csv",
        cohort_sha256=file_sha(args.derived / "cohort_outbreaks.csv"),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        spec_hash=hashlib.sha256("".join(
            (Path(__file__).parent / f).read_text(encoding="utf-8")
            for f in ("analyze.py", "bounds.py", "strata.py", "nors_rules.py", "bootstrap.py", "inference_voi.py")
            if (Path(__file__).parent / f).exists()).encode()).hexdigest()[:16],
        analysis_version="3.0", hardware="CPU only", wall_clock_seconds=None,
    )
    res["summary"] = summary_block(main_res)
    res["groups"] = OrderedDict((g, R.GROUP_LABELS[g]) for g in R.GROUPS)
    res["delta_grid_pp"] = list(DELTA_GRID_PP)
    res["main"] = main_res
    res["delta_fine"] = delta_fine(t_main)
    res["robustness"] = robustness(co, main_res, t_main.cells, main_weights=t_main.w)
    print("robustness done", flush=True)
    res["retention"] = retention(main_df, t_main.cells, args.derived / "strata_counts.csv")
    res["width_decomposition"] = width_decomposition(t_main, main_res)
    if args.bootstrap_json is not None:
        res["bootstrap"] = json.load(open(args.bootstrap_json))
    elif not args.skip_bootstrap:
        import bootstrap as BS
        res["bootstrap"] = BS.run(main_df, t_main, main_res, B=args.bootstrap_B, seed=SEED, workers=args.workers,
                                  jsonl=args.bootstrap_jsonl)
    excl_path = args.derived / "excluded_outbreaks_summary.json"
    res["audit"] = OrderedDict(
        record_audit_summary=json.load(open(args.derived / "record_audit_summary.json")),
        eligibility_flow=json.load(open(args.derived / "eligibility_flow.json")),
        excluded_outbreaks_summary=(json.load(open(excl_path)) if excl_path.exists() else None),
    )
    res["pilot"] = OrderedDict(targets=PILOT_TARGETS, grid=pilot_grid(co), note=TEXT["pilot_note"])
    up = IV.run_all(main_df, t_main, main_res, seed=SEED)
    if res.get("bootstrap") and "pair_dL_worst_cov" in res["bootstrap"]:
        bt_cov_raw = np.asarray(res["bootstrap"]["pair_dL_worst_cov"], dtype=float)
        raw_keys = list(main_res["pairs"].keys())
        raw_idx = {k: i for i, k in enumerate(raw_keys)}
        st_pairs = up["stacked_inference"]["pairs"]
        perm = [raw_idx[k] for k in st_pairs.keys()]
        bt_cov = bt_cov_raw[np.ix_(perm, perm)]
        dL_vec = np.array([st_pairs[k]["dL_worst"] for k in st_pairs], dtype=float)
        bt_se = np.sqrt(np.maximum(np.diag(bt_cov), 1e-20))
        p_bt, _, c_bt = IV.max_t_stepdown(dL_vec / bt_se, bt_cov, alpha=0.05, n_sim=200000, seed=SEED)
        up["stacked_inference"]["bootstrap_cov_fwer"] = OrderedDict(
            c_sim_boot=float(c_bt),
            p_fwer_boot=[float(x) for x in p_bt],
            n_sig_boot=int(sum(x <= 0.05 for x in p_bt)),
            pair_order=list(st_pairs.keys()),
        )
    res["stacked_inference"] = up["stacked_inference"]
    res["cluster_diagnostics"] = up["cluster_diagnostics"]
    res["small_sample_calibration"] = up["small_sample_calibration"]
    res["empirical_calibration"] = up["empirical_calibration"]
    res["followup_masking"] = up["followup_masking"]
    print("stacked inference, calibration & follow-up done", flush=True)
    res["notes"] = TEXT["notes"]
    res["estimates"] = []
    res["estimates_note"] = ("Point-wise identification bounds appear in `main`; uniformly valid Imbens-Manski 95% CIs, "
                             "outbreak-HC1/state-clustered Max-T FWER p-values, and rank confidence sets appear in "
                             "`stacked_inference`.")
    res["provenance"]["wall_clock_seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    print(f"wrote {args.out} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
