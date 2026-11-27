"""Compare a rebuilt results JSON with the canonical analysis results, leaf by leaf.

Usage:
  python compare_results.py --preserved analysis/results.json \
      --rebuilt rebuild_check/results_rebuilt.json --out-json rebuild_check/rebuild_check.json \
      --out-md rebuild_check/rebuild_check.md [--tol 1e-9] [--num-tol 1e-3] [--bootstrap-note TEXT]

Rules: integers, booleans, strings and nulls must be identical; deterministic floats must agree
within --tol (default 1e-9); numerical simulation/root-finding sections (stacked_inference,
small_sample_calibration, cluster_double_bootstrap, cluster_double_bootstrap_by_def,
parametric_calibration, plausible_scenarios, cluster_diagnostics, empirical_calibration) allow
--num-tol (default 1e-3) for cross-platform BLAS/MC differences. Run metadata in `provenance`
is excluded from equality counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True

RUN_METADATA = {"timestamp", "spec_hash", "wall_clock_seconds", "analysis_version", "hardware", "cohort_sha256"}
ADDED_OK = {"estimates", "estimates_note"}
NUM_SECTIONS = {
    "stacked_inference",
    "small_sample_calibration",
    "cluster_double_bootstrap",
    "cluster_double_bootstrap_by_def",
    "parametric_calibration",
    "plausible_scenarios",
    "cluster_diagnostics",
    "empirical_calibration",
    "supplementary_audits",
}
TOL = 1e-9
NUM_TOL = 1e-3


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                out.append((path + [k], "missing_in_rebuilt", a[k], None))
            else:
                walk(a[k], b[k], path + [k], out)
        for k in b:
            if k not in a:
                out.append((path + [k], "extra_in_rebuilt", None, b[k]))
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "length_differs", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [i], out)
        return
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if num(a) and num(b):
        if a == b:
            out.append((path, "exact", a, b))
        elif isinstance(a, float) or isinstance(b, float):
            d = abs(float(a) - float(b))
            tol_here = NUM_TOL if (path and path[0] in NUM_SECTIONS) else TOL
            if math.isfinite(d):
                out.append((path, "float_close" if d <= tol_here else "float_differs", a, b))
            else:
                out.append((path, "float_differs", a, b))
        else:
            out.append((path, "int_differs", a, b))
        return
    out.append((path, "exact" if (a == b and type(a) == type(b)) else "value_differs", a, b))


def bootstrap_mc_check(pa: dict, pb: dict, z_max: float = 3.0, q_sd_max: float = 0.25) -> OrderedDict:
    """Monte Carlo comparison of two bootstrap blocks with the same scheme but (possibly) different
    random streams. Shares: z = |diff| / sqrt((p1(1-p1) + p2(1-p2)) / B), tolerance |z| <= z_max.
    Bound quantiles: |diff| <= q_sd_max x bootstrap SD, SD = (p97.5 - p2.5) / 3.92 (preserved run);
    the Monte Carlo SE of the difference of two 2.5%-quantiles at B = 2000 is about 0.085 SD, so
    0.25 SD is about 3 SE. delta* quantiles are reported, not tested (heavy tails, infinite values)."""
    Bn = int(pa["B"])
    shares, worst = [], None
    for key, x in pa["pairs"].items():
        y = pb["pairs"][key]
        for f, v in x.items():
            if not f.startswith("share"):
                continue
            w = y[f]
            se = math.sqrt((v * (1 - v) + w * (1 - w)) / Bn)
            d = abs(v - w)
            z = 0.0 if d == 0 else (float("inf") if se == 0 else d / se)
            shares.append(OrderedDict(pair=key, field=f, preserved=v, rebuilt=w, abs_diff=d, z=z))
    for kind in ("n_determined_worst_distribution", "n_determined_delta5_distribution"):
        keys = set(pa[kind]) | set(pb[kind])
        for k in sorted(keys, key=int):
            v, w = pa[kind].get(k, 0.0), pb[kind].get(k, 0.0)
            se = math.sqrt((v * (1 - v) + w * (1 - w)) / Bn)
            d = abs(v - w)
            z = 0.0 if d == 0 else (float("inf") if se == 0 else d / se)
            shares.append(OrderedDict(pair=kind, field=k, preserved=v, rebuilt=w, abs_diff=d, z=z))
    quants = []
    for g, x in pa["per_group"].items():
        y = pb["per_group"][g]
        for m in ("L", "U", "theta_delta0"):
            lo, hi = x.get(f"{m}_p2_5"), x.get(f"{m}_p97_5")
            sd = (hi - lo) / 3.92
            for f in [k for k in x if k.startswith(m + "_p")]:
                d = abs(x[f] - y[f])
                quants.append(OrderedDict(group=g, field=f, preserved=x[f], rebuilt=y[f], abs_diff=d,
                                          diff_in_bootstrap_sd=d / sd if sd > 0 else 0.0))
    ds = []
    for key, x in pa["pairs"].items():
        y = pb["pairs"][key]
        for f in ("delta_star_pp_p2_5", "delta_star_pp_p50", "delta_star_pp_p97_5"):
            v, w = x[f], y[f]
            ds.append(OrderedDict(pair=key, field=f, preserved=v, rebuilt=w,
                                  abs_diff_pp=None if (v is None or w is None) else abs(v - w),
                                  infinite_status_same=(v is None) == (w is None)))

    def rng_of(keys, pp):
        vals = [pp["pairs"][k]["share_worst_main_verdict"] for k in keys]
        return [min(vals), max(vals)]
    det = [k for k, x in pa["pairs"].items() if x["share_holds_without_assumption"] >= 0.5]
    und = [k for k in pa["pairs"] if k not in det]
    mode = lambda d: max(d, key=lambda k: d[k])
    conclusions = OrderedDict(
        supported_pairs=det,
        share_supported_reproduced_range=OrderedDict(preserved=rng_of(det, pa), rebuilt=rng_of(det, pb)),
        share_undetermined_stay_undetermined_range=OrderedDict(preserved=rng_of(und, pa), rebuilt=rng_of(und, pb)),
        modal_n_determined_worst=OrderedDict(preserved=int(mode(pa["n_determined_worst_distribution"])),
                                             rebuilt=int(mode(pb["n_determined_worst_distribution"]))),
        modal_n_determined_delta5=OrderedDict(preserved=int(mode(pa["n_determined_delta5_distribution"])),
                                              rebuilt=int(mode(pb["n_determined_delta5_distribution"]))),
    )
    n_bad_s = sum(1 for r in shares if r["z"] > z_max)
    n_bad_q = sum(1 for r in quants if r["diff_in_bootstrap_sd"] > q_sd_max)
    top = sorted(shares, key=lambda r: -r["abs_diff"])[:8]
    return OrderedDict(
        B=Bn, tolerance=OrderedDict(share_z_max=z_max, quantile_sd_max=q_sd_max),
        n_share_fields=len(shares), max_share_abs_diff=max(r["abs_diff"] for r in shares),
        max_share_z=max(r["z"] for r in shares), n_share_fields_beyond_tolerance=n_bad_s,
        n_quantile_fields=len(quants), max_quantile_abs_diff=max(r["abs_diff"] for r in quants),
        max_quantile_diff_in_bootstrap_sd=max(r["diff_in_bootstrap_sd"] for r in quants),
        n_quantile_fields_beyond_tolerance=n_bad_q,
        delta_star_quantiles_infinite_status_same=all(r["infinite_status_same"] for r in ds),
        delta_star_quantiles_max_abs_diff_pp=max([r["abs_diff_pp"] for r in ds if r["abs_diff_pp"] is not None]
                                                 + [0.0]),
        within_monte_carlo_tolerance=(n_bad_s == 0 and n_bad_q == 0),
        conclusions=conclusions, largest_share_differences=top,
        per_ordering=[r for r in shares if r["pair"] in pa["pairs"]],
        quantiles=quants, delta_star_quantiles=ds,
    )


def order_statistic_check(pa: dict, jsonl: Path, groups: list, z: float = 3.0) -> OrderedDict:
    """Supplementary, distribution-free check (added after the first comparison flagged one percentile):
    for each bound percentile q of the preserved run, the preserved value should lie between the rebuilt
    replicates' order statistics of rank q*B -/+ z*sqrt(2)*sqrt(B q (1-q)) (the rank spread of the
    difference of two independent runs). No normality assumption about the bootstrap distribution."""
    import numpy as np
    vals = {m: [] for m in ("L", "U", "theta0")}
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for m in vals:
                vals[m].append(r[m])
    out, n_out = [], 0
    for m, key in (("L", "L"), ("U", "U"), ("theta0", "theta_delta0")):
        arr = np.sort(np.asarray(vals[m]), axis=0)
        Bn = arr.shape[0]
        for gi, g in enumerate(groups):
            for f, q in (("_p2_5", 0.025), ("_p50", 0.5), ("_p97_5", 0.975)):
                fld = key + f
                if fld not in pa["per_group"][g]:
                    continue
                h = z * math.sqrt(2.0) * math.sqrt(Bn * q * (1 - q))
                lo = int(max(0, math.floor(q * Bn - h) - 1))
                hi = int(min(Bn - 1, math.ceil(q * Bn + h) - 1))
                v = pa["per_group"][g][fld]
                inside = bool(arr[lo, gi] <= v <= arr[hi, gi])
                n_out += not inside
                out.append(OrderedDict(group=g, field=fld, preserved=v, rebuilt_rank_interval=[float(arr[lo, gi]),
                                       float(arr[hi, gi])], inside=inside))
    return OrderedDict(z=z, n_fields=len(out), n_outside=n_out, fields=out)


def section_of(path):
    if not path:
        return "(root)"
    if path[0] == "robustness" and len(path) > 1:
        return f"robustness.{path[1]}"
    return str(path[0])


def fmt(path):
    return ".".join(str(p) for p in path)


def main(argv=None):
    global TOL, NUM_TOL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preserved", required=True, type=Path)
    ap.add_argument("--rebuilt", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--num-tol", type=float, default=1e-3)
    ap.add_argument("--bootstrap-note", default="")
    ap.add_argument("--rebuilt-jsonl", type=Path, default=None,
                    help="per-replicate JSONL of the rebuilt bootstrap (enables the supplementary order-statistic check)")
    args = ap.parse_args(argv)
    TOL = args.tol
    NUM_TOL = args.num_tol
    a = json.loads(args.preserved.read_text(encoding="utf-8"))
    b = json.loads(args.rebuilt.read_text(encoding="utf-8"))
    leaves = []
    walk(a, b, [], leaves)

    meta, added, results = [], [], []
    for p, st, x, y in leaves:
        if p and p[0] == "provenance" and len(p) > 1 and p[1] in RUN_METADATA:
            meta.append(OrderedDict(path=fmt(p), status=st, preserved=x, rebuilt=y))
        elif st == "extra_in_rebuilt" and len(p) == 1 and p[0] in ADDED_OK:
            added.append(fmt(p))
        else:
            results.append((p, st, x, y))

    sections = OrderedDict()
    for p, st, x, y in results:
        s = sections.setdefault(section_of(p), OrderedDict(
            n_leaves=0, exact=0, float_close=0, not_matched=0, max_abs_float_deviation=0.0, examples=[]))
        s["n_leaves"] += 1
        if st == "exact":
            s["exact"] += 1
        elif st == "float_close":
            s["float_close"] += 1
            s["max_abs_float_deviation"] = max(s["max_abs_float_deviation"], abs(float(x) - float(y)))
        else:
            s["not_matched"] += 1
            if st == "float_differs":
                try:
                    s["max_abs_float_deviation"] = max(s["max_abs_float_deviation"], abs(float(x) - float(y)))
                except (TypeError, ValueError):
                    pass
            if len(s["examples"]) < 8:
                s["examples"].append(OrderedDict(path=fmt(p), status=st, preserved=x, rebuilt=y))
    for s in sections.values():
        s["status"] = ("exact" if s["not_matched"] == 0 and s["float_close"] == 0 else
                       "within_tolerance" if s["not_matched"] == 0 else "not_matched")
        if not s["examples"]:
            del s["examples"]

    tot = OrderedDict(n_leaves=len(results), exact=sum(s["exact"] for s in sections.values()),
                      float_close=sum(s["float_close"] for s in sections.values()),
                      not_matched=sum(s["not_matched"] for s in sections.values()),
                      max_abs_float_deviation=max([s["max_abs_float_deviation"] for s in sections.values()] + [0.0]))
    non_boot = [k for k in sections if k != "bootstrap"]
    tot_nb = OrderedDict(n_leaves=sum(sections[k]["n_leaves"] for k in non_boot),
                         exact=sum(sections[k]["exact"] for k in non_boot),
                         float_close=sum(sections[k]["float_close"] for k in non_boot),
                         not_matched=sum(sections[k]["not_matched"] for k in non_boot),
                         max_abs_float_deviation=max([sections[k]["max_abs_float_deviation"] for k in non_boot] + [0.0]))
    report = OrderedDict(
        preserved=args.preserved.name, preserved_sha256=hashlib.sha256(args.preserved.read_bytes()).hexdigest(),
        rebuilt=args.rebuilt.name, rebuilt_sha256=hashlib.sha256(args.rebuilt.read_bytes()).hexdigest(),
        tolerance_abs=TOL,
        rules="ints/bools/strings/nulls identical; floats |diff| <= tolerance; run metadata excluded",
        totals=tot,
        totals_excluding_bootstrap=tot_nb,
        all_sections_except_bootstrap_matched=all(sections[k]["status"] != "not_matched" for k in non_boot),
        bootstrap_matched=("bootstrap" in sections and sections["bootstrap"]["status"] != "not_matched"),
        bootstrap_present_in_rebuilt="bootstrap" in b,
        bootstrap_note=args.bootstrap_note,
        bootstrap_monte_carlo_check=(bootstrap_mc_check(a["bootstrap"], b["bootstrap"])
                                     if "bootstrap" in a and "bootstrap" in b else None),
        bootstrap_order_statistic_check=(order_statistic_check(a["bootstrap"], args.rebuilt_jsonl,
                                                               list(a["bootstrap"]["per_group"]))
                                         if args.rebuilt_jsonl and "bootstrap" in a else None),
        sections=sections, run_metadata_differences=[m for m in meta if m["status"] != "exact"],
        added_keys_in_rebuilt=added,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8", newline="\n") as fh_j:
        fh_j.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    L = ["# Rebuild check: rebuilt analysis vs preserved results", "",
         f"- Preserved: `{args.preserved.name}` (sha256 `{report['preserved_sha256'][:12]}...`)",
         f"- Rebuilt: `{args.rebuilt.name}` (sha256 `{report['rebuilt_sha256'][:12]}...`)",
         f"- Rule: integers, booleans, strings and nulls identical; floats within an absolute tolerance of {TOL:g}; "
         "run metadata (timestamp, spec hash, wall clock, version, hardware, cohort file hash) not counted.",
         f"- Everything except the bootstrap: {tot_nb['n_leaves']:,} result leaves; bit-identical {tot_nb['exact']:,}; "
         f"within tolerance {tot_nb['float_close']:,}; not matched {tot_nb['not_matched']:,}; largest float deviation "
         f"{tot_nb['max_abs_float_deviation']:.3g}.",
         f"- Including the bootstrap block: {tot['n_leaves']:,} leaves; not matched {tot['not_matched']:,} (all in the "
         "bootstrap block, which uses a different random-stream order; see the Monte Carlo comparison below).", ""]
    if args.bootstrap_note:
        L += [f"- Bootstrap: {args.bootstrap_note}", ""]
    mc = report["bootstrap_monte_carlo_check"]
    if mc:
        c = mc["conclusions"]
        L += ["## Bootstrap (Monte Carlo comparison)", "",
              f"- Replicates: {mc['B']} in both runs. Exact leaf match is not expected unless the random stream is "
              "consumed in the same order; the tolerance is statistical: every verdict frequency within "
              f"{mc['tolerance']['share_z_max']:g} Monte Carlo standard errors of the difference of two independent runs, "
              f"every bound percentile within {mc['tolerance']['quantile_sd_max']:g} bootstrap SD.",
              f"- Verdict frequencies: {mc['n_share_fields']} compared; largest absolute difference "
              f"{mc['max_share_abs_diff']:.4f} (z = {mc['max_share_z']:.2f}); beyond tolerance: "
              f"{mc['n_share_fields_beyond_tolerance']}.",
              f"- Bound percentiles: {mc['n_quantile_fields']} compared; largest difference "
              f"{100 * mc['max_quantile_abs_diff']:.3f} percentage points ({mc['max_quantile_diff_in_bootstrap_sd']:.3f} "
              f"bootstrap SD); beyond tolerance: {mc['n_quantile_fields_beyond_tolerance']}.",
              f"- delta* percentiles: infinite/finite status identical: {mc['delta_star_quantiles_infinite_status_same']}; "
              f"largest finite difference {mc['delta_star_quantiles_max_abs_diff_pp']:.2f} points (reported, not tested).",
              f"- Supported orderings reproduced: preserved {100 * c['share_supported_reproduced_range']['preserved'][0]:.1f}"
              f"-{100 * c['share_supported_reproduced_range']['preserved'][1]:.1f}%, rebuilt "
              f"{100 * c['share_supported_reproduced_range']['rebuilt'][0]:.1f}-"
              f"{100 * c['share_supported_reproduced_range']['rebuilt'][1]:.1f}%. Undetermined orderings staying "
              f"undetermined: preserved {100 * c['share_undetermined_stay_undetermined_range']['preserved'][0]:.1f}-"
              f"{100 * c['share_undetermined_stay_undetermined_range']['preserved'][1]:.1f}%, rebuilt "
              f"{100 * c['share_undetermined_stay_undetermined_range']['rebuilt'][0]:.1f}-"
              f"{100 * c['share_undetermined_stay_undetermined_range']['rebuilt'][1]:.1f}%. Modal number of "
              f"orderings supported without assumption: preserved {c['modal_n_determined_worst']['preserved']}, "
              f"rebuilt {c['modal_n_determined_worst']['rebuilt']}; at delta = 5: preserved "
              f"{c['modal_n_determined_delta5']['preserved']}, rebuilt {c['modal_n_determined_delta5']['rebuilt']}.",
              f"- Within Monte Carlo tolerance: {mc['within_monte_carlo_tolerance']}."]
        osc = report.get("bootstrap_order_statistic_check")
        if osc:
            bad = [f"{r['group']} {r['field']}" for r in osc["fields"] if not r["inside"]]
            L += [f"- Supplementary distribution-free check (added after the comparison above; not the pre-stated "
                  f"criterion): {osc['n_fields'] - osc['n_outside']} of {osc['n_fields']} preserved bound percentiles lie "
                  f"within the rebuilt replicates' order-statistic interval (rank +/- {osc['z']:g} x sqrt(2) x binomial SD)"
                  + (f"; outside: {', '.join(bad)}." if bad else ".")]
        L += ["",
              "| Ordering | Frequency | Preserved | Rebuilt | Abs. diff. | z |", "|---|---|---:|---:|---:|---:|"]
        for r in mc["per_ordering"]:
            if r["field"] in ("share_worst_main_verdict", "share_delta5_main_verdict", "share_same_delta0_direction"):
                L.append(f"| {r['pair']} | {r['field']} | {r['preserved']:.4f} | {r['rebuilt']:.4f} | "
                         f"{r['abs_diff']:.4f} | {r['z']:.2f} |")
        L.append("")
    L += ["| Section | Leaves | Exact | Within tol. | Not matched | Max abs. float dev. | Status |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for k, s in sections.items():
        L.append(f"| {k} | {s['n_leaves']} | {s['exact']} | {s['float_close']} | {s['not_matched']} | "
                 f"{s['max_abs_float_deviation']:.3g} | {s['status']} |")
    bad = [(k, s) for k, s in sections.items() if s["not_matched"]]
    if bad:
        L += ["", "## Not matched (first examples per section)", ""]
        for k, s in bad:
            for e in s.get("examples", []):
                L.append(f"- `{e['path']}`: preserved {json.dumps(e['preserved'])[:80]}, rebuilt "
                         f"{json.dumps(e['rebuilt'])[:80]} ({e['status']})")
    if report["run_metadata_differences"]:
        L += ["", "## Run metadata (expected to differ, not counted)", ""]
        for m in report["run_metadata_differences"]:
            L.append(f"- `{m['path']}`: preserved {json.dumps(m['preserved'])}, rebuilt {json.dumps(m['rebuilt'])}")
    if added:
        L += ["", f"Keys added by the current pipeline run: {', '.join(added)}."]
    with open(args.out_md, "w", encoding="utf-8", newline="\n") as fh_m:
        fh_m.write("\n".join(L) + "\n")
    print("excluding bootstrap:", json.dumps(tot_nb), "| sections not matched:", [k for k, _ in bad])


if __name__ == "__main__":
    main()
