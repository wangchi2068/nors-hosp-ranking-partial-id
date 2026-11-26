"""Tables, figures, and statistical summary for the NORS hospitalization-ranking analysis (v3.0).

Reads ONLY analysis/results.json (written by analyze.py) so that every number in a table, figure
or sentence comes from that single canonical source.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SLUG = "nors-hosp-ranking-partial-id"
SHORT = {"SAL": "SAL", "STEC": "STEC", "CAMP": "CAMP", "CPERF": "CPERF", "NOV_GI": "NoV GI", "NOV_GII": "NoV GII",
         "NOV": "NoV"}
NAME = {"SAL": "Salmonella", "STEC": "STEC", "CAMP": "Campylobacter", "CPERF": "C. perfringens",
        "NOV_GI": "Norovirus GI", "NOV_GII": "Norovirus GII", "NOV": "Norovirus GI+GII"}
VARIANT_CODES = {"R1a_crude_all": "R1a", "R1b_crude_support": "R1b", "R2_period_only": "R2",
                 "R3a_single_etiology": "R3a", "R3b_complete_only": "R3b", "R3c_pilot_rule": "R3c",
                 "R4a_threshold_2": "R4a", "R4b_threshold_5": "R4b", "R5_named_settings": "R5",
                 "R6a_w_group_balanced": "R6a", "R6b_w_pooled_outbreaks": "R6b", "R6c_w_uniform": "R6c",
                 "R7_norovirus_combined": "R7", "R8_include_suspected": "R8",
                 "R9_excl_healthcare": "R9", "R10_outbreak_weighted": "R10"}
CAT_COLORS = {"A": "#0F4D92", "B": "#8FA8D6", "C": "#E9A6A1", "D": "#E8B04B"}
CAT_TEXT = {"A": "white", "B": "#272727", "C": "#272727"}
CAT_LABEL = {
    "A": "Supported by recorded data alone (worst-case intervals do not overlap)",
    "B": "Illustrative scenario: holds up to δ* ≥ 5 percentage points",
    "C": "Illustrative scenario: fails before δ = 5 percentage points (δ* < 5)",
    "D": "Direction at δ = 0 reversed relative to the main analysis",
}


def load(exp: Path) -> dict:
    return json.loads((exp / "analysis" / "results.json").read_text(encoding="utf-8"))


def pct(x, d=1):
    return "NA" if x is None else f"{100 * x:.{d}f}%"


def pnum(x, d=1):
    return "NA" if x is None else f"{100 * x:.{d}f}"


def fmt_ds(p, unit=" points"):
    ds = p["delta_star_pp"]
    if ds is None:
        return "no assumption needed"
    return f"{ds:.2f}{unit}" if ds < 1 else f"{ds:.1f}{unit}"


def category(p):
    if p["delta_star_pp"] is None:
        return "A"
    return "B" if p["delta_star_pp"] >= 5 else "C"


def pair_label(p, names=NAME):
    return f"{names[p['higher_at_delta0']]} > {names[p['lower_at_delta0']]}"


def sorted_pairs(main):
    return sorted(main["pairs"].items(),
                  key=lambda kv: -(np.inf if kv[1]["delta_star_pp"] is None else kv[1]["delta_star_pp"]))


def tex_escape(s: str) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 (">=", r"$\geq$"), ("<=", r"$\leq$"), ("±", r"$\pm$"),
                 ("δ", r"$\delta$"), ("Γ", r"$\Gamma$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
                 ("–", "--"), ("−", "$-$"), (">", r"$>$"),
                 ("*", r"$^{*}$"), ("×", r"$\times$"), ("∞", r"$\infty$")):
        s = s.replace(a, b)
    return s.replace("$$", "")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def statistical_summary(exp: Path) -> None:
    r = load(exp)
    m = r["main"]
    si = r.get("stacked_inference")
    out = dict(
        source="analysis/results.json", spec_hash=r["provenance"]["spec_hash"],
        analysis_version=r["provenance"].get("analysis_version", "3.0"),
        n_pairs=m["n_pairs"], n_common_support_cells=m["n_cells"],
        determined_pairs=dict(worst_case=m["n_determined"]["worst_case"],
                              **{f"delta_{k}pp": v for k, v in m["n_determined"].items() if k != "worst_case"}),
        determined_worst_case=[pair_label(p) for _, p in m["pairs"].items() if p["worst_case"] != "undetermined"],
        undetermined_at_delta5=[pair_label(p) for _, p in m["pairs"].items()
                                if p["verdict_by_delta_pp"]["5"] == "undetermined"],
        delta_star_pp={pair_label(p): p["delta_star_pp"] for _, p in sorted_pairs(m)},
        groups={NAME[g]: dict(L=v["L_worst"], U=v["U_worst"], theta_delta0=v["theta_delta0"],
                              L_delta5=v["bounds_by_delta_pp"]["5"]["L"], U_delta5=v["bounds_by_delta_pp"]["5"]["U"])
                for g, v in m["per_group"].items()},
        robustness={k: dict(cells=v["n_cells"], determined_worst_case=v["n_determined"]["worst_case"],
                            determined_delta5=v["n_determined"]["5"], n_pairs=v["n_pairs"])
                    for k, v in r["robustness"].items()},
        bootstrap=dict(B=r["bootstrap"]["B"], seed=r["bootstrap"]["seed"],
                       share_reproducing_main_worst_case_verdict={pair_label(m["pairs"][k]): v["share_worst_main_verdict"]
                                                                  for k, v in r["bootstrap"]["pairs"].items()},
                       share_reproducing_main_delta5_verdict={pair_label(m["pairs"][k]): v["share_delta5_main_verdict"]
                                                              for k, v in r["bootstrap"]["pairs"].items()}),
        stacked_inference=dict(
            n_im_excludes_zero=si["n_im_excludes_zero"],
            n_fwer_worst_sig=si["n_fwer_worst_sig"],
            n_fwer_delta5_sig=si["n_fwer_delta5_sig"],
            n_fwer_state_worst_sig=si["n_fwer_state_worst_sig"],
            n_fwer_wcb_worst_sig=si["n_fwer_wcb_worst_sig"],
            c_sim_worst=si["c_sim_worst"],
            c_sim_delta5=si["c_sim_delta5"],
            rank_sets_worst_hc1=si["rank_confidence_sets"]["worst_hc1"],
            rank_sets_delta5_hc1=si["rank_confidence_sets"]["delta5_hc1"],
        ) if si else None,
        note="Sample intervals are identification bounds; Imbens-Manski 95% CIs, outbreak-HC1 and state-clustered "
             "Max-T FWER p-values, and rank confidence sets appear in stacked_inference.",
    )
    (exp / "analysis" / "statistical_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------
def R(w):
    return rf">{{\raggedleft\arraybackslash}}p{{{w}cm}}"


def Lc(w):
    return rf">{{\raggedright\arraybackslash}}p{{{w}cm}}"


def write_table(tdir: Path, manifest: list, tid: str, base: str, df: pd.DataFrame, caption: str, colspec: str,
                notes: str = "", size: str = "scriptsize", sep: str = "3pt"):
    """Paper-sized LaTeX table with normalized LF line endings."""
    (tdir / f"{base}.csv").write_text(df.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="\n")
    lines = [r"\begin{table}[!tbp]", r"\centering", rf"\{size}", rf"\setlength{{\tabcolsep}}{{{sep}}}",
             r"\renewcommand{\arraystretch}{1.12}",
             rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{tid}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\par\smallskip\parbox{{0.97\linewidth}}{{\footnotesize {tex_escape(notes)}}}")
    lines.append(r"\end{table}")
    p = tdir / f"{base}.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest.append(dict(id=tid, path=p.name, csv=f"{base}.csv", sha256=sha256(p), generator="code/make_outputs.py",
                         data_sources=["analysis/results.json"], display="raw", n_seeds=1, std_ddof=0,
                         caption=caption))


def make_tables(r: dict, tdir: Path) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    for f in tdir.glob("table_*"):
        f.unlink()
    m, man = r["main"], []
    order = m["order_by_delta0"]
    bt = r["bootstrap"]
    emp = r.get("empirical_calibration", {})

    # Table 1: per-group bounds
    rows = []
    for g in order:
        v = m["per_group"][g]
        b = bt["per_group"][g]
        rows.append({"Group": NAME[g], "Outbreaks": f"{v['outbreaks']:,}", "Cases": f"{int(v['cases']):,}",
                     "Unrecorded status (%)": pnum(v["unknown_share"]),
                     "Worst-case interval (%)": f"{pnum(v['L_worst'])}–{pnum(v['U_worst'])}",
                     "δ = 5 interval (%)": f"{pnum(v['bounds_by_delta_pp']['5']['L'])}–{pnum(v['bounds_by_delta_pp']['5']['U'])}",
                     "δ = 0 value (%)": pnum(v["theta_delta0"]),
                     "Bootstrap range of lower bound (%)": f"{pnum(b['L_p2_5'])}–{pnum(b['L_p97_5'])}",
                     "Bootstrap range of upper bound (%)": f"{pnum(b['U_p2_5'])}–{pnum(b['U_p97_5'])}"})
    write_table(tdir, man, "tab:group_bounds", "table_1_group_bounds", pd.DataFrame(rows),
                f"Standardized hospitalization proportion among reported outbreak cases, by pathogen group "
                f"({m['n_cells']} common-support setting × period cells, pooled-case reference weights). Sample worst-case "
                f"intervals are identification intervals; bootstrap ranges are the "
                f"2.5th–97.5th percentiles of each bound over {bt['B']:,} outbreak-level resamples.",
                "".join([Lc(2.0), R(1.4), R(1.1), R(1.4), R(1.6), R(1.5), R(1.2), R(1.8), R(1.8)]),
                notes="Unrecorded status is the share of retained cases whose hospitalization status was not recorded, "
                      "including the C − H unrecorded cases from the 52 blank-K records where K was set to H. "
                      "The δ = 0 value is the proportion if unrecorded-status cases had the recorded rate of their group and cell.")

    # Table 2: pairwise (including OR Gamma* and Asym -15/+5)
    or_gs = emp.get("odds_ratio", {}).get("gamma_star_pairs", {})
    asym_15 = emp.get("asymmetric", {}).get("asym_15_5", {}).get("verdicts", {})
    rows = []
    for k, p in sorted_pairs(m):
        b = bt["pairs"][k]
        vd = p["verdict_by_delta_pp"]
        gs_val = or_gs.get(k)
        gs_str = "∞" if gs_val is None else (f"{gs_val:.2f}" if gs_val < 10 else f"{gs_val:.1f}")
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT),
                     "Worst case": "determined" if p["worst_case"] != "undetermined" else "undetermined",
                     "δ* (pp)": "∞" if p["delta_star_pp"] is None else (f"{p['delta_star_pp']:.2f}"
                                                                        if p["delta_star_pp"] < 1 else f"{p['delta_star_pp']:.1f}"),
                     "OR Γ*": gs_str,
                     **{f"δ = {d}": ("yes" if vd[str(d)] != "undetermined" else "no") for d in (1, 2, 5, 10)},
                     "Asym (−15/+5)": ("yes" if asym_15.get(k, "undetermined") != "undetermined" else "no"),
                     "Boot, worst (%)": pnum(b["share_worst_main_verdict"]),
                     "Boot, δ = 5 (%)": pnum(b["share_delta5_main_verdict"])})
    write_table(tdir, man, "tab:pairwise", "table_2_pairwise", pd.DataFrame(rows),
                "Pairwise orderings with worst-case verdicts, critical additive deviations δ*, critical odds-ratio factors Γ*, "
                "illustrative symmetric and calibrated asymmetric (−15/+5 pp) scenario verdicts, and bootstrap reproduction shares.",
                "".join([Lc(2.4), Lc(1.55), R(0.95), R(0.95), R(0.62), R(0.62), R(0.62), R(0.68), R(1.1), R(1.35), R(1.35)]),
                notes="SAL Salmonella; STEC Shiga toxin-producing E. coli; CAMP Campylobacter; CPERF C. perfringens; "
                      "NoV GI/GII norovirus genogroup I/II. At δ = 0 every pair is ordered. Note that wide overlapping worst-case "
                      "intervals remain undetermined across nearly 100% of resamples, inflating bootstrap reproduction shares for "
                      "undetermined pairs. OR Γ* is the smallest multiplicative odds-ratio factor on unrecorded odds where "
                      "intervals overlap (∞ = holds at Γ = 1,000,000).", sep="2.2pt")

    # Table 3: robustness (16 variants)
    rows = [{"Analysis": "Main", "Description": "Setting × period, ≥ 3 outbreaks, pooled-case weights",
             "Cells": m["n_cells"], "Pairs": m["n_pairs"], "Determined, worst case": m["n_determined"]["worst_case"],
             "Determined, δ = 5": m["n_determined"]["5"], "Worst-case verdicts changed": 0,
             "δ = 0 directions changed": 0}]
    for vid, v in r["robustness"].items():
        desc = v["label"].replace("Crude: no standardization", "Crude, no standardization").replace(
            "Cohort: ", "Cohort of ").replace("Weights: ", "Weights from ")
        if vid == "R3c_pilot_rule":
            desc = (f"Cohort of species-level etiology strings; records with one blank outcome field "
                    f"dropped ({v['n_outbreaks_in_cohort']:,} outbreaks)")
        rows.append({"Analysis": VARIANT_CODES[vid], "Description": desc, "Cells": v["n_cells"],
                     "Pairs": v["n_pairs"], "Determined, worst case": v["n_determined"]["worst_case"],
                     "Determined, δ = 5": v["n_determined"]["5"],
                     "Worst-case verdicts changed": v.get("n_worst_case_verdicts_changed", "NA"),
                     "δ = 0 directions changed": v.get("n_delta0_directions_changed", "NA")})
    write_table(tdir, man, "tab:robustness", "table_3_robustness", pd.DataFrame(rows),
                "Sensitivity analyses across 16 cohort, support, weighting, aggregation, and etiology variants, with common-support "
                "cells, determined pairwise orderings under worst-case bounds and at δ = 5 points, and changes relative to the main analysis.",
                "".join([Lc(1.05), Lc(5.15), R(0.85), R(0.85), R(1.5), R(1.45), R(1.5), R(1.5)]),
                notes="R7 combines Norovirus GI and GII into five groups (10 pairs), so six-group pair changes are NA. "
                      "R8 adds 951 suspected study-group outbreaks. R9 excludes 79 hospital/long-term-care outbreaks. "
                      "R10 weights all outbreaks equally within each cell while retaining the main 12-cell reference distribution.")

    # Table 4: retention (widths trimmed slightly so zero overfull hbox)
    ret = r["retention"]["per_group"]
    rows = []
    for g in order:
        v = ret[g]
        top_set = ", ".join(f"{k} {int(c):,}" for k, c in list(v["dropped_cases_by_setting"].items())[:2])
        top_per = ", ".join(f"{k} {int(c):,}" for k, c in list(v["dropped_cases_by_period"].items())[:2])
        rows.append({"Group": NAME[g], "Eligible outbreaks": f"{v['eligible_outbreaks']:,}",
                     "Eligible cases": f"{int(v['eligible_cases']):,}",
                     "Outbreaks retained (%)": f"{v['support_outbreaks']:,} ({v['support_outbreaks_pct']:.1f})",
                     "Cases retained (%)": f"{int(v['support_cases']):,} ({v['support_cases_pct']:.1f})",
                     "Largest dropped settings (cases)": top_set, "Largest dropped periods (cases)": top_per,
                     "Crude δ = 0 before/after (%)": f"{pnum(v['crude_before']['theta_delta0'])}/{pnum(v['crude_after']['theta_delta0'])}",
                     "Crude worst before/after (%)": f"{pnum(v['crude_before']['L'])}–{pnum(v['crude_before']['U'])} / "
                                                     f"{pnum(v['crude_after']['L'])}–{pnum(v['crude_after']['U'])}"})
    write_table(tdir, man, "tab:retention", "table_4_retention", pd.DataFrame(rows),
                "Common-support retention by pathogen group, with eligible and retained outbreaks and cases, the settings "
                "and periods losing most cases, and crude (unstandardized) proportions before and after the restriction.",
                "".join([Lc(1.85), R(1.25), R(0.95), R(1.35), R(1.35), Lc(1.95), Lc(1.6), R(1.2), R(1.75)]), sep="2.3pt")

    # Table 5: record audit
    a = r["audit"]["record_audit_summary"]
    handling = {"complete_consistent": "as recorded (K = 0 means all status unrecorded)",
                "single_missing_K": "K set to H; C − H cases unrecorded",
                "single_missing_H_K0": "H = 0 implied by H ≤ K = 0; all cases unrecorded",
                "single_missing_H_Kpos": "all cases treated as unrecorded",
                "both_missing": "all cases treated as unrecorded",
                "ambiguous_etiology": "excluded (outcome not attributable to one group)"}
    AUDIT_LABEL = {"complete_consistent": "Complete and consistent", "single_missing_K": "K blank",
                   "single_missing_H_K0": "H blank, K = 0", "single_missing_H_Kpos": "H blank, K > 0",
                   "both_missing": "H and K blank", "ambiguous_etiology": "Ambiguous etiology"}
    rows = []
    for cat in ("complete_consistent", "single_missing_K", "single_missing_H_K0", "single_missing_H_Kpos",
                "both_missing", "ambiguous_etiology"):
        c = a["categories"][cat]
        byg = a["by_category_x_group"][cat]
        rows.append({"Audit category": AUDIT_LABEL[cat], "Outbreaks": f"{c['n_outbreaks']:,}",
                     "Cases": f"{int(c['cases']):,}",
                     **{SHORT[g]: ("–" if cat == "ambiguous_etiology" else f"{byg[g]:,}")
                        for g in ("SAL", "STEC", "CAMP", "CPERF", "NOV_GI", "NOV_GII")},
                     "Handling": handling[cat]})
    write_table(tdir, man, "tab:audit", "table_5_audit", pd.DataFrame(rows),
                "Record audit of outbreak outcome fields (C reported cases, H hospitalized, K cases with known "
                "hospitalization status), with category counts and handling. No record had a count contradiction.",
                "".join([Lc(2.1), R(1.4), R(1.1)] + [R(0.92)] * 6 + [Lc(3.4)]),
                notes="Ambiguous etiology denotes confirmed entries of a study group together with entries of another "
                      "group, or norovirus without genogroup; these outbreaks are not assigned to a group (–).")

    # Table 6: bootstrap
    rows = []
    for k, p in sorted_pairs(m):
        b = bt["pairs"][k]
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT),
                     "Same δ = 0 direction (%)": pnum(b["share_same_delta0_direction"]),
                     "Worst case, higher > lower (%)": pnum(b["share_worst_A"] if p["higher_at_delta0"] == p["A"]
                                                             else b["share_worst_B"]),
                     "Worst case, undetermined (%)": pnum(b["share_worst_undetermined"]),
                     "Determined at δ = 5 (%)": pnum(1 - b["share_delta5_undetermined"]),
                     "δ* 2.5th–97.5th percentile (pp)":
                         f"{'∞' if b['delta_star_pp_p2_5'] is None else format(b['delta_star_pp_p2_5'], '.1f')}–"
                         f"{'∞' if b['delta_star_pp_p97_5'] is None else format(b['delta_star_pp_p97_5'], '.1f')}"})
    write_table(tdir, man, "tab:bootstrap", "table_6_bootstrap", pd.DataFrame(rows),
                f"Outbreak-level bootstrap ({bt['B']:,} replicates, seed {bt['seed']}; 0/{bt['B']:,} resamples had an "
                f"unrecorded-rate cell lacking K_rec), with verdict shares and percentile ranges of δ*.",
                "".join([Lc(2.6), R(2.0), R(2.2), R(2.0), R(2.0), R(2.6)]))

    # Table 7: width decomposition
    rows = []
    wd = r["width_decomposition"]["undetermined_pairs"]
    for k, p in sorted_pairs(m):
        if k not in wd:
            continue
        tops = "; ".join(f"{SHORT[t['group']]} {t['cell']} ({100 * t['contribution']:.1f})" for t in wd[k]["top_group_cells"])
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT), "Worst-case overlap (pp)": pnum(wd[k]["overlap"]),
                     "Largest contributions to interval width (group, cell, points)": tops})
    write_table(tdir, man, "tab:width", "table_7_width_decomposition", pd.DataFrame(rows),
                "Undetermined pairs under worst-case bounds, with interval overlap and the group × cell strata "
                "whose unrecorded-status cases contribute most to the interval widths.",
                "".join([Lc(2.6), R(1.6), Lc(10.6)]))

    # Table 8: Joint M-estimation inference (Main Table 3, HC1 + State-clustered + Dynamic delta*)
    if "stacked_inference" in r:
        si = r["stacked_inference"]
        rows = []
        for k, p in sorted_pairs(m):
            sp = si["pairs"][k]
            fmt_ci = lambda a, b: f"[{100 * a:+.1f}, {100 * b:+.1f}]"
            fmt_ds = lambda v: "∞" if v is None else (f"{v:.2f}" if v < 1 else f"{v:.1f}")
            fmt_p = lambda pv: "<0.001" if pv < 0.001 else f"{pv:.3f}"
            rows.append({
                "Ordering at δ = 0": pair_label(p, SHORT),
                "ID bounds (pp)": fmt_ci(sp["dL_worst"], sp["dU_worst"]),
                "IM 95% CI (pp)": fmt_ci(sp["ci95_im_lo"], sp["ci95_im_hi"]),
                "HC1 FWER p (w / δ=5)": f"{fmt_p(sp['p_fwer_worst'])} / {fmt_p(sp['p_fwer_delta5'])}",
                "State FWER p (w / δ=5)": f"{fmt_p(sp['p_fwer_state_worst'])} / {fmt_p(sp['p_fwer_state_delta5'])}",
                "δ* (pt)": fmt_ds(sp["delta_star_pt_pp"]),
                "95% δ* (HC1 / State)": f"{fmt_ds(sp['delta_star_95_sim_pp'])} / {fmt_ds(sp['delta_star_95_sim_state_pp'])}",
            })
        rcs_w = si["rank_confidence_sets"]["worst_hc1"]
        rcs_5 = si["rank_confidence_sets"]["delta5_hc1"]
        rank_txt = "; ".join(f"{SHORT[g]} [{rcs_5[g]['rank_lower']},{rcs_5[g]['rank_upper']}]" for g in order)
        write_table(tdir, man, "tab:joint_inference", "table_8_joint_inference", pd.DataFrame(rows),
                    f"Joint M-estimation inference for the 15 pairwise standardized differences combining outbreak-HC1 "
                    f"and 53-state-clustered sandwich covariances, Imbens–Manski 95% CIs, 200,000-draw Gaussian multiplier "
                    f"Max-T FWER p-values, and dynamic simultaneous 95% confidence tipping points.",
                    "".join([Lc(2.4), R(1.95), R(2.05), R(2.2), R(2.2), R(1.05), R(2.25)]),
                    notes=f"Outbreak-HC1 critical values: c_sim = {si['c_sim_worst']:.2f} (worst case), {si['c_sim_delta5']:.2f} "
                          f"(δ = 5); state-clustered: {si['c_sim_state_worst']:.2f} / {si['c_sim_state_delta5']:.2f} "
                          f"(wild cluster bootstrap STEC > NoV GII p = {si['pairs']['STEC|NOV_GII']['p_fwer_wcb_worst']:.3f}). "
                          f"95% δ* uses dynamic c_sim(δ). Simultaneous 95% rank confidence sets at δ = 5: {rank_txt} "
                          f"(worst case: STEC [{rcs_w['STEC']['rank_lower']},{rcs_w['STEC']['rank_upper']}], NoV GII "
                          f"[{rcs_w['NOV_GII']['rank_lower']},{rcs_w['NOV_GII']['rank_upper']}], others [1,6]).",
                    sep="2.3pt")

    # Table 9: Cluster concentration & true Bootstrap SD vs HC0/HC1/State SE
    if "cluster_diagnostics" in r and "stacked_inference" in r:
        cd = r["cluster_diagnostics"]["per_group"]
        sg = r["stacked_inference"]["per_group"]
        rows = []
        for g in order:
            c_g = cd[g]; s_g = sg[g]; b_g = bt["per_group"][g]
            rows.append({
                "Group": NAME[g],
                "Min/Med/Max n": f"{c_g['min_n']}/{c_g['median_n']:.0f}/{c_g['max_n']}",
                "Max HHI (min N_eff)": f"{c_g['max_hhi']:.3f} ({c_g['min_neff']:.2f})",
                "Max share (%)": f"{100 * c_g['max_single_share']:.1f}",
                "SE(L) HC0/HC1/Boot/St (pp)": (f"{100*s_g['se_L_hc0']:.2f}/{100*s_g['se_L_hc1']:.2f}/"
                                               f"{100*b_g['L_sd']:.2f}/{100*s_g['se_L_state']:.2f}"),
                "SE(U) HC0/HC1/Boot/St (pp)": (f"{100*s_g['se_U_hc0']:.2f}/{100*s_g['se_U_hc1']:.2f}/"
                                               f"{100*b_g['U_sd']:.2f}/{100*s_g['se_U_state']:.2f}"),
            })
        chk = r["stacked_inference"]["checks"]
        write_table(tdir, man, "tab:cluster_diag", "table_9_cluster_diagnostics", pd.DataFrame(rows),
                    "Outbreak-cluster concentration diagnostics across the 12 common-support cells and standard error "
                    "comparison across HC0, HC1, stratified bootstrap SD (2,000 replicates), and 53-state clustering.",
                    "".join([Lc(1.9), R(1.8), R(2.1), R(1.3), R(3.45), R(3.45)]),
                    notes=f"Boot reports the empirical standard deviation of the 2,000 bootstrap bound estimates. "
                          f"Analytic closed-form IF covariances match central-difference Jacobians to "
                          f"{chk['max_abs_diff_analytic_if_vs_sandwich_worst']:.2e} (worst case) and "
                          f"{chk['max_abs_diff_analytic_if_vs_sandwich_delta5']:.2e} (δ = 5).", sep="2.3pt")

    # Table 10: Retrospective masking & follow-up comparison (budget-filling + paired diffs)
    if "followup_masking" in r:
        vm = r["followup_masking"]
        pol_names = [("random", "Random follow-up"), ("volume_greedy", "Largest outbreaks first"),
                     ("width_greedy", "Width-greedy (w_s / C_js)"), ("dpi", "Disambiguation priority (DPI)")]
        rows = []
        for b in vm["budgets_cases"]:
            pd_b = vm["paired_differences"].get(str(b))
            for pk, pname in pol_names:
                s = vm["summary"][pk][str(b)]
                diff_str = "–"
                if b > 0 and pk == "dpi" and pd_b is not None:
                    diff_str = f"{pd_b['dpi_minus_wg_mean']:+.2f} [{pd_b['dpi_minus_wg_ci95'][0]:+.2f}, {pd_b['dpi_minus_wg_ci95'][1]:+.2f}]"
                elif b > 0 and pk == "width_greedy" and pd_b is not None:
                    diff_str = f"vs rand: {pd_b['wg_minus_random_mean']:+.2f}"
                rows.append({
                    "Budget": f"{b:,}",
                    "Policy": pname,
                    "Cases spent": f"{s['mean_cases_spent']:.0f}",
                    "Worst det (/15)": f"{s['mean_worst_det']:.2f} ({s['sd_worst_det']:.2f})",
                    "δ = 5 det (/15)": f"{s['mean_delta5_det']:.2f} ({s['sd_delta5_det']:.2f})",
                    "Overlap (pp)": f"{s['mean_overlap_pp']:.1f}",
                    "Paired Δ vs WG / Rand": diff_str,
                })
        write_table(tdir, man, "tab:voi_masking", "table_10_followup_masking", pd.DataFrame(rows),
                    f"Budget-filling retrospective masking benchmark across {vm['n_trials']} Monte Carlo trials on all "
                    f"{vm['n_complete_outbreaks']:,} fully recorded common-support outbreaks ({vm['total_complete_cases']:,} cases).",
                    "".join([R(1.1), Lc(3.55), R(1.3), R(2.05), R(1.95), R(1.3), Lc(2.9)]),
                    notes="Values are means (SD) across 200 trials using a budget-filling selection rule. Paired Δ for DPI reports "
                          "DPI minus Width-greedy (w_s / C_js) mean difference (95% CI); for Width-greedy, excess over Random.",
                    sep="2.2pt")

    # Table 11 (Supp Table S8): Excluded outbreaks descriptive profile (Major 6c)
    ex = r.get("audit", {}).get("excluded_outbreaks_summary")
    if ex:
        lbls = [("suspected_study_group", "Suspected study-group etiology"),
                ("norovirus_unspecified_confirmed", "Confirmed norovirus without GI/GII"),
                ("multi_cross_group_or_nontarget", "Mixed / non-study multi-agent"),
                ("multi_cross_with_confirmed_target", "  Subset with ≥1 confirmed study group")]
        rows = []
        for k, nm in lbls:
            d = ex[k]
            top_s = ", ".join(f"{t['setting']} ({t['share_pct']:.0f}%)" for t in d["top_settings"][:2])
            rows.append({
                "Excluded category": nm,
                "Outbreaks": f"{d['n_outbreaks']:,}",
                "Cases": f"{int(d['total_cases']):,}",
                "Median [IQR] cases": f"{d['median_cases']:.0f} [{d['iqr_cases'][0]:.0f}, {d['iqr_cases'][1]:.0f}]",
                "Top settings": top_s,
                "Rec. hosp (%)": pnum(d["recorded_hosp_rate"]),
                "Unrecorded (%)": pnum(d["unrecorded_status_share"]),
            })
        write_table(tdir, man, "tab:excluded", "table_11_excluded_outbreaks", pd.DataFrame(rows),
                    "Characteristics of excluded foodborne outbreaks (2009–2023) not retained in the primary six-group "
                    "confirmed cohort.",
                    "".join([Lc(3.8), R(1.25), R(1.1), R(2.1), Lc(3.15), R(1.35), R(1.55)]), sep="2.3pt")

    # Table 12 (Supp Table S9): Empirical within-data delta calibration (Major 1)
    if emp:
        pg = emp["per_group"]
        hc = emp["high_completeness_states"]["per_group"]
        rows = []
        for g in order:
            v = pg[g]; hv = hc[g]
            rows.append({
                "Group": NAME[g],
                "Fully rec. rate (%)": f"{pnum(v['rate_full_pooled'])} ({v['n_full_outbreaks']:,})",
                "Partial known (%)": f"{pnum(v['rate_partial_known_pooled'])} ({v['n_partial_outbreaks']:,})",
                "Partial − Full (pp)": f"{v['full_vs_partial_diff_pp']:+.2f}",
                "Std obs − Full (pp)": f"{v['std_obs_minus_full_pp']:+.2f}",
                "Calib. down (pp)": f"{v['calibrated_down_pp']:.1f}",
                "11 low-miss states (%)": pnum(hv["recorded_hosp_rate"]),
            })
        n_hs = emp["high_completeness_states"]["n_states"]
        or_det = emp["odds_ratio"]["by_gamma"]
        write_table(tdir, man, "tab:delta_cal", "table_12_delta_calibration", pd.DataFrame(rows),
                    f"Within-data empirical calibration of missingness departures across the 12 common-support cells: "
                    f"fully recorded outbreaks (M_i = 0) versus known-status cases in partially recorded outbreaks "
                    f"(M_i > 0, K_rec > 0) and {n_hs} states with <5% unrecorded status.",
                    "".join([Lc(2.1), R(2.2), R(2.2), R(1.6), R(1.6), R(1.5), R(2.2)]),
                    notes=f"Numbers in parentheses give outbreak counts. Asymmetric bands (-10/+5, -15/+5, -20/+5, and "
                          f"group-calibrated box) all determine {emp['asymmetric']['calibrated_group_box']['determined']}/15 pairs; "
                          f"odds-ratio bands determine {or_det['1.5']['determined']}/15 at Γ = 1.5, {or_det['3.0']['determined']}/15 "
                          f"at Γ = 3.0, and {or_det['5.0']['determined']}/15 at Γ = 5.0.", sep="2.3pt")

    # Table 13 (Supp Table S10): Real-data follow-up priority strata (Minor 3)
    if "followup_masking" in r:
        rt = r["followup_masking"]["real_data_top_strata"]
        rows = []
        for idx, s in enumerate(rt, 1):
            rows.append({
                "Rank": str(idx),
                "Pathogen group": NAME[s["group"]],
                "Setting | Period": s["cell"],
                "Weight (%)": f"{100 * s['weight']:.1f}",
                "Cases": f"{int(s['total_cases']):,}",
                "Unrecorded": f"{int(s['unrecorded_cases']):,}",
                "w_s/C_js (pp/case)": f"{s['width_leverage_pp']:.4f}",
                "DPI (score/case)": f"{s['dpi_per_case_pp']:.2f}",
            })
        write_table(tdir, man, "tab:real_followup", "table_13_real_followup_priority", pd.DataFrame(rows),
                    "Top 12 incomplete group × setting-period strata in the observed NORS common-support cohort ranked by "
                    "per-case disambiguation priority index (DPI).",
                    "".join([R(0.75), Lc(2.1), Lc(3.5), R(1.2), R(1.1), R(1.4), R(2.0), R(1.8)]), sep="2.3pt")

    # Table 14 (Supp Table S11): Small-sample Monte Carlo calibration (Major 3a)
    sc = r.get("small_sample_calibration")
    if sc:
        rows = []
        for g in order:
            pg = sc["per_group"][g]
            rows.append({
                "Group": NAME[g],
                "MC SD(L) (pp)": f"{pg['mc_sd_L_pp']:.2f}",
                "Mean HC1 SE(L)": f"{pg['mean_hc1_se_L_pp']:.2f}",
                "Ratio L (HC1/MC)": f"{pg['ratio_hc1_to_mc_L']:.3f}",
                "MC SD(U) (pp)": f"{pg['mc_sd_U_pp']:.2f}",
                "Mean HC1 SE(U)": f"{pg['mean_hc1_se_U_pp']:.2f}",
                "Ratio U (HC1/MC)": f"{pg['ratio_hc1_to_mc_U']:.3f}",
            })
        write_table(tdir, man, "tab:small_cal", "table_14_small_sample_cal", pd.DataFrame(rows),
                    f"Small-sample stratified Monte Carlo calibration ({sc['n_mc']} resamples preserving observed "
                    f"stratum sizes n_js) comparing analytic HC1 sandwich SEs against empirical sampling SDs.",
                    "".join([Lc(2.2), R(1.75), R(1.85), R(1.9), R(1.75), R(1.85), R(1.9)]),
                    notes=f"Across the 15 pairs, Imbens–Manski 95% empirical boundary coverage ranged from "
                          f"{100*sc['im95_min_endpoint_cov_hc1']:.1f}% (min) to {100*sc['im95_median_endpoint_cov_hc1']:.1f}% "
                          f"(median) under HC1 versus {100*sc['im95_min_endpoint_cov_hc0']:.1f}%–"
                          f"{100*sc['im95_median_endpoint_cov_hc0']:.1f}% under HC0; empirical Max-T FWER was "
                          f"{100*sc['fwer_worst_hc1']:.1f}% (worst case) and {100*sc['fwer_delta5_hc1']:.1f}% (δ = 5) under HC1, "
                          f"reflecting sparse high-HHI strata (N_eff ≈ 2).", sep="2.4pt")

    (tdir / "manifest.json").write_text(json.dumps(dict(schema_version=2, tables=man), indent=1, ensure_ascii=False),
                                        encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------
def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return None


def _save(fig, kit, fdir: Path, base: str, manifest: list, caption: str, section: str):
    fig.savefig(fdir / f"{base}.pdf")
    fig.savefig(fdir / f"{base}.png", dpi=300)
    manifest.append(dict(id=base, path_pdf=f"{base}.pdf", path_png=f"{base}.png", caption=caption, section=section,
                         generator="code/make_outputs.py", data_source="analysis/results.json"))
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_matrix(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    m = r["main"]
    order = m["order_by_delta0"]
    n = len(order)
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    lookup = {(p["higher_at_delta0"], p["lower_at_delta0"]): p for p in m["pairs"].values()}
    rows, cols = order[:-1], order[1:]
    for i, gi in enumerate(rows):
        for j, gj in enumerate(cols):
            if j < i:
                continue
            p = lookup[(gi, gj)]
            cat = category(p)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CAT_COLORS[cat], edgecolor="white", lw=2))
            txt = "no assumption\nneeded" if cat == "A" else f"δ* = {fmt_ds(p, unit='')}\npp"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=CAT_TEXT[cat])
    ax.set_xlim(-0.5, n - 1.5); ax.set_ylim(n - 1.5, -0.5)
    ax.set_xticks(range(n - 1)); ax.set_xticklabels([NAME[g] for g in cols])
    ax.set_yticks(range(n - 1)); ax.set_yticklabels([NAME[g] for g in rows])
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Lower group at δ = 0 (column)", labelpad=10)
    ax.set_ylabel("Higher group at δ = 0 (row)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0); ax.set_aspect("equal")
    handles = [Patch(facecolor=CAT_COLORS[c], label=CAT_LABEL[c]) for c in ("A", "B", "C")]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False, fontsize=8)
    nd = m["n_determined"]
    cap = (f"Sample pairwise orderings and critical deviations δ*. Dark blue cells ({nd['worst_case']}/{m['n_pairs']}) "
           f"hold without assumptions on unrecorded outcomes; δ = 5 pp shading is an illustrative scenario threshold.")
    _save(fig, kit, fdir, "fig_02_verdict_matrix", man, cap, "results")


def fig_intervals(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    m, bt = r["main"], r["bootstrap"]
    ret = r["retention"]["per_group"]
    order = m["order_by_delta0"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    for i, g in enumerate(order):
        v = m["per_group"][g]
        b = bt["per_group"][g]
        cr = ret[g]["crude_after"]
        ax.plot([100 * v["L_worst"], 100 * v["U_worst"]], [i, i], color="#B4C0E4", lw=9, solid_capstyle="butt")
        d5 = v["bounds_by_delta_pp"]["5"]
        ax.plot([100 * d5["L"], 100 * d5["U"]], [i, i], color="#0F4D92", lw=9, solid_capstyle="butt")
        ax.plot([100 * v["theta_delta0"]], [i], marker="o", ms=5.0, mfc="white", mec="#272727", mew=1.0, ls="none", zorder=4)
        # Overlay crude H/C (square) and crude H/K (triangle) conventions (Minor 20)
        ax.plot([100 * cr["L"]], [i - 0.19], marker="s", ms=4.5, mfc="#D95F02", mec="#272727", mew=0.6, ls="none", zorder=5)
        ax.plot([100 * cr["recorded_proportion"]], [i + 0.19], marker="^", ms=5.0, mfc="#1B9E77", mec="#272727", mew=0.6, ls="none", zorder=5)
        for lo, hi in ((b["L_p2_5"], b["L_p97_5"]), (b["U_p2_5"], b["U_p97_5"])):
            ax.plot([100 * lo, 100 * hi], [i + 0.34, i + 0.34], color="#4D4D4D", lw=0.9)
            ax.plot([100 * lo, 100 * lo], [i + 0.28, i + 0.40], color="#4D4D4D", lw=0.9)
            ax.plot([100 * hi, 100 * hi], [i + 0.28, i + 0.40], color="#4D4D4D", lw=0.9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[g] for g in order])
    ax.set_ylim(len(order) - 0.35, -0.55)
    xmax = 100 * max(max(bt["per_group"][g]["U_p97_5"], m["per_group"][g]["U_worst"]) for g in order)
    ax.set_xlim(0, 5 * np.ceil(xmax / 5 + 0.2))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hospitalization proportion among reported outbreak cases (%)")
    handles = [
        Line2D([], [], color="#B4C0E4", lw=7, label="Standardized worst-case interval"),
        Line2D([], [], color="#0F4D92", lw=7, label="Standardized δ = ±5 pp scenario"),
        Line2D([], [], marker="o", mfc="white", mec="#272727", ls="none", ms=5, label="Standardized δ = 0 value"),
        Line2D([], [], marker="s", mfc="#D95F02", mec="#272727", ls="none", ms=4.5, label="Crude H/C (unstandardized)"),
        Line2D([], [], marker="^", mfc="#1B9E77", mec="#272727", ls="none", ms=5, label="Crude H/K (unstandardized)"),
        Line2D([], [], color="#4D4D4D", lw=0.9, label=f"Bootstrap 2.5th–97.5th percentile (B = {bt['B']:,})"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7.6)
    cap = ("Standardized hospitalization bounds overlaid with crude H/C and H/K conventions across common-support cells.")
    _save(fig, kit, fdir, "fig_03_bound_intervals", man, cap, "results")


def fig_calibrated_2d(r, kit, fdir, man):
    """Main Text Fig. 4: 2D (delta_h_down, delta_l_up) sensitivity with empirical calibrated box."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    m = r["main"]
    emp = r["empirical_calibration"]["per_group"]
    sp = sorted_pairs(m)
    fig, axes = plt.subplots(3, 5, figsize=(7.4, 5.1), constrained_layout=True, sharex=True, sharey=True)
    for ax, (k, p) in zip(axes.flat, sp):
        td = m["two_d"][k]
        x = np.array(td["d_hi_pp"])
        y = np.array([20.0 if v is None else min(v, 20.0) for v in td["d_lo_boundary_pp"]])
        ax.fill_between(x, 0, y, color=CAT_COLORS[category(p)], alpha=0.85, lw=0)
        # Calibrated box: higher group downward shift <= calibrated_down_pp, lower group upward shift <= 5 pp
        cal_x = min(20.0, emp[p["higher_at_delta0"]]["calibrated_down_pp"])
        rect = Rectangle((0.0, 0.0), cal_x, 5.0, facecolor="none", edgecolor="#D95F02", lw=1.25, ls="--", zorder=4)
        ax.add_patch(rect)
        ax.set_xlim(0, 20); ax.set_ylim(0, 20)
        ax.set_xticks([0, 10, 20]); ax.set_yticks([0, 10, 20])
        ax.grid(alpha=0.3)
        ax.set_title(pair_label(p, SHORT), fontsize=8)
        ax.tick_params(labelsize=7)
    fig.supxlabel("Downward deviation allowed for higher group, δ_h,down (percentage points)", fontsize=9)
    fig.supylabel("Upward deviation allowed for lower group, δ_l,up (percentage points)", fontsize=9)
    handles = [
        Patch(facecolor=CAT_COLORS["A"], alpha=0.85, label="Worst-case determined (all deviations)"),
        Patch(facecolor=CAT_COLORS["B"], alpha=0.85, label="Supported at δ* ≥ 5 pp"),
        Patch(facecolor=CAT_COLORS["C"], alpha=0.85, label="Fails below δ = 5 pp"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#D95F02", lw=1.25, ls="--",
                  label="Empirically calibrated asymmetric region (down ≤ max(5, Δ_full-part), up ≤ 5 pp)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7.6)
    cap = ("Two-dimensional asymmetric sensitivity across all 15 pairs with empirically calibrated downward departure boxes.")
    _save(fig, kit, fdir, "fig_04_calibrated_2d", man, cap, "results")


def fig_inference_followup(r, kit, fdir, man):
    """Main Text Fig. 5: (a) dynamic FWER tipping points (HC1 & State); (b) budget-filling follow-up with 95% CIs."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    si = r["stacked_inference"]
    vm = r["followup_masking"]
    m = r["main"]
    sp = sorted_pairs(m)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    y = np.arange(len(sp))
    for i, (k, p) in enumerate(sp):
        vp = si["pairs"][k]
        pt = 100.0 if vp["delta_star_pt_pp"] is None else vp["delta_star_pt_pp"]
        hc = 100.0 if vp["delta_star_95_sim_pp"] is None else vp["delta_star_95_sim_pp"]
        st = 100.0 if vp["delta_star_95_sim_state_pp"] is None else vp["delta_star_95_sim_state_pp"]
        ax.plot([min(hc, st), pt], [i, i], color="#9AA5B8", lw=1.4, zorder=1)
        ax.scatter([pt], [i], color="#0F4D92", s=24, zorder=3, marker=("^" if vp["delta_star_pt_pp"] is None else "o"))
        ax.scatter([hc], [i - 0.12], color="#C44E52", s=22, zorder=4, marker=("^" if vp["delta_star_95_sim_pp"] is None else "D"))
        ax.scatter([st], [i + 0.12], color="#D95F02", s=22, zorder=4, marker=("^" if vp["delta_star_95_sim_state_pp"] is None else "s"))
    ax.axvline(5.0, color="#666666", ls=":", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([pair_label(p, SHORT) for _, p in sp], fontsize=7)
    ax.set_ylim(len(sp) - 0.5, -0.5); ax.set_xlim(-3, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xticklabels(["0", "20", "40", "60", "80", "100 (∞)"], fontsize=7.5)
    ax.grid(axis="x", alpha=0.35, lw=0.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Tipping point δ* (percentage points)", fontsize=8)
    ax.set_title("(a) Sample δ* vs. dynamic FWER 95% δ*_0.95", fontsize=8.5, pad=6)
    h_a = [
        Line2D([], [], color="#0F4D92", marker="o", ms=4.5, ls="none", label="Sample δ*"),
        Line2D([], [], color="#C44E52", marker="D", ms=4.0, ls="none", label="HC1 FWER δ*_0.95"),
        Line2D([], [], color="#D95F02", marker="s", ms=4.0, ls="none", label="53-state FWER δ*_0.95"),
        Line2D([], [], color="#444444", marker="^", ms=4.5, ls="none", label="Holds at δ = 100 (∞)"),
    ]
    ax.legend(handles=h_a, loc="lower right", frameon=True, facecolor="white", framealpha=0.88, fontsize=6.5)

    ax = axes[1]
    budgets = vm["budgets_cases"]
    styles = [
        ("dpi", "Disambiguation priority (DPI)", "#0F4D92", "-", "o", 2.0),
        ("width_greedy", "Width-greedy (w_s / C_js)", "#2B8CBE", "--", "s", 1.5),
        ("random", "Random follow-up", "#767676", "-.", "^", 1.4),
        ("volume_greedy", "Largest outbreaks first", "#C44E52", ":", "v", 1.6),
    ]
    for pk, lab, col, ls, mk, lw in styles:
        vals = np.array([vm["summary"][pk][str(b)]["mean_worst_det"] for b in budgets])
        ci_lo = np.array([vm["summary"][pk][str(b)]["ci95_worst_det"][0] for b in budgets])
        ci_hi = np.array([vm["summary"][pk][str(b)]["ci95_worst_det"][1] for b in budgets])
        ax.fill_between(budgets, ci_lo, ci_hi, color=col, alpha=0.14, lw=0)
        ax.plot(budgets, vals, color=col, ls=ls, marker=mk, ms=3.8, lw=lw, label=lab)
    ax.set_xlim(-100, 5150); ax.set_ylim(0, 15.3)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000", "4,000", "5,000"], fontsize=7.5)
    ax.set_yticks([0, 3, 6, 9, 12, 15]); ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(alpha=0.35, lw=0.5)
    ax.set_xlabel("Follow-up budget (cases resolved, budget-filling)", fontsize=8)
    ax.set_ylabel("Worst-case determined pairs (of 15)", fontsize=8)
    ax.set_title("(b) Retrospective follow-up (mean & 95% MC CI)", fontsize=8.5, pad=6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.88, fontsize=6.5)

    cap = ("Joint confidence tipping points under dynamic c_sim(δ) and budget-filling follow-up curves with 95% MC intervals.")
    _save(fig, kit, fdir, "fig_05_inference_followup", man, cap, "results")


def fig_retention(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    m, ret = r["main"], r["retention"]["per_group"]
    order = m["order_by_delta0"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True, sharey=True)
    ax = axes[0]
    for i, g in enumerate(order):
        v = ret[g]
        ax.barh(i, v["support_cases_pct"], height=0.6, color="#3775BA")
        ax.text(v["support_cases_pct"] + 1.5, i, f"{v['support_cases_pct']:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 100); ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[g] for g in order])
    ax.set_ylim(len(order) - 0.5, -0.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("(a) Eligible cases inside common support (%)")
    ax = axes[1]
    cols = ("#0F4D92", "#AADCA9", "#CFCECE")
    for i, g in enumerate(order):
        v = m["per_group"][g]
        h = v["hospitalized_known"] / v["cases"]; u = v["unknown_share"]; nh = 1 - h - u
        left = 0
        for val, c in zip((h, nh, u), cols):
            ax.barh(i, 100 * val, left=left, height=0.6, color=c); left += 100 * val
    ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
    ax.set_xlabel("(b) Retained cases by recorded status (%)")
    handles = [Patch(facecolor=c, label=t) for c, t in zip(cols, ("Recorded hospitalized", "Recorded not hospitalized",
                                                                    "Hospitalization status unrecorded"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    _save(fig, kit, fdir, "fig_s1_retention_recording", man, "Common-support retention and status recording.", "supplementary")


def fig_robust(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    m = r["main"]
    sp = sorted_pairs(m)
    variants = [("Main", m)] + [(VARIANT_CODES[k], v) for k, v in r["robustness"].items()
                                if k != "R7_norovirus_combined"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    for j, (code, v) in enumerate(variants):
        for i, (k, p) in enumerate(sp):
            q = v["pairs"][k]
            cat = "D" if q["higher_at_delta0"] != p["higher_at_delta0"] else category(q)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CAT_COLORS[cat], edgecolor="white", lw=1.1))
    ax.set_xlim(-0.5, len(variants) - 0.5); ax.set_ylim(len(sp) - 0.5, -0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([f"{c}\n{v['n_determined']['worst_case']}|{v['n_determined']['5']}" for c, v in variants],
                       fontsize=6.5)
    ax.set_yticks(range(len(sp))); ax.set_yticklabels([pair_label(p) for _, p in sp], fontsize=7.5)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Analysis (second line: determined pairs, worst case | δ = 5 points)")
    handles = [Patch(facecolor=CAT_COLORS[c], label=t) for c, t in
               (("A", "Holds without unrecorded-outcome assumptions"), ("B", "Holds at δ = 5 pp (δ* ≥ 5)"),
                ("C", "Fails before δ = 5 pp (δ* < 5)"), ("D", "δ = 0 direction reversed vs main"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7.8)
    _save(fig, kit, fdir, "fig_s2_robustness", man, "Sensitivity verdicts across all 15 six-group specifications.", "supplementary")


def fig_followup_sens(r, kit, fdir, man):
    """Supp Fig S3: (a) Paired DPI - Width-greedy difference; (b) MNAR, state-clustered, partial yield & adaptive DPI."""
    import matplotlib.pyplot as plt
    vm = r["followup_masking"]
    pd_map = vm["paired_differences"]
    budgets_all = [b for b in vm["budgets_cases"] if b > 0]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), constrained_layout=True)

    # Panel (a): Paired difference DPI minus Width-greedy
    ax = axes[0]
    means = np.array([pd_map[str(b)]["dpi_minus_wg_mean"] for b in budgets_all])
    los = np.array([pd_map[str(b)]["dpi_minus_wg_ci95"][0] for b in budgets_all])
    his = np.array([pd_map[str(b)]["dpi_minus_wg_ci95"][1] for b in budgets_all])
    y = np.arange(len(budgets_all))
    ax.axvline(0.0, color="#666666", ls="--", lw=1.0)
    ax.errorbar(means, y, xerr=[means - los, his - means], fmt="o", color="#0F4D92", ecolor="#0F4D92",
                elinewidth=1.4, capsize=3, ms=4.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{b:,}" for b in budgets_all], fontsize=7.5)
    ax.set_ylim(len(budgets_all) - 0.5, -0.5)
    ax.set_ylabel("Follow-up budget (cases)", fontsize=8)
    ax.set_xlabel("Paired difference: DPI − Width-greedy (pairs, 95% CI)", fontsize=8)
    ax.set_title("(a) Added value of pair-weighting over w_s / C_js", fontsize=8.5, pad=6)
    ax.grid(axis="x", alpha=0.35, lw=0.5)

    # Panel (b): Non-MCAR, 70% yield, Adaptive re-scoring
    ax = axes[1]
    sb = [500, 1000, 2000, 5000]
    sr = vm["sensitivity_regimes"]
    curves = [
        ("Adaptive DPI (baseline MCAR)", [sr["adaptive_rescoring_dpi"][str(b)]["mean_worst_det"] for b in sb], "#0F4D92", "-", "o"),
        ("MNAR outcome+size: DPI", [sr["mnar_outcome_size"]["dpi"][str(b)]["mean_worst_det"] for b in sb], "#2B8CBE", "--", "s"),
        ("MNAR outcome+size: Width-greedy", [sr["mnar_outcome_size"]["width_greedy"][str(b)]["mean_worst_det"] for b in sb], "#1B9E77", "-.", "^"),
        ("MNAR outcome+size: Largest-first", [sr["mnar_outcome_size"]["volume_greedy"][str(b)]["mean_worst_det"] for b in sb], "#C44E52", ":", "v"),
        ("70% partial yield: DPI", [sr["partial_yield_70pct"]["dpi"][str(b)]["mean_worst_det"] for b in sb], "#7570B3", "--", "D"),
    ]
    for lab, vals, col, ls, mk in curves:
        ax.plot(sb, vals, color=col, ls=ls, marker=mk, ms=3.8, lw=1.5, label=lab)
    ax.set_xticks(sb); ax.set_xticklabels([f"{b:,}" for b in sb], fontsize=7.5)
    ax.set_ylim(0, 15.5); ax.set_yticks([0, 3, 6, 9, 12, 15])
    ax.set_xlabel("Follow-up budget (cases)", fontsize=8)
    ax.set_ylabel("Worst-case determined pairs (of 15)", fontsize=8)
    ax.set_title("(b) Adaptive, MNAR, and partial-yield regimes", fontsize=8.5, pad=6)
    ax.grid(alpha=0.35, lw=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.88, fontsize=6.2)

    _save(fig, kit, fdir, "fig_s3_followup_sensitivity", man,
          "Paired DPI vs width-greedy differences and follow-up sensitivity across MNAR, adaptive, and partial-yield designs.",
          "supplementary")


def make_figures(r: dict, fdir: Path) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    for f in list(fdir.glob("fig_*.pdf")) + list(fdir.glob("fig_*.png")):
        if f.name == "fig_01_selection_flow.png":
            continue
        f.unlink()
    kit = _style()
    man = []
    for fn in (fig_matrix, fig_intervals, fig_calibrated_2d, fig_inference_followup,
               fig_retention, fig_robust, fig_followup_sens):
        fn(r, kit, fdir, man)
    (fdir / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")


def make_all(exp: Path, paper_dir: Path) -> None:
    r = load(exp)
    make_tables(r, exp / "data" / "analysis_tables")
    make_tables(r, paper_dir / "tables")
    make_figures(r, paper_dir / "figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=HERE.parent)
    ap.add_argument("--paper-dir", type=Path, default=HERE.parents[1] / "paper")
    a = ap.parse_args()
    statistical_summary(a.exp_dir.resolve())
    make_all(a.exp_dir.resolve(), a.paper_dir.resolve())
