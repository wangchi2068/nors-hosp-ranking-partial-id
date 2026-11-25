"""Tables, figures, statistical summary and generated text for the NORS hospitalization-ranking analysis.

Reads ONLY analysis/results.json (written by analyze.py) so that every number in a table, figure or sentence comes
from that single source.  Called by analyze.py; can also be run alone:

    python make_outputs.py [--exp-dir <experiment dir>] [--work-dir <.work/<slug>/experiment>]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
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
                 "R7_norovirus_combined": "R7"}
CAT_COLORS = {"A": "#0F4D92", "B": "#8FA8D6", "C": "#E9A6A1", "D": "#E8B04B"}
CAT_TEXT = {"A": "white", "B": "#272727", "C": "#272727"}
CAT_LABEL = {
    "A": "Supported by recorded data alone (worst-case intervals do not overlap)",
    "B": "Needs an assumption on unrecorded cases; holds for any δ below δ* ≥ 5 points",
    "C": "Needs an assumption; fails once δ reaches δ* < 5 points (needs more complete records)",
    "D": "Direction at δ = 0 reversed relative to the main analysis",
}


# ------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------
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
                 (">=", r"$\geq$"), ("<=", r"$\leq$"),
                 ("δ", r"$\delta$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("–", "--"), (">", r"$>$"),
                 ("*", r"$^{*}$"), ("×", r"$\times$"), ("∞", r"$\infty$")):
        s = s.replace(a, b)
    return s.replace("$$", "")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------
# statistical summary
# ------------------------------------------------------------------------------------------------------------
def statistical_summary(exp: Path) -> None:
    r = load(exp)
    m = r["main"]
    out = dict(
        source="analysis/results.json", spec_hash=r["provenance"]["spec_hash"],
        n_pairs=m["n_pairs"], n_common_support_cells=m["n_cells"],
        determined_pairs=dict(worst_case=m["n_determined"]["worst_case"],
                              **{f"delta_{k}pp": v for k, v in m["n_determined"].items() if k != "worst_case"}),
        determined_worst_case=[pair_label(p) for k, p in m["pairs"].items() if p["worst_case"] != "undetermined"],
        undetermined_at_delta5=[pair_label(p) for k, p in m["pairs"].items()
                                if p["verdict_by_delta_pp"]["5"] == "undetermined"],
        delta_star_pp={pair_label(p): p["delta_star_pp"] for k, p in sorted_pairs(m)},
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
            n_im_excludes_zero=r["stacked_inference"]["n_im_excludes_zero"],
            n_fwer_worst_sig=r["stacked_inference"]["n_fwer_worst_sig"],
            n_fwer_delta5_sig=r["stacked_inference"]["n_fwer_delta5_sig"],
            c_sim_worst=r["stacked_inference"]["c_sim_worst"],
            c_sim_delta5=r["stacked_inference"]["c_sim_delta5"],
        ) if "stacked_inference" in r else None,
        note="Identification intervals are not confidence intervals; bootstrap shares describe sampling variability. "
             "Uniformly valid Imbens-Manski 95% CIs and Max-T FWER p-values appear in stacked_inference.")
    (exp / "analysis" / "statistical_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------
def R(w):  # right-aligned wrapping column of width w (cm), for numbers and their headers
    return rf">{{\raggedleft\arraybackslash}}p{{{w}cm}}"


def Lc(w):  # left-aligned wrapping column of width w (cm)
    return rf">{{\raggedright\arraybackslash}}p{{{w}cm}}"


def write_table(tdir: Path, manifest: list, tid: str, base: str, df: pd.DataFrame, caption: str, colspec: str,
                notes: str = "", size: str = "scriptsize", sep: str = "3pt"):
    """Paper-sized LaTeX table: fixed-width wrapping columns so every table fits the single-column text width."""
    df.to_csv(tdir / f"{base}.csv", index=False)
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
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
                f"({m['n_cells']} common-support setting × period cells, pooled-case reference weights). Worst-case "
                f"intervals are identification intervals, not confidence intervals; bootstrap ranges are the "
                f"2.5th–97.5th percentiles of each bound over {bt['B']:,} outbreak-level resamples.",
                "".join([Lc(2.0), R(1.4), R(1.1), R(1.4), R(1.6), R(1.5), R(1.2), R(1.8), R(1.8)]),
                notes="Unrecorded status is the share of retained cases whose hospitalization status was not recorded. "
                      "The δ = 0 value is the proportion if unrecorded-status cases had the recorded rate of their group "
                      "and cell.")
    # Table 2: pairwise
    rows = []
    for k, p in sorted_pairs(m):
        b = bt["pairs"][k]
        vd = p["verdict_by_delta_pp"]
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT),
                     "Worst case": "determined" if p["worst_case"] != "undetermined" else "undetermined",
                     "δ* (points)": "∞" if p["delta_star_pp"] is None else (f"{p['delta_star_pp']:.2f}"
                                                                           if p["delta_star_pp"] < 1 else f"{p['delta_star_pp']:.1f}"),
                     **{f"δ = {d}": ("yes" if vd[str(d)] != "undetermined" else "no") for d in (1, 2, 5, 10)},
                     "Bootstrap, worst-case verdict reproduced (%)": pnum(b["share_worst_main_verdict"]),
                     "Bootstrap, δ = 5 verdict reproduced (%)": pnum(b["share_delta5_main_verdict"])})
    write_table(tdir, man, "tab:pairwise", "table_2_pairwise", pd.DataFrame(rows),
                "Pairwise orderings with their verdict under worst-case bounds and the critical deviation δ*. δ* is the "
                "smallest deviation of the unrecorded-status hospitalization rate from the recorded rate at which the "
                "δ = 0 ordering is no longer guaranteed (∞, holds with no assumption). Further columns show whether the "
                "ordering is determined at δ = 1, 2, 5 and 10 points and the share of bootstrap replicates reproducing "
                "the main verdicts.",
                "".join([Lc(2.6), Lc(1.8), R(1.1), R(0.8), R(0.8), R(0.8), R(0.8), R(2.2), R(2.2)]),
                notes="SAL Salmonella; STEC Shiga toxin-producing E. coli; CAMP Campylobacter; CPERF C. perfringens; "
                      "NoV GI/GII norovirus genogroup I/II. At δ = 0 every pair is ordered (point values differ). "
                      "The last two columns report the share of bootstrap replicates reproducing the main-analysis "
                      "verdict (determined or undetermined under worst-case bounds; yes or no at δ = 5 points).")
    # Table 3: robustness
    rows = [{"Analysis": "Main", "Description": "Setting × period, ≥ 3 outbreaks, pooled-case weights",
             "Cells": m["n_cells"], "Pairs": m["n_pairs"], "Determined, worst case": m["n_determined"]["worst_case"],
             "Determined, δ = 5": m["n_determined"]["5"], "Worst-case verdicts changed": 0,
             "δ = 0 directions changed": 0}]
    for vid, v in r["robustness"].items():
        desc = v["label"].replace("Crude: no standardization", "Crude, no standardization").replace(
            "Cohort: ", "Cohort of ").replace("Weights: ", "Weights from ")
        if vid == "R3c_pilot_rule":  # describe the rule itself in the paper table
            desc = (f"Cohort of species-level etiology strings; records with exactly one blank outcome field "
                    f"dropped ({v['n_outbreaks_in_cohort']:,} outbreaks)")
        rows.append({"Analysis": VARIANT_CODES[vid], "Description": desc, "Cells": v["n_cells"],
                     "Pairs": v["n_pairs"], "Determined, worst case": v["n_determined"]["worst_case"],
                     "Determined, δ = 5": v["n_determined"]["5"],
                     "Worst-case verdicts changed": v.get("n_worst_case_verdicts_changed", "NA"),
                     "δ = 0 directions changed": v.get("n_delta0_directions_changed", "NA")})
    write_table(tdir, man, "tab:robustness", "table_3_robustness", pd.DataFrame(rows),
                "Robustness analyses, with the number of common-support cells, the number of determined pairwise "
                "orderings under worst-case bounds and at δ = 5 points, and changes relative to the main analysis.",
                "".join([Lc(1.1), Lc(4.9), R(0.9), R(0.9), R(1.5), R(1.5), R(1.5), R(1.5)]),
                notes="R7 has five groups (10 pairs), so changes versus the main analysis are not defined (NA).")
    # Table 4: retention
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
                     "Crude δ = 0 before / after (%)": f"{pnum(v['crude_before']['theta_delta0'])} / {pnum(v['crude_after']['theta_delta0'])}",
                     "Crude worst case before / after (%)": f"{pnum(v['crude_before']['L'])}–{pnum(v['crude_before']['U'])} / "
                                                           f"{pnum(v['crude_after']['L'])}–{pnum(v['crude_after']['U'])}"})
    write_table(tdir, man, "tab:retention", "table_4_retention", pd.DataFrame(rows),
                "Common-support retention by pathogen group, with eligible and retained outbreaks and cases, the settings "
                "and periods losing most cases, and crude (unstandardized) proportions before and after the restriction.",
                "".join([Lc(1.75), R(1.4), R(1.05), R(1.45), R(1.45), Lc(1.85), Lc(1.75), R(1.3), R(1.8)]))
    # Table 5: record audit
    a = r["audit"]["record_audit_summary"]
    handling = {"complete_consistent": "as recorded (K = 0 means all status unrecorded)",
                "single_missing_K": "K set to H; remaining cases unrecorded",
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
                     "Worst case, higher group > lower (%)": pnum(b["share_worst_A"] if p["higher_at_delta0"] == p["A"]
                                                                 else b["share_worst_B"]),
                     "Worst case, undetermined (%)": pnum(b["share_worst_undetermined"]),
                     "Determined at δ = 5 (%)": pnum(1 - b["share_delta5_undetermined"]),
                     "δ* 2.5th–97.5th percentile (points)":
                         f"{'∞' if b['delta_star_pp_p2_5'] is None else format(b['delta_star_pp_p2_5'], '.1f')}–"
                         f"{'∞' if b['delta_star_pp_p97_5'] is None else format(b['delta_star_pp_p97_5'], '.1f')}"})
    write_table(tdir, man, "tab:bootstrap", "table_6_bootstrap", pd.DataFrame(rows),
                f"Outbreak-level bootstrap ({bt['B']:,} replicates, seed {bt['seed']}; outbreaks resampled within "
                f"group × common-support cell, reference weights recomputed), with the share of replicates giving each "
                f"verdict and the percentile range of δ* for the main-analysis direction. These describe sampling "
                f"variability and are not confidence intervals.",
                "".join([Lc(2.6), R(2.0), R(2.2), R(2.0), R(2.0), R(2.6)]))
    # Table 7: width decomposition (supplementary)
    rows = []
    wd = r["width_decomposition"]["undetermined_pairs"]
    for k, p in sorted_pairs(m):
        if k not in wd:
            continue
        tops = "; ".join(f"{SHORT[t['group']]} {t['cell']} ({100 * t['contribution']:.1f})" for t in wd[k]["top_group_cells"])
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT), "Worst-case overlap (points)": pnum(wd[k]["overlap"]),
                     "Largest contributions to interval width (group, cell, points)": tops})
    write_table(tdir, man, "tab:width", "table_7_width_decomposition", pd.DataFrame(rows),
                "Undetermined pairs under worst-case bounds, with the overlap of the two intervals and the group × cell "
                "combinations whose unrecorded-status cases contribute most to the interval widths. The decomposition "
                "is descriptive and indicates where more complete hospitalization records would narrow the intervals "
                "most.",
                "".join([Lc(2.6), R(1.6), Lc(10.6)]))
    # Table 8: Joint M-estimation inference (Main Table 3)
    if "stacked_inference" in r:
        si = r["stacked_inference"]
        rows = []
        for k, p in sorted_pairs(m):
            sp = si["pairs"][k]
            fmt_ci = lambda a, b: f"[{100 * a:+.1f}, {100 * b:+.1f}]"
            fmt_ds = lambda v: "∞" if v is None else (f"{v:.2f}" if v < 1 else f"{v:.1f}")
            fmt_p = lambda pv: "<0.0001" if pv < 0.0001 else f"{pv:.4f}"
            rows.append({
                "Ordering at δ = 0": pair_label(p, SHORT),
                "ID bounds (pp)": fmt_ci(sp["dL_worst"], sp["dU_worst"]),
                "IM 95% CI (pp)": fmt_ci(sp["ci95_im_lo"], sp["ci95_im_hi"]),
                "FWER p (worst)": fmt_p(sp["p_fwer_worst"]),
                "FWER p (δ = 5)": fmt_p(sp["p_fwer_delta5"]),
                "δ* (pt)": fmt_ds(sp["delta_star_pt_pp"]),
                "95% δ* (pw)": fmt_ds(sp["delta_star_95_pw_pp"]),
                "95% δ* (sim)": fmt_ds(sp["delta_star_95_sim_pp"]),
            })
        write_table(tdir, man, "tab:joint_inference", "table_8_joint_inference", pd.DataFrame(rows),
                    f"Joint M-estimation statistical inference for the 15 pairwise standardized differences, combining "
                    f"outbreak-cluster HC1 sandwich variance, Imbens–Manski 95% confidence intervals, Gaussian "
                    f"multiplier Max-T stepdown FWER p-values (c = {si['c_sim_worst']:.3f} worst case; "
                    f"{si['c_sim_delta5']:.3f} at δ = 5 points), and 95% confidence tipping points.",
                    "".join([Lc(2.4), R(2.0), R(2.1), R(1.6), R(1.6), R(1.1), R(1.3), R(1.3)]),
                    notes="ID bounds are sample worst-case differences [L - U, U - L]; IM 95% CI is the "
                          "Imbens–Manski uniformly valid confidence interval; FWER p tests H0: higher <= lower "
                          "across all 15 pairs; 95% δ* (pw) and 95% δ* (sim) are pairwise and 15-pair FWER-adjusted "
                          "95% confidence tipping points in percentage points (∞ = holds at δ = 100).")
    # Table 9: Cluster concentration & SE calibration (Supp Table S5)
    if "cluster_diagnostics" in r and "stacked_inference" in r:
        cd = r["cluster_diagnostics"]["per_group"]
        sg = r["stacked_inference"]["per_group"]
        rows = []
        for g in order:
            c_g = cd[g]
            s_g = sg[g]
            b_g = bt["per_group"][g]
            boot_se_L = (b_g["L_p97_5"] - b_g["L_p2_5"]) / 3.9199
            boot_se_U = (b_g["U_p97_5"] - b_g["U_p2_5"]) / 3.9199
            rows.append({
                "Group": NAME[g],
                "Min / Med / Max n": f"{c_g['min_n']} / {c_g['median_n']:.0f} / {c_g['max_n']}",
                "Mean / Max HHI": f"{c_g['mean_hhi']:.3f} / {c_g['max_hhi']:.3f}",
                "Min N eff": f"{c_g['min_neff']:.2f}",
                "Max share (%)": f"{100 * c_g['max_single_share']:.1f}",
                "SE(L) HC0 / HC1 / Boot (pp)": f"{100 * s_g['se_L_hc0']:.2f} / {100 * s_g['se_L_hc1']:.2f} / {100 * boot_se_L:.2f}",
                "SE(U) HC0 / HC1 / Boot (pp)": f"{100 * s_g['se_U_hc0']:.2f} / {100 * s_g['se_U_hc1']:.2f} / {100 * boot_se_U:.2f}",
            })
        write_table(tdir, man, "tab:cluster_diag", "table_9_cluster_diagnostics", pd.DataFrame(rows),
                    "Outbreak-cluster concentration diagnostics across the 12 common-support cells and standard error "
                    "comparison between stacked M-estimation sandwich estimators (HC0, HC1) and outbreak-level "
                    "bootstrap (2,000 replicates).",
                    "".join([Lc(1.9), R(2.0), R(1.8), R(1.2), R(1.3), R(2.5), R(2.5)]),
                    notes="HHI is the Herfindahl–Hirschman index of case shares within a group × cell stratum; N eff = "
                          "1/HHI is the effective cluster count; Max share is the largest single outbreak's share of "
                          "cases in any cell. Analytic IF and central-difference sandwich covariances match to "
                          f"{r['stacked_inference']['checks']['max_abs_diff_analytic_if_vs_sandwich']:.2e}.")
    # Table 10: Retrospective masking & VoI follow-up (Supp Table S7)
    if "voi_masking" in r:
        vm = r["voi_masking"]
        pol_names = [("random", "Random follow-up"), ("volume_greedy", "Largest outbreaks first"),
                     ("width_greedy", "Width reduction greedy"), ("voi_disambiguation", "VoI disambiguation (proposed)")]
        rows = []
        for b in vm["budgets_cases"]:
            for pk, pname in pol_names:
                s = vm["summary"][pk][str(b)]
                rows.append({
                    "Budget (cases)": f"{b:,}",
                    "Policy": pname,
                    "Worst-case det (/15)": f"{s['mean_worst_det']:.2f} ({s['sd_worst_det']:.2f})",
                    "δ = 5 det (/15)": f"{s['mean_delta5_det']:.2f} ({s['sd_delta5_det']:.2f})",
                    "Total overlap (pp)": f"{s['mean_overlap_pp']:.1f} ({s['sd_overlap_pp']:.1f})",
                })
        write_table(tdir, man, "tab:voi_masking", "table_10_voi_masking", pd.DataFrame(rows),
                    f"Retrospective masking experiment across {vm['n_trials']} Monte Carlo trials on all "
                    f"{vm['n_complete_outbreaks']:,} complete-status outbreaks ({vm['total_complete_cases']:,} cases) "
                    f"in the common support. Values are means (SD) across trials by follow-up case budget.",
                    "".join([R(1.6), Lc(4.2), R(2.4), R(2.4), R(2.4)]))
    (tdir / "manifest.json").write_text(json.dumps(dict(schema_version=1, tables=man), indent=1, ensure_ascii=False),
                                        encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    root = HERE
    while root != root.parent and not (root / ".claude").is_dir():
        root = root.parent
    kit_dir = root / ".claude" / "skills" / "experiment-suite" / "figure_examples"
    if kit_dir.is_dir():
        sys.path.insert(0, str(kit_dir))
        import style_kit as kit
        kit.apply_publication_style(font_size=9, language="english")
        return kit
    import matplotlib.pyplot as plt  # stand-alone fallback outside the task repository
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    return None


def _save(fig, kit, fdir: Path, base: str, manifest: list, caption: str, section: str):
    if kit is not None:
        fig.canvas.draw()
        kit.assert_manuscript_fonts(fig)
        kit.assert_no_text_collisions(fig)
        kit.assert_no_legend_reference_line_collisions(fig)
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
    fig, ax = plt.subplots(figsize=(7.0, 5.3), constrained_layout=True)
    lookup = {}
    for k, p in m["pairs"].items():
        lookup[(p["higher_at_delta0"], p["lower_at_delta0"])] = p
    rows, cols = order[:-1], order[1:]  # the lowest group is never a row, the highest never a column
    for i, gi in enumerate(rows):
        for j, gj in enumerate(cols):
            if j < i:
                continue
            p = lookup[(gi, gj)]
            cat = category(p)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CAT_COLORS[cat], edgecolor="white", lw=2))
            txt = "no assumption\nneeded" if cat == "A" else f"δ* = {fmt_ds(p, unit='')}\npoints"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=CAT_TEXT[cat])
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 1.5, -0.5)
    ax.set_xticks(range(n - 1))
    ax.set_xticklabels([NAME[g] for g in cols])
    ax.set_yticks(range(n - 1))
    ax.set_yticklabels([NAME[g] for g in rows])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Lower group at δ = 0 (column)", labelpad=10)
    ax.set_ylabel("Higher group at δ = 0 (row)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    handles = [Patch(facecolor=CAT_COLORS[c], label=CAT_LABEL[c]) for c in ("A", "B", "C")]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False, fontsize=8)
    nd = m["n_determined"]
    cap = (f"Which pairwise orderings of the standardized hospitalization proportion among reported outbreak cases are "
           f"supported by the recorded NORS data. Each cell compares the row group with the column group "
           f"(groups ordered by the value obtained if unrecorded-status cases had the recorded rate, δ = 0; this order is "
           f"not a ranking). Dark cells: the row group's worst-case lower bound exceeds the column group's worst-case "
           f"upper bound ({nd['worst_case']} of {m['n_pairs']} pairs). Other cells show δ*, the smallest deviation (percentage "
           f"points) of the unrecorded-status hospitalization rate from the recorded rate, allowed independently per group, "
           f"at which the ordering is no longer guaranteed. {m['n_cells']} setting × period cells, pooled-case weights; "
           f"CDC NORS foodborne outbreaks 2009–2023.")
    _save(fig, kit, fdir, "fig_01_verdict_matrix", man, cap, "results")


def fig_delta(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    m = r["main"]
    sp = sorted_pairs(m)
    n = len(sp)
    fig, ax = plt.subplots(figsize=(7.0, 4.9), constrained_layout=True)
    for row, (k, p) in enumerate(sp):
        ds = p["delta_star_pp"]
        end = 10.0 if ds is None else min(ds, 10.0)
        ax.barh(row, end, left=0, height=0.62, color=CAT_COLORS[category(p)], edgecolor="none")
        if end < 10:
            ax.barh(row, 10 - end, left=end, height=0.62, color="#E6E6E6", edgecolor="none")
            ax.plot([end, end], [row - 0.36, row + 0.36], color="#272727", lw=1.2)
        lab = "no assumption needed" if ds is None else f"δ* = {fmt_ds(p, unit='')}"
        ax.text(10.25, row, lab, ha="left", va="center", fontsize=8, clip_on=False)
    ax.set_yticks(range(n))
    ax.set_yticklabels([pair_label(p) for _, p in sp])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 1, 2, 5, 10])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.6, lw=0.6, color="#767676", ls=":")
    ax.set_axisbelow(False)
    ax.set_xlabel("Allowed deviation δ of the unrecorded-status hospitalization rate from the recorded rate (points)")
    handles = [Patch(facecolor=CAT_COLORS[c], label=t) for c, t in
               (("A", "Determined at every δ (no assumption needed)"), ("B", "Determined below δ* (δ* ≥ 5 points)"),
                ("C", "Determined below δ* (δ* < 5 points)"))] + [Patch(facecolor="#E6E6E6", label="Undetermined")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    ax.text(10.25, -0.95, "critical δ*", ha="left", va="bottom", fontsize=8, style="italic", clip_on=False)
    cap = ("Stability of each pairwise ordering as the assumption about unrecorded-status cases is relaxed. Rows are "
           "the 15 pairs (higher group at δ = 0 first), sorted by δ*; a coloured bar means the ordering is guaranteed "
           "when every group's unrecorded-status hospitalization rate lies within δ points of its recorded rate; the "
           "black tick and the right-hand label give the exact δ*. Vertical lines mark the reported scenario values "
           "δ = 1, 2, 5 and 10 points. δ is a scenario parameter, not an estimate.")
    _save(fig, kit, fdir, "fig_02_delta_stability", man, cap, "results")


def fig_intervals(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    m, bt = r["main"], r["bootstrap"]
    order = m["order_by_delta0"]
    fig, ax = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    for i, g in enumerate(order):
        v = m["per_group"][g]
        b = bt["per_group"][g]
        ax.plot([100 * v["L_worst"], 100 * v["U_worst"]], [i, i], color="#B4C0E4", lw=9, solid_capstyle="butt")
        d5 = v["bounds_by_delta_pp"]["5"]
        ax.plot([100 * d5["L"], 100 * d5["U"]], [i, i], color="#0F4D92", lw=9, solid_capstyle="butt")
        ax.plot([100 * v["theta_delta0"]], [i], marker="o", ms=4.5, mfc="white", mec="#272727", mew=0.9, ls="none")
        for lo, hi in ((b["L_p2_5"], b["L_p97_5"]), (b["U_p2_5"], b["U_p97_5"])):
            ax.plot([100 * lo, 100 * hi], [i + 0.3, i + 0.3], color="#4D4D4D", lw=0.9)
            ax.plot([100 * lo, 100 * lo], [i + 0.24, i + 0.36], color="#4D4D4D", lw=0.9)
            ax.plot([100 * hi, 100 * hi], [i + 0.24, i + 0.36], color="#4D4D4D", lw=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[g] for g in order])
    ax.set_ylim(len(order) - 0.45, -0.55)
    xmax = 100 * max(max(bt["per_group"][g]["U_p97_5"], m["per_group"][g]["U_worst"]) for g in order)
    ax.set_xlim(0, 5 * np.ceil(xmax / 5 + 0.2))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Standardized hospitalization proportion among reported outbreak cases (%)")
    handles = [Line2D([], [], color="#B4C0E4", lw=7, label="Worst-case interval (no assumption)"),
               Line2D([], [], color="#0F4D92", lw=7, label="Interval with δ = 5 points"),
               Line2D([], [], marker="o", mfc="white", mec="#272727", ls="none", label="δ = 0 value"),
               Line2D([], [], color="#4D4D4D", lw=0.9,
                      label=f"Bootstrap 2.5th–97.5th percentile of each bound (B = {bt['B']:,})")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    cap = ("Standardized bounds per pathogen group. Light bars are worst-case identification intervals (all "
           "unrecorded-status cases not hospitalized / all hospitalized), dark bars the intervals when the "
           "unrecorded-status rate is within 5 points of the recorded rate, circles the δ = 0 values. Thin lines show "
           "outbreak-level bootstrap percentile ranges of each worst-case bound, describing sampling variability; "
           "neither the intervals nor these ranges are confidence intervals.")
    _save(fig, kit, fdir, "fig_03_bound_intervals", man, cap, "results")


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
    ax.set_xlim(0, 100)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[g] for g in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("(a) Eligible cases inside the common support (%)")
    ax = axes[1]
    cols = ("#0F4D92", "#AADCA9", "#CFCECE")
    for i, g in enumerate(order):
        v = m["per_group"][g]
        h = v["hospitalized_known"] / v["cases"]
        u = v["unknown_share"]
        nh = 1 - h - u
        left = 0
        for val, c in zip((h, nh, u), cols):
            ax.barh(i, 100 * val, left=left, height=0.6, color=c)
            left += 100 * val
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("(b) Retained cases by recorded status (%)")
    handles = [Patch(facecolor=c, label=t) for c, t in zip(cols, ("Recorded hospitalized", "Recorded not hospitalized",
                                                                    "Hospitalization status unrecorded"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    tot = r["retention"]["total"]
    cap = (f"Common-support retention and outcome recording. (a) Share of each group's eligible cases in the "
           f"{m['n_cells']} setting × period cells covered by all six groups (overall {tot['support_outbreaks']:,} of "
           f"{tot['eligible_outbreaks']:,} outbreaks, {int(tot['support_cases']):,} of {int(tot['eligible_cases']):,} "
           f"cases). (b) Retained cases split into recorded hospitalized, recorded not hospitalized and unrecorded "
           f"status; the unrecorded share sets the width of the worst-case interval.")
    _save(fig, kit, fdir, "fig_04_retention_recording", man, cap, "results")


def fig_2d(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    m = r["main"]
    sp = sorted_pairs(m)
    fig, axes = plt.subplots(3, 5, figsize=(7.0, 4.9), constrained_layout=True, sharex=True, sharey=True)
    for ax, (k, p) in zip(axes.flat, sp):
        td = m["two_d"][k]
        x = np.array(td["d_hi_pp"])
        y = np.array([20.0 if v is None else min(v, 20.0) for v in td["d_lo_boundary_pp"]])
        ax.fill_between(x, 0, y, color=CAT_COLORS[category(p)], lw=0)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
        ax.grid(alpha=0.3)
        ax.set_title(pair_label(p, SHORT), fontsize=8)
        ax.tick_params(labelsize=7)
    fig.supxlabel("δ allowed for the higher group (points)", fontsize=9)
    fig.supylabel("δ allowed for the lower group (points)", fontsize=9)
    cap = ("Supplementary: two-dimensional sensitivity. For each pair (higher group at δ = 0 first), the shaded region "
           "contains the combinations of separate deviations for the two groups (0–20 points each) under which the "
           "ordering stays guaranteed; a fully shaded panel means it holds with no assumption. SAL Salmonella; CAMP "
           "Campylobacter; CPERF C. perfringens; NoV GI/GII norovirus genogroup I/II.")
    _save(fig, kit, fdir, "fig_s1_delta_2d", man, cap, "supplementary")


def fig_robust(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    m = r["main"]
    sp = sorted_pairs(m)
    variants = [("Main", m)] + [(VARIANT_CODES[k], v) for k, v in r["robustness"].items()
                                if k != "R7_norovirus_combined"]
    fig, ax = plt.subplots(figsize=(7.0, 4.9), constrained_layout=True)
    for j, (code, v) in enumerate(variants):
        for i, (k, p) in enumerate(sp):
            q = v["pairs"][k]
            cat = "D" if q["higher_at_delta0"] != p["higher_at_delta0"] else category(q)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CAT_COLORS[cat], edgecolor="white", lw=1.2))
    ax.set_xlim(-0.5, len(variants) - 0.5)
    ax.set_ylim(len(sp) - 0.5, -0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([f"{c}\n{v['n_determined']['worst_case']}|{v['n_determined']['5']}" for c, v in variants],
                       fontsize=7)
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels([pair_label(p) for _, p in sp])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Analysis (second line: determined pairs, worst case | δ = 5 points)")
    handles = [Patch(facecolor=CAT_COLORS[c], label=t) for c, t in
               (("A", "Holds with no assumption"), ("B", "Needs assumption, δ* ≥ 5 points"),
                ("C", "Needs assumption, δ* < 5 points"), ("D", "δ = 0 direction reversed vs main"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    cap = ("Supplementary: robustness of each pairwise verdict. Columns: main analysis and pre-specified variants "
           "(R1a crude, all eligible outbreaks; R1b crude within the common support; R2 period-only standardization; "
           "R3a single-etiology cohort; R3b complete records only; R3c pilot cohort rule; R4a/R4b support threshold 2/5 "
           "outbreaks; R5 named settings only; R6a/R6b/R6c group-balanced, outbreak-mix and uniform weights). The "
           "norovirus GI+GII combined analysis (R7) has different pairs and is reported in Table 3.")
    _save(fig, kit, fdir, "fig_s2_robustness", man, cap, "supplementary")


def fig_boot(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    m, bt = r["main"], r["bootstrap"]
    sp = sorted_pairs(m)
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    for i, (k, p) in enumerate(sp):
        b = bt["pairs"][k]
        ax.barh(i - 0.18, 100 * b["share_worst_main_verdict"], height=0.34, color="#0F4D92")
        ax.barh(i + 0.18, 100 * b["share_delta5_main_verdict"], height=0.34, color="#8FA8D6")
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels([f"{pair_label(p)} ({'determined' if p['worst_case'] != 'undetermined' else 'undetermined'})"
                        for _, p in sp])
    ax.set_ylim(len(sp) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Bootstrap replicates reproducing the main-analysis verdict (%)")
    handles = [Patch(facecolor="#0F4D92", label="Worst-case verdict"), Patch(facecolor="#8FA8D6", label="Verdict at δ = 5 points")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    cap = (f"Supplementary: sampling variability of the pairwise verdicts. Share of {bt['B']:,} outbreak-level bootstrap "
           f"replicates (seed {bt['seed']}) that reproduce the main-analysis verdict (in parentheses, the worst-case "
           f"verdict) under worst-case bounds and at δ = 5 points.")
    _save(fig, kit, fdir, "fig_s3_bootstrap", man, cap, "supplementary")


def fig_inference_voi(r, kit, fdir, man):
    if "stacked_inference" not in r or "voi_masking" not in r:
        return
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    si = r["stacked_inference"]
    vm = r["voi_masking"]
    m = r["main"]
    sp = sorted_pairs(m)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel (a): Point delta* vs Simultaneous 95% Confidence delta*_0.95
    ax = axes[0]
    y = np.arange(len(sp))
    for i, (k, p) in enumerate(sp):
        vp = si["pairs"][k]
        pt = 100.0 if vp["delta_star_pt_pp"] is None else vp["delta_star_pt_pp"]
        sim = 100.0 if vp["delta_star_95_sim_pp"] is None else vp["delta_star_95_sim_pp"]
        ax.plot([sim, pt], [i, i], color="#8E9AAE", lw=1.6, zorder=1)
        ax.scatter([pt], [i], color="#0F4D92", s=26, zorder=2,
                   marker=("^" if vp["delta_star_pt_pp"] is None else "o"))
        ax.scatter([sim], [i], color="#C44E52", s=26, zorder=3,
                   marker=("^" if vp["delta_star_95_sim_pp"] is None else "D"))
    ax.axvline(5.0, color="#666666", ls=":", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([pair_label(p, SHORT) for _, p in sp], fontsize=7)
    ax.set_ylim(len(sp) - 0.5, -0.5)
    ax.set_xlim(-3, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100 (∞)"], fontsize=7.5)
    ax.grid(axis="x", alpha=0.35, lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Tipping point δ* (percentage points)", fontsize=8)
    ax.set_title("(a) Sample $\\delta^{*}$ vs. 15-pair FWER 95% $\\delta^{*}_{0.95}$", fontsize=8.5, pad=6)
    h_a = [
        Line2D([], [], color="#0F4D92", marker="o", ms=4.5, ls="none", label="Sample $\\delta^{*}$"),
        Line2D([], [], color="#C44E52", marker="D", ms=4.2, ls="none", label="FWER 95% $\\delta^{*}_{0.95}$"),
        Line2D([], [], color="#444444", marker="^", ms=4.8, ls="none", label="Holds at δ = 100 (∞)"),
    ]
    ax.legend(handles=h_a, loc="lower right", frameon=True, facecolor="white", framealpha=0.85, fontsize=6.8)

    # Panel (b): Retrospective masking & VoI follow-up curve
    ax = axes[1]
    budgets = vm["budgets_cases"]
    styles = [
        ("voi_disambiguation", "VoI disambiguation (proposed)", "#0F4D92", "-", "o", 2.0),
        ("width_greedy", "Width reduction greedy", "#2B8CBE", "--", "s", 1.5),
        ("random", "Random follow-up", "#767676", "-.", "^", 1.4),
        ("volume_greedy", "Largest outbreaks first", "#C44E52", ":", "v", 1.6),
    ]
    for pk, lab, col, ls, mk, lw in styles:
        vals = [vm["summary"][pk][str(b)]["mean_worst_det"] for b in budgets]
        ax.plot(budgets, vals, color=col, ls=ls, marker=mk, ms=4.0, lw=lw, label=lab)
    ax.set_xlim(-100, 5150)
    ax.set_ylim(0, 15.3)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000", "4,000", "5,000"], fontsize=7.5)
    ax.set_yticks([0, 3, 6, 9, 12, 15])
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(alpha=0.35, lw=0.5)
    ax.set_xlabel("Follow-up budget (masked cases resolved)", fontsize=8)
    ax.set_ylabel("Worst-case determined pairs (of 15)", fontsize=8)
    ax.set_title("(b) Retrospective masking follow-up gain", fontsize=8.5, pad=6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.85, fontsize=6.8)

    cap = (f"Methodological extension: simultaneous 95% confidence tipping points and Value-of-Information follow-up. "
           f"(a) Sample tipping point δ* compared with the 15-pair FWER-adjusted 95% confidence tipping point δ*_95 "
           f"(c = {si['c_sim_delta5']:.3f}; triangles at 100 denote δ* = ∞). (b) Mean number of worst-case "
           f"determined pairs across {vm['n_trials']} retrospective masking trials on {vm['n_complete_outbreaks']:,} "
           f"complete-status outbreaks under four follow-up allocation strategies.")
    _save(fig, kit, fdir, "fig_04_inference_voi", man, cap, "results")


def make_figures(r: dict, fdir: Path) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    for f in list(fdir.glob("fig_*.pdf")) + list(fdir.glob("fig_*.png")):
        f.unlink()
    kit = _style()
    man = []
    for fn in (fig_matrix, fig_delta, fig_intervals, fig_retention, fig_2d, fig_robust, fig_boot, fig_inference_voi):
        fn(r, kit, fdir, man)
    (fdir / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")


def make_all(exp: Path, work: Path) -> None:
    r = load(exp)
    make_tables(r, work / "tables")
    make_figures(r, work / "figures")
    import make_report
    make_report.write_report(r, exp)
    make_report.write_pilot_comparison(r, exp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=HERE.parent)
    ap.add_argument("--work-dir", type=Path, default=HERE.parents[3] / ".work" / SLUG / "experiment")
    a = ap.parse_args()
    statistical_summary(a.exp_dir.resolve())
    make_all(a.exp_dir.resolve(), a.work_dir.resolve())
