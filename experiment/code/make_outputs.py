"""Tables, figures, and statistical summary for the NORS hospitalization-ranking analysis (v6.0).

Reads ONLY analysis/results.json so every table, figure, and text value shares one canonical source.
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
SHORT = {"SAL": "SAL", "STEC": "STEC", "CAMP": "CAMP", "CPERF": "CPERF", "NOV_GI": "NoV GI", "NOV_GII": "NoV GII",
         "NOV": "NoV"}
NAME = {"SAL": "Salmonella", "STEC": "STEC", "CAMP": "Campylobacter", "CPERF": "C. perfringens",
        "NOV_GI": "Norovirus GI", "NOV_GII": "Norovirus GII", "NOV": "Norovirus"}
VARIANT_CODES = {"R1a_crude_all": "R1a", "R1b_crude_support": "R1b", "R2_period_only": "R2",
                 "R3a_single_etiology": "R3a", "R3b_complete_only": "R3b", "R3c_pilot_rule": "R3c",
                 "R4a_threshold_2": "R4a", "R4b_threshold_5": "R4b", "R5_named_settings": "R5",
                 "R6a_w_group_balanced": "R6a", "R6b_w_pooled_outbreaks": "R6b", "R6c_w_uniform": "R6c",
                 "R7_norovirus_combined": "R7", "R7b_norovirus_all_confirmed": "R7b",
                 "R8_include_suspected": "R8", "R9_excl_healthcare": "R9",
                 "R10_outbreak_weighted": "R10", "R11_single_state_only": "R11",
                 "R12_stec_o157_only": "R12", "R13_outbreak_level_mar": "R13",
                 "R14a_Ci_extrap25": "R14a", "R14b_low_unrec_50": "R14b"}
CAT_COLORS = {"A": "#0F4D92", "B": "#8FA8D6", "C": "#E9A6A1", "D": "#E8B04B"}
CAT_TEXT = {"A": "white", "B": "#272727", "C": "#272727"}
CAT_LABEL = {
    "A": "Worst-case determined (disjoint bounds without outcome assumptions)",
    "B": "Separation scenario: holds up to δ* ≥ 5 pp",
    "C": "Separation scenario: fails below δ = 5 pp (δ* < 5 pp)",
    "D": "Direction at δ = 0 reversed relative to main analysis",
}


def load(exp: Path) -> dict:
    return json.loads((exp / "analysis" / "results.json").read_text(encoding="utf-8"))


def pnum(x, d=1):
    return "NA" if x is None else f"{100 * x:.{d}f}"


def fmt_ds(p, unit=" pp"):
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
                 ("δ", r"$\delta$"), ("Γ", r"$\Gamma$"), ("α", r"$\alpha$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
                 ("–", "--"), ("−", "$-$"), (">", r"$>$"), ("†", r"$^{\dagger}$"),
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
        analysis_version=r["provenance"].get("analysis_version", "6.0"),
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
                       share_determined_worst_case={pair_label(m["pairs"][k]): v["share_holds_without_assumption"]
                                                    for k, v in r["bootstrap"]["pairs"].items()},
                       share_determined_delta5={pair_label(m["pairs"][k]): 1.0 - v["share_delta5_undetermined"]
                                                for k, v in r["bootstrap"]["pairs"].items()}),
        stacked_inference=dict(
            n_hypotheses=si["n_hypotheses"],
            n_fwer_worst_sig=si["n_fwer_worst_sig"],
            n_fwer_boot_t_worst_sig=si["n_fwer_boot_t_worst_sig"],
            n_fwer_boot_t_delta5_sig=si["n_fwer_boot_t_delta5_sig"],
            c_bt_worst=si["c_bt_worst"],
            c_bt_delta5=si["c_bt_delta5"],
            rank_breakdown_points_pp=si["rank_breakdown_points_pp"],
        ) if si else None,
        note="Canonical summary across sample bounds, 30-hypothesis Gaussian & studentized bootstrap-t inference, and 22 variants.",
    )
    (exp / "analysis" / "statistical_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")


def R(w):
    return rf">{{\raggedleft\arraybackslash}}p{{{w}cm}}"


def Lc(w):
    return rf">{{\raggedright\arraybackslash}}p{{{w}cm}}"


def write_table(tdir: Path, manifest: list, tid: str, base: str, df: pd.DataFrame, caption: str, colspec: str,
                notes: str = "", size: str = "scriptsize", sep: str = "3pt"):
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


def make_tables(r: dict, tdir: Path, write_tex: bool = True) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    for f in tdir.glob("table_*"):
        f.unlink()
    m, man = r["main"], []
    order = m["order_by_delta0"]
    bt = r["bootstrap"]
    emp = r.get("empirical_calibration", {})

    def _emit(tid, base, df, caption, colspec, notes="", size="scriptsize", sep="3pt"):
        if write_tex:
            write_table(tdir, man, tid, base, df, caption, colspec, notes=notes, size=size, sep=sep)
        else:
            (tdir / f"{base}.csv").write_text(df.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="\n")

    # Table 1: per-group bounds
    rows = []
    for g in order:
        v = m["per_group"][g]; b = bt["per_group"][g]
        rows.append({"Group": NAME[g], "Outbreaks": f"{v['outbreaks']:,}", "Cases": f"{int(v['cases']):,}",
                     "Unrecorded status (%)": pnum(v["unknown_share"]),
                     "Worst-case interval (%)": f"{pnum(v['L_worst'])}–{pnum(v['U_worst'])}",
                     "δ = 5 pp interval (%)": f"{pnum(v['bounds_by_delta_pp']['5']['L'])}–{pnum(v['bounds_by_delta_pp']['5']['U'])}",
                     "δ = 0 value (%)": pnum(v["theta_delta0"]),
                     "Bootstrap range of lower bound (%)": f"{pnum(b['L_p2_5'])}–{pnum(b['L_p97_5'])}",
                     "Bootstrap range of upper bound (%)": f"{pnum(b['U_p2_5'])}–{pnum(b['U_p97_5'])}"})
    _emit("tab:group_bounds", "table_1_group_bounds", pd.DataFrame(rows),
          f"Standardized hospitalization proportion among reported outbreak cases across six pathogen groups "
          f"({m['n_cells']} common-support setting × period cells, pooled-case weights).",
          "".join([Lc(2.0), R(1.4), R(1.1), R(1.4), R(1.6), R(1.5), R(1.2), R(1.8), R(1.8)]),
          notes="STEC Shiga toxin-producing E. coli. Unrecorded status includes C − H cases from 52 blank-K records. "
                "δ = 0 value uses within-cell recorded rates; δ = 5 pp is an illustrative symmetric scenario. "
                f"Bootstrap ranges give 2.5th–97.5th percentiles over {bt['B']:,} outbreak-level resamples.")

    # Table 2: pairwise breakdown points & stress tests (descriptive stability, no CM column)
    or_gs = emp.get("odds_ratio", {}).get("gamma_star_pairs", {})
    or_fl_gs = emp.get("odds_ratio_floor_001", {}).get("gamma_star_pairs", {})
    asym_15 = emp.get("asymmetric", {}).get("asym_15_5", {}).get("verdicts", {})
    rows = []
    for k, p in sorted_pairs(m):
        b = bt["pairs"][k]
        vd = p["verdict_by_delta_pp"]
        gs_val = or_gs.get(k)
        if gs_val is None:
            gs_str = "∞" if p["delta_star_pp"] is None else "∞†"
        else:
            gs_str = f"{gs_val:.2f}" if gs_val < 10 else f"{gs_val:.1f}"
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT),
                     "Worst case": "determined" if p["worst_case"] != "undetermined" else "undetermined",
                     "δ* (pp)": "∞" if p["delta_star_pp"] is None else (f"{p['delta_star_pp']:.2f}"
                                                                        if p["delta_star_pp"] < 1 else f"{p['delta_star_pp']:.1f}"),
                     "OR Γ*": gs_str,
                     "δ = 5 pp": ("yes" if vd["5"] != "undetermined" else "no"),
                     "δ = 10 pp": ("yes" if vd["10"] != "undetermined" else "no"),
                     "Asym (−15/+5)": ("yes" if asym_15.get(k, "undetermined") != "undetermined" else "no"),
                     "Boot stab., worst (%)": pnum(b["share_holds_without_assumption"]),
                     "Boot stab., δ = 5 (%)": pnum(1.0 - b["share_delta5_undetermined"])})
    sal_cp_fl = or_fl_gs.get("SAL|CPERF")
    _emit("tab:pairwise", "table_2_pairwise", pd.DataFrame(rows),
          "Pairwise sample breakdown points (δ*, Γ*), asymmetric (−15/+5 pp) checks, and descriptive bootstrap stability.",
          "".join([Lc(2.6), Lc(1.7), R(1.0), R(1.0), R(0.95), R(1.0), R(1.3), R(1.55), R(1.55)]),
          notes=f"SAL Salmonella; STEC Shiga toxin-producing E. coli; CAMP Campylobacter; CPERF C. perfringens; "
                f"NoV GI/GII norovirus genogroup I/II. Boot stab. columns are descriptive bootstrap stability frequencies "
                f"of disjoint sample bounds across {bt['B']:,} resamples, not confidence probabilities. "
                f"†OR Γ* = ∞† for SAL > CPERF reflects 9/72 cells with p_js = 0; a 0.1% continuity floor yields Γ* = {sal_cp_fl:.1f}.",
          sep="2.3pt")

    # Table 3: robustness (22 variants)
    rows = [{"Analysis": "Main", "Description": "Setting × period, ≥ 3 outbreaks, pooled-case weights",
             "Cells": m["n_cells"], "Pairs": m["n_pairs"], "Determined, worst case": m["n_determined"]["worst_case"],
             "Determined, δ = 5 pp": m["n_determined"]["5"], "Worst-case changed": 0,
             "δ = 0 reversed": 0}]
    for vid, v in r["robustness"].items():
        desc = v["label"].replace("Crude: no standardization", "Crude, no standardization").replace(
            "Cohort: ", "Cohort of ").replace("Weights: ", "Weights from ")
        if vid == "R3c_pilot_rule":
            desc = (f"Cohort of species-level strings; records with one blank outcome field "
                    f"dropped ({v['n_outbreaks_in_cohort']:,} outbreaks)")
        rows.append({"Analysis": VARIANT_CODES[vid], "Description": desc, "Cells": v["n_cells"],
                     "Pairs": v["n_pairs"], "Determined, worst case": v["n_determined"]["worst_case"],
                     "Determined, δ = 5 pp": v["n_determined"]["5"],
                     "Worst-case changed": v.get("n_worst_case_verdicts_changed", "NA"),
                     "δ = 0 reversed": v.get("n_delta0_directions_changed", "NA")})
    _emit("tab:robustness", "table_3_robustness", pd.DataFrame(rows),
          "Sensitivity analyses across all 22 cohort, support, weighting, serogroup, outbreak-MAR, and denominator variants.",
          "".join([Lc(1.0), Lc(5.25), R(0.85), R(0.85), R(1.5), R(1.45), R(1.45), R(1.45)]),
          notes="R7 and R7b combine norovirus into 5 groups (10 pairs), so 6-group changes are NA. "
                "R12 restricts STEC to confirmed O157; R13 uses outbreak-level H_i/K_i MAR anchors; "
                "R14a–R14b evaluate C_i contamination/extrapolation sensitivity.")

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
                     "Crude δ = 0 before/after (%)": f"{pnum(v['crude_before']['theta_delta0'])}/{pnum(v['crude_after']['theta_delta0'])}",
                     "Crude worst before/after (%)": f"{pnum(v['crude_before']['L'])}–{pnum(v['crude_before']['U'])} / "
                                                     f"{pnum(v['crude_after']['L'])}–{pnum(v['crude_after']['U'])}"})
    _emit("tab:retention", "table_4_retention", pd.DataFrame(rows),
          "Common-support retention by pathogen group, dropped strata, and crude proportions before/after restriction.",
          "".join([Lc(1.85), R(1.25), R(0.95), R(1.35), R(1.35), Lc(1.95), Lc(1.6), R(1.2), R(1.75)]), sep="2.3pt")

    # Table 5: record audit
    a = r["audit"]["record_audit_summary"]
    handling = {"complete_consistent": "as recorded (K = 0 means all status unrecorded)",
                "single_missing_K": "K set to H; C − H cases unrecorded (exact worst case; upper δ conservative)",
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
        c = a["categories"][cat]; byg = a["by_category_x_group"][cat]
        rows.append({"Audit category": AUDIT_LABEL[cat], "Outbreaks": f"{c['n_outbreaks']:,}",
                     "Cases": f"{int(c['cases']):,}",
                     **{SHORT[g]: ("–" if cat == "ambiguous_etiology" else f"{byg[g]:,}")
                        for g in ("SAL", "STEC", "CAMP", "CPERF", "NOV_GI", "NOV_GII")},
                     "Handling": handling[cat]})
    _emit("tab:audit", "table_5_audit", pd.DataFrame(rows),
          "Record audit of outbreak outcome fields (C reported cases, H hospitalized, K known status). Zero count contradictions.",
          "".join([Lc(2.1), R(1.35), R(1.1)] + [R(0.88)] * 6 + [Lc(3.6)]), sep="2.3pt")

    # Table 6: bootstrap (2 decimals to eliminate 100.0 + 0.1 = 100.1 rounding)
    rows = []
    for k, p in sorted_pairs(m):
        b = bt["pairs"][k]
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT),
                     "Same δ = 0 direction (%)": pnum(b["share_same_delta0_direction"], d=2),
                     "Worst case, higher > lower (%)": pnum(b["share_worst_A"] if p["higher_at_delta0"] == p["A"]
                                                             else b["share_worst_B"], d=2),
                     "Worst case, undetermined (%)": pnum(b["share_worst_undetermined"], d=2),
                     "Determined at δ = 5 pp (%)": pnum(1 - b["share_delta5_undetermined"], d=2),
                     "δ* 2.5th–97.5th percentile (pp)":
                         f"{'∞' if b['delta_star_pp_p2_5'] is None else format(b['delta_star_pp_p2_5'], '.1f')}–"
                         f"{'∞' if b['delta_star_pp_p97_5'] is None else format(b['delta_star_pp_p97_5'], '.1f')}"})
    _emit("tab:bootstrap", "table_6_bootstrap", pd.DataFrame(rows),
          f"Outbreak-level bootstrap ({bt['B']:,} replicates, seed {bt['seed']}; 0/{bt['B']:,} resamples lacked K_rec), "
          f"reporting two-decimal verdict frequencies and δ* percentile intervals.",
          "".join([Lc(2.6), R(2.0), R(2.2), R(2.0), R(2.0), R(2.6)]))

    # Table 7: width decomposition
    rows = []
    wd = r["width_decomposition"]["undetermined_pairs"]
    for k, p in sorted_pairs(m):
        if k not in wd:
            continue
        tops = "; ".join(f"{SHORT[t['group']]} {t['cell']} ({100 * t['contribution']:.1f})" for t in wd[k]["top_group_cells"])
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT), "Worst-case overlap (pp)": pnum(wd[k]["overlap"]),
                     "Largest contributions to interval width (group, cell, pp)": tops})
    _emit("tab:width", "table_7_width_decomposition", pd.DataFrame(rows),
          "Undetermined pairs under worst-case bounds and top unrecorded-status width contributors.",
          "".join([Lc(2.6), R(1.6), Lc(10.6)]))

    # Table 8: Joint 30-hypothesis inference (Main Table 3: Pointwise IM + HC1/State + Studentized Boot-t)
    if "stacked_inference" in r:
        si = r["stacked_inference"]
        sc_cal = r.get("small_sample_calibration", {})
        rows = []
        for k, p in sorted_pairs(m):
            sp = si["pairs"][k]
            fmt_ci = lambda a, b: f"[{100 * a:+.1f}, {100 * b:+.1f}]"
            fmt_ds = lambda v: "∞" if v is None else (f"{v:.2f}" if v < 1 else f"{v:.1f}")
            fmt_p = lambda pv: "<0.001" if (pv is not None and pv < 0.001) else ("NA" if pv is None else f"{pv:.3f}")
            rows.append({
                "Ordering (δ = 0)": pair_label(p, SHORT),
                "Pointwise IM 95% CI (pp)†": fmt_ci(sp["ci95_im_lo"], sp["ci95_im_hi"]),
                "HC1 / State 30-test p (w)": f"{fmt_p(sp['p_fwer_worst'])} / {fmt_p(sp['p_fwer_state_worst'])}",
                "Boot-t p (w / δ=5)": f"{fmt_p(sp['p_fwer_boot_t_worst'])} / {fmt_p(sp['p_fwer_boot_t_delta5'])}",
                "Sample δ* (pp)": fmt_ds(sp["delta_star_pt_pp"]),
                "95% δ* (HC1 / Boot-t, pp)": f"{fmt_ds(sp['delta_star_95_sim_pp'])} / {fmt_ds(sp['delta_star_95_bt_pp'])}",
            })
        rcs_w = si["rank_confidence_sets"]["worst_hc1"]
        rb_hc = si["rank_breakdown_points_pp"]["hc1"]
        rb_bt = si["rank_breakdown_points_pp"]["bootstrap_t"]
        rb_btu = si["rank_breakdown_points_pp"].get("bootstrap_t_unif", {})
        im_min = 100.0 * sc_cal.get("im95_min_endpoint_cov_hc1", 0.76)
        im_med = 100.0 * sc_cal.get("im95_median_endpoint_cov_hc1", 0.87)
        _emit("tab:joint_inference", "table_8_joint_inference", pd.DataFrame(rows),
              "Joint 30-hypothesis one-sided (15 pairs × 2 directions) FWER inference: pointwise Imbens–Manski 95% CIs, "
              "outbreak-HC1 and 53-cluster exposure-state Gaussian Max-T p-values, and studentized bootstrap-t p-values "
              "and δ-specific breakdown points.",
              "".join([Lc(2.4), R(2.2), R(2.35), R(2.45), R(1.15), R(2.7)]),
              notes=f"SAL Salmonella; STEC Shiga toxin-producing E. coli; CAMP Campylobacter; CPERF C. perfringens; "
                    f"NoV GI/GII norovirus GI/GII. †Pointwise IM 95% CIs under-cover in sparse strata ({im_min:.1f}% min to "
                    f"{im_med:.1f}% median endpoint coverage under HC1; Supp. Table S8). "
                    f"Critical values (worst / δ = 5 pp): HC1 {si['c_sim_worst']:.2f}/{si['c_sim_delta5']:.2f}, "
                    f"53-cluster {si['c_sim_state_worst']:.2f}/{si['c_sim_state_delta5']:.2f}, studentized bootstrap-t "
                    f"{si['c_bt_worst']:.2f}/{si['c_bt_delta5']:.2f} (c_bt(δ) uses δ-specific quantiles for 95% δ*). "
                    f"Worst-case 95% rank sets: HC1 STEC [{rcs_w['STEC']['rank_lower']},{rcs_w['STEC']['rank_upper']}], "
                    f"NoV GII [{rcs_w['NOV_GII']['rank_lower']},{rcs_w['NOV_GII']['rank_upper']}], others [1,6]; studentized bootstrap-t all [1,6]. "
                    f"At δ = 5 pp (and −15/+5 pp), rank sets are STEC [1,1], SAL [2,2], CAMP [3,3], lower [4,6], "
                    f"holding jointly up to {rb_bt['joint_upper_tier_min']:.1f} pp ({rb_bt['CAMP_3_3']:.1f} pp CAMP alone) under δ-specific boot-t, "
                    f"{rb_btu.get('joint_upper_tier_min', 7.9):.1f} pp under conservative worst-case c_bt = {si['c_bt_worst']:.2f}, and "
                    f"{rb_hc['joint_upper_tier_min']:.1f} pp under HC1.",
              sep="2.1pt")

    # Table 9: Cluster concentration, Multistate shares & cluster score variance share
    if "cluster_diagnostics" in r and "stacked_inference" in r:
        cd = r["cluster_diagnostics"]["per_group"]
        sg = r["stacked_inference"]["per_group"]
        cvs = r["cluster_diagnostics"].get("cluster_variance_share") or r["cluster_diagnostics"].get("mnw_cluster_leverage", {})
        ms = r["cluster_diagnostics"].get("multistate_sensitivity", {})
        cwi = r["stacked_inference"].get("coarser_and_weighting_inference", {})
        loo = r["stacked_inference"].get("loo_influence", {})
        rows = []
        for g in order:
            c_g = cd[g]; s_g = sg[g]; b_g = bt["per_group"][g]
            rows.append({
                "Group": NAME[g],
                "Min/Med/Max n": f"{c_g['min_n']}/{c_g['median_n']:.0f}/{c_g['max_n']}",
                "Max HHI (min N_eff)": f"{c_g['max_hhi']:.3f} ({c_g['min_neff']:.2f})",
                "Multistate unrec (%)": f"{100 * c_g.get('multistate_unrecorded_share', 0.0):.1f}",
                "SE(L) HC0/HC1/Boot/St (pp)": (f"{100*s_g['se_L_hc0']:.2f}/{100*s_g['se_L_hc1']:.2f}/"
                                               f"{100*b_g['L_sd']:.2f}/{100*s_g['se_L_state']:.2f}"),
                "SE(U) HC0/HC1/Boot/St (pp)": (f"{100*s_g['se_U_hc0']:.2f}/{100*s_g['se_U_hc1']:.2f}/"
                                               f"{100*b_g['U_sd']:.2f}/{100*s_g['se_U_state']:.2f}"),
            })
        cw_w = cvs.get("worst", {}); cw_5 = cvs.get("delta5", {})
        sh_w = cw_w.get("max_variance_share", cw_w.get("max_leverage_share", 0.0))
        sh_5 = cw_5.get("max_variance_share", cw_5.get("max_leverage_share", 0.0))
        r2 = cwi.get("R2_period_only", {}); r10 = cwi.get("R10_outbreak_weighted", {})
        llo = loo.get("leave_largest_per_group", {})
        _emit("tab:cluster_diag", "table_9_cluster_diagnostics", pd.DataFrame(rows),
              "Outbreak-cluster concentration, Multistate unrecorded case share, cluster share of score variance, "
              "coarser/outbreak-weighted inference, and leave-one-outbreak-out influence.",
              "".join([Lc(1.85), R(1.75), R(2.05), R(1.55), R(3.4), R(3.4)]),
              notes=f"Cluster share of score variance across 53 exposure-state categories (Carter et al. 2017; MacKinnon et al. 2023): "
                    f"Multistate share = {100*sh_w:.1f}% (worst case) and {100*sh_5:.1f}% (δ = 5 pp), G_eff = "
                    f"{cw_w.get('effective_clusters_g_eff',0):.2f} / {cw_5.get('effective_clusters_g_eff',0):.2f}. "
                    f"Splitting 219 Multistate outbreaks as singletons (271 clusters) or single-state R11 (52 states) gives "
                    f"{ms.get('n_sig_singleton_worst',0)}/{ms.get('n_sig_r11_state_worst',0)} worst-case and "
                    f"{ms.get('n_sig_singleton_delta5',12)}/{ms.get('n_sig_r11_state_delta5',12)} δ = 5 significant hypotheses. "
                    f"Coarser R2 (3 periods, min cell n = {r2.get('min_cell_outbreaks',24)}) gives "
                    f"{r2.get('n_sig_worst',{}).get('hc1',2)} HC1 and {r2.get('n_sig_worst',{}).get('boot_t',0)} boot-t worst-case rejections "
                    f"(δ = 5: 12 boot-t, upper tier [1,1]/[2,2]/[3,3]); equal-outbreak R10 gives "
                    f"{r10.get('n_sig_worst',{}).get('boot_t',5)} boot-t worst-case and 12 δ = 5 rejections (identical under R4a/R4b). "
                    f"Across all {loo.get('n_outbreaks_checked',3176):,} single-outbreak deletions (and simultaneous 6-group LLO removal), "
                    f"worst-case determined pairs stay {loo.get('wc_determined_range',[4,4])[0]}–{loo.get('wc_determined_range',[4,4])[1]} "
                    f"({llo.get('wc_determined',4)} LLO) and δ = 5 stays {loo.get('delta5_determined_range',[12,13])[0]}–{loo.get('delta5_determined_range',[12,13])[1]} "
                    f"({llo.get('delta5_determined',12)} LLO).",
              sep="2.1pt")

    # Table 10: Cost-conditional follow-up benchmark (ALL mixed budgets 500..5000 included)
    if "followup_masking" in r:
        vm = r["followup_masking"]
        rows = []
        case_pols = [("random", "Random"), ("volume_greedy", "Largest first"),
                     ("width_greedy", "Per-case w_s/C_js"), ("dpi", "DPI (heuristic)")]
        for b in (250, 500, 1000, 2000, 5000):
            for pk, pname in case_pols:
                s = vm["summary"][pk][str(b)]
                rows.append({"Cost model / Budget": f"Per-case: {b:,} cases", "Policy": pname,
                             "Spent": f"{s['mean_cases_spent']:.0f} cases",
                             "Worst det (/15)": f"{s['mean_worst_det']:.2f} ({s['sd_worst_det']:.2f})",
                             "95% MC CI": f"[{s['ci95_worst_det'][0]:.2f}, {s['ci95_worst_det'][1]:.2f}]"})
        ob_pols = [("random", "Random"), ("volume_greedy", "Largest first"),
                   ("width_greedy", "Per-case w_s/C_js"), ("outbreak_width", "Per-outbreak w_s U_i/C_js")]
        for b in (25, 50, 100, 200):
            for pk, pname in ob_pols:
                s = vm["outbreak_budget_summary"][pk][str(b)]
                rows.append({"Cost model / Budget": f"Per-outbreak: {b} ob", "Policy": pname,
                             "Spent": f"{s['mean_spent_cases']:.0f} cases",
                             "Worst det (/15)": f"{s['mean_worst_det']:.2f} ({s['sd_worst_det']:.2f})",
                             "95% MC CI": f"[{s['ci95_worst_det'][0]:.2f}, {s['ci95_worst_det'][1]:.2f}]"})
        mix_pols = [("random", "Random"), ("volume_greedy", "Largest first"), ("width_greedy", "Per-case w_s/C_js"),
                    ("outbreak_width", "Per-outbreak w_s U_i/C_js"), ("mixed_leverage", "Mixed (w_s U_i/C_js)/(20+U_i)")]
        for b in (500, 1000, 2000, 5000):
            for pk, pname in mix_pols:
                s = vm["mixed_cost_summary"][pk][str(b)]
                rows.append({"Cost model / Budget": f"Mixed (20+U_i): {b:,}", "Policy": pname,
                             "Spent": f"{s['mean_spent']:.0f} eq",
                             "Worst det (/15)": f"{s['mean_worst_det']:.2f} ({s['sd_worst_det']:.2f})",
                             "95% MC CI": f"[{s['ci95_worst_det'][0]:.2f}, {s['ci95_worst_det'][1]:.2f}]"})
        _emit("tab:voi_masking", "table_10_followup_masking", pd.DataFrame(rows),
              f"Cost-conditional follow-up benchmark across {vm['n_trials']} MCAR masking trials on all "
              f"{vm['n_complete_outbreaks']:,} fully recorded outbreaks (baseline pre-follow-up mean = 1.55 pairs).",
              "".join([Lc(3.35), Lc(4.25), R(1.85), R(2.2), R(2.45)]),
              notes="All rules use budget-filling selection. Note that at 5,000 mixed-cost units, per-case w_s/C_js (9.72) "
                    "exceeds the mixed greedy ratio (9.22) on the discrete determined-pair count.", sep="2.2pt")

    # Table 11: Excluded outbreaks
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
            rows.append({"Excluded category": nm, "Outbreaks": f"{d['n_outbreaks']:,}", "Cases": f"{int(d['total_cases']):,}",
                         "Median [IQR] cases": f"{d['median_cases']:.0f} [{d['iqr_cases'][0]:.0f}, {d['iqr_cases'][1]:.0f}]",
                         "Top settings": top_s, "Rec. hosp (%)": pnum(d["recorded_hosp_rate"]),
                         "Unrecorded (%)": pnum(d["unrecorded_status_share"])})
        _emit("tab:excluded", "table_11_excluded_outbreaks", pd.DataFrame(rows),
              "Characteristics of excluded foodborne outbreaks (2009–2023) not retained in the primary six-group cohort.",
              "".join([Lc(3.8), R(1.25), R(1.1), R(2.1), Lc(3.15), R(1.35), R(1.55)]), sep="2.3pt")

    # Table 12: Composition-adjusted Partial vs Full Comparison + Multistate Δ + 11-state Wilson CIs
    if emp:
        pg = emp["per_group"]; hc = emp["high_completeness_states"]["per_group"]
        rows = []
        for g in order:
            v = pg[g]; hv = hc[g]
            ciw = hv.get("ci95_wilson")
            ci_str = f"{pnum(hv['recorded_hosp_rate'])} [{pnum(ciw[0])},{pnum(ciw[1])}]" if ciw else "NA"
            m_str = "–" if v.get("multistate_diff_pp") is None else f"{v['multistate_diff_pp']:+.2f}"
            rows.append({
                "Group": NAME[g],
                "Full vs Part (%)": f"{pnum(v['rate_full_pooled'])} vs {pnum(v['rate_partial_known_pooled'])}",
                "Within-cell Δ (pp)": f"{v['within_cell_diff_pp']:+.2f}",
                "Std obs − Full (pp)": f"{v['std_obs_minus_full_pp']:+.2f}",
                "Single-state Δ (pp)": f"{v['single_state_diff_pp']:+.2f}",
                "Multistate Δ (pp)": m_str,
                "11 states n / C": f"{hv.get('outbreaks', 0)} / {int(hv.get('cases', 0)):,}",
                "11 states rate [95% CI]": ci_str,
            })
        cm = emp.get("common_mechanism", {})
        cm_0 = cm.get("0.00", {}).get("n_main_direction_preserved", 14)
        cm_5 = cm.get("0.50", {}).get("n_main_direction_preserved", 15)
        or_fl = emp.get("odds_ratio_floor_001", {}).get("by_gamma", {})
        _emit("tab:delta_cal", "table_12_delta_calibration", pd.DataFrame(rows),
              "Composition-adjusted comparison of fully recorded (M_i = 0) vs. known-status cases in partially recorded "
              "outbreaks (M_i > 0, K_rec > 0), and benchmark rates with 95% Wilson CIs in 11 low-missingness states "
              "(<5% unrecorded and ≥200 common-support cases).",
              "".join([Lc(1.75), R(2.0), R(1.35), R(1.45), R(1.45), R(1.35), R(1.45), R(2.9)]),
              notes=f"Within-cell Δ (+0.05 to +9.02 pp) and single-state Δ are positive across all six groups, whereas "
                    f"Std obs − Full is negative for STEC (−2.53 pp) and Campylobacter (−0.27 pp) and Multistate Δ is negative "
                    f"for STEC (−1.28 pp). Common-mechanism scaling (q_js = α p_js) preserves {cm_0}/15 directions at α = 0 and "
                    f"{cm_5}/15 for α ≥ 0.50 (Supp. Fig. S4). With a 0.1% zero-cell floor under OR scaling, 14/15 pairs hold at Γ = 3.0 and "
                    f"{or_fl.get('5.0', {}).get('determined', 13)}/15 at Γ = 5.0.", sep="1.8pt")

    # Table 13: Real-data follow-up strata
    if "followup_masking" in r:
        rt = r["followup_masking"]["real_data_top_strata"]
        rows = []
        for idx, s in enumerate(rt, 1):
            rows.append({
                "Rank": str(idx), "Pathogen group": NAME[s["group"]], "Setting | Period": s["cell"],
                "Weight (%)": f"{100 * s['weight']:.1f}", "Cases": f"{int(s['total_cases']):,}",
                "Unrecorded": f"{int(s['unrecorded_cases']):,}",
                "w_s/C_js (pp/case)": f"{s['width_leverage_pp']:.4f}",
                "DPI (score/case)": f"{s['dpi_per_case_pp']:.2f}",
            })
        _emit("tab:real_followup", "table_13_real_followup_priority", pd.DataFrame(rows),
              "Top 12 incomplete strata in the observed NORS common-support cohort ranked by heuristic DPI and stratum leverage w_s/C_js.",
              "".join([R(0.75), Lc(2.1), Lc(3.5), R(1.2), R(1.1), R(1.4), R(2.0), R(1.8)]), sep="2.3pt")

    # Table 14: Double-bootstrap 6-method FWER + top drivers
    sc = r.get("small_sample_calibration")
    if sc:
        rows = []
        for g in order:
            pg = sc["per_group"][g]
            rows.append({
                "Group": NAME[g], "MC SD(L) (pp)": f"{pg['mc_sd_L_pp']:.2f}",
                "Mean HC1 SE(L)": f"{pg['mean_hc1_se_L_pp']:.2f}", "Ratio L (HC1/MC)": f"{pg['ratio_hc1_to_mc_L']:.3f}",
                "MC SD(U) (pp)": f"{pg['mc_sd_U_pp']:.2f}", "Mean HC1 SE(U)": f"{pg['mean_hc1_se_U_pp']:.2f}",
                "Ratio U (HC1/MC)": f"{pg['ratio_hc1_to_mc_U']:.3f}",
            })
        fw = sc["fwer_worst"]; fd5 = sc["fwer_delta5"]
        sew = sc.get("fwer_worst_mc_se", {}); sed5 = sc.get("fwer_delta5_mc_se", {})
        cw = sc.get("inner_c_bt_worst_summary", {}); c5 = sc.get("inner_c_bt_delta5_summary", {})
        drv = "; ".join(f"{d['hypothesis']} ({100*d['exceed_rate']:.1f}%)" for d in sc.get("top_fwer_drivers_wc", [])[:3])
        bi = sc.get("B_inner", 399)
        _emit("tab:small_cal", "table_14_small_sample_cal", pd.DataFrame(rows),
              f"Stratified double-bootstrap evaluation ({sc['n_mc']} outer worlds × {bi} inner resamples preserving n_js; "
              f"no state dependence simulated) of 30-hypothesis FWER with every procedure re-estimating its critical value inside each world.",
              "".join([Lc(2.2), R(1.75), R(1.85), R(1.9), R(1.75), R(1.85), R(1.9)]),
              notes=f"IM 95% boundary coverage (15 primary pairs): {100*sc['im95_min_endpoint_cov_hc1']:.1f}% min to "
                    f"{100*sc['im95_median_endpoint_cov_hc1']:.1f}% median (HC1). Empirical 30-test Max-T FWER (nominal 5%; worst / δ = 5; ± MC SE): "
                    f"HC0 {100*fw['hc0']:.1f}±{100*sew.get('hc0',0):.1f}%/{100*fd5['hc0']:.1f}%; "
                    f"HC1 {100*fw['hc1']:.1f}±{100*sew.get('hc1',0):.1f}%/{100*fd5['hc1']:.1f}%; "
                    f"53-cluster {100*fw['state']:.1f}±{100*sew.get('state',0):.1f}%/{100*fd5['state']:.1f}%; "
                    f"unrestricted WCB {100*fw['wcb']:.1f}±{100*sew.get('wcb',0):.1f}%/{100*fd5['wcb']:.1f}%; "
                    f"fixed-scale boot {100*fw['fixed_scale_boot']:.1f}±{100*sew.get('fixed_scale_boot',0):.1f}%/{100*fd5['fixed_scale_boot']:.1f}%; "
                    f"studentized boot-t {100*fw['bootstrap_t']:.1f}±{100*sew.get('bootstrap_t',0):.1f}%/{100*fd5['bootstrap_t']:.1f}±{100*sed5.get('bootstrap_t',0):.1f}%. "
                    f"Inner c_bt 5th–95th percentile: {cw.get('p5',3.5):.2f}–{cw.get('p95',5.4):.2f} (worst), "
                    f"{c5.get('p5',3.0):.2f}–{c5.get('p95',4.1):.2f} (δ = 5). Top HC1 exceedance drivers: {drv}.", sep="2.3pt")

    if write_tex:
        (tdir / "manifest.json").write_text(json.dumps(dict(schema_version=2, tables=man), indent=1, ensure_ascii=False),
                                            encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------
# figures (zero raw underscores; clean mathtext subscripts; fixed legend/axis layout)
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
        "mathtext.fontset": "stix",
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
    m = r["main"]; order = m["order_by_delta0"]; n = len(order)
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    lookup = {(p["higher_at_delta0"], p["lower_at_delta0"]): p for p in m["pairs"].values()}
    rows, cols = order[:-1], order[1:]
    for i, gi in enumerate(rows):
        for j, gj in enumerate(cols):
            if j < i:
                continue
            p = lookup[(gi, gj)]; cat = category(p)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CAT_COLORS[cat], edgecolor="white", lw=2))
            txt = "no assumption\nneeded" if cat == "A" else f"$\\delta^{{*}} = ${fmt_ds(p, unit='')}\npp"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=CAT_TEXT[cat])
    ax.set_xlim(-0.5, n - 1.5); ax.set_ylim(n - 1.5, -0.5)
    ax.set_xticks(range(n - 1)); ax.set_xticklabels([NAME[g] for g in cols])
    ax.set_yticks(range(n - 1)); ax.set_yticklabels([NAME[g] for g in rows])
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Lower group at $\\delta = 0$ (column)", labelpad=10)
    ax.set_ylabel("Higher group at $\\delta = 0$ (row)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0); ax.set_aspect("equal")
    handles = [Patch(facecolor=CAT_COLORS[c], label=CAT_LABEL[c]) for c in ("A", "B", "C")]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False, fontsize=8)
    _save(fig, kit, fdir, "fig_02_verdict_matrix", man, "Pairwise ordering verdicts and $\\delta^*$.", "results")


def fig_intervals(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    m, bt = r["main"], r["bootstrap"]; ret = r["retention"]["per_group"]; order = m["order_by_delta0"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    for i, g in enumerate(order):
        v = m["per_group"][g]; b = bt["per_group"][g]; cr = ret[g]["crude_after"]
        ax.plot([100 * v["L_worst"], 100 * v["U_worst"]], [i, i], color="#B4C0E4", lw=9, solid_capstyle="butt")
        d5 = v["bounds_by_delta_pp"]["5"]
        ax.plot([100 * d5["L"], 100 * d5["U"]], [i, i], color="#0F4D92", lw=9, solid_capstyle="butt")
        ax.plot([100 * v["theta_delta0"]], [i], marker="o", ms=5.0, mfc="white", mec="#272727", mew=1.0, ls="none", zorder=4)
        ax.plot([100 * cr["L"]], [i - 0.19], marker="s", ms=4.5, mfc="#D95F02", mec="#272727", mew=0.6, ls="none", zorder=5)
        ax.plot([100 * cr["recorded_proportion"]], [i + 0.19], marker="^", ms=5.0, mfc="#1B9E77", mec="#272727", mew=0.6, ls="none", zorder=5)
        for lo, hi in ((b["L_p2_5"], b["L_p97_5"]), (b["U_p2_5"], b["U_p97_5"])):
            ax.plot([100 * lo, 100 * hi], [i + 0.34, i + 0.34], color="#4D4D4D", lw=0.9)
            ax.plot([100 * lo, 100 * lo], [i + 0.28, i + 0.40], color="#4D4D4D", lw=0.9)
            ax.plot([100 * hi, 100 * hi], [i + 0.28, i + 0.40], color="#4D4D4D", lw=0.9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[g] for g in order])
    ax.set_ylim(len(order) - 0.35, -0.55)
    xmax = 100 * max(max(bt["per_group"][g]["U_p97_5"], m["per_group"][g]["U_worst"]) for g in order)
    ax.set_xlim(0, 5 * np.ceil(xmax / 5 + 0.2)); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hospitalization proportion among reported outbreak cases (%)")
    handles = [
        Line2D([], [], color="#B4C0E4", lw=7, label="Standardized worst-case interval"),
        Line2D([], [], color="#0F4D92", lw=7, label="Standardized $\\delta = \\pm 5$ pp scenario"),
        Line2D([], [], marker="o", mfc="white", mec="#272727", ls="none", ms=5, label="Standardized $\\delta = 0$ value"),
        Line2D([], [], marker="s", mfc="#D95F02", mec="#272727", ls="none", ms=4.5, label="Crude $H/C$ (unstandardized)"),
        Line2D([], [], marker="^", mfc="#1B9E77", mec="#272727", ls="none", ms=5, label="Crude $H/K$ (unstandardized)"),
        Line2D([], [], color="#4D4D4D", lw=0.9, label=f"Bootstrap 2.5th–97.5th percentile ($B = {bt['B']:,}$)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7.6)
    _save(fig, kit, fdir, "fig_03_bound_intervals", man, "Standardized bounds with crude H/C and H/K.", "results")


def fig_calibrated_2d(r, kit, fdir, man):
    """Main Fig. 4: 2x3 2D breakdown frontiers — top row: 3 lower-tier pairs [0,10]^2;
    bottom row: 3 key upper-tier boundary pairs [0,60]^2 (m14, zero legend overlap)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    m = r["main"]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2), constrained_layout=True)

    lower_keys = ["CPERF|NOV_GII", "NOV_GI|NOV_GII", "CPERF|NOV_GI"]
    # Order by delta_star descending among lower tier: GII>GI (4.20), GII>CPERF (3.68), CPERF>GI (0.19)
    lower_keys = ["NOV_GI|NOV_GII", "CPERF|NOV_GII", "CPERF|NOV_GI"]
    upper_keys = ["SAL|STEC", "SAL|CAMP", "CAMP|NOV_GII"]

    for ax, k in zip(axes[0], lower_keys):
        p = m["pairs"][k]
        td = m["two_d"][k]
        x = np.array(td["d_hi_pp"])
        y = np.array([10.0 if v is None else min(v, 10.0) for v in td["d_lo_boundary_pp"]])
        mask = x <= 10.05
        ax.fill_between(x[mask], 0, y[mask], color=CAT_COLORS[category(p)], alpha=0.85, lw=0)
        ax.add_patch(Rectangle((0.0, 0.0), 5.0, 5.0, facecolor="none", edgecolor="#272727", lw=1.0, ls=":", zorder=4))
        if k == "CPERF|NOV_GI":
            ax.annotate("$\\delta^{*} = 0.19$ pp", xy=(0.19, 0.19), xytext=(2.8, 3.8),
                        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8),
                        fontsize=7.2, color="#333333")
        else:
            ax.text(5.8, 7.6, f"$\\delta^{{*}} = {p['delta_star_pp']:.1f}$ pp", fontsize=7.2, color="#272727")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
        ax.grid(alpha=0.3); ax.set_title(pair_label(p, SHORT), fontsize=8.3); ax.tick_params(labelsize=7.2)

    two_ext = m.get("two_d_ext", {})
    for ax, k in zip(axes[1], upper_keys):
        p = m["pairs"][k]
        td = two_ext.get(k) or m["two_d"][k]
        x = np.array(td["d_hi_pp"])
        y = np.array([60.0 if v is None else min(v, 60.0) for v in td["d_lo_boundary_pp"]])
        ax.fill_between(x, 0, y, color=CAT_COLORS[category(p)], alpha=0.85, lw=0)
        ax.add_patch(Rectangle((0.0, 0.0), 5.0, 5.0, facecolor="none", edgecolor="#272727", lw=1.0, ls=":", zorder=4))
        ax.add_patch(Rectangle((0.0, 0.0), 15.0, 5.0, facecolor="none", edgecolor="#D95F02", lw=1.25, ls="--", zorder=4))
        ax.text(32.0, 48.0, f"$\\delta^{{*}} = {p['delta_star_pp']:.1f}$ pp", fontsize=7.2, color="#272727")
        ax.set_xlim(0, 60); ax.set_ylim(0, 60); ax.set_xticks([0, 20, 40, 60]); ax.set_yticks([0, 20, 40, 60])
        ax.grid(alpha=0.3); ax.set_title(pair_label(p, SHORT), fontsize=8.3); ax.tick_params(labelsize=7.2)

    fig.supxlabel("Downward deviation allowed for higher group, $\\delta_{h,\\mathrm{down}}$ (pp)", fontsize=8.8)
    fig.supylabel("Upward deviation allowed for lower group, $\\delta_{l,\\mathrm{up}}$ (pp)", fontsize=8.8)
    handles = [
        Patch(facecolor=CAT_COLORS["B"], alpha=0.85, label="Upper tier ($\\delta^{*} \\geq 35.5$ pp)"),
        Patch(facecolor=CAT_COLORS["C"], alpha=0.85, label="Lower tier ($\\delta^{*} < 5$ pp)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#272727", lw=1.0, ls=":", label="Symmetric $5 \\times 5$ pp box"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#D95F02", lw=1.25, ls="--", label="Asymmetric $-15 / +5$ pp stress test"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7.5)
    _save(fig, kit, fdir, "fig_04_calibrated_2d", man,
          "2D breakdown frontiers across lower-tier pairs [0,10] pp and key upper-tier boundary pairs [0,60] pp.", "results")


def fig_inference_followup(r, kit, fdir, man):
    """Main Fig. 5: (a) Sample & 30-test statistical breakdown points (HC1, 53-cluster, dynamic & uniform Boot-t);
    (b) Dynamic Max-T critical value curves c_bt(delta) vs c_HC1(delta) and c_state(delta)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    si = r["stacked_inference"]; m = r["main"]; sp = sorted_pairs(m)
    rb_bt = si["rank_breakdown_points_pp"]["bootstrap_t"]
    joint_min_bt = rb_bt["joint_upper_tier_min"]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.22, 1.0]})

    ax = axes[0]
    y = np.arange(len(sp))
    for i, (k, p) in enumerate(sp):
        vp = si["pairs"][k]
        pt = 100.0 if vp["delta_star_pt_pp"] is None else vp["delta_star_pt_pp"]
        hc = 100.0 if vp["delta_star_95_sim_pp"] is None else vp["delta_star_95_sim_pp"]
        st = 100.0 if vp["delta_star_95_sim_state_pp"] is None else vp["delta_star_95_sim_state_pp"]
        bt = 100.0 if vp["delta_star_95_bt_pp"] is None else vp["delta_star_95_bt_pp"]
        btu = 100.0 if vp.get("delta_star_95_bt_unif_pp") is None else vp["delta_star_95_bt_unif_pp"]
        ax.plot([min(hc, st, bt, btu), pt], [i, i], color="#9AA5B8", lw=1.3, zorder=1)
        ax.scatter([pt], [i], color="#0F4D92", s=22, zorder=3, marker=("^" if vp["delta_star_pt_pp"] is None else "o"))
        ax.scatter([hc], [i - 0.14], color="#2B8CBE", s=18, zorder=4, marker=("^" if vp["delta_star_95_sim_pp"] is None else "s"))
        ax.scatter([st], [i - 0.04], color="#D95F02", s=18, zorder=4, marker=("^" if vp["delta_star_95_sim_state_pp"] is None else "D"))
        ax.scatter([bt], [i + 0.08], color="#C44E52", s=21, zorder=5, marker=("^" if vp["delta_star_95_bt_pp"] is None else "P"))
        ax.scatter([btu], [i + 0.18], color="#7570B3", s=16, zorder=4, marker="X")
    ax.axvline(5.0, color="#666666", ls=":", lw=0.9)
    ax.axvline(joint_min_bt, color="#C44E52", ls="--", lw=1.0, alpha=0.8)
    ax.text(joint_min_bt + 1.5, 13.7, f"{joint_min_bt:.1f} pp\n(joint)", fontsize=6.4, color="#C44E52")
    ax.set_yticks(y); ax.set_yticklabels([pair_label(p, SHORT) for _, p in sp], fontsize=6.9)
    ax.set_ylim(len(sp) - 0.5, -0.5); ax.set_xlim(-3, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xticklabels(["0", "20", "40", "60", "80", "100 ($\\infty$)"], fontsize=7.3)
    ax.grid(axis="x", alpha=0.35, lw=0.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Breakdown deviation $\\delta^{*}$ (pp)", fontsize=8)
    ax.set_title("(a) Sample & 30-test 95% breakdown points", fontsize=8.3, pad=5)
    h_a = [
        Line2D([], [], color="#0F4D92", marker="o", ms=4.2, ls="none", label="Sample $\\delta^{*}$"),
        Line2D([], [], color="#2B8CBE", marker="s", ms=4.0, ls="none", label="HC1 $\\delta^{*}_{0.95}$"),
        Line2D([], [], color="#D95F02", marker="D", ms=3.8, ls="none", label="53-cluster $\\delta^{*}_{0.95}$"),
        Line2D([], [], color="#C44E52", marker="P", ms=4.2, ls="none", label="Boot-$t$ $\\delta^{*}_{0.95}$ ($c_{\\mathrm{bt}}(\\delta)$)"),
        Line2D([], [], color="#7570B3", marker="X", ms=4.0, ls="none", label="Boot-$t$ ($c_{\\mathrm{bt}}(100) = 4.52$)"),
    ]
    ax.legend(handles=h_a, loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.1)

    ax = axes[1]
    dyn = si["dynamic_csim_grid"]
    d_gauss = np.array(dyn["delta_pp"])
    ax.plot(d_gauss, dyn["hc1"], color="#2B8CBE", ls="-", lw=1.7, marker="s", ms=2.8, markevery=2, label="Gaussian HC1 $c_{\\mathrm{HC1}}(\\delta)$")
    ax.plot(d_gauss, dyn["state"], color="#D95F02", ls="--", lw=1.7, marker="D", ms=2.8, markevery=2, label="Gaussian 53-cluster $c_{\\mathrm{state}}(\\delta)$")
    bt_g = si["studentized_bootstrap_t"]
    ax.plot(bt_g["c_bt_grid_pp"], bt_g["c_bt_grid_vals"], color="#C44E52", ls="-", lw=2.0, marker="P", ms=4.2, label="Studentized boot-$t$ $c_{\\mathrm{bt}}(\\delta)$")
    ax.axvline(5.0, color="#666666", ls=":", lw=0.9)
    ax.axvline(joint_min_bt, color="#C44E52", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlim(-2, 102); ax.set_ylim(2.2, 4.75)
    ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.tick_params(labelsize=7.2)
    ax.grid(alpha=0.35, lw=0.5)
    ax.set_xlabel("Missingness bound deviation $\\delta$ (pp)", fontsize=8)
    ax.set_ylabel("30-hypothesis 95% Max-$T$ critical value", fontsize=8)
    ax.set_title("(b) $\\delta$-specific Max-$T$ critical values", fontsize=8.3, pad=5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.4)

    _save(fig, kit, fdir, "fig_05_inference_followup", man,
          "Sample and 30-hypothesis statistical breakdown points alongside delta-specific Max-T critical value curves.", "results")


def fig_common_mechanism(r, kit, fdir, man):
    """Supp Fig. S4: Common-mechanism proportional scaling trajectories across full alpha in [0, 2.0] (m15)."""
    import matplotlib.pyplot as plt
    emp = r["empirical_calibration"]; m = r["main"]; cm = emp["common_mechanism"]
    alphas = sorted(float(k) for k in cm.keys())
    grp_cols = {"STEC": "#0F4D92", "SAL": "#2B8CBE", "CAMP": "#1B9E77",
                "NOV_GII": "#D95F02", "CPERF": "#7570B3", "NOV_GI": "#E7298A"}
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for g in m["order_by_delta0"]:
        vals = [100.0 * cm[f"{al:.2f}"]["rates"][g] for al in alphas]
        ax.plot(alphas, vals, color=grp_cols[g], marker="o", ms=3.5, lw=1.7, label=NAME[g])
    ax.axvline(1.0, color="#666666", ls=":", lw=1.0)
    ax.set_xlim(-0.05, 2.05); ax.set_ylim(-1.0, 38.0)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.00])
    ax.grid(alpha=0.35, lw=0.5)
    ax.set_xlabel("Common rate multiplier $\\alpha$ ($q_{js} = \\min(1, \\alpha p_{js})$)", fontsize=8.5)
    ax.set_ylabel("Standardized hospitalization (%)", fontsize=8.5)
    ax.set_title("Common-mechanism proportional scaling across all six pathogen groups ($\\alpha \\in [0, 2]$)", fontsize=9)
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.92, fontsize=7.2)
    _save(fig, kit, fdir, "fig_s4_common_mechanism", man,
          "Standardized hospitalization under uniform common-mechanism rate scaling alpha in [0, 2].", "supplementary")


def fig_retention(r, kit, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    m, ret = r["main"], r["retention"]["per_group"]; order = m["order_by_delta0"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True, sharey=True)
    ax = axes[0]
    for i, g in enumerate(order):
        v = ret[g]; ax.barh(i, v["support_cases_pct"], height=0.6, color="#3775BA")
        ax.text(v["support_cases_pct"] + 1.5, i, f"{v['support_cases_pct']:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 100); ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[g] for g in order])
    ax.set_ylim(len(order) - 0.5, -0.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("(a) Eligible cases inside common support (%)")
    ax = axes[1]; cols = ("#0F4D92", "#AADCA9", "#CFCECE")
    for i, g in enumerate(order):
        v = m["per_group"][g]
        h = v["hospitalized_known"] / v["cases"]; u = v["unknown_share"]; nh = 1 - h - u; left = 0
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
    m = r["main"]; sp = sorted_pairs(m)
    variants = [("Main", m)] + [(VARIANT_CODES[k], v) for k, v in r["robustness"].items()
                                if k not in ("R7_norovirus_combined", "R7b_norovirus_all_confirmed")]
    fig, ax = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    for j, (code, v) in enumerate(variants):
        for i, (k, p) in enumerate(sp):
            q = v["pairs"][k]
            cat = "D" if q["higher_at_delta0"] != p["higher_at_delta0"] else category(q)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CAT_COLORS[cat], edgecolor="white", lw=1.0))
    ax.set_xlim(-0.5, len(variants) - 0.5); ax.set_ylim(len(sp) - 0.5, -0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([f"{c}\n{v['n_determined']['worst_case']}|{v['n_determined']['5']}" for c, v in variants],
                       fontsize=5.8)
    ax.set_yticks(range(len(sp))); ax.set_yticklabels([pair_label(p) for _, p in sp], fontsize=7.4)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Analysis (second line: determined pairs, worst case | $\\delta = 5$ pp)")
    handles = [Patch(facecolor=CAT_COLORS[c], label=t) for c, t in
               (("A", "Holds without unrecorded-outcome assumptions"), ("B", "Holds at $\\delta = 5$ pp ($\\delta^{*} \\geq 5$)"),
                ("C", "Fails before $\\delta = 5$ pp ($\\delta^{*} < 5$)"), ("D", "$\\delta = 0$ direction reversed vs. main"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7.6)
    _save(fig, kit, fdir, "fig_s2_robustness", man, "Sensitivity verdicts across all 20 six-group specifications.", "supplementary")


def fig_followup_sens(r, kit, fdir, man):
    """Supp Fig S3 (3 panels, zero raw underscores, MNAR baseline & random explicitly plotted):
    (a) Per-case cost; (b) Per-outbreak cost; (c) Mixed cost & MNAR comparison."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    vm = r["followup_masking"]
    fig = plt.figure(figsize=(7.6, 6.2), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel (a): Per-case cost
    cb = vm["budgets_cases"]
    for pk, lab, col, ls, mk, lw in [
        ("width_greedy", "Per-case $w_s/C_{js}$", "#0F4D92", "-", "o", 1.8),
        ("dpi", "DPI (heuristic)", "#2B8CBE", "--", "s", 1.3),
        ("random", "Random", "#767676", "-.", "^", 1.4),
        ("volume_greedy", "Largest first", "#C44E52", ":", "v", 1.5),
    ]:
        vals = [vm["summary"][pk][str(b)]["mean_worst_det"] for b in cb]
        ax_a.plot(cb, vals, color=col, ls=ls, marker=mk, ms=3.3, lw=lw, label=lab)
    ax_a.set_xticks([0, 1000, 2000, 3000, 4000, 5000]); ax_a.set_xticklabels(["0", "1k", "2k", "3k", "4k", "5k"], fontsize=7.2)
    ax_a.set_ylim(0, 15.3); ax_a.set_yticks([0, 3, 6, 9, 12, 15]); ax_a.tick_params(axis="y", labelsize=7.2)
    ax_a.grid(alpha=0.35, lw=0.5); ax_a.set_xlabel("Budget (cases)", fontsize=8); ax_a.set_ylabel("Determined pairs", fontsize=8)
    ax_a.set_title("(a) Per-case cost model", fontsize=8.3, pad=4)
    ax_a.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=6.2)

    # Panel (b): Per-outbreak cost
    ob = vm["budgets_outbreaks"]
    for pk, lab, col, ls, mk, lw in [
        ("outbreak_width", "Per-outbreak $w_s U_i/C_{js}$", "#1B9E77", "-", "D", 1.8),
        ("volume_greedy", "Largest first", "#C44E52", ":", "v", 1.6),
        ("width_greedy", "Per-case $w_s/C_{js}$", "#0F4D92", "--", "o", 1.4),
        ("random", "Random", "#767676", "-.", "^", 1.4),
    ]:
        vals = [vm["outbreak_budget_summary"][pk][str(b)]["mean_worst_det"] for b in ob]
        ax_b.plot(ob, vals, color=col, ls=ls, marker=mk, ms=3.3, lw=lw, label=lab)
    ax_b.set_xticks([0, 25, 50, 100, 150, 200]); ax_b.tick_params(labelsize=7.2)
    ax_b.set_ylim(0, 15.3); ax_b.set_yticks([0, 3, 6, 9, 12, 15])
    ax_b.grid(alpha=0.35, lw=0.5); ax_b.set_xlabel("Budget (outbreaks)", fontsize=8)
    ax_b.set_title("(b) Per-outbreak cost model", fontsize=8.3, pad=4)
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=6.2)

    # Panel (c): Mixed cost model (all 500..5000)
    mb = vm["budgets_mixed"]
    for pk, lab, col, ls, mk in [
        ("mixed_leverage", "Mixed $(w_s U_i/C_{js})/(20+U_i)$", "#0F4D92", "-", "o"),
        ("width_greedy", "Per-case $w_s/C_{js}$", "#2B8CBE", "--", "s"),
        ("outbreak_width", "Per-outbreak $w_s U_i/C_{js}$", "#1B9E77", "-.", "D"),
        ("volume_greedy", "Largest first", "#C44E52", ":", "v"),
        ("random", "Random", "#767676", "--", "^"),
    ]:
        vals = [vm["mixed_cost_summary"][pk][str(b)]["mean_worst_det"] for b in mb]
        ax_c.plot(mb, vals, color=col, ls=ls, marker=mk, ms=3.3, lw=1.4, label=lab)
    ax_c.set_xticks([0, 1000, 2000, 3500, 5000]); ax_c.set_xticklabels(["0", "1k", "2k", "3.5k", "5k"], fontsize=7.2)
    ax_c.set_ylim(0, 15.3); ax_c.set_yticks([0, 3, 6, 9, 12, 15]); ax_c.tick_params(axis="y", labelsize=7.2)
    ax_c.grid(alpha=0.35, lw=0.5); ax_c.set_xlabel("Mixed budget ($20 + U_i$ eq.)", fontsize=8); ax_c.set_ylabel("Determined pairs", fontsize=8)
    ax_c.set_title("(c) Mixed fixed + per-case cost", fontsize=8.3, pad=4)
    ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=5.9)

    # Panel (d): MNAR with baseline 0 & Random explicitly plotted!
    sb_all = [0, 500, 1000, 2000, 5000]
    sr = vm["sensitivity_regimes"]["mnar_outcome_size"]
    for pk, lab, col, ls, mk in [
        ("dpi", "MNAR: Heuristic DPI", "#0F4D92", "-", "o"),
        ("width_greedy", "MNAR: Per-case $w_s/C_{js}$", "#1B9E77", "--", "s"),
        ("random", "MNAR: Random (baseline 6.61)", "#767676", "-.", "^"),
        ("volume_greedy", "MNAR: Largest first", "#C44E52", ":", "v"),
    ]:
        vals = [sr[pk][str(b)]["mean_worst_det"] for b in sb_all]
        ax_d.plot(sb_all, vals, color=col, ls=ls, marker=mk, ms=3.3, lw=1.5, label=lab)
    ax_d.set_xticks([0, 1000, 2000, 3500, 5000]); ax_d.set_xticklabels(["0", "1k", "2k", "3.5k", "5k"], fontsize=7.2)
    ax_d.set_ylim(5.5, 15.5); ax_d.set_yticks([6, 8, 10, 12, 14]); ax_d.tick_params(axis="y", labelsize=7.2)
    ax_d.grid(alpha=0.35, lw=0.5); ax_d.set_xlabel("Budget (cases under MNAR)", fontsize=8)
    ax_d.set_title("(d) MNAR masking (with baseline & random)", fontsize=8.3, pad=4)
    ax_d.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=6.1)

    _save(fig, kit, fdir, "fig_s3_followup_sensitivity", man,
          "Follow-up benchmarks across per-case, per-outbreak, mixed-cost, and MNAR masking regimes.", "supplementary")


def make_figures(r: dict, fdir: Path) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    for f in list(fdir.glob("fig_*.pdf")) + list(fdir.glob("fig_*.png")):
        if f.name == "fig_01_selection_flow.png":
            continue
        f.unlink()
    kit = _style()
    man = []
    for fn in (fig_matrix, fig_intervals, fig_calibrated_2d, fig_inference_followup,
               fig_retention, fig_robust, fig_followup_sens, fig_common_mechanism):
        fn(r, kit, fdir, man)
    (fdir / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")


def make_all(exp: Path, paper_dir: Path) -> None:
    r = load(exp)
    make_tables(r, exp / "data" / "analysis_tables", write_tex=False)
    make_tables(r, paper_dir / "tables", write_tex=True)
    make_figures(r, paper_dir / "figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, default=HERE.parent)
    ap.add_argument("--paper-dir", type=Path, default=HERE.parents[1] / "paper")
    a = ap.parse_args()
    statistical_summary(a.exp_dir.resolve())
    make_all(a.exp_dir.resolve(), a.paper_dir.resolve())
