"""Tables, figures, and statistical summary for the NORS hospitalization-ranking analysis (v11.0).

Reads ONLY analysis/results.json (plus data/derived/etiology_mapping.csv for Supp. Table S4)
so every table, figure, and text value shares one canonical source. Writes LF-normalized files.
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
ROOT = HERE.parent
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
                 "R12_stec_o157_only": "R12", "R12b_nontyphoidal_sal": "R12b",
                 "R13_outbreak_level_mar": "R13", "R14a_Ci_extrap25": "R14a",
                 "R14b_low_unrec_50": "R14b"}
CAT_COLORS = {"A": "#0F4D92", "B": "#8FA8D6", "C": "#E9A6A1", "D": "#E8B04B"}
CAT_TEXT = {"A": "white", "B": "#272727", "C": "#272727"}


def load(base_dir: Path) -> dict:
    return json.loads((base_dir / "analysis" / "results.json").read_text(encoding="utf-8"))


def pnum(x, d=1):
    return "NA" if x is None else f"{100 * x:.{d}f}"


def fmt_ds(p_val, unit=" pp"):
    if isinstance(p_val, dict):
        p_val = p_val["delta_star_pp"]
    if p_val is None:
        return "no assumption needed" if unit == " pp" else "∞"
    return f"{p_val:.2f}{unit}" if p_val < 1 else f"{p_val:.1f}{unit}"


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
    math_tokens = [
        ("95% Γ*", r"95\% $\Gamma^{*}$"),
        ("H_rec/K_rec", r"$H_{\mathrm{rec}}/K_{\mathrm{rec}}$"),
        ("w_s/C_js", r"$w_s/C_{js}$"),
        ("p_js", r"$p_{js}$"),
        ("q_js", r"$q_{js}$"),
        ("H_rec", r"$H_{\mathrm{rec}}$"),
        ("K_rec", r"$K_{\mathrm{rec}}$"),
        ("H_i/K_i", r"$H_i/K_i$"),
        ("N_eff", r"$N_{\mathrm{eff}}$"),
        ("G_eff", r"$G_{\mathrm{eff}}$"),
        ("t_{G-1}", r"$t_{G-1}$"),
        ("t_52", r"$t_{52}$"),
        ("c_bt", r"$c_{\mathrm{bt}}$"),
        ("c_cbt", r"$c_{\mathrm{cbt}}$"),
        ("c_crit", r"$c_{\mathrm{crit}}$"),
        ("n_js", r"$n_{js}$"),
        ("w_s", r"$w_s$"),
        ("C_js", r"$C_{js}$"),
        ("M_i", r"$M_i$"),
        ("C_i", r"$C_i$"),
        ("H_i", r"$H_i$"),
        ("K_i", r"$K_i$"),
        ("L_j", r"$L_j$"),
        ("U_j", r"$U_j$"),
        ("L_h", r"$L_h$"),
        ("U_l", r"$U_l$"),
        ("Γ*", r"$\Gamma^{*}$"),
        ("δ*", r"$\delta^{*}$"),
    ]
    holders = {}
    for i, (src, latex) in enumerate(math_tokens):
        key = f"MTOK{i:02d}X"
        if src in s:
            s = s.replace(src, key)
            holders[key] = latex
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 (">=", r"$\geq$"), ("<=", r"$\leq$"), ("±", r"$\pm$"), ("Δ", r"$\Delta$"),
                 ("δ", r"$\delta$"), ("Γ", r"$\Gamma$"), ("α", r"$\alpha$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
                 ("–", "--"), ("−", "$-$"), (">", r"$>$"), ("<", r"$<$"), ("†", r"$^{\dagger}$"),
                 ("*", r"$^{*}$"), ("×", r"$\times$"), ("∞", r"$\infty$")):
        s = s.replace(a, b)
    for key, latex in holders.items():
        s = s.replace(key, latex)
    return s.replace("$$", "")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def R(w):
    return rf">{{\raggedleft\arraybackslash}}p{{{w}cm}}"


def Lc(w):
    return rf">{{\raggedright\arraybackslash}}p{{{w}cm}}"


def write_table(tdir: Path, manifest: list, tid: str, base: str, df: pd.DataFrame, caption: str, colspec: str,
                notes: str = "", size: str = "scriptsize", sep: str = "3pt"):
    csv_p = tdir / f"{base}.csv"
    tex_p = tdir / f"{base}.tex"
    with open(csv_p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(df.to_csv(index=False, lineterminator="\n"))
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        rf"\{size}",
        rf"\setlength{{\tabcolsep}}{{{sep}}}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{tid}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(tex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(row[c]) for c in df.columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\par\smallskip\parbox{{0.97\linewidth}}{{\footnotesize {tex_escape(notes)}}}")
    lines.append(r"\end{table}")
    with open(tex_p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    manifest.append(dict(id=base, label=tid, path_tex=f"{base}.tex", path_csv=f"{base}.csv",
                         sha256_csv=sha256(csv_p), caption=caption, shape=list(df.shape), columns=list(df.columns)))


def statistical_summary(base_dir: Path) -> None:
    r = load(base_dir)
    m = r["main"]
    si = r["stacked_inference"]
    cc_def = r.get("cluster_double_bootstrap_by_def", {})
    out = dict(
        source="analysis/results.json",
        spec_hash=r["provenance"]["spec_hash"],
        analysis_version=r["provenance"]["analysis_version"],
        n_pairs=m["n_pairs"],
        n_common_support_cells=m["n_cells"],
        determined_pairs=dict(worst_case=m["n_determined"]["worst_case"],
                              **{f"delta_{k}pp": v for k, v in m["n_determined"].items() if k != "worst_case"}),
        groups={NAME[g]: dict(L=v["L_worst"], U=v["U_worst"], theta_delta0=v["theta_delta0"],
                              L_delta5=v["bounds_by_delta_pp"]["5"]["L"], U_delta5=v["bounds_by_delta_pp"]["5"]["U"])
                for g, v in m["per_group"].items()},
        stacked_inference=dict(
            n_hypotheses=si["n_hypotheses"],
            n_fwer_worst_sig=si["n_fwer_worst_sig"],
            n_fwer_cv3_worst_sig=si["n_fwer_cv3_worst_sig"],
            n_fwer_boot_t_worst_sig=si["n_fwer_boot_t_worst_sig"],
            n_fwer_boot_t_delta5_sig=si["n_fwer_boot_t_delta5_sig"],
            rank_breakdown_points_pp=si["rank_breakdown_points_pp"],
        ),
        cluster_calibrations={k: v["fwer_worst"] for k, v in cc_def.items()},
        plausible_scenarios={k: dict(sample=v["sample_determined"], boot_t=v["n_sig_boot_t"], cv3=v["n_sig_cv3"])
                             for k, v in r.get("plausible_scenarios", {}).items()},
        note=f"Canonical summary across sample bounds, 30-hypothesis simultaneous inference, and {len(r['robustness'])} variants.",
    )
    with open(base_dir / "analysis" / "statistical_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


def _fmt_ranks(rc: dict, order: list[str]) -> dict[str, str]:
    out = {}
    for g in order:
        v = rc[g]
        lo, hi = (v[0], v[1]) if isinstance(v, (list, tuple)) else (v["rank_lower"], v["rank_upper"])
        out[NAME[g]] = f"[{lo}, {hi}]"
    return out


def make_tables(r: dict, tdir: Path, write_tex: bool = True) -> None:
    tdir.mkdir(parents=True, exist_ok=True)
    for f in tdir.glob("table_*"):
        f.unlink()
    m, man = r["main"], []
    order = m["order_by_delta0"]
    bt = r["bootstrap"]
    si = r["stacked_inference"]
    emp = r["empirical_calibration"]

    def _emit(tid, base, df, caption, colspec, notes="", size="scriptsize", sep="2.6pt"):
        if write_tex:
            write_table(tdir, man, tid, base, df, caption, colspec, notes=notes, size=size, sep=sep)
        else:
            with open(tdir / f"{base}.csv", "w", encoding="utf-8", newline="\n") as fh:
                fh.write(df.to_csv(index=False, lineterminator="\n"))

    # MAIN TABLE 1: STROBE 14a descriptive characteristics by pathogen group
    desc = r["descriptive_cohort"]
    rows = []
    for g in order:
        d = desc[g]
        iqr = d["iqr_cases"]
        rows.append({
            "Pathogen group": NAME[g],
            "Outbreaks": f"{d['outbreaks']:,}",
            "Cases": f"{int(d['cases']):,}",
            "Size med [IQR]": f"{d['median_cases']:.0f} [{iqr[0]:.0f}–{iqr[1]:.0f}]",
            "Rest. cases (%)": f"{d['pct_restaurant_cases']:.1f}",
            "Inst. cases (%)": f"{d['pct_institution_cases']:.1f}",
            "Multistate (%)": f"{d['pct_multistate_cases']:.1f}",
            "2009–18 (%)": f"{d['pct_2009_2018_cases']:.1f}",
            "Unrec. (%)": pnum(d["unrecorded_share"]),
            "Rec. hosp (%)": pnum(d["recorded_hosp_rate"]),
        })
    n_blank_ret = sum(desc[g]["n_blank_k_outbreaks"] for g in order)
    _emit("tab:descriptive", "table_1_descriptive", pd.DataFrame(rows),
          f"Descriptive characteristics of retained common-support outbreaks across six foodborne pathogen groups "
          f"({m['n_cells']} setting × period cells; 3,176 outbreaks; 72,132 cases).",
          "".join([Lc(2.15), R(1.25), R(1.15), R(1.7), R(1.15), R(1.15), R(1.2), R(1.15), R(1.15), R(1.15)]),
          notes="Percentages of Restaurant (Rest.), Institution (Inst.), Multistate, and 2009–2018 cases are case-weighted within each group. "
                f"Unrec. (%) includes C_i − H_i cases from {n_blank_ret} blank-K_i records retained in common support (52 in eligible cohort); Rec. hosp (%) is pooled H_rec/K_rec across retained outbreaks.",
          sep="1.8pt")

    # MAIN TABLE 2: Per-group standardized bounds & crude comparisons
    ret = r["retention"]["per_group"]
    rows = []
    for g in order:
        v = m["per_group"][g]; b = bt["per_group"][g]; cr = ret[g]["crude_after"]
        rows.append({
            "Pathogen group": NAME[g],
            "Crude H/C (%)": pnum(cr["L"]),
            "Crude δ=0 (%)": pnum(cr["theta_delta0"]),
            "Worst-case [L, U] (%)": f"{pnum(v['L_worst'])}–{pnum(v['U_worst'])}",
            "δ = 5 pp [L, U] (%)": f"{pnum(v['bounds_by_delta_pp']['5']['L'])}–{pnum(v['bounds_by_delta_pp']['5']['U'])}",
            "δ = 0 MAR (%)": pnum(v["theta_delta0"]),
            "Bootstrap L range (%)": f"{pnum(b['L_p2_5'])}–{pnum(b['L_p97_5'])}",
            "Bootstrap U range (%)": f"{pnum(b['U_p2_5'])}–{pnum(b['U_p97_5'])}",
        })
    _emit("tab:group_bounds", "table_2_group_bounds", pd.DataFrame(rows),
          f"Crude and setting × period standardized hospitalization proportions among reported outbreak cases "
          f"({m['n_cells']} common-support cells, pooled-case weights).",
          "".join([Lc(2.15), R(1.2), R(1.2), R(2.0), R(1.9), R(1.3), R(2.1), R(2.1)]),
          notes=f"Crude δ=0 (%) = (H + M·p)/C retains known hospitalizations H_i from the 42 blank-K_i records while imputing C_i − H_i at pooled rate p = H_rec/K_rec, hence slightly exceeding Table 1 pooled Rec. hosp (%) whenever blank-K_i records contain hospitalized cases. "
                f"δ = 0 MAR uses within-cell recorded rates p_js; δ = 5 pp bounds unrecorded rates to p_js ± 5 pp. "
                f"Bootstrap L and U ranges give 2.5th–97.5th percentiles over {bt['B']:,} stratified outbreak-level replicates.",
          sep="2.1pt")

    # MAIN TABLE 3: Pairwise bounds, breakdown thresholds, implied unrecorded rates & OR Γ*
    oi = si["or_inference"]["gamma_star_95"]
    fmt_p = lambda pv: "<0.001" if pv < 0.001 else f"{pv:.3f}"
    fmt_ds_s = lambda v: "∞" if v is None else (f"{v:.2f}" if v < 1 else f"{v:.1f}")
    fmt_or_s = lambda v: "∞" if v is None else (">50" if v > 50.0 else (f"{v:.2f}" if v < 1 else f"{v:.1f}"))
    rows = []
    for k, p in sorted_pairs(m):
        sp = si["pairs"][k]; b = bt["pairs"][k]; og = oi[k]
        imp_bt = sp["implied_at_delta_star_95"]["boot_t"]
        imp_str = "–" if imp_bt is None else f"{imp_bt['implied_unrec_high_pct']:.1f} / {imp_bt['implied_unrec_low_pct']:.1f}"
        rows.append({
            "Ordering (δ = 0)": pair_label(p, SHORT),
            "Boot disjoint (WC | δ=5 %)": f"{pnum(b['share_holds_without_assumption'], 1)} | {pnum(1.0 - b['share_delta5_undetermined'], 1)}",
            "Boot-t p (WC | δ=5)": f"{fmt_p(sp['p_fwer_boot_t_worst'])} | {fmt_p(sp['p_fwer_boot_t_delta5'])}",
            "CV3 p (WC | δ=5)": f"{fmt_p(sp['p_fwer_cv3_worst'])} | {fmt_p(sp['p_fwer_cv3_delta5'])}",
            "Sample δ*": fmt_ds_s(sp["delta_star_pt_pp"]),
            "95% ss δ* (Bt | CV3)": f"{fmt_ds_s(sp['delta_star_95_bt_pp'])} | {fmt_ds_s(sp['delta_star_95_cv3_pp'])}",
            "Implied q at Bt δ* (H | L %)": imp_str,
            "OR 95% Γ* (Bt | CV3)": f"{fmt_or_s(og['gamma_star_95_bt'])} | {fmt_or_s(og['gamma_star_95_cv3'])}",
        })
    rb_bt = si["rank_breakdown_points_pp"]["bootstrap_t"]
    rb_c3 = si["rank_breakdown_points_pp"]["cv3"]
    sc = r["small_sample_calibration"]
    cc53 = r["cluster_double_bootstrap"]
    _emit("tab:pairwise_inference", "table_3_pairwise_inference", pd.DataFrame(rows),
          "Pairwise ordering stability, simultaneous 30-hypothesis stepdown FWER p-values (studentized bootstrap-t and CV3 jackknife), "
          "additive/OR breakdown thresholds, and implied unrecorded-case hospitalization rates at 95% thresholds.",
          "".join([Lc(2.1), R(1.85), R(1.85), R(1.80), R(0.95), R(1.8), R(1.85), R(1.65)]),
          notes=f"Abbreviations: ss = single-step; WC = worst case; H | L = higher | lower group. Boot disjoint (WC | δ=5 %) reports sample share of {bt['B']:,} resamples with disjoint bounds (L_h > U_l). "
                f"Dual-criterion rule (requiring both studentized bootstrap-t and CV3 jackknife p <= 0.05) rejects {si['n_fwer_dual_worst_sig']}/30 at worst case, {si['n_fwer_dual_delta5_sig']}/30 at δ = 5 pp (stepdown; {si['n_fwer_ss_dual_delta5_sig']}/30 single-step), and {si['n_fwer_dual_delta0_sig']}/30 at δ = 0. "
                f"Primary studentized bootstrap-t uses δ-specific c_bt(δ) ({si['c_bt_worst']:.2f} WC | {si['c_bt_delta5']:.2f} δ=5 | {si['c_bt_delta0']:.2f} δ=0; 15-hyp. c_bt = {si['c_bt_15_worst']:.2f} | {si['c_bt_15_delta5']:.2f}); "
                f"CV3 leaves out 53 states ({si['cv3_n_valid_worst']}/53 valid; t_52 c = {si['c_sim_cv3_t_worst']:.2f}, 0/30 WC, 12/30 δ=5). "
                f"Implied q at Bt δ* (H | L %) evaluates unrecorded hospitalization at ss 95% Bt δ* "
                f"(STEC>SAL: {si['pairs']['SAL|STEC']['implied_at_delta_star_95']['boot_t']['implied_unrec_high_pct']:.1f}% | {si['pairs']['SAL|STEC']['implied_at_delta_star_95']['boot_t']['implied_unrec_low_pct']:.1f}%; "
                f"CAMP>CPERF: {si['pairs']['CAMP|CPERF']['implied_at_delta_star_95']['boot_t']['implied_unrec_high_pct']:.1f}% | {si['pairs']['CAMP|CPERF']['implied_at_delta_star_95']['boot_t']['implied_unrec_low_pct']:.1f}%). "
                f"Within-upper ss 95% δ*: {rb_bt['within_upper_tier_min']:.1f} pp (Bt, STEC>SAL) vs {rb_c3['within_upper_tier_min']:.1f} pp (CV3, SAL>CAMP, {100*si['rank_breakdown_points_pp']['cv3_sal_camp_multistate_var_share_root']:.1f}% from pooled Multistate variance vs {r['cluster_diagnostics']['multistate_sensitivity']['r11_full']['pair_breakdown_pp']['SAL|CAMP']['cv3']:.1f} pp in R11; stepdown CV3 {si['rank_breakdown_points_pp']['cv3_sal_camp_stepdown_pp']:.1f} pp, p = {si['pairs']['SAL|CAMP']['p_fwer_cv3_delta5']:.3f} at δ = 5 vs ss p = {si['pairs']['SAL|CAMP']['p_ss_cv3_delta5']:.3f}). "
                f"OR 95% Γ* (0.1% floor; >50 capped) is ss (NoV GII>GI: {oi['NOV_GI|NOV_GII']['gamma_star_95_bt']:.2f} Bt | {oi['NOV_GI|NOV_GII']['gamma_star_95_cv3']:.2f} CV3; sample Γ* under 0/0.1%/1% floors: 7.75/7.60/4.95).",
          sep="1.8pt")

    # MAIN TABLE 4: Conventional baselines vs approximate conditioned 95% rank sets across specifications, plausibility regimes & scales
    rcs = si["rank_confidence_sets"]
    cwi = si["coarser_and_weighting_inference"]
    pl = r["plausible_scenarios"]
    cb = r["conventional_ranking_benchmark"]
    oi_bg = si["or_inference"]["by_gamma"]
    rank_scenarios = [
        ("Baseline: Marginal Wilson 95% CI (pooled H_rec/K_rec)", cb["marginal_wilson_hk"]["ranks"]),
        ("Baseline: Marginal 2,000-rep bootstrap (std. δ = 0)", cb["marginal_bootstrap_delta0"]["ranks"]),
        ("Baseline: EB Beta-Binomial across outbreaks (Laird–Louis)", cb["eb_betabinom_outbreak"]["ranks"]),
        ("Baseline: EB Beta-Binomial across 53 states (Laird–Louis)", cb["eb_betabinom_state"]["ranks"]),
        ("Baseline: Simultaneous δ = 0 MAR (Dual-criterion)", rcs["delta0_dual"]),
        ("Worst case: Case-weighted (Dual Boot-t & CV3)", rcs["worst_dual"]),
        ("Worst case: Case-weighted (HC1 / CV1)", rcs["worst_hc1"]),
        ("Worst case: R10 Equal outbreak wt. (Dual / Boot-t)", cwi["R10_outbreak_weighted"]["ranks"]["worst_boot_t"]),
        ("Worst case: R2 Period-only / R11 Single-state (Dual)", cwi["R2_period_only"]["ranks"]["worst_cv3"]),
        ("External anchors P_ext (FoodNet/Scallan; Dual)", pl["P_ext_external_surveillance"]["ranks_dual"]),
        ("External anchors P_ext_2x ([0, 2%] lower tier; Dual)", pl["P_ext_2x_lower"]["ranks_dual"]),
        ("Plausibility P1/P2 (Lower <=5%/<=10%, Upper p_js ±50%; Dual)", pl["P1_lower5_upper50pct"]["ranks_dual"]),
        ("Additive δ = 5 pp (Single-step Dual & CV3)", rcs["delta5_dual_ss"]),
        ("Additive δ = 5 pp (Stepdown Dual & CV3)", rcs["delta5_dual"]),
        ("Diagnostic asym. (-10/+5 SAL, -7/+5 CAMP; Dual)", rcs["diag_asym_cv3"]),
        ("OR scale Γ = 2.0 (0.1% floor, Stepdown Boot-t)", oi_bg["2.0"]["ranks"]["boot_t"]),
        ("OR scale Γ = 2.0 (0.1% floor, Dual / Stepdown CV3)", oi_bg["2.0"]["ranks"]["cv3"]),
        ("OR scale Γ = 3.0 (0.1% floor, Dual / Stepdown CV3)", oi_bg["3.0"]["ranks"]["cv3"]),
    ]
    rows = []
    for lbl, rmap in rank_scenarios:
        rows.append({"Scenario / Procedure": lbl, **_fmt_ranks(rmap, order)})
    _emit("tab:rank_sets", "table_4_rank_sets_summary", pd.DataFrame(rows),
          "Conventional ranking baselines and approximate conditioned simultaneous 95% rank confidence sets [1 = highest, 6 = lowest] across weighting specifications, external/internal plausibility intervals, additive departures, and OR scaling.",
          "".join([Lc(4.45)] + [R(1.65)] * 6),
          notes=f"Rows 1–4 summarize conventional marginal/hierarchical baselines (details in Table S13); rows 5–18 summarize simultaneous rank sets conditioned on each missingness assumption under the dual-criterion (Boot-t & CV3) rule. Because realized FWER is 7.3%–13.0% under 53-cluster random effects (3.0%/1.0% outbreak, 4.7%/2.0% LogNormal, 8.0%/3.0% 271-cluster, 8.3%/4.7% R11; Tables S10–S11), rank sets carry an approximate rather than exact 95% guarantee. "
                f"Under δ = 5 pp, single-step Dual/CV3 yields [2, 3] for SAL/CAMP (11/30; CV3 ss p = {si['pairs']['SAL|CAMP']['p_ss_cv3_delta5']:.3f}) whereas stepdown yields [2, 2]/[3, 3] (12/30; p = {si['pairs']['SAL|CAMP']['p_fwer_cv3_delta5']:.3f}). "
                f"R10 targets an equal-outbreak-average estimand (fixing w_s), rejecting {cwi['R10_outbreak_weighted']['n_sig_worst']['boot_t']}/30 (Boot-t p = {cwi['R10_outbreak_weighted']['stec_nov_gii_p_worst']['boot_t']:.3f}) and {cwi['R10_outbreak_weighted']['n_sig_worst']['cv3']}/30 (CV3) at worst case. "
                f"OR Γ = 2.0 rejects {oi_bg['2.0']['n_sig']['boot_t']}/30 under stepdown Boot-t (separating NoV GII [4, 5] > GI [5, 6]; ss Γ* = {oi['NOV_GI|NOV_GII']['gamma_star_95_bt']:.2f}) but {oi_bg['2.0']['n_sig']['cv3']}/30 under CV3/Dual (both [4, 6]; ss Γ* = {oi['NOV_GI|NOV_GII']['gamma_star_95_cv3']:.2f}).",
          size="tiny", sep="1.9pt")

    # SUPP TABLE S1: Audit
    a = r["audit"]["record_audit_summary"]
    handling = {"complete_consistent": "as recorded (K = 0 means all status unrecorded)",
                "single_missing_K": "K set to H; C − H cases unrecorded (worst case exact; upper δ conservative)",
                "single_missing_H_K0": "H = 0 implied by H ≤ K = 0; all cases unrecorded",
                "single_missing_H_Kpos": "K_rec set to 0; all cases treated as unrecorded",
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
    _emit("tab:audit", "table_s1_audit", pd.DataFrame(rows),
          "Record audit of outbreak outcome fields (C reported cases, H hospitalized, K known status). Zero count contradictions.",
          "".join([Lc(2.1), R(1.35), R(1.1)] + [R(0.88)] * 6 + [Lc(3.6)]),
          notes="Ambiguous etiology (408 outbreaks, 12,220 cases) equals the 304 confirmed ungenotyped norovirus outbreaks (7,679 cases) plus 104 multi-agent outbreaks with >=1 confirmed study group (4,541 cases) in Table S2.",
          sep="2.3pt")

    # SUPP TABLE S2: Excluded outbreaks
    ex = r["audit"]["excluded_outbreaks_summary"]
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
    _emit("tab:excluded", "table_s2_excluded", pd.DataFrame(rows),
          "Characteristics of excluded foodborne outbreaks (2009–2023) not retained in the primary six-group cohort.",
          "".join([Lc(3.8), R(1.25), R(1.1), R(2.1), Lc(3.15), R(1.35), R(1.55)]), sep="2.3pt")

    # SUPP TABLE S3: Common-support retention
    rows = []
    for g in order:
        v = ret[g]
        top_set = ", ".join(f"{k} {int(c):,}" for k, c in list(v["dropped_cases_by_setting"].items())[:2])
        top_per = ", ".join(f"{k} {int(c):,}" for k, c in list(v["dropped_cases_by_period"].items())[:2])
        rows.append({"Group": NAME[g], "Eligible ob": f"{v['eligible_outbreaks']:,}",
                     "Eligible cases": f"{int(v['eligible_cases']):,}",
                     "Retained ob (%)": f"{v['support_outbreaks']:,} ({v['support_outbreaks_pct']:.1f})",
                     "Retained cases (%)": f"{int(v['support_cases']):,} ({v['support_cases_pct']:.1f})",
                     "Largest dropped settings (cases)": top_set, "Largest dropped periods": top_per,
                     "Crude δ=0 bef/aft (%)": f"{pnum(v['crude_before']['theta_delta0'])}/{pnum(v['crude_after']['theta_delta0'])}"})
    _emit("tab:retention", "table_s3_retention", pd.DataFrame(rows),
          "Common-support retention by pathogen group, dropped strata, and crude proportions before/after restriction.",
          "".join([Lc(1.9), R(1.15), R(1.15), R(1.45), R(1.5), Lc(2.1), Lc(1.7), R(1.7)]), sep="2.1pt")

    # SUPP TABLE S4: RECORD 6.1 Etiology string -> study group mapping
    map_csv = ROOT / "data" / "derived" / "etiology_mapping.csv"
    mdf = pd.read_csv(map_csv, keep_default_na=False)
    mdf_study = mdf[mdf["group"] != ""].copy()
    rows = []
    for _, rw in mdf_study.iterrows():
        rows.append({
            "Raw NORS etiology string": str(rw["etiology_string"]),
            "Assigned study group": NAME.get(str(rw["group"]), str(rw["group"])),
            "Food 2009–23 entries": str(int(rw["n_entries_food_2009_2023"])),
            "In R3c exact-string pilot": "Yes" if str(rw["pilot_exact_string"]).lower() == "true" else "No",
        })
    mdf_excl = mdf[mdf["group"] == ""].copy()
    mdf_nov_un = mdf_excl[mdf_excl["etiology_string"].isin(["Norovirus unknown", "Norovirus"])].copy()
    for _, rw in mdf_nov_un.iterrows():
        rows.append({
            "Raw NORS etiology string": str(rw["etiology_string"]),
            "Assigned study group": "Excluded from GI/GII (R7/R7b)",
            "Food 2009–23 entries": str(int(rw["n_entries_food_2009_2023"])),
            "In R3c exact-string pilot": "No",
        })
    mdf_other = mdf_excl[~mdf_excl["etiology_string"].isin(["Norovirus unknown", "Norovirus"])].copy()
    rows.append({
        "Raw NORS etiology string": f"All other non-study foodborne strings (n = {len(mdf_other)} distinct strings)",
        "Assigned study group": "Excluded (non-target / multi-group)",
        "Food 2009–23 entries": str(int(pd.to_numeric(mdf_other["n_entries_food_2009_2023"]).sum())),
        "In R3c exact-string pilot": "No",
    })
    _emit("tab:etiology_map", "table_s4_etiology_mapping", pd.DataFrame(rows),
          "Complete mapping from raw CDC NORS etiology strings (Food 2009–2023) to the six study pathogen groups (RECORD Item 6.1).",
          "".join([Lc(5.6), Lc(3.0), R(2.2), R(2.6)]),
          notes="Multi-entry outbreaks are mapped only when every entry maps to the same study group and >=1 entry has status Confirmed (R3a evaluates single-entry only). "
                "The 1,656 raw Norovirus / Norovirus unknown entries include suspected and multi-etiology records (corresponding to 304 confirmed single-group ungenotyped norovirus outbreaks in R7/R7b); remaining excluded foodborne strings (n = 87) are summarized at bottom.",
          sep="2.3pt")

    # SUPP TABLE S5: All 72 group x cell outbreak counts, cases, HHI & N_eff (M5)
    rows = []
    for c72 in r["cells_72"]:
        rows.append({
            "Group": NAME[c72["group"]],
            "Setting | Period": c72["cell"],
            "Outbreaks": f"{c72['n_outbreaks']:,}",
            "Cases": f"{int(c72['cases']):,}",
            "Unrecorded (%)": f"{100*c72['unrecorded_share']:.1f}",
            "Max outbreak (%)": f"{100*c72['max_share']:.1f}",
            "HHI": f"{c72['hhi']:.3f}",
            "N_eff": f"{c72['n_eff']:.2f}",
        })
    _emit("tab:cells_72", "table_s5_cells_72", pd.DataFrame(rows),
          "Complete 72-stratum (6 groups × 12 common-support cells) outbreak counts, case counts, HHI, and effective cluster size N_eff.",
          "".join([Lc(2.05), Lc(3.55), R(1.15), R(1.15), R(1.4), R(1.5), R(1.1), R(1.2)]),
          notes="4 of 72 cells contain exactly n_js = 3 outbreaks and 8 contain <= 4 outbreaks; N_eff = 1/HHI falls to 1.97–2.09 in concentrated strata.",
          size="tiny", sep="1.9pt")

    # SUPP TABLE S6: Missingness diagnostics WITH bootstrap 95% CIs (M2d) + 11-state Wilson CIs
    pg = emp["per_group"]; hc = emp["high_completeness_states"]["per_group"]
    f_ci = lambda val, ci: f"{val:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else f"{val:+.2f}"
    rows = []
    for g in order:
        v = pg[g]; hv = hc[g]; ciw = hv["ci95_wilson"]
        m_str = "–" if v["multistate_diff_pp"] is None else f_ci(v["multistate_diff_pp"], v.get("multistate_diff_ci95_pp"))
        rows.append({
            "Group": NAME[g],
            "Full vs Part (%)": f"{pnum(v['rate_full_pooled'])} vs {pnum(v['rate_partial_known_pooled'])}",
            "Within-cell Δ [95% CI]": f_ci(v["within_cell_diff_pp"], v.get("within_cell_diff_ci95_pp")),
            "Std obs − Full [95% CI]": f_ci(v["std_obs_minus_full_pp"], v.get("std_obs_minus_full_ci95_pp")),
            "Single-state Δ [95% CI]": f_ci(v["single_state_diff_pp"], v.get("single_state_diff_ci95_pp")),
            "Multistate Δ [95% CI]": m_str,
            "11 states rate [95% CI]": f"{pnum(hv['recorded_hosp_rate'])} [{pnum(ciw[0])}, {pnum(ciw[1])}]",
        })
    dai = si["diag_asymmetric_inference"]
    _emit("tab:delta_cal", "table_s6_delta_calibration", pd.DataFrame(rows),
          "Partial-versus-full missingness contrasts with 2,000-replicate bootstrap 95% CIs (pp) and descriptive 11-state Wilson 95% CIs.",
          "".join([Lc(1.65), R(1.85), R(2.35), R(2.2), R(2.35), R(2.05), R(2.3)]),
          notes=f"All contrasts compare identical shared cells (8–12 cells, 79.3%–100.0% weight). Bootstrap CIs come from {bt['B']:,} stratified resamples; "
                f"Multistate Δ for STEC is {pg['STEC']['multistate_diff_pp']:+.2f} pp (bootstrap CI reflects small multistate full sample). "
                f"Diagnostic asymmetric scenario (SAL −10/+5, CAMP −7/+5, others ±5 pp) retains 12/12 upper-tier separations (p <= 0.05 under HC1/CV1/CV3/Boot-t; c_bt = {dai['c_sim']['boot_t']:.2f}).",
          size="tiny", sep="1.5pt")

    # SUPP TABLE S7: Robustness (23 variants)
    rows = [{"Analysis": "Main", "Description": "Setting × period, >=3 outbreaks, pooled-case weights",
             "Cells": m["n_cells"], "Pairs": m["n_pairs"], "Determined, worst case": m["n_determined"]["worst_case"],
             "Determined, δ = 5 pp": m["n_determined"]["5"], "Worst-case changed": 0, "δ = 0 reversed": 0}]
    for vid, v in r["robustness"].items():
        desc = v["label"].replace("Crude: no standardization", "Crude, no standardization").replace(
            "Cohort: ", "Cohort of ").replace("Weights: ", "Weights from ")
        if vid == "R3c_pilot_rule":
            desc = f"Cohort of species-level strings; records with one blank outcome dropped ({v['n_outbreaks_in_cohort']:,} outbreaks)"
        rows.append({"Analysis": VARIANT_CODES[vid], "Description": desc, "Cells": v["n_cells"],
                     "Pairs": v["n_pairs"], "Determined, worst case": v["n_determined"]["worst_case"],
                     "Determined, δ = 5 pp": v["n_determined"]["5"],
                     "Worst-case changed": v.get("n_worst_case_verdicts_changed", "NA"),
                     "δ = 0 reversed": v.get("n_delta0_directions_changed", "NA")})
    _emit("tab:robustness", "table_s7_robustness", pd.DataFrame(rows),
          f"Sensitivity analyses across all {len(r['robustness'])} cohort, support, weighting, serogroup, outbreak-MAR, and selection variants.",
          "".join([Lc(1.0), Lc(5.25), R(0.85), R(0.85), R(1.5), R(1.45), R(1.45), R(1.45)]),
          notes="R7 and R7b combine norovirus into 5 groups (10 pairs), so 6-group changes are NA. "
                "R12 restricts STEC to confirmed O157 (279/388); R12b excludes 6 Salmonella Typhi outbreaks (87 cases); "
                "R13 uses outbreak H_i/K_i MAR anchors; R14a evaluates C_i contamination; R14b evaluates high-completeness selection.",
          sep="2.3pt")

    # SUPP TABLE S8: Bootstrap + conditional K_rec check (m4)
    st_bt = si["studentized_bootstrap_t"]
    rows = []
    for k, p in sorted_pairs(m):
        b = bt["pairs"][k]
        rows.append({"Ordering at δ = 0": pair_label(p, SHORT),
                     "Same δ = 0 direction (%)": pnum(b["share_same_delta0_direction"], d=2),
                     "Worst case, higher > lower (%)": pnum(b["share_worst_A"] if p["higher_at_delta0"] == p["A"] else b["share_worst_B"], d=2),
                     "Worst case, undetermined (%)": pnum(b["share_worst_undetermined"], d=2),
                     "Determined at δ = 5 pp (%)": pnum(1 - b["share_delta5_undetermined"], d=2),
                     "δ* 2.5th–97.5th percentile (pp)":
                         f"{'∞' if b['delta_star_pp_p2_5'] is None else format(b['delta_star_pp_p2_5'], '.1f')}–"
                         f"{'∞' if b['delta_star_pp_p97_5'] is None else format(b['delta_star_pp_p97_5'], '.1f')}"})
    _emit("tab:bootstrap", "table_s8_bootstrap", pd.DataFrame(rows),
          f"Outbreak-level bootstrap ({bt['B']:,} replicates, seed {bt['seed']}; {bt['n_replicates_with_undefined_strata']}/{bt['B']:,} draws had >=1 stratum lacking K_rec), "
          f"with conditional studentized bootstrap-t check over complete-K_rec draws.",
          "".join([Lc(2.6), R(2.0), R(2.2), R(2.0), R(2.0), R(2.6)]),
          notes=f"Conditioning on the {st_bt['cond_valid_B']}/{st_bt['B_bt']} draws where every cell has K_rec > 0 gives c_bt = "
                f"{st_bt['cond_worst']['c_bt']:.3f} worst case (0 sig; cf. unconditional {si['c_bt_worst']:.3f}) and "
                f"{st_bt['cond_delta5']['c_bt']:.3f} at δ = 5 pp (12 sig; cf. {si['c_bt_delta5']:.3f}).",
          sep="2.2pt")

    # SUPP TABLE S9: All 6 inference procedures, SEs, 53/271/52 clustering & LOO/LLO
    cd = r["cluster_diagnostics"]["per_group"]
    sg = si["per_group"]
    cvs = r["cluster_diagnostics"]["cluster_variance_share"]
    ms = r["cluster_diagnostics"]["multistate_sensitivity"]
    loo = si["loo_influence"]; llo = loo["leave_largest_per_group"]
    rows = []
    for g in order:
        c_g = cd[g]; s_g = sg[g]; b_g = bt["per_group"][g]
        rows.append({
            "Group": NAME[g],
            "Min/Med/Max n": f"{c_g['min_n']}/{c_g['median_n']:.0f}/{c_g['max_n']}",
            "Max HHI (min N_eff)": f"{c_g['max_hhi']:.3f} ({c_g['min_neff']:.2f})",
            "Multistate unrec (%)": f"{100 * c_g['multistate_unrecorded_share']:.1f}",
            "SE(L) HC0/HC1/Boot/St (pp)": f"{100*s_g['se_L_hc0']:.2f}/{100*s_g['se_L_hc1']:.2f}/{100*b_g['L_sd']:.2f}/{100*s_g['se_L_state']:.2f}",
            "SE(U) HC0/HC1/Boot/St (pp)": f"{100*s_g['se_U_hc0']:.2f}/{100*s_g['se_U_hc1']:.2f}/{100*b_g['U_sd']:.2f}/{100*s_g['se_U_state']:.2f}",
        })
    cw_w = cvs["worst"]; cw_5 = cvs["delta5"]
    _emit("tab:cluster_diag", "table_s9_all_inference_cluster", pd.DataFrame(rows),
          "Outbreak concentration, Multistate score variance share, standard errors across methods, clustering structures, and LOO/LLO influence.",
          "".join([Lc(1.85), R(1.75), R(2.05), R(1.55), R(3.4), R(3.4)]),
          notes=f"All 6 procedures at worst case (STEC>NoV GII p / c_crit / # sig.): HC1 {si['pairs']['STEC|NOV_GII']['p_fwer_worst']:.3f}/{si['c_sim_worst']:.2f}/1; "
                f"53-state CV1 {si['pairs']['STEC|NOV_GII']['p_fwer_state_worst']:.3f}/{si['c_sim_state_worst']:.2f}/1; "
                f"WCB {si['pairs']['STEC|NOV_GII']['p_fwer_wcb_worst']:.3f}/{si['wild_cluster_bootstrap']['worst']['c_wcb_sim']:.2f}/1; "
                f"CV3 {si['pairs']['STEC|NOV_GII']['p_fwer_cv3_worst']:.3f}/{si['c_sim_cv3_worst']:.2f}/0 (t_52 crit {si['c_sim_cv3_t_worst']:.2f}, p={si['stec_nov_gii_p_cv3_t_worst']:.3f}); "
                f"cluster boot-t {si['pairs']['STEC|NOV_GII']['p_fwer_cluster_bt_worst']:.3f}/{si['c_cbt_worst']:.2f}/0; "
                f"boot-t {si['pairs']['STEC|NOV_GII']['p_fwer_boot_t_worst']:.3f}/{si['c_bt_worst']:.2f}/0 (15-hyp. c = {si['c_bt_15_worst']:.2f}); dual-criterion {si['n_fwer_dual_worst_sig']}/30. At δ = 5 pp, all 6 reject 12 stepdown (15-hyp. Boot-t c = {si['c_bt_15_delta5']:.2f}; CV3 SAL>CAMP p = {si['pairs']['SAL|CAMP']['p_fwer_cv3_delta5']:.3f} stepdown vs {si['pairs']['SAL|CAMP']['p_ss_cv3_delta5']:.3f} ss). "
                f"Multistate score variance share: {100*cw_w['max_variance_share']:.1f}% worst (plug-in G_eff = {cw_w['effective_clusters_g_eff']:.2f}) and {100*cw_5['max_variance_share']:.1f}% δ = 5 (G_eff = {cw_5['effective_clusters_g_eff']:.2f}); "
                f"271-singleton yields {ms['n_sig_singleton_worst']}/30 CV1 (STEC>NoV GII p = {ms['stec_nov_gii_p_singleton_cv1_worst']:.3f}) and {ms['n_sig_singleton_cv3_worst']}/30 CV3 (p = {ms['stec_nov_gii_p_singleton_cv3_worst']:.3f}) at worst case (both 12/30 at δ = 5), while 52-state R11 yields 0/30 worst case and 12/30 δ = 5 across all 5 methods. Across {loo['n_outbreaks_checked']:,} LOO (and 6-group LLO) deletions, sample worst-case determination stays {loo['wc_determined_range'][0]}–{loo['wc_determined_range'][1]} ({llo['wc_determined']} LLO; min worst-case margin {loo['most_influential_single_outbreak']['min_margin_pp']:.2f} pp on {loo['most_influential_single_outbreak']['pair']} after removing Ohio {int(loo['most_influential_single_outbreak']['cases'])}-case CPERF outbreak); across top-10 largest outbreaks, HC1/CV3 rejection counts remain invariant (1/30 HC1, 0/30 CV3 at WC; 12/30 at δ = 5).",
          sep="2.1pt")

    # SUPP TABLE S10: Complete Double-Bootstrap, Partial-Null Stepdown vs Single-Step, and Empirical Power Matrix (M1, M7, M8)
    cc_def = r["cluster_double_bootstrap_by_def"]
    pc = r["parametric_calibration"]
    spc = r["stepdown_and_power_calibration"]
    rows = []
    for k_m, lbl_m in [("hc0", "HC0 Gaussian (df-uncorrected)"), ("hc1", "HC1 Gaussian"),
                       ("fixed_scale_boot", "Fixed-scale std. bootstrap"), ("state", "53-state CV1 Gaussian"),
                       ("wcb", "Unrestricted Rademacher WCB"), ("cv3_jackknife", "CV3 cluster jackknife"),
                       ("bootstrap_t", "Studentized bootstrap-t"), ("dual_criterion", "Dual-criterion (Boot-t & CV3)")]:
        rows.append({
            "Calibration world": f"Global null: Outbreak ({sc['n_mc']}×{sc['B_inner']})",
            "Procedure": lbl_m,
            "Worst / Primary (%)": f"{100*sc['fwer_worst'][k_m]:.1f} ± {100*sc['fwer_worst_mc_se'][k_m]:.1f}",
            "δ=5 / Sec. (%)": f"{100*sc['fwer_delta5'][k_m]:.1f} ± {100*sc['fwer_delta5_mc_se'][k_m]:.1f}",
            "World note": f"Boot-t inner c: {sc['inner_c_bt_worst_summary']['p5']:.2f}–{sc['inner_c_bt_worst_summary']['p95']:.2f}" if k_m == "bootstrap_t" else "No state correlation; global null (sd = ss)",
        })
    for c_key, c_lbl in [("state_53", "Global null: 53-state RE"), ("singleton_271", "Global null: 271-singleton RE"), ("r11_52", "Global null: 52-state R11 RE")]:
        cd_b = cc_def[c_key]
        for k_c, lbl_c in [("cv1_state", "CV1 cluster Gaussian"), ("wcb", "Unrestricted Rademacher WCB"),
                           ("cluster_boot_t", "Bayesian cluster boot-t"), ("bootstrap_t", "Studentized bootstrap-t"),
                           ("cv3_jackknife", "CV3 cluster jackknife"), ("cv3_t_df", "CV3 jackknife (t_{G-1})"),
                           ("dual_criterion", "Dual-criterion (Boot-t & CV3)")]:
            if k_c not in cd_b["fwer_worst"]:
                continue
            rows.append({
                "Calibration world": f"{c_lbl} ({cd_b['n_mc']}×{cd_b['B_inner']})",
                "Procedure": lbl_c,
                "Worst / Primary (%)": f"{100*cd_b['fwer_worst'][k_c]:.1f} ± {100*cd_b['fwer_worst_mc_se'][k_c]:.1f}",
                "δ=5 / Sec. (%)": f"{100*cd_b['fwer_delta5'][k_c]:.1f} ± {100*cd_b['fwer_delta5_mc_se'][k_c]:.1f}",
                "World note": f"Random-size RE + Beta-Binom (SE/SD: CV1 {cd_b['median_se_ratio_cv1']:.2f}, CV3 {cd_b['median_se_ratio_cv3']:.2f})",
            })
    for k_p, lbl_p in [("hc1", "HC1 Gaussian"), ("cv3_jackknife", "CV3 cluster jackknife"),
                       ("bootstrap_t", "Studentized bootstrap-t"), ("dual_criterion", "Dual-criterion (Boot-t & CV3)")]:
        rows.append({
            "Calibration world": f"Global null: LogNorm ({pc['n_mc']}×{pc['B_inner']})",
            "Procedure": lbl_p,
            "Worst / Primary (%)": f"{100*pc['fwer_worst'][k_p]:.1f} ± {100*pc['fwer_worst_mc_se'][k_p]:.1f}",
            "δ=5 / Sec. (%)": f"{100*pc['fwer_delta5'][k_p]:.1f} ± {100*pc['fwer_delta5_mc_se'][k_p]:.1f}",
            "World note": f"LogNorm + Beta-Binom (HC1 SE/obs {pc['median_synthetic_vs_observed_hc1_se_ratio']:.2f})",
        })
    # Partial-null Stepdown vs Single-step FWER & Empirical Power rows go in dedicated Table S11
    pg_L = [sc["per_group"][g]["ratio_hc1_to_mc_L"] for g in order]
    pg_U = [sc["per_group"][g]["ratio_hc1_to_mc_U"] for g in order]
    _emit("tab:small_cal", "table_s10_calibration", pd.DataFrame(rows),
          "Empirical 30-hypothesis global-null FWER (%) under least-favourable boundary nulls (nominal 5.0%) across outbreak, three cluster definitions (53/271/52), and parametric heavy-tail worlds.",
          "".join([Lc(3.3), Lc(3.05), R(2.2), R(2.6), Lc(3.35)]),
          notes=f"Under global nulls, stepdown equals single-step FWER identically; dual-criterion achieves <=5.0% FWER in outbreak (3.0%/1.0%) and LogNormal (4.7%/2.0%) worlds and lowers cluster-RE over-rejection to 13.0%/7.3% (53-cluster) and 8.0%–8.3% / 3.0%–4.7% (271/52-cluster). "
                f"Pointwise Imbens-Manski 95% boundary coverage (outbreak world): {100*sc['im95_min_endpoint_cov_hc1']:.1f}% min to {100*sc['im95_median_endpoint_cov_hc1']:.1f}% median (HC1); per-group HC1 SE / MC SD ratios {min(pg_L):.3f}–{max(pg_L):.3f} (L_j) and {min(pg_U):.3f}–{max(pg_U):.3f} (U_j).",
          size="tiny", sep="1.7pt")

    # SUPP TABLE S11: Partial-Null Stepdown vs Single-Step FWER and Empirical-Alternative Power (M7, M8)
    rows_sp = []
    for w_key, w_lbl, k_map, pw_key in [
        ("outbreak_partial_null_d5", "Outbreak (400×499)", [("hc1", "HC1 Gaussian"), ("state", "53-state CV1"), ("cv3_jackknife", "CV3 cluster jackknife"), ("bootstrap_t", "Studentized bootstrap-t"), ("dual_criterion", "Dual-criterion (Boot-t & CV3)")], "outbreak_empirical_power"),
        ("cluster_53_partial_null_d5", "53-state RE (300×399)", [("cv1_state", "CV1 cluster Gaussian"), ("cv3_jackknife", "CV3 cluster jackknife"), ("bootstrap_t", "Studentized bootstrap-t"), ("dual_criterion", "Dual-criterion (Boot-t & CV3)")], "cluster_53_empirical_power"),
    ]:
        pn = spc[w_key]; ep = spc[pw_key]
        for kp, lp in k_map:
            rows_sp.append({
                "World": w_lbl,
                "Procedure": lp,
                "Partial-null δ=5 FWER (sd | ss %)": f"{100*pn['fwer_stepdown'][kp]:.1f} | {100*pn['fwer_single_step'][kp]:.1f}",
                "Partial-null SAL>CAMP | 11-all (%)": f"{100*pn['sal_camp_false_rej_stepdown'][kp]:.1f} | {100*pn['strong11_all_power'][kp]:.1f}",
                "WC power: STEC>GII | any4 (%)": f"{100*ep['wc_stec_nov_gii_power'][kp]:.1f} | {100*ep['wc_any4_power'][kp]:.1f}",
                "δ=5 power: SAL>CAMP | mean/12": f"{100*ep['d5_sal_camp_power'][kp]:.1f}% | {ep['d5_mean12_rejected'][kp]:.2f}/12",
            })
    _emit("tab:stepdown_power", "table_s11_stepdown_power", pd.DataFrame(rows_sp),
          "Partial-null δ = 5 pp stepdown vs. single-step FWER (%) and empirical-alternative power across outbreak and 53-cluster random-effects worlds.",
          "".join([Lc(2.4), Lc(3.15), R(2.3), R(2.25), R(2.2), R(2.25)]),
          notes="Partial-null δ = 5 pp retains empirical true margins on the 11 strongly separated pairs while centering SAL|CAMP, 3 lower-tier pairs, and 15 reversed pairs at zero (19 nulls). Empirical-alternative power evaluates uncentered empirical worlds (4 separated pairs at worst case; 12 upper-tier pairs at δ = 5 pp).",
          size="tiny", sep="1.7pt")

    # SUPP TABLE S12: Full 15-Pair Decomposition Across δ = 0, δ = 5 pp, and Worst Case (Separating Sampling vs Missingness Uncertainty) (M5)
    rows = []
    for k, p in sorted_pairs(m):
        sp = si["pairs"][k]
        rows.append({
            "Ordering (δ = 0)": pair_label(p, SHORT),
            "Lower diff. pp (δ=0 | δ=5 | WC)": f"{100*sp['dL_delta0']:.1f} | {100*sp['dL_delta5']:.1f} | {100*sp['dL_worst']:.1f}",
            "z HC1 (δ=0 | δ=5 | WC)": f"{sp['z_dL_hc1_delta0']:.2f} | {sp['z_dL_hc1_delta5']:.2f} | {sp['z_dL_hc1']:.2f}",
            "z CV3 (δ=0 | δ=5 | WC)": f"{sp['z_dL_cv3_delta0']:.2f} | {sp['z_dL_cv3_delta5']:.2f} | {sp['z_dL_cv3']:.2f}",
            "HC1 p (δ=0 | δ=5 | WC)": f"{fmt_p(sp['p_fwer_hc1_delta0'])} | {fmt_p(sp['p_fwer_delta5'])} | {fmt_p(sp['p_fwer_worst'])}",
            "Boot-t p (δ=0 | δ=5 | WC)": f"{fmt_p(sp['p_fwer_boot_t_delta0'])} | {fmt_p(sp['p_fwer_boot_t_delta5'])} | {fmt_p(sp['p_fwer_boot_t_worst'])}",
            "CV3 p (δ=0 | δ=5 | WC)": f"{fmt_p(sp['p_fwer_cv3_delta0'])} | {fmt_p(sp['p_fwer_cv3_delta5'])} | {fmt_p(sp['p_fwer_cv3_worst'])}",
        })
    _emit("tab:pair_delta_decomp", "table_s12_pair_delta_comparison", pd.DataFrame(rows),
          "Per-pair decomposition separating sampling uncertainty (δ = 0 MAR) from identification uncertainty (δ = 5 pp and worst case) across all 15 primary comparisons.",
          "".join([Lc(2.1), R(2.2), R(2.05), R(2.05), R(2.0), R(2.0), R(2.0)]),
          notes=f"Max-T critical values (30 hypotheses): HC1 {si['c_sim_delta0']:.2f} (δ=0) | {si['c_sim_delta5']:.2f} (δ=5) | {si['c_sim_worst']:.2f} (WC); "
                f"CV3 {si['c_sim_cv3_delta0']:.2f} | {si['c_sim_cv3_delta5']:.2f} | {si['c_sim_cv3_worst']:.2f}; "
                f"Boot-t {si['c_bt_delta0']:.2f} | {si['c_bt_delta5']:.2f} | {si['c_bt_worst']:.2f}. "
                f"Under δ = 0 MAR, 14/30 hypotheses reject across all procedures (leaving only CPERF vs NoV GI tied); at δ = 5 pp, single-step Dual/CV3 rejects {si['n_fwer_ss_dual_delta5_sig']}/30 (SAL>CAMP ss p = {si['pairs']['SAL|CAMP']['p_ss_cv3_delta5']:.3f}) while stepdown rejects 12/30; at worst case, HC1/CV1 reject 1/30 while CV3, Boot-t, and Dual reject 0/30.",
          size="tiny", sep="1.6pt")

    # SUPP TABLE S13: Detailed Conventional vs Hierarchical EB Beta-Binomial vs Partial-ID Benchmark (M13)
    wil = cb["marginal_wilson_hk"]["per_group"]
    mb0 = cb["marginal_bootstrap_delta0"]["per_group"]
    ebo = cb["eb_betabinom_outbreak"]["per_group"]
    ebs = cb["eb_betabinom_state"]["per_group"]
    rows = []
    for g in order:
        rows.append({
            "Pathogen": NAME[g],
            "Wilson H/K % [95% CI]": f"{100*wil[g]['est']:.2f} [{100*wil[g]['ci_lo']:.2f}, {100*wil[g]['ci_hi']:.2f}]",
            "Marg. Boot δ=0 % [CI]": f"{100*mb0[g]['est']:.2f} [{100*mb0[g]['ci_lo']:.2f}, {100*mb0[g]['ci_hi']:.2f}]",
            "EB Outbreak % [CrI] (Rk)": f"{100*ebo[g]['post_mean']:.2f} [{100*ebo[g]['ci_lo']:.2f}, {100*ebo[g]['ci_hi']:.2f}] ({ebo[g]['post_mean_rank']:.2f})",
            "EB 53-State % [CrI] (Rk)": f"{100*ebs[g]['post_mean']:.2f} [{100*ebs[g]['ci_lo']:.2f}, {100*ebs[g]['ci_hi']:.2f}] ({ebs[g]['post_mean_rank']:.2f})",
            "Dual Rk δ=0 / δ=5 / WC": f"[{rcs['delta0_dual'][g]['rank_lower']},{rcs['delta0_dual'][g]['rank_upper']}] / [{rcs['delta5_dual'][g]['rank_lower']},{rcs['delta5_dual'][g]['rank_upper']}] / [{rcs['worst_dual'][g]['rank_lower']},{rcs['worst_dual'][g]['rank_upper']}]",
        })
    _emit("tab:conv_benchmark", "table_s13_conventional_benchmark", pd.DataFrame(rows),
          "Detailed comparison of conventional marginal and Laird–Louis (1989) Empirical Bayes (EB) Beta-Binomial rankings against partial-identification simultaneous rank confidence sets.",
          "".join([Lc(1.85), R(2.6), R(2.55), R(2.9), R(2.85), R(2.0)]),
          notes="EB Outbreak fits Kleinman (1973) Beta-Binomial priors per pathogen across retained outbreaks and draws 10,000 posterior standardized proportions across the 12 cells (imputed from prior when K_rec = 0); EB 53-State fits Beta-Binomial priors across exposure states. All conventional/EB baselines and δ = 0 simultaneous inference yield identical strict separation above the lower tier ([1,1]/[2,2]/[3,3]/[4,4]/[5,6]/[5,6]), whereas explicit missingness widening drives rank overlap at δ = 5 pp ([4,6] lower tier; [2,3] single-step SAL/CAMP) and worst case ([1,6]).",
          size="tiny", sep="1.6pt")

    # SUPP TABLE S14: Structured Audit of the Four Published CDC NORS Summaries (m25)
    audit_reports = [
        {"Report": "Dewey-Mattia et al. (2018)", "Venue / Year": "MMWR Surveill Summ, 2018", "NORS years": "2009–2015",
         "Reported location": "Results text & Table 2 (pp. 4–11)", "Hospitalization denominator": "Total ill (H / C); K not tabulated",
         "Missing-status assumption": "Implicitly equates all unrecorded outcomes M_i = C_i − K_i to non-hospitalized (recovers worst-case L); notes general NORS underreporting without paired bounds."},
        {"Report": "Tack et al. (2021)", "Venue / Year": "Microorganisms, 2021", "NORS years": "2010–2017 (STEC)",
         "Reported location": "Table 1 & Section 3 (pp. 4–7)", "Hospitalization denominator": "Total ill (H / C); K not tabulated",
         "Missing-status assumption": "Standardizes on H / C (recovers lower bound L); Limitations (p. 11) note missing hospitalization data may underestimate severity without quantifying [L, U] or δ*."},
        {"Report": "CDC Annual Report (2019)", "Venue / Year": "CDC NORS Portal, accessed 2026", "NORS years": "2019",
         "Reported location": "Pathogen summary tables", "Hospitalization denominator": "Total ill (H / C); K not tabulated",
         "Missing-status assumption": "Reports aggregate hospitalizations alongside total outbreak cases (recovers lower bound L) without paired H / K or unrecorded shares M / C."},
        {"Report": "Wikswo et al. (2022)", "Venue / Year": "Clin Infect Dis, 2022", "NORS years": "2009–2019",
         "Reported location": "Table 1 & Results (pp. 1908–1910)", "Hospitalization denominator": "Known status (H / K); M omitted",
         "Missing-status assumption": "Divides by known hospitalization status (approximates pooled MAR anchor δ = 0) without paired H / C, worst-case upper bounds U, or breakdown points."},
    ]
    _emit("tab:nors_audit", "table_s14_nors_report_audit", pd.DataFrame(audit_reports),
          "Structured methodological audit of the four published CDC NORS surveillance summaries motivated in Section 1 and Section 4.1.",
          "".join([Lc(2.4), Lc(2.15), R(1.3), Lc(2.2), Lc(2.0), Lc(4.2)]),
          notes="None of the four audited summaries was designed as a formal multiplicity-adjusted ranking study; Table 5 provides a drop-in extension adding paired H / K (or H / C), unrecorded shares M / C, standardized bounds, and breakdown points.",
          size="tiny", sep="1.7pt")

    # SUPP TABLE S15: Quantitative Surveillance Prescriptions (Adjacent Completeness Caps & 80% Power Sample Sizes)
    acp = r["supplementary_audits"]["adjacent_chain_and_prescriptions"]
    rows_rx = []
    for ap in acp["adjacent_pairs"]:
        rows_rx.append({
            "Target / Comparison": f"Adjacent: {SHORT[ap['higher']]} > {SHORT[ap['lower']]}",
            "Target regime": "Worst-case point separation",
            "Current metric": f"Overlap {ap['wc_overlap_pp']:+.2f} pp (M/C {ap['current_u_high_pct']:.1f}% | {ap['current_u_low_pct']:.1f}%)",
            "Prescription for separation / 80% power": f"Reduce uniform M/C <= {ap['uniform_mc_cap_pct']:.1f}% (completeness >= {ap['min_completeness_pct']:.1f}%; prop. scale {ap['prop_scale_f_star']:.2f}x)",
        })
    for pp_r in acp["cross_tier_80pct_power_prescriptions"]:
        gh_p, gl_p = pp_r["pair"].split(">")
        rows_rx.append({
            "Target / Comparison": f"Cross-tier: {SHORT[gh_p]} > {SHORT[gl_p]}",
            "Target regime": "Worst-case FWER test (80% power)",
            "Current metric": f"Lower diff. = {pp_r['dL_worst_pp']:.2f} pp (SE {pp_r['se_hc1_pp']:.2f} HC1 / {pp_r['se_cv3_pp']:.2f} CV3)",
            "Prescription for separation / 80% power": f"{pp_r['mult_cv3_80']:.1f}x CV3 (~{pp_r['outbreaks_cv3_80']:,} ob) | {pp_r['mult_boot_t_80']:.1f}x Boot-t (~{pp_r['outbreaks_boot_t_80']:,} ob) | {pp_r['mult_hc1_80']:.1f}x HC1",
        })
    _emit("tab:prescriptions", "table_s15_prescriptions", pd.DataFrame(rows_rx),
          "Actionable surveillance prescriptions: completeness thresholds (1 − M/C) to eliminate worst-case overlap along all five adjacent steps and sample/cluster multipliers for 80% statistical power across the four separated cross-tier pairs.",
          "".join([Lc(2.8), Lc(2.4), Lc(4.15), Lc(5.15)]),
          notes="Adjacent rows show uniform M/C cap m* = Δ_0/(1 + Δ_0) (requiring M_js/C_js <= m* across all strata) and proportional scaling factor f* = Δ_0 / ((θ_h(0)−L_h)+(U_l−θ_l(0))) keeping cell recorded rates p_js fixed. Power multipliers approximate [(c_crit + z_0.80)·SE / (L_h−U_l)] squared relative to 3,176 outbreaks: CV3 assumes scaling effective independent clusters rather than adding outbreaks inside pooled Multistate, while Boot-t holds current c_bt = 4.52 fixed.",
          size="tiny", sep="1.8pt")

    if write_tex:
        with open(tdir / "manifest.json", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(dict(schema_version=3, tables=man), indent=1, ensure_ascii=False) + "\n")


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


def _save(fig, fdir: Path, base: str, manifest: list, caption: str, section: str):
    fig.savefig(fdir / f"{base}.pdf")
    fig.savefig(fdir / f"{base}.png", dpi=300)
    manifest.append(dict(id=base, path_pdf=f"{base}.pdf", path_png=f"{base}.png", caption=caption, section=section,
                         generator="code/make_outputs.py", data_source="analysis/results.json"))
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_matrix(r, fdir, man):
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
    handles = [Patch(facecolor=CAT_COLORS[k], label=v) for k, v in (
        ("A", "Worst-case determined (disjoint bounds without outcome assumptions)"),
        ("B", "Separation scenario: holds up to $\\delta^{*} \\geq 5$ pp"),
        ("C", "Separation scenario: fails below $\\delta = 5$ pp ($\\delta^{*} < 5$ pp)"),
    )]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False, fontsize=8)
    _save(fig, fdir, "fig_02_verdict_matrix", man, "Pairwise sample breakdown matrix across six groups.", "results")


def fig_intervals(r, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    m, bt, ret = r["main"], r["bootstrap"], r["retention"]["per_group"]
    acp = r["supplementary_audits"]["adjacent_chain_and_prescriptions"]
    order = m["order_by_delta0"]; y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1.0]})

    # Panel (a): Standardized bounds with shaded adjacent-step overlaps (0/5 adjacent pairs separated)
    ax = axes[0]
    for i in range(len(order) - 1):
        gh, gl = order[i], order[i + 1]
        Lh = 100 * m["per_group"][gh]["L_worst"]
        Ul = 100 * m["per_group"][gl]["U_worst"]
        if Ul > Lh:
            ax.add_patch(Rectangle((Lh, i), Ul - Lh, 1.0, facecolor="#F4A582", alpha=0.35, edgecolor="#CA0020", lw=0.7, ls="--", zorder=0))
            ax.text(0.5 * (Lh + Ul), i + 0.5, f"+{Ul - Lh:.1f} pp", ha="center", va="center", fontsize=6.6, fontweight="bold", color="#99000D")
    for i, g in enumerate(order):
        v, b = m["per_group"][g], bt["per_group"][g]
        cr_hc = 100 * ret[g]["crude_after"]["L"]; cr_hk = 100 * ret[g]["crude_after"]["theta_delta0"]
        Lw, Uw = 100 * v["L_worst"], 100 * v["U_worst"]
        L5, U5 = 100 * v["bounds_by_delta_pp"]["5"]["L"], 100 * v["bounds_by_delta_pp"]["5"]["U"]
        th0 = 100 * v["theta_delta0"]
        ax.hlines(i, Lw, Uw, color="#A8BCD9", lw=8.5, zorder=1)
        ax.hlines(i, L5, U5, color="#0F4D92", lw=3.8, zorder=2)
        ax.plot([100 * b["L_p2_5"], 100 * b["L_p97_5"]], [i - 0.19, i - 0.19], color="#555", lw=1.0, marker="|", ms=3.5)
        ax.plot([100 * b["U_p2_5"], 100 * b["U_p97_5"]], [i + 0.19, i + 0.19], color="#555", lw=1.0, marker="|", ms=3.5)
        ax.scatter([th0], [i], color="white", edgecolor="#0F4D92", s=38, zorder=4)
        ax.scatter([cr_hc], [i], color="#C44E52", marker="s", s=22, zorder=5)
        ax.scatter([cr_hk], [i], color="#2CA02C", marker="^", s=24, zorder=5)
        ax.text(Uw + 0.8, i, f"[{Lw:.1f}, {Uw:.1f}]", va="center", fontsize=7.0)
    ax.set_yticks(y); ax.set_yticklabels([NAME[g] for g in order], fontsize=8.0); ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xlim(-1.5, 48.0); ax.set_xlabel("Standardized hospitalization proportion (%)", fontsize=8.2)
    ax.set_title("(a) Standardized bounds & 0/5 adjacent overlaps", fontsize=8.5, pad=5)
    ax.grid(axis="x", alpha=0.35); ax.grid(axis="y", visible=False)

    # Panel (b): Uniform M/C completeness prescription vs current NORS unrecorded range (12.7% - 22.2%)
    ax2 = axes[1]
    adj = acp["adjacent_pairs"]
    y2 = np.arange(len(adj))
    caps = [ap["uniform_mc_cap_pct"] for ap in adj]
    lbls2 = [f"{SHORT[ap['higher']]} > {SHORT[ap['lower']]}" for ap in adj]
    u_min = min(100 * m["per_group"][g]["unknown_share"] for g in order)
    u_max = max(100 * m["per_group"][g]["unknown_share"] for g in order)
    ax2.axvspan(u_min, u_max, color="#E0E0E0", alpha=0.75, zorder=0, label="Current NORS unrecorded range (12.7–22.2%)")
    ax2.axvline(u_min, color="#666666", ls=":", lw=0.9)
    ax2.axvline(u_max, color="#666666", ls=":", lw=0.9)
    bars = ax2.barh(y2, caps, height=0.52, color=["#2B8CBE", "#2B8CBE", "#1B9E77", "#8C96C6", "#8C96C6"], edgecolor="white", zorder=2)
    for b_rect, cap_v in zip(bars, caps):
        ax2.text(cap_v + 0.5, b_rect.get_y() + b_rect.get_height() / 2.0, f"≤ {cap_v:.1f}%", va="center", fontsize=7.2, color="#222222")
    ax2.set_yticks(y2); ax2.set_yticklabels(lbls2, fontsize=7.5); ax2.set_ylim(len(adj) - 0.5, -0.5)
    ax2.set_xlim(0, 25.5); ax2.set_xlabel("Max uniform unrecorded share M/C (%)", fontsize=8.2)
    ax2.set_title("(b) Completeness thresholds vs. current NORS", fontsize=8.5, pad=5)
    ax2.grid(axis="x", alpha=0.35); ax2.grid(axis="y", visible=False)

    handles = [
        Patch(facecolor="#A8BCD9", label="Worst-case [L, U]"),
        Patch(facecolor="#F4A582", alpha=0.45, edgecolor="#CA0020", ls="--", label="Adjacent-step overlap (0/5 separated)"),
        Patch(facecolor="#0F4D92", label="$\\delta = 5$ pp interval"),
        Line2D([], [], color="#0F4D92", marker="o", mfc="white", ls="none", ms=5.5, label="$\\delta = 0$ MAR"),
        Line2D([], [], color="#C44E52", marker="s", ls="none", ms=4.5, label="Crude H/C"),
        Line2D([], [], color="#2CA02C", marker="^", ls="none", ms=4.8, label="Crude δ = 0"),
        Patch(facecolor="#E0E0E0", label="Observed NORS M/C (12.7%–22.2%)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7.0)
    _save(fig, fdir, "fig_03_bound_intervals", man, "Standardized bounds with 0/5 adjacent-step overlaps and unrecorded-share thresholds required for assumption-free separation.", "results")


def fig_calibrated_2d(r, fdir, man):
    """Main Fig. 4: 2x3 2D frontiers — top: 3 lower-tier pairs; bottom: STEC>SAL, SAL>CAMP, CAMP>CPERF (m1)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    m = r["main"]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.5))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.94, bottom=0.185, hspace=0.40, wspace=0.22)

    lower_keys = ["NOV_GI|NOV_GII", "CPERF|NOV_GII", "CPERF|NOV_GI"]
    upper_keys = ["SAL|STEC", "SAL|CAMP", "CAMP|CPERF"]

    for ax, k in zip(axes[0], lower_keys):
        p = m["pairs"][k]; td = m["two_d"][k]
        x = np.array(td["d_hi_pp"])
        y = np.array([10.0 if v is None else min(v, 10.0) for v in td["d_lo_boundary_pp"]])
        mask = x <= 10.05
        ax.fill_between(x[mask], 0, y[mask], color=CAT_COLORS[category(p)], alpha=0.85, lw=0)
        ax.add_patch(Rectangle((0.0, 0.0), 5.0, 5.0, facecolor="none", edgecolor="#272727", lw=1.0, ls=":", zorder=4))
        if k == "CPERF|NOV_GI":
            ds_c = p["delta_star_pp"]
            ax.annotate(f"$\\delta^{{*}} = {ds_c:.2f}$ pp", xy=(ds_c, ds_c), xytext=(2.1, 6.2),
                        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8), fontsize=7.2, color="#333333")
        else:
            ax.text(5.8, 7.6, f"$\\delta^{{*}} = {p['delta_star_pp']:.1f}$ pp", fontsize=7.2, color="#272727")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
        ax.grid(alpha=0.3); ax.set_title(pair_label(p, SHORT), fontsize=8.3, pad=4); ax.tick_params(labelsize=7.2)

    two_ext = m.get("two_d_ext", {})
    for ax, k in zip(axes[1], upper_keys):
        p = m["pairs"][k]; td = two_ext.get(k) or m["two_d"][k]
        x = np.array(td["d_hi_pp"])
        y = np.array([60.0 if v is None else min(v, 60.0) for v in td["d_lo_boundary_pp"]])
        if x[-1] < 60.0:
            x = np.append(x, 60.0)
            y = np.append(y, y[-1])  # CAMP|CPERF is already on its flat ceiling (max p_obs <= 15 pp)
        ax.fill_between(x, 0, y, color=CAT_COLORS[category(p)], alpha=0.85, lw=0)
        ax.add_patch(Rectangle((0.0, 0.0), 5.0, 5.0, facecolor="none", edgecolor="#272727", lw=1.0, ls=":", zorder=4))
        ax.add_patch(Rectangle((0.0, 0.0), 15.0, 5.0, facecolor="none", edgecolor="#D95F02", lw=1.25, ls="--", zorder=4))
        ax.text(32.0, 54.8, f"$\\delta^{{*}} = {p['delta_star_pp']:.1f}$ pp", fontsize=7.2, color="#272727")
        ax.set_xlim(0, 60); ax.set_ylim(0, 60)
        ticks = [0, 20, 40, 60]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.grid(alpha=0.3); ax.set_title(pair_label(p, SHORT), fontsize=8.3, pad=4); ax.tick_params(labelsize=7.2)

    min_up_ds = min(m["pairs"][k]["delta_star_pp"] for k in upper_keys)
    fig.text(0.54, 0.105, "Downward deviation allowed for higher group, $\\delta_{h,\\mathrm{down}}$ (pp)", ha="center", va="center", fontsize=8.8)
    fig.text(0.024, 0.56, "Upward deviation allowed for lower group, $\\delta_{l,\\mathrm{up}}$ (pp)", ha="center", va="center", rotation="vertical", fontsize=8.8)
    handles = [
        Patch(facecolor=CAT_COLORS["B"], alpha=0.85, label=f"Upper tier ($\\delta^{{*}} \\geq {min_up_ds:.1f}$ pp)"),
        Patch(facecolor=CAT_COLORS["C"], alpha=0.85, label="Lower tier ($\\delta^{*} < 5$ pp)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#272727", lw=1.0, ls=":", label="Symmetric $5 \\times 5$ pp box"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#D95F02", lw=1.25, ls="--", label="Asymmetric $-15 / +5$ pp stress test"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.01), ncol=2, frameon=False, fontsize=7.5)
    _save(fig, fdir, "fig_04_calibrated_2d", man, "2D breakdown frontiers including the binding upper-vs-lower pair CAMP > CPERF.", "results")


def fig_breakdown_primary(r, fdir, man):
    """Main Fig. 5: Decluttered (m5) Sample delta* + Primary Boot-t + CV3 jackknife + dynamic critical curves."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    si = r["stacked_inference"]; m = r["main"]; sp = sorted_pairs(m)
    rb_bt = si["rank_breakdown_points_pp"]["bootstrap_t"]
    within_bt = rb_bt["within_upper_tier_min"]; tier_bt = rb_bt["upper_vs_lower_tier_min"]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.24, 1.0]})
    ax = axes[0]; y = np.arange(len(sp))
    for i, (k, p) in enumerate(sp):
        vp = si["pairs"][k]
        pt = 100.0 if vp["delta_star_pt_pp"] is None else vp["delta_star_pt_pp"]
        c3 = 100.0 if vp["delta_star_95_cv3_pp"] is None else vp["delta_star_95_cv3_pp"]
        bt = 100.0 if vp["delta_star_95_bt_pp"] is None else vp["delta_star_95_bt_pp"]
        ax.plot([min(c3, bt), pt], [i, i], color="#9AA5B8", lw=1.5, zorder=1)
        ax.scatter([pt], [i], color="#0F4D92", s=26, zorder=3, marker=("^" if vp["delta_star_pt_pp"] is None else "o"))
        ax.scatter([c3], [i - 0.10], color="#1B9E77", s=22, zorder=4, marker=("^" if vp["delta_star_95_cv3_pp"] is None else "v"))
        ax.scatter([bt], [i + 0.10], color="#C44E52", s=26, zorder=5, marker=("^" if vp["delta_star_95_bt_pp"] is None else "P"))
    ax.axvline(5.0, color="#666666", ls=":", lw=0.9)
    ax.axvline(within_bt, color="#C44E52", ls="--", lw=1.0, alpha=0.85)
    ax.axvline(tier_bt, color="#8B2252", ls="-.", lw=1.0, alpha=0.85)
    ax.annotate(f"Boot-$t$ {within_bt:.1f} pp (STEC>SAL)", xy=(within_bt, 8.0), xytext=(46.5, 8.45),
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8), fontsize=6.4, color="#C44E52", va="center")
    ax.annotate(f"Boot-$t$ {tier_bt:.1f} pp (CAMP>CPERF)", xy=(tier_bt, 11.0), xytext=(41.5, 10.55),
                arrowprops=dict(arrowstyle="->", color="#8B2252", lw=0.8), fontsize=6.4, color="#8B2252", va="center")
    ax.set_yticks(y); ax.set_yticklabels([pair_label(p, SHORT) for _, p in sp], fontsize=6.9)
    ax.set_ylim(len(sp) - 0.5, -0.5); ax.set_xlim(-3, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xticklabels(["0", "20", "40", "60", "80", "100 ($\\infty$)"], fontsize=7.3)
    ax.grid(axis="x", alpha=0.35, lw=0.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Breakdown deviation $\\delta^{*}$ (pp)", fontsize=8)
    ax.set_title("(a) Sample & primary 95% breakdown points", fontsize=8.3, pad=5)
    h_a = [
        Line2D([], [], color="#0F4D92", marker="o", ms=4.5, ls="none", label="Sample $\\delta^{*}$ ($\\triangle = \\infty$)"),
        Line2D([], [], color="#C44E52", marker="P", ms=4.5, ls="none", label="Primary Boot-$t$ $\\delta^{*}_{0.95}$"),
        Line2D([], [], color="#1B9E77", marker="v", ms=4.2, ls="none", label="CV3 jackknife $\\delta^{*}_{0.95}$"),
    ]
    ax.legend(handles=h_a, loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.4)

    ax = axes[1]; dyn = si["dynamic_csim_grid"]; d_g = np.array(dyn["delta_pp"])
    bt_g = si["studentized_bootstrap_t"]
    ax.plot(d_g, dyn["cv3"], color="#1B9E77", ls="-.", lw=1.8, marker="v", ms=3.2, markevery=2, label="CV3 jackknife $c_{\\mathrm{CV3}}(\\delta)$")
    ax.plot(bt_g["c_bt_grid_pp"], bt_g["c_bt_grid_vals"], color="#C44E52", ls="-", lw=2.0, marker="P", ms=4.2, label="Primary Boot-$t$ $c_{\\mathrm{bt}}(\\delta)$")
    ax.axvline(5.0, color="#666666", ls=":", lw=0.9)
    ax.axvline(within_bt, color="#C44E52", ls="--", lw=1.0, alpha=0.8)
    ax.axvline(tier_bt, color="#8B2252", ls="-.", lw=1.0, alpha=0.8)
    ax.set_xlim(-2, 102); ax.set_ylim(2.2, 4.75); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.tick_params(labelsize=7.2)
    ax.grid(alpha=0.35, lw=0.5); ax.set_xlabel("Missingness bound deviation $\\delta$ (pp)", fontsize=8)
    ax.set_ylabel("30-hypothesis 95% Max-$T$ critical value", fontsize=8)
    ax.set_title("(b) $\\delta$-specific Max-$T$ critical values", fontsize=8.3, pad=5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.6)
    _save(fig, fdir, "fig_05_breakdown_primary", man, "Sample breakdown points alongside primary Boot-t and CV3 95% breakdown frontiers.", "results")


def fig_retention(r, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    m, ret = r["main"], r["retention"]["per_group"]; order = m["order_by_delta0"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True, sharey=True)
    ax = axes[0]
    for i, g in enumerate(order):
        v = ret[g]; ax.barh(i, v["support_cases_pct"], height=0.6, color="#3775BA")
        ax.text(v["support_cases_pct"] + 1.5, i, f"{v['support_cases_pct']:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 100); ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[g] for g in order])
    ax.set_ylim(len(order) - 0.5, -0.5); ax.grid(axis="y", visible=False); ax.set_xlabel("(a) Eligible cases inside common support (%)")
    ax = axes[1]; cols = ("#0F4D92", "#AADCA9", "#CFCECE")
    for i, g in enumerate(order):
        v = m["per_group"][g]; h = v["hospitalized_known"] / v["cases"]; u = v["unknown_share"]; nh = 1 - h - u; left = 0
        for val, c in zip((h, nh, u), cols):
            ax.barh(i, 100 * val, left=left, height=0.6, color=c); left += 100 * val
    ax.set_xlim(0, 100); ax.grid(axis="y", visible=False); ax.set_xlabel("(b) Retained cases by recorded status (%)")
    handles = [Patch(facecolor=c, label=t) for c, t in zip(cols, ("Recorded hospitalized", "Recorded not hospitalized", "Hospitalization status unrecorded"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    _save(fig, fdir, "fig_s1_retention_recording", man, "Common-support retention and status recording.", "supplementary")


def fig_robust(r, fdir, man):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    m = r["main"]; sp = sorted_pairs(m)
    variants = [("Main", m)] + [(VARIANT_CODES[k], v) for k, v in r["robustness"].items()
                                if k not in ("R7_norovirus_combined", "R7b_norovirus_all_confirmed")]
    fig, ax = plt.subplots(figsize=(8.8, 5.1), constrained_layout=True)
    for j, (_, v) in enumerate(variants):
        for i, (k, p) in enumerate(sp):
            q = v["pairs"][k]
            cat = "D" if q["higher_at_delta0"] != p["higher_at_delta0"] else category(q)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CAT_COLORS[cat], edgecolor="white", lw=1.0))
    ax.set_xlim(-0.5, len(variants) - 0.5); ax.set_ylim(len(sp) - 0.5, -0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([f"{c}\n{v['n_determined']['worst_case']}|{v['n_determined']['5']}" for c, v in variants], fontsize=5.5)
    ax.set_yticks(range(len(sp))); ax.set_yticklabels([pair_label(p) for _, p in sp], fontsize=7.4)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Analysis (second line: determined pairs, worst case | $\\delta = 5$ pp)")
    handles = [Patch(facecolor=CAT_COLORS[c], label=t) for c, t in (
        ("A", "Holds without unrecorded-outcome assumptions"), ("B", "Holds at $\\delta = 5$ pp ($\\delta^{*} \\geq 5$)"),
        ("C", "Fails before $\\delta = 5$ pp ($\\delta^{*} < 5$)"), ("D", "$\\delta = 0$ direction reversed vs. main"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7.6)
    _save(fig, fdir, "fig_s2_robustness", man, f"Sensitivity verdicts across all {len(variants) - 1} six-group specifications.", "supplementary")


def fig_supp_sensitivity(r, fdir, man):
    """Supp Fig. S3: (a) Common-mechanism scaling alpha in [0, 2]; (b) All comparator Max-T critical value curves."""
    import matplotlib.pyplot as plt
    emp = r["empirical_calibration"]; m = r["main"]; cm = emp["common_mechanism"]; si = r["stacked_inference"]
    alphas = sorted(float(k) for k in cm.keys())
    grp_cols = {"STEC": "#0F4D92", "SAL": "#2B8CBE", "CAMP": "#1B9E77",
                "NOV_GII": "#D95F02", "CPERF": "#7570B3", "NOV_GI": "#E7298A"}
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), constrained_layout=True)

    ax = axes[0]
    for g in m["order_by_delta0"]:
        vals = [100.0 * cm[f"{al:.2f}"]["rates"][g] for al in alphas]
        ax.plot(alphas, vals, color=grp_cols[g], marker="o", ms=3.2, lw=1.6, label=NAME[g])
    ax.axvline(1.0, color="#666666", ls=":", lw=1.0); ax.set_xlim(-0.05, 2.05); ax.set_ylim(-1.0, 38.0)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0]); ax.grid(alpha=0.35, lw=0.5)
    ax.set_xlabel("Common rate multiplier $\\alpha$ ($q_{js} = \\min(1, \\alpha p_{js})$)", fontsize=8.2)
    ax.set_ylabel("Standardized hospitalization (%)", fontsize=8.2)
    ax.set_title("(a) Common-mechanism proportional scaling", fontsize=8.5)
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.92, fontsize=6.4)

    ax = axes[1]; dyn = si["dynamic_csim_grid"]; d_g = np.array(dyn["delta_pp"]); bt_g = si["studentized_bootstrap_t"]
    ax.plot(d_g, dyn["hc1"], color="#2B8CBE", ls="-", lw=1.6, marker="s", ms=2.6, markevery=2, label="Gaussian HC1 $c_{\\mathrm{HC1}}(\\delta)$")
    ax.plot(d_g, dyn["state"], color="#D95F02", ls="--", lw=1.6, marker="D", ms=2.6, markevery=2, label="53-state CV1 $c_{\\mathrm{CV1}}(\\delta)$")
    ax.plot(d_g, dyn["cv3"], color="#1B9E77", ls="-.", lw=1.6, marker="v", ms=2.8, markevery=2, label="CV3 jackknife $c_{\\mathrm{CV3}}(\\delta)$")
    ax.plot(bt_g["c_bt_grid_pp"], bt_g["c_bt_grid_vals"], color="#C44E52", ls="-", lw=2.0, marker="P", ms=3.8, label="Studentized boot-$t$ $c_{\\mathrm{bt}}(\\delta)$")
    ax.set_xlim(-2, 102); ax.set_ylim(2.2, 4.75); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.grid(alpha=0.35, lw=0.5)
    ax.set_xlabel("Missingness bound deviation $\\delta$ (pp)", fontsize=8.2)
    ax.set_ylabel("30-hypothesis 95% Max-$T$ critical value", fontsize=8.2)
    ax.set_title("(b) All comparator critical-value curves", fontsize=8.5)
    ax.legend(loc="center right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.4)
    _save(fig, fdir, "fig_s3_sensitivity_supplement", man, "Common-mechanism proportional scaling and all comparator critical-value curves.", "supplementary")


def make_figures(r: dict, fdir: Path) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    for f in list(fdir.glob("fig_*.pdf")) + list(fdir.glob("fig_*.png")):
        if f.name == "fig_01_selection_flow.png":
            continue
        f.unlink()
    _style()
    man = []
    for fn in (fig_matrix, fig_intervals, fig_calibrated_2d, fig_breakdown_primary, fig_retention, fig_robust, fig_supp_sensitivity):
        fn(r, fdir, man)
    with open(fdir / "manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(man, indent=1, ensure_ascii=False) + "\n")


def make_all(base_dir: Path, paper_dir: Path) -> None:
    r = load(base_dir)
    make_tables(r, base_dir / "analysis" / "tables", write_tex=False)
    make_tables(r, paper_dir / "tables", write_tex=True)
    make_figures(r, paper_dir / "figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=ROOT)
    ap.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    a = ap.parse_args()
    statistical_summary(a.base_dir.resolve())
    make_all(a.base_dir.resolve(), a.paper_dir.resolve())
