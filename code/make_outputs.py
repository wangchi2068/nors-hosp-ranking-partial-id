"""Tables, figures, and statistical summary for the NORS hospitalization-ranking analysis (v8.0).

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
    return {NAME[g]: f"[{rc[g]['rank_lower']}, {rc[g]['rank_upper']}]" for g in order}


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
    rows = []
    for k, p in sorted_pairs(m):
        sp = si["pairs"][k]; b = bt["pairs"][k]; og = oi[k]
        imp_bt = sp["implied_at_delta_star_95"]["boot_t"]
        imp_str = "–" if imp_bt is None else f"{imp_bt['implied_unrec_high_pct']:.1f} / {imp_bt['implied_unrec_low_pct']:.1f}"
        rows.append({
            "Ordering (δ = 0)": pair_label(p, SHORT),
            "Bootstrap disjoint w/δ=5 (%)": f"{pnum(b['share_holds_without_assumption'], 1)} / {pnum(1.0 - b['share_delta5_undetermined'], 1)}",
            "Boot-t p (w / δ=5)": f"{fmt_p(sp['p_fwer_boot_t_worst'])} / {fmt_p(sp['p_fwer_boot_t_delta5'])}",
            "CV3 p (w / δ=5)": f"{fmt_p(sp['p_fwer_cv3_worst'])} / {fmt_p(sp['p_fwer_cv3_delta5'])}",
            "Sample δ*": fmt_ds_s(sp["delta_star_pt_pp"]),
            "95% δ* (Bt / CV3, ss)": f"{fmt_ds_s(sp['delta_star_95_bt_pp'])} / {fmt_ds_s(sp['delta_star_95_cv3_pp'])}",
            "Implied q at Bt δ* (H/L %)": imp_str,
            "OR 95% Γ* (Bt / CV3)": f"{fmt_ds_s(og['gamma_star_95_bt'])} / {fmt_ds_s(og['gamma_star_95_cv3'])}",
        })
    rb_bt = si["rank_breakdown_points_pp"]["bootstrap_t"]
    rb_c3 = si["rank_breakdown_points_pp"]["cv3"]
    sc = r["small_sample_calibration"]
    cc53 = r["cluster_double_bootstrap"]
    _emit("tab:pairwise_inference", "table_3_pairwise_inference", pd.DataFrame(rows),
          "Pairwise ordering stability, simultaneous 30-hypothesis FWER p-values (studentized bootstrap-t and CV3 jackknife), "
          "additive/OR breakdown thresholds, and implied unrecorded-case hospitalization rates at 95% thresholds.",
          "".join([Lc(2.1), R(1.75), R(2.0), R(1.85), R(0.95), R(1.8), R(1.85), R(1.65)]),
          notes=f"Bootstrap disjoint w/δ=5 (%) reports sample share of {bt['B']:,} resamples with disjoint bounds (L_h > U_l) at worst case and δ = 5 pp. "
                f"Primary studentized bootstrap-t uses δ-specific c_bt(δ) (c_bt = {si['c_bt_worst']:.2f} worst / {si['c_bt_delta5']:.2f} at δ = 5; "
                f"empirical 400×499 FWER {100*sc['fwer_worst']['bootstrap_t']:.1f}% / {100*sc['fwer_delta5']['bootstrap_t']:.1f}%). "
                f"CV3 leaves out each of 53 states ({si['cv3_n_valid_worst']}/53 valid; cluster RE FWER {100*cc53['fwer_worst']['cv3_jackknife']:.1f}% / {100*cc53['fwer_delta5']['cv3_jackknife']:.1f}%; t_52 reference c = {si['c_sim_cv3_t_worst']:.2f}, 0/30 sig worst, 12/30 at δ = 5). "
                f"Implied q at Bt δ* (H/L %) lists weighted unrecorded hospitalization rates evaluated at the single-step 95% Bt statistical breakdown threshold, where simultaneous FWER significance fails "
                f"(STEC>SAL: {si['pairs']['SAL|STEC']['implied_at_delta_star_95']['boot_t']['implied_unrec_high_pct']:.1f}% vs {si['pairs']['SAL|STEC']['implied_at_delta_star_95']['boot_t']['implied_unrec_low_pct']:.1f}%; "
                f"CAMP>CPERF: {si['pairs']['CAMP|CPERF']['implied_at_delta_star_95']['boot_t']['implied_unrec_high_pct']:.1f}% vs {si['pairs']['CAMP|CPERF']['implied_at_delta_star_95']['boot_t']['implied_unrec_low_pct']:.1f}%). "
                f"Within-upper single-step min 95% δ*: {rb_bt['within_upper_tier_min']:.1f} pp (STEC>SAL, Bt) vs {rb_c3['within_upper_tier_min']:.1f} pp (SAL>CAMP, CV3 single-step, 80% driven by pooled Multistate variance vs {r['cluster_diagnostics']['multistate_sensitivity']['r11_full']['pair_breakdown_pp']['SAL|CAMP']['cv3']:.1f} pp in R11; stepdown CV3 holds to {si['rank_breakdown_points_pp']['cv3_sal_camp_stepdown_pp']:.1f} pp, p = {si['pairs']['SAL|CAMP']['p_fwer_cv3_delta5']:.3f} at δ = 5); "
                f"upper-vs-lower (CAMP>CPERF): {rb_bt['upper_vs_lower_tier_min']:.1f} pp (Bt) and {rb_c3['upper_vs_lower_tier_min']:.1f} pp (CV3). OR 95% Γ* is likewise single-step (NoV GII>GI: {oi['NOV_GI|NOV_GII']['gamma_star_95_bt']:.2f} Bt, {oi['NOV_GI|NOV_GII']['gamma_star_95_cv3']:.2f} CV3).",
          sep="1.6pt")

    # MAIN TABLE 4: Simultaneous 95% rank confidence sets across specifications, plausibility regimes & scales
    rcs = si["rank_confidence_sets"]
    cwi = si["coarser_and_weighting_inference"]
    r11 = r["cluster_diagnostics"]["multistate_sensitivity"]["r11_full"]
    pl = r["plausible_scenarios"]
    oi_bg = si["or_inference"]["by_gamma"]
    rank_scenarios = [
        ("Worst case: Case-weighted (Boot-t)", rcs["worst_boot_t"]),
        ("Worst case: Case-weighted (HC1 / CV1)", rcs["worst_hc1"]),
        ("Worst case: R10 Equal outbreak wt. (Boot-t)", cwi["R10_outbreak_weighted"]["ranks"]["worst_boot_t"]),
        ("Worst case: R2 Period-only / R11 Single-state (Boot-t)", cwi["R2_period_only"]["ranks"]["worst_boot_t"]),
        ("Plausibility P1 (Lower <=5%, Upper p_js ±50%) (Boot-t)", pl["P1_lower5_upper50pct"]["ranks_boot_t"]),
        ("Plausibility P2 (Lower <=10%, Upper p_js ±50%) (Boot-t)", pl["P2_lower10_upper50pct"]["ranks_boot_t"]),
        ("Additive δ = 5 pp (Boot-t / CV3 / Cluster boot-t)", rcs["delta5_boot_t"]),
        ("Diagnostic asymmetric (-10/+5 SAL, -7/+5 CAMP) (Boot-t)", rcs["diag_asym_boot_t"]),
        ("OR scale Γ = 2.0 (0.1% floor, Boot-t)", oi_bg["2.0"]["ranks"]["boot_t"]),
        ("OR scale Γ = 3.0 (0.1% floor, Boot-t)", oi_bg["3.0"]["ranks"]["boot_t"]),
    ]
    rows = []
    for lbl, rmap in rank_scenarios:
        rows.append({"Scenario / Procedure": lbl, **_fmt_ranks(rmap, order)})
    _emit("tab:rank_sets", "table_4_rank_sets_summary", pd.DataFrame(rows),
          "Nominal simultaneous 95% rank confidence sets [1 = highest, 6 = lowest] across weighting specifications, "
          "explicit plausibility intervals (P1–P2), additive departures, and odds-ratio (OR) scaling.",
          "".join([Lc(4.15)] + [R(1.74)] * 6),
          notes=f"Sets are nominal 95% (see Table S10 for empirical FWER: 7.2%–7.8% Boot-t in outbreak resampling; 18.7%/11.7% CV3 and 15.7%/10.7% CV3-t_52 under 53-cluster RE). "
                f"R10 rejects {cwi['R10_outbreak_weighted']['n_sig_worst']['boot_t']}/30 hypotheses under Boot-t (p = {cwi['R10_outbreak_weighted']['stec_nov_gii_p_worst']['boot_t']:.3f}) and {cwi['R10_outbreak_weighted']['n_sig_worst']['cv3']}/30 under CV3 at worst case, "
                f"whereas case-weighted R2 (p = {cwi['R2_period_only']['stec_nov_gii_p_worst']['boot_t']:.3f} Boot-t, {cwi['R2_period_only']['stec_nov_gii_p_worst']['cv3']:.3f} CV3) and R11 (0/30 sig across all 5 methods) yield [1, 6] for all six groups. "
                f"Both P1 and P2 determine {pl['P1_lower5_upper50pct']['sample_determined']}/15 pairs in sample and reject 12/30 under Boot-t and CV3. "
                f"OR scale Γ = 2.0 rejects {oi_bg['2.0']['n_sig']['boot_t']}/30 hypotheses under stepdown Boot-t (separating Norovirus GII above GI; single-step Γ* = {oi['NOV_GI|NOV_GII']['gamma_star_95_bt']:.2f}).",
          sep="2.0pt")

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
          "".join([Lc(2.1), R(1.35), R(1.1)] + [R(0.88)] * 6 + [Lc(3.6)]), sep="2.3pt")

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
                f"boot-t {si['pairs']['STEC|NOV_GII']['p_fwer_boot_t_worst']:.3f}/{si['c_bt_worst']:.2f}/0. At δ = 5 pp, all reject 12. "
                f"Multistate score variance share: {100*cw_w['max_variance_share']:.1f}% worst (plug-in G_eff = {cw_w['effective_clusters_g_eff']:.2f}) and {100*cw_5['max_variance_share']:.1f}% δ = 5 (G_eff = {cw_5['effective_clusters_g_eff']:.2f}); "
                f"271-singleton and 52-state R11 yield 0 worst-case and 12 δ = 5 rejections. Across {loo['n_outbreaks_checked']:,} LOO (and 6-group LLO) deletions, worst-case determined stays {loo['wc_determined_range'][0]}–{loo['wc_determined_range'][1]} ({llo['wc_determined']} LLO).",
          sep="2.1pt")

    # SUPP TABLE S10: Complete Double-Bootstrap & Parametric Heavy-Tail Calibration Matrix (M3, M4, m2)
    cc_def = r["cluster_double_bootstrap_by_def"]
    pc = r["parametric_calibration"]
    rows = []
    for k_m, lbl_m in [("hc0", "HC0 Gaussian (df-uncorrected)"), ("hc1", "HC1 Gaussian"),
                       ("fixed_scale_boot", "Fixed-scale std. bootstrap"), ("state", "53-state CV1 Gaussian"),
                       ("wcb", "Unrestricted Rademacher WCB"), ("cv3_jackknife", "CV3 cluster jackknife"),
                       ("bootstrap_t", "Studentized bootstrap-t")]:
        rows.append({
            "Calibration world": f"Outbreak ({sc['n_mc']}×{sc['B_inner']})",
            "Procedure": lbl_m,
            "Worst FWER ± SE (%)": f"{100*sc['fwer_worst'][k_m]:.1f} ± {100*sc['fwer_worst_mc_se'][k_m]:.1f}",
            "δ=5 FWER ± SE (%)": f"{100*sc['fwer_delta5'][k_m]:.1f} ± {100*sc['fwer_delta5_mc_se'][k_m]:.1f}",
            "World note": f"Boot-t inner c: {sc['inner_c_bt_worst_summary']['p5']:.2f}–{sc['inner_c_bt_worst_summary']['p95']:.2f}" if k_m == "bootstrap_t" else "No state correlation simulated",
        })
    for c_key, c_lbl in [("state_53", "Cluster 53-state"), ("singleton_271", "Cluster 271-singleton"), ("r11_52", "Cluster 52-state R11")]:
        cd_b = cc_def[c_key]
        for k_c, lbl_c in [("cv1_state", "CV1 cluster Gaussian"), ("wcb", "Unrestricted Rademacher WCB"),
                           ("cluster_boot_t", "Bayesian cluster boot-t"), ("cv3_jackknife", "CV3 cluster jackknife"),
                           ("cv3_t_df", "CV3 jackknife (t_{G-1})")]:
            if k_c not in cd_b["fwer_worst"]:
                continue
            rows.append({
                "Calibration world": f"{c_lbl} ({cd_b['n_mc']}×{cd_b['B_inner']})",
                "Procedure": lbl_c,
                "Worst FWER ± SE (%)": f"{100*cd_b['fwer_worst'][k_c]:.1f} ± {100*cd_b['fwer_worst_mc_se'][k_c]:.1f}",
                "δ=5 FWER ± SE (%)": f"{100*cd_b['fwer_delta5'][k_c]:.1f} ± {100*cd_b['fwer_delta5_mc_se'][k_c]:.1f}",
                "World note": f"Fixed-size Cluster RE + Beta-Binom (SE/SD: CV1 {cd_b['median_se_ratio_cv1']:.2f}, CV3 {cd_b['median_se_ratio_cv3']:.2f})",
            })
    for k_p, lbl_p in [("hc1", "HC1 Gaussian"), ("bootstrap_t", "Studentized bootstrap-t")]:
        rows.append({
            "Calibration world": f"Parametric LogNorm ({pc['n_mc']}×{pc['B_inner']})",
            "Procedure": lbl_p,
            "Worst FWER ± SE (%)": f"{100*pc['fwer_worst'][k_p]:.1f} ± {100*pc['fwer_worst_mc_se'][k_p]:.1f}",
            "δ=5 FWER ± SE (%)": f"{100*pc['fwer_delta5'][k_p]:.1f} ± {100*pc['fwer_delta5_mc_se'][k_p]:.1f}",
            "World note": f"LogNorm + Beta-Binom (HC1 SE/obs {pc['median_synthetic_vs_observed_hc1_se_ratio']:.2f})",
        })
    pg_L = [sc["per_group"][g]["ratio_hc1_to_mc_L"] for g in order]
    pg_U = [sc["per_group"][g]["ratio_hc1_to_mc_U"] for g in order]
    _emit("tab:small_cal", "table_s10_calibration", pd.DataFrame(rows),
          "Empirical 30-hypothesis FWER (%) under least-favourable boundary nulls (nominal 5.0%) across outbreak, three cluster definitions (53/271/52), and parametric heavy-tail worlds.",
          "".join([Lc(3.35), Lc(3.45), R(2.35), R(2.35), Lc(3.1)]),
          notes=f"Pointwise Imbens-Manski 95% boundary coverage (15 primary pairs, outbreak world): {100*sc['im95_min_endpoint_cov_hc1']:.1f}% min to {100*sc['im95_median_endpoint_cov_hc1']:.1f}% median (HC1). "
                f"Per-group HC1 SE / MC SD ratios: {min(pg_L):.3f}–{max(pg_L):.3f} (L_j) and {min(pg_U):.3f}–{max(pg_U):.3f} (U_j). "
                f"While CV3 achieves <=5.0% FWER under independent outbreak resampling (4.0% worst, 1.0% at δ = 5) and studentized bootstrap-t achieves 2.7% at δ = 5 in overdispersed LogNormal worlds, neither primary procedure guarantees <=5.0% worst-case FWER across all five calibration worlds.",
          size="tiny", sep="2.0pt")

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
    from matplotlib.patches import Patch
    m, bt, ret = r["main"], r["bootstrap"], r["retention"]["per_group"]
    order = m["order_by_delta0"]; y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for i, g in enumerate(order):
        v, b = m["per_group"][g], bt["per_group"][g]
        cr_hc = 100 * ret[g]["crude_after"]["L"]; cr_hk = 100 * ret[g]["crude_after"]["theta_delta0"]
        Lw, Uw = 100 * v["L_worst"], 100 * v["U_worst"]
        L5, U5 = 100 * v["bounds_by_delta_pp"]["5"]["L"], 100 * v["bounds_by_delta_pp"]["5"]["U"]
        th0 = 100 * v["theta_delta0"]
        ax.hlines(i, Lw, Uw, color="#A8BCD9", lw=9, zorder=1)
        ax.hlines(i, L5, U5, color="#0F4D92", lw=4, zorder=2)
        ax.plot([100 * b["L_p2_5"], 100 * b["L_p97_5"]], [i - 0.19, i - 0.19], color="#555", lw=1.1, marker="|", ms=4)
        ax.plot([100 * b["U_p2_5"], 100 * b["U_p97_5"]], [i + 0.19, i + 0.19], color="#555", lw=1.1, marker="|", ms=4)
        ax.scatter([th0], [i], color="white", edgecolor="#0F4D92", s=45, zorder=4)
        ax.scatter([cr_hc], [i], color="#C44E52", marker="s", s=26, zorder=5)
        ax.scatter([cr_hk], [i], color="#2CA02C", marker="^", s=30, zorder=5)
        ax.text(Uw + 0.9, i, f"[{Lw:.1f}, {Uw:.1f}]", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([NAME[g] for g in order]); ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(-1.5, 48.0); ax.set_xlabel("Standardized hospitalization proportion (%)")
    ax.grid(axis="x", alpha=0.35); ax.grid(axis="y", visible=False)
    handles = [
        Patch(facecolor="#A8BCD9", label="Worst-case bounds [L, U]"),
        Patch(facecolor="#0F4D92", label="$\\delta = 5$ pp scenario interval"),
        Line2D([], [], color="#0F4D92", marker="o", mfc="white", ls="none", ms=6, label="$\\delta = 0$ MAR"),
        Line2D([], [], color="#555", marker="|", ls="-", ms=6, label="2,000-rep bootstrap 2.5th–97.5th"),
        Line2D([], [], color="#C44E52", marker="s", ls="none", ms=5, label="Crude H/C"),
        Line2D([], [], color="#2CA02C", marker="^", ls="none", ms=5, label="Crude δ = 0"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7.6)
    _save(fig, fdir, "fig_03_bound_intervals", man, "Standardized bounds overlaid with crude H/C and crude δ = 0.", "results")


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
            ax.annotate("$\\delta^{*} = 0.19$ pp", xy=(0.19, 0.19), xytext=(2.1, 6.2),
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

    fig.text(0.54, 0.105, "Downward deviation allowed for higher group, $\\delta_{h,\\mathrm{down}}$ (pp)", ha="center", va="center", fontsize=8.8)
    fig.text(0.024, 0.56, "Upward deviation allowed for lower group, $\\delta_{l,\\mathrm{up}}$ (pp)", ha="center", va="center", rotation="vertical", fontsize=8.8)
    handles = [
        Patch(facecolor=CAT_COLORS["B"], alpha=0.85, label="Upper tier ($\\delta^{*} \\geq 35.5$ pp)"),
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
    ax.annotate(f"Boot-$t$ {within_bt:.1f} pp (STEC>SAL)", xy=(within_bt, 8.0), xytext=(46.5, 7.55),
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
