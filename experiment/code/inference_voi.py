"""Stacked M-estimation inference, cluster-concentration diagnostics, and VoI follow-up allocation.

Implements the unified methodological upgrade for the NORS partial-identification study:
1. Stratified outbreak-cluster stacked M-estimator (central-difference Jacobian sandwich with
   HC0 and HC1 finite-cluster scaling), verified against closed-form outbreak influence functions.
2. Imbens-Manski (2004) uniformly valid 95% confidence intervals for pairwise differences,
   Gaussian multiplier Max-T stepdown FWER p-values across all 15 pairs, and 95% confidence
   tipping points (pairwise and simultaneous FWER-adjusted delta*_0.95).
3. Cell-level cluster concentration diagnostics (Herfindahl-Hirschman index HHI, effective sample
   size N_eff = 1/HHI, and maximum single-outbreak share).
4. Value-of-Information (VoI) follow-up targeting policy evaluated via retrospective masking on
   all complete-status NORS outbreaks in the common support, plus real-data priority strata.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import bounds as B
import nors_rules as R
import strata as S


class StackedInference:
    """Stratified outbreak-cluster M-estimator over G groups x K common-support cells."""

    def __init__(self, main_df: pd.DataFrame, t_main: S.Table, ordered_pairs: list[tuple[int, int]]):
        self.groups = list(t_main.groups)
        self.cells = list(t_main.cells)
        self.G = len(self.groups)
        self.K = len(self.cells)
        self.t_main = t_main
        self.ordered_pairs = list(ordered_pairs)

        self.n = np.zeros((self.G, self.K), dtype=int)
        self.mu = np.zeros((self.G, self.K, 5), dtype=float)
        self.omega_hc0 = np.zeros((self.G, self.K, 5, 5), dtype=float)
        self.omega_hc1 = np.zeros((self.G, self.K, 5, 5), dtype=float)
        self.cell_arrays: dict[tuple[int, int], np.ndarray] = {}

        for gi, g in enumerate(self.groups):
            for ci, c in enumerate(self.cells):
                sub = main_df[(main_df["group"] == g) & (main_df["cell"] == c)]
                Z = sub[["C", "H", "U", "H_rec", "K_rec"]].to_numpy(dtype=float)
                m = Z.shape[0]
                self.n[gi, ci] = m
                self.cell_arrays[(gi, ci)] = Z
                mean_z = Z.mean(axis=0)
                self.mu[gi, ci] = mean_z
                centered = Z - mean_z
                S_zz = centered.T @ centered
                self.omega_hc0[gi, ci] = S_zz / (m * m)
                self.omega_hc1[gi, ci] = S_zz / (m * (m - 1)) if m > 1 else S_zz / (m * m)

    def table_from_mu(self, mu: np.ndarray) -> S.Table:
        totals = mu * self.n[:, :, None]
        t = S.Table(
            groups=self.groups,
            cells=self.cells,
            C=totals[:, :, 0],
            H=totals[:, :, 1],
            U=totals[:, :, 2],
            H_rec=totals[:, :, 3],
            K_rec=totals[:, :, 4],
            n=self.n.astype(float),
        )
        t.w = S.weights(t, "pooled_cases")
        return t

    def functionals(self, mu: np.ndarray, delta: float) -> np.ndarray:
        """Return [L_0..L_{G-1}, U_0..U_{G-1}, theta0_0..theta0_{G-1}, dL_0..dL_{P-1}, dU_0..dU_{P-1}]."""
        t = self.table_from_mu(mu)
        L = np.zeros(self.G)
        U = np.zeros(self.G)
        T0 = np.zeros(self.G)
        for i in range(self.G):
            l0, u0 = B.point_delta0(t, i)
            T0[i] = 0.5 * (l0 + u0)
            if delta >= 1.0:
                L[i], U[i] = B.worst_bounds(t, i)
            else:
                L[i], U[i] = B.group_bounds(t, i, delta)
        dL = [L[h] - U[l] for h, l in self.ordered_pairs]
        dU = [U[h] - L[l] for h, l in self.ordered_pairs]
        return np.concatenate([L, U, T0, np.asarray(dL), np.asarray(dU)])

    def sandwich_cov(self, delta: float, hc1: bool = True, rel_step: float = 1e-6):
        """Central-difference Jacobian sandwich covariance over the (G, K, 5) cell means."""
        theta_hat = self.functionals(self.mu, delta)
        q = theta_hat.size
        p = self.G * self.K * 5
        J = np.zeros((q, p))
        flat_mu = self.mu.reshape(-1)
        for idx in range(p):
            step = rel_step * max(1.0, abs(flat_mu[idx]))
            mp = flat_mu.copy()
            mm = flat_mu.copy()
            mp[idx] += step
            mm[idx] -= step
            fp = self.functionals(mp.reshape(self.G, self.K, 5), delta)
            fm = self.functionals(mm.reshape(self.G, self.K, 5), delta)
            J[:, idx] = (fp - fm) / (2.0 * step)

        omega = self.omega_hc1 if hc1 else self.omega_hc0
        V = np.zeros((q, q))
        for gi in range(self.G):
            for ci in range(self.K):
                s = (gi * self.K + ci) * 5
                Jg = J[:, s:s + 5]
                V += Jg @ omega[gi, ci] @ Jg.T
        return theta_hat, V, J

    def analytic_worst_case_cov(self, hc1: bool = True) -> np.ndarray:
        """Closed-form outbreak-level influence function covariance for worst-case (L_j, U_j)."""
        t = self.t_main
        C_tot = float(t.C.sum())
        rL = t.H / t.C
        rU = (t.H + t.U) / t.C
        L_bar = (rL * t.w[None, :]).sum(axis=1)
        U_bar = (rU * t.w[None, :]).sum(axis=1)

        V = np.zeros((2 * self.G, 2 * self.G))
        for ki in range(self.G):
            for ci in range(self.K):
                Z = self.cell_arrays[(ki, ci)]
                m = Z.shape[0]
                Cc = Z[:, 0] - Z[:, 0].mean()
                Hc = Z[:, 1] - Z[:, 1].mean()
                Uc = Z[:, 2] - Z[:, 2].mean()
                psi = np.zeros((m, 2 * self.G))
                for j in range(self.G):
                    psi[:, j] = ((rL[j, ci] - L_bar[j]) / C_tot) * Cc
                    psi[:, self.G + j] = ((rU[j, ci] - U_bar[j]) / C_tot) * Cc
                psi[:, ki] += t.w[ci] * (Hc - rL[ki, ci] * Cc) / t.C[ki, ci]
                psi[:, self.G + ki] += t.w[ci] * ((Hc + Uc) - rU[ki, ci] * Cc) / t.C[ki, ci]
                scale = m / (m - 1.0) if (hc1 and m > 1) else 1.0
                V += scale * (psi.T @ psi)
        return V


def imbens_manski_crit(width: float, se_L: float, se_U: float, alpha: float = 0.05) -> float:
    """Solve Phi(c + width / max(se_L, se_U)) - Phi(-c) = 1 - alpha (Imbens & Manski 2004)."""
    s_max = max(se_L, se_U, 1e-15)
    ratio = max(0.0, width) / s_max
    lo = float(norm.ppf(1.0 - alpha))
    hi = float(norm.ppf(1.0 - alpha / 2.0))
    if norm.cdf(lo + ratio) - norm.cdf(-lo) >= 1.0 - alpha:
        return lo
    return float(brentq(lambda c: norm.cdf(c + ratio) - norm.cdf(-c) - (1.0 - alpha), lo, hi))


def max_t_stepdown(t_stats: np.ndarray, cov_L: np.ndarray, alpha: float = 0.05,
                   n_sim: int = 50000, seed: int = 20261125):
    """Gaussian multiplier stepdown FWER p-values and simultaneous 1-sided critical value."""
    P = len(t_stats)
    sd = np.sqrt(np.maximum(np.diag(cov_L), 1e-20))
    corr = cov_L / np.outer(sd, sd)
    corr = 0.5 * (corr + corr.T)
    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 0.0)
    L_chol = vecs @ np.diag(np.sqrt(vals))

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sim, P)) @ L_chol.T
    c_sim_all = float(np.quantile(Z.max(axis=1), 1.0 - alpha))

    order = np.argsort(-t_stats, kind="mergesort")
    adj_p = np.zeros(P)
    prev_p = 0.0
    for step in range(P):
        active = order[step:]
        max_z = Z[:, active].max(axis=1)
        p_raw = float(np.mean(max_z >= t_stats[order[step]]))
        p_mon = max(prev_p, p_raw)
        adj_p[order[step]] = p_mon
        prev_p = p_mon
    return adj_p, c_sim_all


def confidence_tipping_point(inf: StackedInference, pair_idx: int, z_crit: float) -> float | None:
    """Smallest delta (pp) where lower confidence limit dL(delta) - z_crit * SE(dL(delta)) <= 0."""
    off = 3 * inf.G

    def margin(d: float) -> float:
        th, V, _ = inf.sandwich_cov(d, hc1=True)
        dL = th[off + pair_idx]
        seL = math.sqrt(max(V[off + pair_idx, off + pair_idx], 0.0))
        return dL - z_crit * seL

    if margin(1.0) > 0.0:
        return None
    if margin(0.0) <= 0.0:
        return 0.0
    return 100.0 * float(brentq(margin, 0.0, 1.0, xtol=1e-5))


def cluster_diagnostics(main_df: pd.DataFrame, t_main: S.Table) -> OrderedDict:
    """Compute outbreak-cluster HHI, effective clusters N_eff = 1/HHI, and max single-outbreak share."""
    cell_rows = []
    per_group = OrderedDict()
    for g in t_main.groups:
        g_rows = []
        for c in t_main.cells:
            sub = main_df[(main_df["group"] == g) & (main_df["cell"] == c)]
            shares = sub["C"].to_numpy(dtype=float) / float(sub["C"].sum())
            hhi = float(np.sum(shares ** 2))
            msh = float(shares.max())
            rec = OrderedDict(
                group=g, cell=c, n_outbreaks=int(len(sub)), cases=float(sub["C"].sum()),
                hhi=hhi, n_eff=float(1.0 / hhi), max_share=msh,
            )
            cell_rows.append(rec)
            g_rows.append(rec)
        hhis = [r["hhi"] for r in g_rows]
        neffs = [r["n_eff"] for r in g_rows]
        ns = [r["n_outbreaks"] for r in g_rows]
        mshs = [r["max_share"] for r in g_rows]
        max_cell = max(g_rows, key=lambda r: r["hhi"])
        per_group[g] = OrderedDict(
            min_n=int(min(ns)), median_n=float(np.median(ns)), max_n=int(max(ns)),
            mean_hhi=float(np.mean(hhis)), max_hhi=float(max(hhis)),
            min_neff=float(min(neffs)), max_single_share=float(max(mshs)),
            most_concentrated_cell=max_cell["cell"],
        )
    return OrderedDict(per_group=per_group, cells=cell_rows)


def voi_scores_for_table(tb: S.Table) -> np.ndarray:
    """Marginal expected overlap reduction per case followed up across unresolved pairs."""
    G, K = len(tb.groups), len(tb.cells)
    Lw = np.zeros(G); Uw = np.zeros(G); th0 = np.zeros(G)
    for i in range(G):
        Lw[i], Uw[i] = B.worst_bounds(tb, i)
        l0, u0 = B.point_delta0(tb, i)
        th0[i] = 0.5 * (l0 + u0)
    p_hat = np.where(np.isnan(tb.p_obs), 0.0, tb.p_obs)
    cell_val = np.zeros((G, K))
    for a in range(G):
        for b in range(a + 1, G):
            if Lw[a] > Uw[b] or Lw[b] > Uw[a]:
                continue
            h, l = (a, b) if th0[a] >= th0[b] else (b, a)
            overlap = Uw[l] - Lw[h]
            urgency = 1.0 / (overlap + 0.015)
            cell_val[h] += urgency * tb.w * p_hat[h] / tb.C[h]
            cell_val[l] += urgency * tb.w * (1.0 - p_hat[l]) / tb.C[l]
    return cell_val


def run_masking_experiment(main_df: pd.DataFrame, t_main: S.Table,
                           budgets=(250, 500, 1000, 1500, 2000, 3500, 5000),
                           n_trials: int = 200, seed: int = 20261125) -> OrderedDict:
    """Retrospective masking experiment on all complete-status outbreaks in the common support."""
    sup_df = main_df[main_df["cell"].isin(t_main.cells)].copy()
    complete_df = sup_df[sup_df["U"] == 0].copy().reset_index(drop=True)
    t_true = S.build(complete_df, R.GROUPS, "setting_period", fixed_cells=t_main.cells)
    gi_map = {g: i for i, g in enumerate(t_true.groups)}
    ci_map = {c: i for i, c in enumerate(t_true.cells)}
    true_rates = OrderedDict((g, float(np.sum(t_true.w * t_true.H[i] / t_true.C[i])))
                             for i, g in enumerate(R.GROUPS))

    emp_inc_prob = {}
    for g in R.GROUPS:
        for c in t_main.cells:
            sub = sup_df[(sup_df["group"] == g) & (sup_df["cell"] == c)]
            p_mask = float((sub["U"] > 0).mean())
            emp_inc_prob[(g, c)] = min(max(p_mask, 0.12), 0.45)

    C_arr = complete_df["C"].to_numpy(dtype=float)
    H_true = complete_df["H"].to_numpy(dtype=float)
    g_idx = np.array([gi_map[g] for g in complete_df["group"]], dtype=int)
    c_idx = np.array([ci_map[c] for c in complete_df["cell"]], dtype=int)
    mask_probs = np.array([emp_inc_prob[(g, c)] for g, c in zip(complete_df["group"], complete_df["cell"])])

    G, K = len(R.GROUPS), len(t_main.cells)
    C_cell = t_true.C.copy()
    w_vec = t_true.w.copy()
    policies = ("random", "volume_greedy", "width_greedy", "voi_disambiguation")
    all_b = (0,) + tuple(budgets)
    results = {pol: {b: {"worst_det": [], "overlap_sum": [], "delta5_det": []} for b in all_b}
               for pol in policies}

    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        masked = rng.random(len(complete_df)) < mask_probs
        for gi in range(G):
            for ci in range(K):
                idx_gc = np.where((g_idx == gi) & (c_idx == ci))[0]
                if np.all(masked[idx_gc]):
                    masked[rng.choice(idx_gc)] = False

        def eval_state(cur_masked: np.ndarray):
            H_obs = np.where(cur_masked, 0.0, H_true)
            U_obs = np.where(cur_masked, C_arr, 0.0)
            K_obs = np.where(cur_masked, 0.0, C_arr)
            H_mat = np.zeros((G, K)); U_mat = np.zeros((G, K)); K_mat = np.zeros((G, K))
            np.add.at(H_mat, (g_idx, c_idx), H_obs)
            np.add.at(U_mat, (g_idx, c_idx), U_obs)
            np.add.at(K_mat, (g_idx, c_idx), K_obs)
            tb = S.Table(groups=list(R.GROUPS), cells=list(t_main.cells),
                         C=C_cell, H=H_mat, U=U_mat, H_rec=H_mat, K_rec=K_mat, n=t_true.n, w=w_vec)
            Lw = np.zeros(G); Uw = np.zeros(G); L5 = np.zeros(G); U5 = np.zeros(G); th0 = np.zeros(G)
            for i in range(G):
                Lw[i], Uw[i] = B.worst_bounds(tb, i)
                L5[i], U5[i] = B.group_bounds(tb, i, 0.05)
                l0, u0 = B.point_delta0(tb, i)
                th0[i] = 0.5 * (l0 + u0)
            w_det = 0; d5_det = 0; ov_sum = 0.0
            for i in range(G):
                for j in range(i + 1, G):
                    if Lw[i] > Uw[j] or Lw[j] > Uw[i]:
                        w_det += 1
                    else:
                        h, l = (i, j) if th0[i] >= th0[j] else (j, i)
                        ov_sum += (Uw[l] - Lw[h])
                    if L5[i] > U5[j] or L5[j] > U5[i]:
                        d5_det += 1
            return w_det, d5_det, ov_sum, tb

        w0, d50, ov0, tb0 = eval_state(masked)
        m_indices = np.where(masked)[0]

        for pol in policies:
            results[pol][0]["worst_det"].append(w0)
            results[pol][0]["delta5_det"].append(d50)
            results[pol][0]["overlap_sum"].append(100.0 * ov0)

            if pol == "random":
                order = rng.permutation(m_indices)
            elif pol == "volume_greedy":
                order = m_indices[np.argsort(-C_arr[m_indices], kind="mergesort")]
            elif pol == "width_greedy":
                score = w_vec[c_idx[m_indices]] / C_cell[g_idx[m_indices], c_idx[m_indices]]
                order = m_indices[np.argsort(-score, kind="mergesort")]
            else:
                cell_val = voi_scores_for_table(tb0)
                score = cell_val[g_idx[m_indices], c_idx[m_indices]]
                order = m_indices[np.argsort(-score, kind="mergesort")]

            cum_cases = np.cumsum(C_arr[order])
            for b in budgets:
                k_take = int(np.searchsorted(cum_cases, b, side="right"))
                cur_m = masked.copy()
                cur_m[order[:k_take]] = False
                wd, d5d, ov, _ = eval_state(cur_m)
                results[pol][b]["worst_det"].append(wd)
                results[pol][b]["delta5_det"].append(d5d)
                results[pol][b]["overlap_sum"].append(100.0 * ov)

    summary = OrderedDict()
    for pol in policies:
        summary[pol] = OrderedDict()
        for b in all_b:
            wd_arr = np.asarray(results[pol][b]["worst_det"], dtype=float)
            d5_arr = np.asarray(results[pol][b]["delta5_det"], dtype=float)
            ov_arr = np.asarray(results[pol][b]["overlap_sum"], dtype=float)
            summary[pol][str(b)] = OrderedDict(
                mean_worst_det=float(wd_arr.mean()),
                sd_worst_det=float(wd_arr.std(ddof=1)),
                mean_delta5_det=float(d5_arr.mean()),
                sd_delta5_det=float(d5_arr.std(ddof=1)),
                mean_overlap_pp=float(ov_arr.mean()),
                sd_overlap_pp=float(ov_arr.std(ddof=1)),
            )

    # Real-data top priority strata on the actual NORS incomplete records
    real_scores = voi_scores_for_table(t_main)
    real_top = []
    for gi, g in enumerate(t_main.groups):
        for ci, c in enumerate(t_main.cells):
            unk = float(t_main.U[gi, ci])
            if unk > 0:
                real_top.append(OrderedDict(
                    group=g, cell=c, unrecorded_cases=unk,
                    total_cases=float(t_main.C[gi, ci]), weight=float(t_main.w[ci]),
                    voi_per_case_pp=100.0 * float(real_scores[gi, ci]),
                ))
    real_top.sort(key=lambda r: -r["voi_per_case_pp"])

    return OrderedDict(
        n_trials=n_trials, seed=seed, budgets_cases=list(all_b),
        n_complete_outbreaks=int(len(complete_df)), total_complete_cases=int(C_arr.sum()),
        true_rates=true_rates, summary=summary, real_data_top_strata=real_top[:10],
    )


def run_all(main_df: pd.DataFrame, t_main: S.Table, main_res: dict, seed: int = 20261125) -> OrderedDict:
    """Run stacked inference, HHI cluster diagnostics, and VoI masking validation."""
    sorted_p = sorted(
        main_res["pairs"].items(),
        key=lambda kv: -(np.inf if kv[1]["delta_star_pp"] is None else kv[1]["delta_star_pp"]),
    )
    gi_map = {g: i for i, g in enumerate(t_main.groups)}
    ordered_pairs = [(gi_map[p["higher_at_delta0"]], gi_map[p["lower_at_delta0"]]) for _, p in sorted_p]

    inf = StackedInference(main_df, t_main, ordered_pairs)
    th_hc1, V_hc1, _ = inf.sandwich_cov(1.0, hc1=True)
    _, V_hc0, _ = inf.sandwich_cov(1.0, hc1=False)
    V_ana = inf.analytic_worst_case_cov(hc1=True)
    G = inf.G
    P = len(ordered_pairs)
    max_if_diff = float(np.max(np.abs(V_hc1[:2 * G, :2 * G] - V_ana)))

    per_group = OrderedDict()
    for i, g in enumerate(t_main.groups):
        per_group[g] = OrderedDict(
            se_L_hc0=float(math.sqrt(max(V_hc0[i, i], 0.0))),
            se_L_hc1=float(math.sqrt(max(V_hc1[i, i], 0.0))),
            se_U_hc0=float(math.sqrt(max(V_hc0[G + i, G + i], 0.0))),
            se_U_hc1=float(math.sqrt(max(V_hc1[G + i, G + i], 0.0))),
            se_theta0_hc1=float(math.sqrt(max(V_hc1[2 * G + i, 2 * G + i], 0.0))),
        )

    cov_dL_wc = V_hc1[3 * G:3 * G + P, 3 * G:3 * G + P]
    dL_wc = th_hc1[3 * G:3 * G + P]
    dU_wc = th_hc1[3 * G + P:3 * G + 2 * P]
    seL_wc = np.sqrt(np.maximum(np.diag(cov_dL_wc), 1e-20))
    seU_wc = np.sqrt(np.maximum(np.diag(V_hc1[3 * G + P:3 * G + 2 * P, 3 * G + P:3 * G + 2 * P]), 1e-20))
    t_wc = dL_wc / seL_wc
    p_fwer_wc, c_sim_wc = max_t_stepdown(t_wc, cov_dL_wc, alpha=0.05, n_sim=50000, seed=seed)

    th_d5, V_d5, _ = inf.sandwich_cov(0.05, hc1=True)
    cov_dL_d5 = V_d5[3 * G:3 * G + P, 3 * G:3 * G + P]
    dL_d5 = th_d5[3 * G:3 * G + P]
    seL_d5 = np.sqrt(np.maximum(np.diag(cov_dL_d5), 1e-20))
    t_d5 = dL_d5 / seL_d5
    p_fwer_d5, c_sim_d5 = max_t_stepdown(t_d5, cov_dL_d5, alpha=0.05, n_sim=50000, seed=seed)

    z_pw = float(norm.ppf(0.95))
    pairs = OrderedDict()
    for idx, (key, p) in enumerate(sorted_p):
        c_im = imbens_manski_crit(float(dU_wc[idx] - dL_wc[idx]), float(seL_wc[idx]), float(seU_wc[idx]), 0.05)
        ci_lo = float(dL_wc[idx] - c_im * seL_wc[idx])
        ci_hi = float(dU_wc[idx] + c_im * seU_wc[idx])
        ds_pw = confidence_tipping_point(inf, idx, z_pw)
        ds_sim = confidence_tipping_point(inf, idx, c_sim_d5)
        pairs[key] = OrderedDict(
            higher_at_delta0=p["higher_at_delta0"],
            lower_at_delta0=p["lower_at_delta0"],
            dL_worst=float(dL_wc[idx]),
            dU_worst=float(dU_wc[idx]),
            se_dL_hc1=float(seL_wc[idx]),
            se_dU_hc1=float(seU_wc[idx]),
            c_im=float(c_im),
            ci95_im_lo=ci_lo,
            ci95_im_hi=ci_hi,
            im_excludes_zero=bool(ci_lo > 0.0),
            p_fwer_worst=float(p_fwer_wc[idx]),
            p_fwer_delta5=float(p_fwer_d5[idx]),
            delta_star_pt_pp=p["delta_star_pp"],
            delta_star_95_pw_pp=ds_pw,
            delta_star_95_sim_pp=ds_sim,
        )

    clust = cluster_diagnostics(main_df, t_main)
    voi = run_masking_experiment(main_df, t_main, n_trials=200, seed=seed)

    return OrderedDict(
        stacked_inference=OrderedDict(
            alpha=0.05,
            n_multiplier_draws=50000,
            c_sim_worst=float(c_sim_wc),
            c_sim_delta5=float(c_sim_d5),
            c_bonferroni_one_sided=float(norm.ppf(1.0 - 0.05 / P)),
            n_im_excludes_zero=sum(v["im_excludes_zero"] for v in pairs.values()),
            n_fwer_worst_sig=sum(v["p_fwer_worst"] < 0.05 for v in pairs.values()),
            n_fwer_delta5_sig=sum(v["p_fwer_delta5"] < 0.05 for v in pairs.values()),
            per_group=per_group,
            pairs=pairs,
            checks=OrderedDict(max_abs_diff_analytic_if_vs_sandwich=max_if_diff),
        ),
        cluster_diagnostics=clust,
        voi_masking=voi,
    )
