"""Stacked M-estimation, 30-hypothesis two-sided inference, state clustering, calibration, and cost-conditional follow-up (v4.0).

Implements the full v4.0 methodological suite:
1. Closed-form outbreak-level influence functions for arbitrary symmetric or asymmetric delta bands
   (delta_down, delta_up) in [0, 1], verified against central-difference Jacobians.
2. Outbreak-HC1 and 53-state-clustered sandwich covariances + 9,999-draw studentized Rademacher
   wild cluster bootstrap over all 30 two-sided pairwise hypotheses (plus 15-directed check).
3. Imbens-Manski 95% CIs, 200,000-draw Gaussian multiplier Max-T stepdown (30 hypotheses),
   dynamic c_sim_30(delta) 95% confidence tipping points, and simultaneous 95% rank confidence
   sets at worst case, delta = 5 pp, and asymmetric (-15/+5 pp).
4. Stratified bootstrap-world calibration (500 draws preserving n_js) evaluating HC0, HC1,
   53-state clustering, 53-state wild cluster bootstrap, and non-Gaussian bootstrap critical values.
5. Within-data empirical missingness decomposition (crude vs within-cell standardized, multistate vs
   single-state stratified, outbreak size), asymmetric stress tests, and OR scaling (pure & 0.1% floor).
6. Unified cost-normalized width-reduction follow-up benchmark across per-case, per-outbreak, and
   mixed-cost budgets alongside heuristic DPI and non-MCAR/adaptive sensitivity checks.
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
    """Stratified outbreak-cluster and state-clustered M-estimator over G groups x K cells."""

    def __init__(self, main_df: pd.DataFrame, t_main: S.Table, ordered_pairs: list[tuple[int, int]],
                 fixed_w: np.ndarray | None = None):
        self.groups = list(t_main.groups)
        self.cells = list(t_main.cells)
        self.G = len(self.groups)
        self.K = len(self.cells)
        self.P = len(ordered_pairs)
        self.t_main = t_main
        self.ordered_pairs = list(ordered_pairs)
        self.fixed_w = None if fixed_w is None else np.asarray(fixed_w, dtype=float)

        self.n = np.zeros((self.G, self.K), dtype=int)
        self.mu = np.zeros((self.G, self.K, 5), dtype=float)
        self.omega_hc0 = np.zeros((self.G, self.K, 5, 5), dtype=float)
        self.omega_hc1 = np.zeros((self.G, self.K, 5, 5), dtype=float)
        self.cell_arrays: dict[tuple[int, int], np.ndarray] = {}
        self.cell_states: dict[tuple[int, int], np.ndarray] = {}

        sup_df = main_df[main_df["cell"].isin(self.cells)].copy()
        self.unique_states = sorted(sup_df["state"].astype(str).unique().tolist())
        self.state_map = {st: idx for idx, st in enumerate(self.unique_states)}
        self.n_states = len(self.unique_states)

        for gi, g in enumerate(self.groups):
            for ci, c in enumerate(self.cells):
                sub = sup_df[(sup_df["group"] == g) & (sup_df["cell"] == c)]
                Z = sub[["C", "H", "U", "H_rec", "K_rec"]].to_numpy(dtype=float)
                m = Z.shape[0]
                self.n[gi, ci] = m
                self.cell_arrays[(gi, ci)] = Z
                self.cell_states[(gi, ci)] = np.array(
                    [self.state_map[str(s)] for s in sub["state"]], dtype=int
                )
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
        t.w = self.fixed_w.copy() if self.fixed_w is not None else S.weights(t, "pooled_cases")
        return t

    def functionals(self, mu: np.ndarray, delta: float, d_up: float | None = None) -> np.ndarray:
        """Return [L_0..L_{G-1}, U_0..U_{G-1}, theta0_0..theta0_{G-1}, dL_0..dL_{P-1}, dU_0..dU_{P-1}]."""
        du = delta if d_up is None else d_up
        t = self.table_from_mu(mu)
        L = np.zeros(self.G)
        U = np.zeros(self.G)
        T0 = np.zeros(self.G)
        for i in range(self.G):
            l0, u0 = B.point_delta0(t, i)
            T0[i] = 0.5 * (l0 + u0)
            if delta >= 1.0 and du >= 1.0:
                L[i], U[i] = B.worst_bounds(t, i)
            else:
                L[i], U[i] = B.group_bounds_asym(t, i, min(delta, 1.0), min(du, 1.0))
        dL = [L[h] - U[l] for h, l in self.ordered_pairs]
        dU = [U[h] - L[l] for h, l in self.ordered_pairs]
        return np.concatenate([L, U, T0, np.asarray(dL), np.asarray(dU)])

    def sandwich_cov(self, delta: float, hc1: bool = True, rel_step: float = 1e-6, d_up: float | None = None):
        """Central-difference Jacobian sandwich covariance over the (G, K, 5) cell means."""
        theta_hat = self.functionals(self.mu, delta, d_up=d_up)
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
            fp = self.functionals(mp.reshape(self.G, self.K, 5), delta, d_up=d_up)
            fm = self.functionals(mm.reshape(self.G, self.K, 5), delta, d_up=d_up)
            J[:, idx] = (fp - fm) / (2.0 * step)

        omega = self.omega_hc1 if hc1 else self.omega_hc0
        V = np.zeros((q, q))
        for gi in range(self.G):
            for ci in range(self.K):
                s = (gi * self.K + ci) * 5
                Jg = J[:, s:s + 5]
                V += Jg @ omega[gi, ci] @ Jg.T
        return theta_hat, V, J

    def analytic_covariances(self, delta: float, return_state_scores: bool = False, d_up: float | None = None):
        """Fast closed-form influence covariance at any (delta_down, delta_up) in [0, 1]^2."""
        du = delta if d_up is None else d_up
        t = self.t_main
        G, K, P = self.G, self.K, self.P
        q = 3 * G + 2 * P
        C_tot = float(t.C.sum())
        w = t.w
        p_obs = t.p_obs
        undef = np.isnan(p_obs)
        pp = np.where(undef, 0.0, p_obs)

        if delta >= 1.0:
            qL = np.zeros((G, K))
            dL_dp = np.zeros((G, K))
        else:
            qL = np.where(undef, 0.0, np.maximum(0.0, pp - delta))
            dL_dp = np.where((~undef) & (pp > delta), 1.0, 0.0)

        if du >= 1.0:
            qU = np.ones((G, K))
            dU_dp = np.zeros((G, K))
        else:
            qU = np.where(undef, 1.0, np.minimum(1.0, pp + du))
            dU_dp = np.where((~undef) & (pp < 1.0 - du), 1.0, 0.0)

        q0 = np.where(undef, 0.5, pp)
        d0_dp = np.where(~undef, 1.0, 0.0)

        rL = (t.H + t.U * qL) / t.C
        rU = (t.H + t.U * qU) / t.C
        r0 = (t.H + t.U * q0) / t.C

        L_bar = rL @ w
        U_bar = rU @ w
        T0_bar = r0 @ w
        dL_hat = np.array([L_bar[h] - U_bar[l] for h, l in self.ordered_pairs])
        dU_hat = np.array([U_bar[h] - L_bar[l] for h, l in self.ordered_pairs])
        theta_hat = np.concatenate([L_bar, U_bar, T0_bar, dL_hat, dU_hat])

        V_hc0 = np.zeros((q, q))
        V_hc1 = np.zeros((q, q))
        state_scores = np.zeros((self.n_states, q))

        pair_h = np.array([h for h, _ in self.ordered_pairs], dtype=int)
        pair_l = np.array([l for _, l in self.ordered_pairs], dtype=int)

        w_L_coeff = np.zeros((G, K)) if self.fixed_w is not None else (rL - L_bar[:, None]) / C_tot
        w_U_coeff = np.zeros((G, K)) if self.fixed_w is not None else (rU - U_bar[:, None]) / C_tot
        w_0_coeff = np.zeros((G, K)) if self.fixed_w is not None else (r0 - T0_bar[:, None]) / C_tot

        for ki in range(G):
            for ci in range(K):
                Z = self.cell_arrays[(ki, ci)]
                m = Z.shape[0]
                Zc = Z - self.mu[ki, ci]
                Cc, Hc, Uc, Hrc, Krc = Zc[:, 0], Zc[:, 1], Zc[:, 2], Zc[:, 3], Zc[:, 4]

                psi = np.zeros((m, q))
                psi[:, :G] = Cc[:, None] * w_L_coeff[:, ci][None, :]
                psi[:, G:2 * G] = Cc[:, None] * w_U_coeff[:, ci][None, :]
                psi[:, 2 * G:3 * G] = Cc[:, None] * w_0_coeff[:, ci][None, :]

                k_rec = float(t.K_rec[ki, ci])
                p_tilde = (Hrc - pp[ki, ci] * Krc) / k_rec if k_rec > 0.0 else np.zeros(m)
                fac = w[ci] / float(t.C[ki, ci])

                psi[:, ki] += fac * (
                    Hc + qL[ki, ci] * Uc + dL_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rL[ki, ci] * Cc
                )
                psi[:, G + ki] += fac * (
                    Hc + qU[ki, ci] * Uc + dU_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rU[ki, ci] * Cc
                )
                psi[:, 2 * G + ki] += fac * (
                    Hc + q0[ki, ci] * Uc + d0_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - r0[ki, ci] * Cc
                )

                psi[:, 3 * G:3 * G + P] = psi[:, pair_h] - psi[:, G + pair_l]
                psi[:, 3 * G + P:] = psi[:, G + pair_h] - psi[:, pair_l]

                gram = psi.T @ psi
                V_hc0 += gram
                scale = (m / (m - 1.0)) if m > 1 else 1.0
                V_hc1 += scale * gram
                np.add.at(state_scores, self.cell_states[(ki, ci)], psi)

        st_scale = (self.n_states / (self.n_states - 1.0)) if self.n_states > 1 else 1.0
        V_state = st_scale * (state_scores.T @ state_scores)

        if return_state_scores:
            return theta_hat, V_hc1, V_hc0, V_state, state_scores
        return theta_hat, V_hc1, V_hc0, V_state


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
                   n_sim: int = 200000, seed: int = 20261125, Z_std: np.ndarray | None = None):
    """Gaussian multiplier stepdown FWER p-values, Monte Carlo SEs, and Max-T critical value."""
    M_hyp = len(t_stats)
    sd = np.sqrt(np.maximum(np.diag(cov_L), 1e-20))
    corr = cov_L / np.outer(sd, sd)
    corr = 0.5 * (corr + corr.T)
    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 0.0)
    L_chol = vecs @ np.diag(np.sqrt(vals))

    if Z_std is None:
        rng = np.random.default_rng(seed)
        Z_std = rng.standard_normal((n_sim, M_hyp))
    Z = Z_std @ L_chol.T
    M = Z.shape[0]
    c_sim_all = float(np.quantile(Z.max(axis=1), 1.0 - alpha))

    order = np.argsort(-t_stats, kind="mergesort")
    adj_p = np.zeros(M_hyp)
    mc_se = np.zeros(M_hyp)
    prev_p = 0.0
    prev_se = 0.0
    for step in range(M_hyp):
        active = order[step:]
        max_z = Z[:, active].max(axis=1)
        p_raw = float(np.mean(max_z >= t_stats[order[step]]))
        se_raw = math.sqrt(max(p_raw * (1.0 - p_raw), 0.0) / M)
        if p_raw >= prev_p:
            adj_p[order[step]] = p_raw
            mc_se[order[step]] = se_raw
            prev_p = p_raw
            prev_se = se_raw
        else:
            adj_p[order[step]] = prev_p
            mc_se[order[step]] = prev_se
    return adj_p, mc_se, c_sim_all


def wild_cluster_bootstrap_fwer(theta_dL: np.ndarray, scores_dL: np.ndarray,
                                alpha: float = 0.05, B_wcb: int = 9999,
                                seed: int = 20261125) -> OrderedDict:
    """Cameron-Gelbach-Miller (2008) studentized Rademacher wild cluster bootstrap across states."""
    G_st, M_hyp = scores_dL.shape
    scale = math.sqrt(G_st / (G_st - 1.0)) if G_st > 1 else 1.0
    se_obs = np.sqrt(np.maximum(scale * scale * np.sum(scores_dL ** 2, axis=0), 1e-20))
    t_obs = theta_dL / se_obs

    rng = np.random.default_rng(seed)
    rad = rng.choice(np.array([-1.0, 1.0]), size=(B_wcb, G_st))
    boot_diff = scale * (rad @ scores_dL)

    # Vectorized across (B_wcb, G_st, M_hyp) via sum of squares:
    # var_centered = sum_g (w_g s_gp - mean_g)^2 = sum_g s_gp^2 - G_st * (mean_g)^2
    sum_sq = np.sum(scores_dL ** 2, axis=0)[None, :]  # (1, M_hyp)
    mean_term = (rad @ scores_dL) / G_st             # (B_wcb, M_hyp)
    ss_centered = np.maximum(sum_sq - G_st * (mean_term ** 2), 1e-24)
    boot_se = scale * np.sqrt(ss_centered)
    T_boot = boot_diff / boot_se
    c_wcb = float(np.quantile(T_boot.max(axis=1), 1.0 - alpha))

    order = np.argsort(-t_obs, kind="mergesort")
    adj_p = np.zeros(M_hyp)
    prev_p = 0.0
    for step in range(M_hyp):
        active = order[step:]
        max_t = T_boot[:, active].max(axis=1)
        p_raw = float((1.0 + np.sum(max_t >= t_obs[order[step]])) / (B_wcb + 1.0))
        prev_p = max(prev_p, p_raw)
        adj_p[order[step]] = prev_p

    return OrderedDict(
        B_wcb=int(B_wcb),
        n_state_clusters=int(G_st),
        c_wcb_sim=float(c_wcb),
        se_state_pp=(100.0 * se_obs).tolist(),
        p_fwer_wcb=adj_p.tolist(),
    )


def dynamic_csim_grid(inf: StackedInference, cov_kind: str, Z_std: np.ndarray,
                      alpha: float = 0.05, n_grid: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Compute Max-T critical value curve c_sim(delta) on delta in [0, 1] over all hypotheses."""
    grid = np.linspace(0.0, 1.0, n_grid)
    c_vals = np.zeros(n_grid)
    off = 3 * inf.G
    M_hyp = inf.P
    dummy_t = np.zeros(M_hyp)
    for k, d in enumerate(grid):
        _, V_hc1, _, V_st = inf.analytic_covariances(float(d))
        V = V_hc1 if cov_kind == "hc1" else V_st
        cov_dL = V[off:off + M_hyp, off:off + M_hyp]
        _, _, c_vals[k] = max_t_stepdown(dummy_t, cov_dL, alpha=alpha, Z_std=Z_std)
    return grid, c_vals


def confidence_tipping_point_dynamic(inf: StackedInference, pair_idx: int, cov_kind: str,
                                     grid_d: np.ndarray | None = None, grid_c: np.ndarray | None = None,
                                     fixed_c: float | None = None) -> tuple[float | None, float | None]:
    """Smallest delta (pp) where dL(delta) - c(delta) * SE(dL(delta)) <= 0."""
    off = 3 * inf.G

    def crit(d: float) -> float:
        if fixed_c is not None:
            return fixed_c
        return float(np.interp(d, grid_d, grid_c))

    def margin(d: float) -> float:
        th, V_hc1, _, V_st = inf.analytic_covariances(d)
        V = V_hc1 if cov_kind == "hc1" else V_st
        dL = th[off + pair_idx]
        seL = math.sqrt(max(V[off + pair_idx, off + pair_idx], 0.0))
        return dL - crit(d) * seL

    if margin(1.0) > 0.0:
        return None, crit(1.0)
    if margin(0.0) <= 0.0:
        return 0.0, crit(0.0)
    root = float(brentq(margin, 0.0, 1.0, xtol=1e-6))
    return 100.0 * root, crit(root)


def rank_confidence_sets(groups: list[str], ordered_pairs: list[tuple[int, int]],
                         p_fwer: np.ndarray | list[float], alpha: float = 0.05) -> OrderedDict:
    """Simultaneous (1 - alpha) confidence set for each pathogen's rank via joint two-sided tests."""
    G = len(groups)
    out = OrderedDict()
    for j, g in enumerate(groups):
        n_above = 0
        n_below = 0
        for idx, (h, l) in enumerate(ordered_pairs):
            if p_fwer[idx] <= alpha:
                if l == j:
                    n_above += 1
                elif h == j:
                    n_below += 1
        r_lo = 1 + n_above
        r_hi = G - n_below
        out[g] = OrderedDict(
            rank_lower=int(r_lo),
            rank_upper=int(r_hi),
            n_significantly_higher=int(n_above),
            n_significantly_lower=int(n_below),
        )
    return out


def cluster_diagnostics(main_df: pd.DataFrame, t_main: S.Table) -> OrderedDict:
    """Compute outbreak-cluster HHI, N_eff, max share, and jurisdiction missingness concentration."""
    sup_df = main_df[main_df["cell"].isin(t_main.cells)].copy()
    cell_rows = []
    per_group = OrderedDict()
    for g in t_main.groups:
        g_rows = []
        for c in t_main.cells:
            sub = sup_df[(sup_df["group"] == g) & (sup_df["cell"] == c)]
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
        sub_g = sup_df[sup_df["group"] == g]
        u_by_st = sub_g.groupby("state")["U"].sum().sort_values(ascending=False)
        u_tot_g = float(sub_g["U"].sum())
        top3_share = float(u_by_st.iloc[:3].sum() / u_tot_g) if u_tot_g > 0 else 0.0
        multi_share = float(u_by_st.get("Multistate", 0.0) / u_tot_g) if u_tot_g > 0 else 0.0
        top3_named = [OrderedDict(state=str(st), share_pct=float(100.0 * val / u_tot_g))
                      for st, val in u_by_st.iloc[:3].items()] if u_tot_g > 0 else []
        per_group[g] = OrderedDict(
            min_n=int(min(ns)), median_n=float(np.median(ns)), max_n=int(max(ns)),
            mean_hhi=float(np.mean(hhis)), max_hhi=float(max(hhis)),
            min_neff=float(min(neffs)), max_single_share=float(max(mshs)),
            most_concentrated_cell=max_cell["cell"],
            multistate_unrecorded_share=multi_share,
            top3_exposure_states=top3_named,
            top3_states_unrecorded_share=top3_share,
        )

    st_grp = sup_df.groupby("state").agg(C=("C", "sum"), U=("U", "sum"), n=("C", "size"))
    st_500 = st_grp[st_grp["C"] >= 500].copy()
    st_500["u_share"] = st_500["U"] / st_500["C"]
    state_summary = OrderedDict(
        n_states_common_support=int(len(st_grp)),
        n_states_ge_500_cases=int(len(st_500)),
        unrecorded_share_min_ge500=float(st_500["u_share"].min()),
        unrecorded_share_median_ge500=float(st_500["u_share"].median()),
        unrecorded_share_max_ge500=float(st_500["u_share"].max()),
    )
    return OrderedDict(per_group=per_group, cells=cell_rows, state_missingness=state_summary)


def _eval_resampled_table(inf: StackedInference, res_arrays: dict, delta_val: float,
                          res_states: dict | None = None):
    """Fast vectorized closed-form evaluation of bounds, HC1/HC0, and optional state scores."""
    G, K, P = inf.G, inf.K, inf.P
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    sums = np.zeros((G, K, 5))
    mu_r = np.zeros((G, K, 5))
    for gi in range(G):
        for ci in range(K):
            arr = res_arrays[(gi, ci)]
            mu_r[gi, ci] = arr.mean(axis=0)
            sums[gi, ci] = arr.sum(axis=0)
    tb = S.Table(groups=inf.groups, cells=inf.cells, C=sums[:, :, 0], H=sums[:, :, 1],
                 U=sums[:, :, 2], H_rec=sums[:, :, 3], K_rec=sums[:, :, 4], n=inf.n.astype(float))
    tb.w = S.weights(tb, "pooled_cases")
    q = 3 * G + 2 * P
    off = 3 * G
    C_tot = float(tb.C.sum())
    w = tb.w
    p_obs = tb.p_obs
    undef = np.isnan(p_obs)
    pp = np.where(undef, 0.0, p_obs)
    if delta_val >= 1.0:
        qL = np.zeros((G, K)); qU = np.ones((G, K))
        dL_dp = np.zeros((G, K)); dU_dp = np.zeros((G, K))
    else:
        qL = np.where(undef, 0.0, np.maximum(0.0, pp - delta_val))
        qU = np.where(undef, 1.0, np.minimum(1.0, pp + delta_val))
        dL_dp = np.where((~undef) & (pp > delta_val), 1.0, 0.0)
        dU_dp = np.where((~undef) & (pp < 1.0 - delta_val), 1.0, 0.0)
    rL = (tb.H + tb.U * qL) / tb.C
    rU = (tb.H + tb.U * qU) / tb.C
    L_bar = rL @ w
    U_bar = rU @ w
    dL_hat = L_bar[pair_h] - U_bar[pair_l]
    dU_hat = U_bar[pair_h] - L_bar[pair_l]

    V_hc0 = np.zeros((q, q))
    V_hc1 = np.zeros((q, q))
    state_scores = np.zeros((inf.n_states, P)) if res_states is not None else None
    w_L_coeff = (rL - L_bar[:, None]) / C_tot
    w_U_coeff = (rU - U_bar[:, None]) / C_tot

    for ki in range(G):
        for ci in range(K):
            Z = res_arrays[(ki, ci)]
            m = Z.shape[0]
            Zc = Z - mu_r[ki, ci]
            Cc, Hc, Uc, Hrc, Krc = Zc[:, 0], Zc[:, 1], Zc[:, 2], Zc[:, 3], Zc[:, 4]
            psi = np.zeros((m, q))
            psi[:, :G] = Cc[:, None] * w_L_coeff[:, ci][None, :]
            psi[:, G:2 * G] = Cc[:, None] * w_U_coeff[:, ci][None, :]
            k_rec = float(tb.K_rec[ki, ci])
            p_tilde = (Hrc - pp[ki, ci] * Krc) / k_rec if k_rec > 0.0 else np.zeros(m)
            fac = w[ci] / float(tb.C[ki, ci])
            psi[:, ki] += fac * (Hc + qL[ki, ci] * Uc + dL_dp[ki, ci] * float(tb.U[ki, ci]) * p_tilde - rL[ki, ci] * Cc)
            psi[:, G + ki] += fac * (Hc + qU[ki, ci] * Uc + dU_dp[ki, ci] * float(tb.U[ki, ci]) * p_tilde - rU[ki, ci] * Cc)
            psi_dL = psi[:, pair_h] - psi[:, G + pair_l]
            psi[:, off:off + P] = psi_dL
            psi[:, off + P:] = psi[:, G + pair_h] - psi[:, pair_l]
            gram = psi.T @ psi
            V_hc0 += gram
            V_hc1 += ((m / (m - 1.0)) if m > 1 else 1.0) * gram
            if res_states is not None:
                np.add.at(state_scores, res_states[(ki, ci)], psi_dL)

    if res_states is not None:
        st_scale = (inf.n_states / (inf.n_states - 1.0)) if inf.n_states > 1 else 1.0
        V_st_dL = st_scale * (state_scores.T @ state_scores)
        return L_bar, U_bar, dL_hat, dU_hat, V_hc1, V_hc0, V_st_dL, state_scores
    return L_bar, U_bar, dL_hat, dU_hat, V_hc1, V_hc0


def _batched_boot_t_dL(inf: StackedInference, cell_arrays: dict, delta_vals: list[float],
                         B: int, rng: np.random.Generator | None = None,
                         batch_size: int = 200,
                         fixed_indices: dict[tuple[int, int], np.ndarray] | None = None) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Vectorized HC1 dL and se(dL) across B stratified resamples via exact 5-moment algebraic expansion."""
    G, K, P = inf.G, inf.K, inf.P
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    idx_h = [np.where(pair_h == ki)[0] for ki in range(G)]
    idx_l = [np.where(pair_l == ki)[0] for ki in range(G)]
    fixed_w = inf.fixed_w

    out = {float(d): (np.zeros((B, P)), np.zeros((B, P))) for d in delta_vals}

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        bs = end - start
        cell_Z = {}
        sums = np.zeros((bs, G, K, 5))
        mu_r = np.zeros((bs, G, K, 5))
        for gi in range(G):
            for ci in range(K):
                arr = cell_arrays[(gi, ci)]
                m = arr.shape[0]
                if fixed_indices is not None:
                    idx = fixed_indices[(gi, ci)][start:end]
                else:
                    idx = rng.integers(0, m, size=(bs, m))
                Z = arr[idx]  # (bs, m, 5)
                cell_Z[(gi, ci)] = Z
                sums[:, gi, ci, :] = Z.sum(axis=1)
                mu_r[:, gi, ci, :] = Z.mean(axis=1)

        C_mat = sums[:, :, :, 0]
        H_mat = sums[:, :, :, 1]
        U_mat = sums[:, :, :, 2]
        Hr_mat = sums[:, :, :, 3]
        Kr_mat = sums[:, :, :, 4]
        if fixed_w is not None:
            w = np.broadcast_to(fixed_w[None, :], (bs, K))
            C_tot = None
        else:
            C_cell_sum = C_mat.sum(axis=1)  # (bs, K)
            C_tot = np.maximum(C_cell_sum.sum(axis=1), 1e-15)  # (bs,)
            w = C_cell_sum / C_tot[:, None]
        undef = Kr_mat <= 0.0
        pp = np.where(undef, 0.0, np.divide(Hr_mat, np.maximum(Kr_mat, 1e-15)))

        for d_raw in delta_vals:
            delta_val = float(d_raw)
            if delta_val >= 1.0:
                qL = np.zeros((bs, G, K)); qU = np.ones((bs, G, K))
                dL_dp = np.zeros((bs, G, K)); dU_dp = np.zeros((bs, G, K))
            else:
                qL = np.where(undef, 0.0, np.maximum(0.0, pp - delta_val))
                qU = np.where(undef, 1.0, np.minimum(1.0, pp + delta_val))
                dL_dp = np.where((~undef) & (pp > delta_val), 1.0, 0.0)
                dU_dp = np.where((~undef) & (pp < 1.0 - delta_val), 1.0, 0.0)

            C_safe = np.maximum(C_mat, 1e-15)
            rL = (H_mat + U_mat * qL) / C_safe
            rU = (H_mat + U_mat * qU) / C_safe
            L_bar = np.einsum("bsk,bk->bs", rL, w)
            U_bar = np.einsum("bsk,bk->bs", rU, w)
            dL_hat = L_bar[:, pair_h] - U_bar[:, pair_l]

            if fixed_w is None:
                w_L_coeff = (rL - L_bar[:, :, None]) / C_tot[:, None, None]  # (bs, G, K)
                w_U_coeff = (rU - U_bar[:, :, None]) / C_tot[:, None, None]
                w_pair_coeff = w_L_coeff[:, pair_h, :] - w_U_coeff[:, pair_l, :]  # (bs, P, K)

            var_dL = np.zeros((bs, P))
            for ki in range(G):
                ih = idx_h[ki]
                il = idx_l[ki]
                for ci in range(K):
                    Z = cell_Z[(ki, ci)]
                    m = Z.shape[1]
                    Zc = Z - mu_r[:, ki, ci, :][:, None, :]
                    Cc, Hc, Uc, Hrc, Krc = Zc[:, :, 0], Zc[:, :, 1], Zc[:, :, 2], Zc[:, :, 3], Zc[:, :, 4]
                    k_rec = Kr_mat[:, ki, ci][:, None]
                    p_tilde = np.where(k_rec > 0.0, (Hrc - pp[:, ki, ci][:, None] * Krc) / np.maximum(k_rec, 1e-15), 0.0)
                    fac = (w[:, ci] / C_safe[:, ki, ci])[:, None]
                    own_L = fac * (Hc + qL[:, ki, ci][:, None] * Uc + (dL_dp[:, ki, ci] * U_mat[:, ki, ci])[:, None] * p_tilde - rL[:, ki, ci][:, None] * Cc)
                    own_U = fac * (Hc + qU[:, ki, ci][:, None] * Uc + (dU_dp[:, ki, ci] * U_mat[:, ki, ci])[:, None] * p_tilde - rU[:, ki, ci][:, None] * Cc)

                    scale = (m / (m - 1.0)) if m > 1 else 1.0
                    s_LL = np.sum(own_L * own_L, axis=1)
                    s_UU = np.sum(own_U * own_U, axis=1)
                    if fixed_w is None:
                        s_cc = np.sum(Cc * Cc, axis=1)
                        s_cL = np.sum(Cc * own_L, axis=1)
                        s_cU = np.sum(Cc * own_U, axis=1)
                        a = w_pair_coeff[:, :, ci]  # (bs, P)
                        var_dL += scale * (a * a) * s_cc[:, None]
                        if len(ih):
                            var_dL[:, ih] += scale * (2.0 * a[:, ih] * s_cL[:, None] + s_LL[:, None])
                        if len(il):
                            var_dL[:, il] += scale * (-2.0 * a[:, il] * s_cU[:, None] + s_UU[:, None])
                    else:
                        if len(ih):
                            var_dL[:, ih] += scale * s_LL[:, None]
                        if len(il):
                            var_dL[:, il] += scale * s_UU[:, None]

            out[delta_val][0][start:end] = dL_hat
            out[delta_val][1][start:end] = np.sqrt(np.maximum(var_dL, 0.0))

    return out


def run_studentized_bootstrap_t(inf: StackedInference, B_bt: int = 2000,
                                seed: int = 20261125,
                                grid_pp: tuple[float, ...] = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0)
                                ) -> OrderedDict:
    """True replicate-studentized bootstrap-t Max-T stepdown over all ordered hypotheses + delta-specific c_bt(delta)."""
    G, K, P = inf.G, inf.K, inf.P
    off = 3 * G
    th_wc, V_wc, _, _ = inf.analytic_covariances(1.0)
    dL_hat_wc = th_wc[off:off + P]
    se_hat_wc = np.sqrt(np.maximum(np.diag(V_wc[off:off + P, off:off + P]), 1e-20))
    t_obs_wc = dL_hat_wc / se_hat_wc

    th_d5, V_d5, _, _ = inf.analytic_covariances(0.05)
    dL_hat_d5 = th_d5[off:off + P]
    se_hat_d5 = np.sqrt(np.maximum(np.diag(V_d5[off:off + P, off:off + P]), 1e-20))
    t_obs_d5 = dL_hat_d5 / se_hat_d5

    fixed_indices = {(gi, ci): np.zeros((B_bt, inf.cell_arrays[(gi, ci)].shape[0]), dtype=np.int32)
                     for gi in range(G) for ci in range(K)}
    for b in range(B_bt):
        rng_b = np.random.default_rng([seed, b])
        for gi in range(G):
            for ci in range(K):
                m = inf.cell_arrays[(gi, ci)].shape[0]
                fixed_indices[(gi, ci)][b] = rng_b.integers(0, m, size=m)

    d_vals = sorted({round(pp / 100.0, 6) for pp in grid_pp} | {0.05, 1.0})
    evals = _batched_boot_t_dL(inf, inf.cell_arrays, d_vals, B_bt, batch_size=100, fixed_indices=fixed_indices)

    dL_bw, se_bw = evals[1.0]
    n_degen_wc = int(np.sum(se_bw < 1e-12))
    T_wc = (dL_bw - dL_hat_wc[None, :]) / np.maximum(se_bw, 1e-12)

    dL_b5, se_b5 = evals[0.05]
    n_degen_d5 = int(np.sum(se_b5 < 1e-12))
    T_d5 = (dL_b5 - dL_hat_d5[None, :]) / np.maximum(se_b5, 1e-12)

    def summarize_bt(t_obs, T_mat):
        c_crit = float(np.quantile(T_mat.max(axis=1), 0.95))
        order = np.argsort(-t_obs, kind="mergesort")
        adj_p = np.zeros(P); mc_se = np.zeros(P)
        prev_p = 0.0; prev_se = 0.0
        for step in range(P):
            active = order[step:]
            max_t = T_mat[:, active].max(axis=1)
            p_raw = float((1.0 + np.sum(max_t >= t_obs[order[step]])) / (B_bt + 1.0))
            se_raw = float(math.sqrt(max(p_raw * (1.0 - p_raw), 0.0) / B_bt))
            if p_raw >= prev_p:
                adj_p[order[step]] = p_raw; mc_se[order[step]] = se_raw
                prev_p = p_raw; prev_se = se_raw
            else:
                adj_p[order[step]] = prev_p; mc_se[order[step]] = prev_se
        ranks = rank_confidence_sets(inf.groups, inf.ordered_pairs, adj_p, 0.05)
        return OrderedDict(c_bt=c_crit, p_fwer=adj_p.tolist(), mc_se_p=mc_se.tolist(),
                           n_sig=int(sum(x <= 0.05 for x in adj_p)), rank_confidence_sets=ranks)

    w_res = summarize_bt(t_obs_wc, T_wc)
    w_res["n_degenerate_se_entries"] = int(n_degen_wc)
    d_res = summarize_bt(t_obs_d5, T_d5)
    d_res["n_degenerate_se_entries"] = int(n_degen_d5)

    c_bt_grid_vals = []
    for pp in grid_pp:
        dv = round(pp / 100.0, 6)
        th_d, _, _, _ = inf.analytic_covariances(dv)
        dL_hat_d = th_d[off:off + P]
        dL_bd, se_bd = evals[dv]
        T_d = (dL_bd - dL_hat_d[None, :]) / np.maximum(se_bd, 1e-12)
        c_bt_grid_vals.append(float(np.quantile(T_d.max(axis=1), 0.95)))

    return OrderedDict(
        B_bt=int(B_bt), worst=w_res, delta5=d_res,
        c_bt_grid_pp=list(grid_pp), c_bt_grid_vals=c_bt_grid_vals,
    )


def run_small_sample_calibration(inf: StackedInference,
                                 n_mc: int = 400,
                                 B_inner: int = 399,
                                 seed: int = 20261125) -> OrderedDict:
    """Non-circular stratified double bootstrap (Beran 1988) over all 30 hypotheses.

    Inside every outer bootstrap world, ALL six procedures re-estimate their own critical value:
      (1) HC0 Gaussian, (2) HC1 Gaussian, (3) Fixed-scale standardized bootstrap (B_inner draws),
      (4) Studentized bootstrap-t (B_inner draws with analytic inner HC1 SEs),
      (5) 53-exposure-state Gaussian, (6) Unrestricted Rademacher WCB (B_inner draws).
    Note: Stratified outbreak resampling assumes independent strata and does not simulate state-level correlation.
    """
    G, K, P = inf.G, inf.K, inf.P
    P15 = P // 2
    rng = np.random.default_rng(seed + 17)
    Z_std_mc = np.random.default_rng(seed + 19).standard_normal((10000, P))
    off = 3 * G

    th0_wc, _, _, _ = inf.analytic_covariances(1.0)
    dL0_wc, dU0_wc = th0_wc[off:off + P], th0_wc[off + P:]
    th0_d5, _, _, _ = inf.analytic_covariances(0.05)
    dL0_d5 = th0_d5[off:off + P]

    def wcb_exceeds(dL_diff: np.ndarray, state_scores_dL: np.ndarray, rng_wcb, B_wcb_cal: int) -> bool:
        G_st = state_scores_dL.shape[0]
        scale = math.sqrt(G_st / (G_st - 1.0))
        se_obs = scale * np.sqrt(np.maximum(np.sum(state_scores_dL ** 2, axis=0), 1e-20))
        t_max_obs = float(np.max(dL_diff / se_obs))
        rad = rng_wcb.choice(np.array([-1.0, 1.0]), size=(B_wcb_cal, G_st))
        boot_diff = scale * (rad @ state_scores_dL)
        sum_sq = np.sum(state_scores_dL ** 2, axis=0)[None, :]
        mean_term = (rad @ state_scores_dL) / G_st
        boot_se = scale * np.sqrt(np.maximum(sum_sq - G_st * (mean_term ** 2), 1e-24))
        c_loc = float(np.quantile((boot_diff / boot_se).max(axis=1), 0.95))
        return bool(t_max_obs > c_loc)

    im_cov_lo_hc1 = np.zeros(P15); im_cov_hi_hc1 = np.zeros(P15)
    im_cov_lo_hc0 = np.zeros(P15); im_cov_hi_hc0 = np.zeros(P15)
    keys_m = ("hc0", "hc1", "fixed_scale_boot", "bootstrap_t", "state", "wcb")
    fwer_wc = {k: 0 for k in keys_m}
    fwer_d5 = {k: 0 for k in keys_m}
    exceed_hc1_wc = np.zeros(P, dtype=int)
    exceed_hc1_d5 = np.zeros(P, dtype=int)
    c_bt_in_wc_list = []
    c_bt_in_d5_list = []

    L_draws = np.zeros((n_mc, G)); U_draws = np.zeros((n_mc, G))
    seL_draws_hc1 = np.zeros((n_mc, G)); seU_draws_hc1 = np.zeros((n_mc, G))
    dummy_t = np.zeros(P)
    rng_wcb = np.random.default_rng(seed + 31)
    rng_in = np.random.default_rng(seed + 43)

    for r_idx in range(n_mc):
        res_arrs = {}
        res_sts = {}
        for gi in range(G):
            for ci in range(K):
                arr = inf.cell_arrays[(gi, ci)]
                st_arr = inf.cell_states[(gi, ci)]
                m = arr.shape[0]
                idx = rng.integers(0, m, size=m)
                res_arrs[(gi, ci)] = arr[idx]
                res_sts[(gi, ci)] = st_arr[idx]

        L_r, U_r, dL_r, dU_r, V1_wc, V0_wc, Vst_wc, st_sc_wc = _eval_resampled_table(inf, res_arrs, 1.0, res_sts)
        L_draws[r_idx] = L_r; U_draws[r_idx] = U_r
        seL_draws_hc1[r_idx] = np.sqrt(np.maximum(np.diag(V1_wc[:G, :G]), 0.0))
        seU_draws_hc1[r_idx] = np.sqrt(np.maximum(np.diag(V1_wc[G:2 * G, G:2 * G]), 0.0))

        seL1 = np.sqrt(np.maximum(np.diag(V1_wc[off:off + P, off:off + P]), 1e-20))
        seU1 = np.sqrt(np.maximum(np.diag(V1_wc[off + P:, off + P:]), 1e-20))
        seL0 = np.sqrt(np.maximum(np.diag(V0_wc[off:off + P, off:off + P]), 1e-20))
        seU0 = np.sqrt(np.maximum(np.diag(V0_wc[off + P:, off + P:]), 1e-20))
        seLst = np.sqrt(np.maximum(np.diag(Vst_wc), 1e-20))

        for p in range(P15):
            c1 = imbens_manski_crit(float(dU_r[p] - dL_r[p]), float(seL1[p]), float(seU1[p]), 0.05)
            im_cov_lo_hc1[p] += ((dL_r[p] - c1 * seL1[p]) <= dL0_wc[p] <= (dU_r[p] + c1 * seU1[p]))
            im_cov_hi_hc1[p] += ((dL_r[p] - c1 * seL1[p]) <= dU0_wc[p] <= (dU_r[p] + c1 * seU1[p]))
            c0 = imbens_manski_crit(float(dU_r[p] - dL_r[p]), float(seL0[p]), float(seU0[p]), 0.05)
            im_cov_lo_hc0[p] += ((dL_r[p] - c0 * seL0[p]) <= dL0_wc[p] <= (dU_r[p] + c0 * seU0[p]))
            im_cov_hi_hc0[p] += ((dL_r[p] - c0 * seL0[p]) <= dU0_wc[p] <= (dU_r[p] + c0 * seU0[p]))

        _, _, cmax1 = max_t_stepdown(dummy_t, V1_wc[off:off + P, off:off + P], 0.05, Z_std=Z_std_mc)
        _, _, cmax0 = max_t_stepdown(dummy_t, V0_wc[off:off + P, off:off + P], 0.05, Z_std=Z_std_mc)
        _, _, cmax_st = max_t_stepdown(dummy_t, Vst_wc, 0.05, Z_std=Z_std_mc)
        tvec1 = (dL_r - dL0_wc) / seL1
        exceed_hc1_wc += (tvec1 > cmax1).astype(int)
        tmax1 = float(np.max(tvec1))
        fwer_wc["hc1"] += int(tmax1 > cmax1)
        fwer_wc["hc0"] += int(np.max((dL_r - dL0_wc) / seL0) > cmax0)
        fwer_wc["state"] += int(np.max((dL_r - dL0_wc) / seLst) > cmax_st)
        fwer_wc["wcb"] += int(wcb_exceeds(dL_r - dL0_wc, st_sc_wc, rng_wcb, B_inner))

        _, _, dL_5r, _, V1_5, V0_5, Vst_5, st_sc_5 = _eval_resampled_table(inf, res_arrs, 0.05, res_sts)
        seL1_5 = np.sqrt(np.maximum(np.diag(V1_5[off:off + P, off:off + P]), 1e-20))
        seL0_5 = np.sqrt(np.maximum(np.diag(V0_5[off:off + P, off:off + P]), 1e-20))
        seLst_5 = np.sqrt(np.maximum(np.diag(Vst_5), 1e-20))
        _, _, cmax1_5 = max_t_stepdown(dummy_t, V1_5[off:off + P, off:off + P], 0.05, Z_std=Z_std_mc)
        _, _, cmax0_5 = max_t_stepdown(dummy_t, V0_5[off:off + P, off:off + P], 0.05, Z_std=Z_std_mc)
        _, _, cmaxst_5 = max_t_stepdown(dummy_t, Vst_5, 0.05, Z_std=Z_std_mc)
        tvec1_5 = (dL_5r - dL0_d5) / seL1_5
        exceed_hc1_d5 += (tvec1_5 > cmax1_5).astype(int)
        tmax1_5 = float(np.max(tvec1_5))
        fwer_d5["hc1"] += int(tmax1_5 > cmax1_5)
        fwer_d5["hc0"] += int(np.max((dL_5r - dL0_d5) / seL0_5) > cmax0_5)
        fwer_d5["state"] += int(np.max((dL_5r - dL0_d5) / seLst_5) > cmaxst_5)
        fwer_d5["wcb"] += int(wcb_exceeds(dL_5r - dL0_d5, st_sc_5, rng_wcb, B_inner))

        # Inner bootstrap inside outer world res_arrs (evaluates both 1.0 and 0.05 simultaneously)
        in_eval = _batched_boot_t_dL(inf, res_arrs, [1.0, 0.05], B_inner, rng=rng_in, batch_size=399)
        dL_in_wc, se_in_wc = in_eval[1.0]
        sd_in_wc = np.maximum(np.std(dL_in_wc, axis=0, ddof=1), 1e-12)
        c_std_in_wc = float(np.quantile(np.max((dL_in_wc - dL_r[None, :]) / sd_in_wc[None, :], axis=1), 0.95))
        fwer_wc["fixed_scale_boot"] += int(float(np.max((dL_r - dL0_wc) / sd_in_wc)) > c_std_in_wc)
        c_bt_in_wc = float(np.quantile(np.max((dL_in_wc - dL_r[None, :]) / np.maximum(se_in_wc, 1e-12), axis=1), 0.95))
        c_bt_in_wc_list.append(c_bt_in_wc)
        fwer_wc["bootstrap_t"] += int(tmax1 > c_bt_in_wc)

        dL_in_d5, se_in_d5 = in_eval[0.05]
        sd_in_d5 = np.maximum(np.std(dL_in_d5, axis=0, ddof=1), 1e-12)
        c_std_in_d5 = float(np.quantile(np.max((dL_in_d5 - dL_5r[None, :]) / sd_in_d5[None, :], axis=1), 0.95))
        fwer_d5["fixed_scale_boot"] += int(float(np.max((dL_5r - dL0_d5) / sd_in_d5)) > c_std_in_d5)
        c_bt_in_d5 = float(np.quantile(np.max((dL_in_d5 - dL_5r[None, :]) / np.maximum(se_in_d5, 1e-12), axis=1), 0.95))
        c_bt_in_d5_list.append(c_bt_in_d5)
        fwer_d5["bootstrap_t"] += int(tmax1_5 > c_bt_in_d5)

    def top_drivers(counts):
        idx_sort = np.argsort(-counts, kind="mergesort")[:5]
        return [OrderedDict(hypothesis=f"{inf.groups[inf.ordered_pairs[i][0]]}->{inf.groups[inf.ordered_pairs[i][1]]}",
                            exceed_rate=float(counts[i] / n_mc)) for i in idx_sort]

    cov_worst_hc1 = np.minimum(im_cov_lo_hc1, im_cov_hi_hc1) / n_mc
    cov_worst_hc0 = np.minimum(im_cov_lo_hc0, im_cov_hi_hc0) / n_mc

    mc_sd_L = np.std(L_draws, axis=0, ddof=1)
    mc_sd_U = np.std(U_draws, axis=0, ddof=1)
    per_grp_cal = OrderedDict()
    for i, g in enumerate(inf.groups):
        per_grp_cal[g] = OrderedDict(
            mc_sd_L_pp=100.0 * float(mc_sd_L[i]),
            mean_hc1_se_L_pp=100.0 * float(seL_draws_hc1[:, i].mean()),
            ratio_hc1_to_mc_L=float(seL_draws_hc1[:, i].mean() / mc_sd_L[i]),
            mc_sd_U_pp=100.0 * float(mc_sd_U[i]),
            mean_hc1_se_U_pp=100.0 * float(seU_draws_hc1[:, i].mean()),
            ratio_hc1_to_mc_U=float(seU_draws_hc1[:, i].mean() / mc_sd_U[i]),
        )

    fwer_w_rates = OrderedDict((k, float(v / n_mc)) for k, v in fwer_wc.items())
    fwer_d5_rates = OrderedDict((k, float(v / n_mc)) for k, v in fwer_d5.items())
    mc_se_w = OrderedDict((k, float(math.sqrt(max(p * (1.0 - p), 0.0) / n_mc))) for k, p in fwer_w_rates.items())
    mc_se_d5 = OrderedDict((k, float(math.sqrt(max(p * (1.0 - p), 0.0) / n_mc))) for k, p in fwer_d5_rates.items())
    cw_arr = np.asarray(c_bt_in_wc_list)
    c5_arr = np.asarray(c_bt_in_d5_list)

    return OrderedDict(
        n_mc=int(n_mc),
        B_inner=int(B_inner),
        n_hypotheses=int(P),
        seed=int(seed + 17),
        nominal_alpha=0.05,
        design=(f"double bootstrap ({n_mc} outer worlds x {B_inner} inner resamples preserving n_js; "
                "all 6 procedures re-estimate critical values in every outer world; no state dependence simulated)"),
        im95_min_endpoint_cov_hc1=float(cov_worst_hc1.min()),
        im95_median_endpoint_cov_hc1=float(np.median(cov_worst_hc1)),
        im95_mean_endpoint_cov_hc1=float(cov_worst_hc1.mean()),
        im95_min_endpoint_cov_hc0=float(cov_worst_hc0.min()),
        im95_median_endpoint_cov_hc0=float(np.median(cov_worst_hc0)),
        im95_per_pair_min_cov_hc1=[float(x) for x in cov_worst_hc1],
        fwer_worst=fwer_w_rates,
        fwer_worst_mc_se=mc_se_w,
        fwer_delta5=fwer_d5_rates,
        fwer_delta5_mc_se=mc_se_d5,
        inner_c_bt_worst_summary=OrderedDict(
            mean=float(cw_arr.mean()), p5=float(np.quantile(cw_arr, 0.05)),
            p50=float(np.quantile(cw_arr, 0.50)), p95=float(np.quantile(cw_arr, 0.95)),
        ),
        inner_c_bt_delta5_summary=OrderedDict(
            mean=float(c5_arr.mean()), p5=float(np.quantile(c5_arr, 0.05)),
            p50=float(np.quantile(c5_arr, 0.50)), p95=float(np.quantile(c5_arr, 0.95)),
        ),
        top_fwer_drivers_wc=top_drivers(exceed_hc1_wc),
        top_fwer_drivers_d5=top_drivers(exceed_hc1_d5),
        per_group=per_grp_cal,
    )


def empirical_delta_calibration(main_df: pd.DataFrame, t_main: S.Table,
                                ordered_pairs_15: list[tuple[int, int]]) -> OrderedDict:
    """Decompose fully vs partially recorded comparisons by cell, multistate status, and size,
    plus scenario stress tests and OR (pure & 0.1%-floor) sensitivity."""
    sup_df = main_df[main_df["cell"].isin(t_main.cells)].copy()
    sup_df["is_multi"] = sup_df["state"].astype(str) == "Multistate"
    full_df = sup_df[sup_df["U"] == 0]
    part_df = sup_df[(sup_df["U"] > 0) & (sup_df["K_rec"] > 0)]

    per_group = OrderedDict()
    for i, g in enumerate(t_main.groups):
        fg = full_df[full_df["group"] == g]
        pg = part_df[part_df["group"] == g]
        r_full_pool = float(fg["H_rec"].sum() / fg["K_rec"].sum())
        r_part_pool = float(pg["H_rec"].sum() / pg["K_rec"].sum()) if len(pg) else r_full_pool

        num_f = 0.0; num_p = 0.0; den = 0.0
        for ci, c in enumerate(t_main.cells):
            fc = fg[fg["cell"] == c]
            pc = pg[pg["cell"] == c]
            if len(fc) > 0 and len(pc) > 0 and fc["K_rec"].sum() > 0 and pc["K_rec"].sum() > 0:
                num_f += t_main.w[ci] * float(fc["H_rec"].sum() / fc["K_rec"].sum())
                num_p += t_main.w[ci] * float(pc["H_rec"].sum() / pc["K_rec"].sum())
                den += t_main.w[ci]
        r_full_std = num_f / den if den > 0 else r_full_pool
        r_part_std = num_p / den if den > 0 else r_part_pool

        r_all_std = float(B.point_delta0(t_main, i)[0])

        # Single-state vs multistate stratification
        fs = fg[~fg["is_multi"]]; ps = pg[~pg["is_multi"]]
        fm = fg[fg["is_multi"]]; pm = pg[pg["is_multi"]]
        r_fs = float(fs["H_rec"].sum() / fs["K_rec"].sum()) if fs["K_rec"].sum() > 0 else None
        r_ps = float(ps["H_rec"].sum() / ps["K_rec"].sum()) if ps["K_rec"].sum() > 0 else None
        r_fm = float(fm["H_rec"].sum() / fm["K_rec"].sum()) if fm["K_rec"].sum() > 0 else None
        r_pm = float(pm["H_rec"].sum() / pm["K_rec"].sum()) if pm["K_rec"].sum() > 0 else None

        per_group[g] = OrderedDict(
            n_full_outbreaks=int(len(fg)),
            cases_full=float(fg["C"].sum()),
            median_size_full=float(fg["C"].median()),
            pct_multistate_full=float(100.0 * fg["is_multi"].mean()),
            rate_full_pooled=r_full_pool,
            n_partial_outbreaks=int(len(pg)),
            known_cases_partial=float(pg["K_rec"].sum()),
            median_size_partial=float(pg["C"].median()) if len(pg) else None,
            pct_multistate_partial=float(100.0 * pg["is_multi"].mean()) if len(pg) else 0.0,
            rate_partial_known_pooled=r_part_pool,
            full_vs_partial_diff_pp=100.0 * (r_part_pool - r_full_pool),
            rate_full_std=r_full_std,
            rate_partial_known_std=r_part_std,
            within_cell_diff_pp=100.0 * (r_part_std - r_full_std),
            std_obs_minus_full_pp=100.0 * (r_all_std - r_full_std),
            single_state_full_rate=r_fs,
            single_state_partial_rate=r_ps,
            single_state_diff_pp=(100.0 * (r_ps - r_fs)) if (r_fs is not None and r_ps is not None) else None,
            multistate_full_rate=r_fm,
            multistate_partial_rate=r_pm,
            multistate_diff_pp=(100.0 * (r_pm - r_fm)) if (r_fm is not None and r_pm is not None) else None,
        )

    st_grp = sup_df.groupby("state").agg(C=("C", "sum"), U=("U", "sum"))
    hi_states = st_grp[(st_grp["C"] >= 200) & (st_grp["U"] / st_grp["C"] < 0.05)].index.tolist()
    hi_df = sup_df[sup_df["state"].isin(hi_states)]
    hi_rates = OrderedDict()
    z975 = float(norm.ppf(0.975))
    for g in t_main.groups:
        hg = hi_df[hi_df["group"] == g]
        kr = float(hg["K_rec"].sum())
        hr = float(hg["H_rec"].sum())
        if kr > 0:
            p_w = hr / kr
            denom_w = 1.0 + z975 * z975 / kr
            center_w = (p_w + z975 * z975 / (2.0 * kr)) / denom_w
            half_w = (z975 * math.sqrt((p_w * (1.0 - p_w) + z975 * z975 / (4.0 * kr)) / kr)) / denom_w
            ci_w = [max(0.0, center_w - half_w), min(1.0, center_w + half_w)]
        else:
            p_w = None; ci_w = None
        hi_rates[g] = OrderedDict(
            outbreaks=int(len(hg)),
            cases=float(hg["C"].sum()),
            known_cases=kr,
            unrecorded_share=float(hg["U"].sum() / hg["C"].sum()) if hg["C"].sum() > 0 else None,
            recorded_hosp_rate=p_w,
            ci95_wilson=ci_w,
        )

    # Common-mechanism proportional scaling scenario: q_js = clip(alpha * p_js, 0, 1) across ALL groups
    cm_alphas = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.00]
    cm_results = OrderedDict()
    p_obs_valid = np.where(np.isnan(t_main.p_obs), 0.0, t_main.p_obs)
    for al in cm_alphas:
        q_mat = np.clip(al * p_obs_valid, 0.0, 1.0)
        rates_al = OrderedDict()
        for gi, g in enumerate(t_main.groups):
            rates_al[g] = float(np.sum(t_main.w * (t_main.H[gi] + t_main.U[gi] * q_mat[gi]) / t_main.C[gi]))
        pair_pos = sum(1 for h, l in ordered_pairs_15 if rates_al[t_main.groups[h]] > rates_al[t_main.groups[l]])
        cm_results[f"{al:.2f}"] = OrderedDict(
            alpha=float(al), rates=rates_al, n_main_direction_preserved=int(pair_pos),
        )

    asym_scenarios = [
        ("asym_10_5", "Asymmetric scenario (-10 / +5 pp)", 0.10, 0.05),
        ("asym_15_5", "Asymmetric scenario (-15 / +5 pp)", 0.15, 0.05),
        ("asym_20_5", "Asymmetric scenario (-20 / +5 pp)", 0.20, 0.05),
    ]
    asym_results = OrderedDict()
    for code, label, dd, du in asym_scenarios:
        det = 0
        verdicts = OrderedDict()
        for h, l in ordered_pairs_15:
            key = f"{t_main.groups[min(h, l)]}|{t_main.groups[max(h, l)]}"
            Lh, Uh = B.group_bounds_asym(t_main, h, dd, du)
            Ll, Ul = B.group_bounds_asym(t_main, l, dd, du)
            v = B.verdict(Lh, Uh, Ll, Ul, t_main.groups[h], t_main.groups[l])
            if v != B.UNDET:
                det += 1
            verdicts[key] = v
        asym_results[code] = OrderedDict(label=label, down_pp=100.0 * dd, up_pp=100.0 * du,
                                        determined=int(det), verdicts=verdicts)

    def eval_or(floor: float):
        or_gammas = [1.25, 1.5, 2.0, 3.0, 5.0]
        or_by_gamma = OrderedDict()
        for gm in or_gammas:
            det_g = 0
            vmap = OrderedDict()
            for h, l in ordered_pairs_15:
                gh, gl = t_main.groups[h], t_main.groups[l]
                key = f"{t_main.groups[min(h, l)]}|{t_main.groups[max(h, l)]}"
                Lh, _ = B.group_bounds_or(t_main, h, gm, floor=floor)
                _, Ul = B.group_bounds_or(t_main, l, gm, floor=floor)
                v = gh if Lh > Ul else B.UNDET
                if v != B.UNDET:
                    det_g += 1
                vmap[key] = v
            or_by_gamma[str(gm)] = OrderedDict(gamma=gm, determined=int(det_g), verdicts=vmap)
        gs_pairs = OrderedDict()
        for h, l in ordered_pairs_15:
            key = f"{t_main.groups[min(h, l)]}|{t_main.groups[max(h, l)]}"
            gs_pairs[key] = B.gamma_star(t_main, h, l, floor=floor)
        return OrderedDict(floor=floor, by_gamma=or_by_gamma, gamma_star_pairs=gs_pairs)

    n_zero_rec = int(np.sum((~np.isnan(t_main.p_obs)) & (t_main.p_obs == 0.0)))
    or_pure = eval_or(0.0)
    or_pure["n_cells_zero_recorded_rate"] = n_zero_rec
    or_floor = eval_or(0.001)

    # NORS confirmed STEC O157 serogroup case mix & Scallan Walter (2025) weighted STEC benchmark
    stec_sup = sup_df[sup_df["group"] == "STEC"].copy()
    sg_u = stec_sup["serotype_or_genotype"].astype(str).str.upper() if "serotype_or_genotype" in stec_sup.columns else pd.Series("", index=stec_sup.index)
    stec_o157_cases = float(stec_sup.loc[sg_u.str.contains("O157"), "C"].sum())
    stec_total_cases = float(stec_sup["C"].sum())
    w_o157 = stec_o157_cases / stec_total_cases if stec_total_cases > 0 else 0.0
    scallan2025_stec_mix_rate = 0.431 * w_o157 + 0.162 * (1.0 - w_o157)

    return OrderedDict(
        per_group=per_group,
        high_completeness_states=OrderedDict(n_states=int(len(hi_states)), states=hi_states, per_group=hi_rates),
        common_mechanism=cm_results,
        asymmetric=asym_results,
        odds_ratio=or_pure,
        odds_ratio_floor_001=or_floor,
        stec_o157_case_share=float(w_o157),
        scallan_walter_2025_stec_weighted_rate=float(scallan2025_stec_mix_rate),
    )


def disambiguation_scores(tb: S.Table, eps: float = 0.015) -> np.ndarray:
    """Heuristic Disambiguation Priority Index (DPI) per case followed up across unresolved pairs."""
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
            urgency = 1.0 / (overlap + eps)
            cell_val[h] += urgency * tb.w * p_hat[h] / tb.C[h]
            cell_val[l] += urgency * tb.w * (1.0 - p_hat[l]) / tb.C[l]
    return cell_val


voi_scores_for_table = disambiguation_scores


def _take_budget_filling(order: np.ndarray, cost_arr: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
    """Budget-filling selection: admit every ranked outbreak fitting remaining budget."""
    chosen = []
    spent = 0.0
    for idx in order:
        c = float(cost_arr[idx])
        if spent + c <= budget + 1e-12:
            chosen.append(int(idx))
            spent += c
    return np.asarray(chosen, dtype=int), spent


def run_followup_masking(main_df: pd.DataFrame, t_main: S.Table,
                         n_trials: int = 200, seed: int = 20261125) -> OrderedDict:
    """Cost-conditional retrospective masking benchmark across per-case, per-outbreak, and mixed-cost
    budgets, plus MNAR, state-clustered, partial-yield, and adaptive DPI sensitivity checks."""
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

    st_grp = sup_df.groupby("state").agg(U=("U", "sum"), C=("C", "sum"))
    st_u_rate = (st_grp["U"] / st_grp["C"]).to_dict()
    mean_u_all = float(sup_df["U"].sum() / sup_df["C"].sum())

    C_arr = complete_df["C"].to_numpy(dtype=float)
    H_true = complete_df["H"].to_numpy(dtype=float)
    g_idx = np.array([gi_map[g] for g in complete_df["group"]], dtype=int)
    c_idx = np.array([ci_map[c] for c in complete_df["cell"]], dtype=int)
    mask_probs_mcar = np.array([emp_inc_prob[(g, c)] for g, c in zip(complete_df["group"], complete_df["cell"])])
    mask_probs_state = np.array([
        min(max(0.8 * st_u_rate.get(str(st), mean_u_all) + 0.2 * emp_inc_prob[(g, c)], 0.03), 0.65)
        for st, g, c in zip(complete_df["state"], complete_df["group"], complete_df["cell"])
    ])
    logC = np.log1p(C_arr)
    size_factor = np.clip(0.55 + 0.35 * (logC - logC.mean()) / max(logC.std(), 1e-6), 0.25, 1.85)

    G, K = len(R.GROUPS), len(t_main.cells)
    C_cell = t_true.C.copy()
    w_vec = t_true.w.copy()

    case_budgets = (250, 500, 1000, 1500, 2000, 3500, 5000)
    outbreak_budgets = (25, 50, 100, 200)
    mixed_budgets = (500, 1000, 2000, 5000)
    fixed_eq_cases = 20.0

    def make_table(H_obs, U_obs, K_obs):
        H_mat = np.zeros((G, K)); U_mat = np.zeros((G, K)); K_mat = np.zeros((G, K))
        np.add.at(H_mat, (g_idx, c_idx), H_obs)
        np.add.at(U_mat, (g_idx, c_idx), U_obs)
        np.add.at(K_mat, (g_idx, c_idx), K_obs)
        return S.Table(groups=list(R.GROUPS), cells=list(t_main.cells),
                       C=C_cell, H=H_mat, U=U_mat, H_rec=H_mat, K_rec=K_mat, n=t_true.n, w=w_vec)

    def eval_table(tb: S.Table):
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
        return w_det, d5_det, ov_sum

    def summarize_array(arr):
        a = np.asarray(arr, dtype=float)
        return float(a.mean()), float(a.std(ddof=1))

    # Storage across the three cost models on identical 200 MCAR draws
    case_pols = ("random", "volume_greedy", "width_greedy", "dpi")
    ob_pols = ("random", "volume_greedy", "width_greedy", "outbreak_width")
    mix_pols = ("random", "volume_greedy", "width_greedy", "outbreak_width", "mixed_leverage")

    res_case = {p: {b: {"worst_det": [], "overlap_sum": [], "delta5_det": [], "spent": [], "lev_pp": []}
                    for b in (0,) + case_budgets} for p in case_pols}
    res_ob = {p: {b: {"worst_det": [], "spent_ob": [], "spent_cases": []}
                  for b in (0,) + outbreak_budgets} for p in ob_pols}
    res_mix = {p: {b: {"worst_det": [], "spent": []}
                   for b in (0,) + mixed_budgets} for p in mix_pols}

    sens_budgets = (500, 1000, 2000, 5000)
    partial_res = {p: {b: [] for b in sens_budgets} for p in case_pols}
    adaptive_res = {b: [] for b in sens_budgets}
    eps_vals = (0.005, 0.015, 0.05, 0.10)
    eps_res = {str(ep): {b: [] for b in (1000, 2000)} for ep in eps_vals}

    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        masked = rng.random(len(complete_df)) < mask_probs_mcar
        for gi in range(G):
            for ci in range(K):
                idx_gc = np.where((g_idx == gi) & (c_idx == ci))[0]
                if np.all(masked[idx_gc]):
                    masked[rng.choice(idx_gc)] = False

        H_obs0 = np.where(masked, 0.0, H_true)
        U_obs0 = np.where(masked, C_arr, 0.0)
        K_obs0 = np.where(masked, 0.0, C_arr)
        tb0 = make_table(H_obs0, U_obs0, K_obs0)
        w0, d50, ov0 = eval_table(tb0)
        m_indices = np.where(masked)[0]

        lev_per_case = w_vec[c_idx[m_indices]] / C_cell[g_idx[m_indices], c_idx[m_indices]]
        ob_width = lev_per_case * C_arr[m_indices]
        dpi_tbl = disambiguation_scores(tb0, eps=0.015)
        dpi_sc = dpi_tbl[g_idx[m_indices], c_idx[m_indices]]
        mix_cost = fixed_eq_cases + C_arr[m_indices]
        mix_sc = ob_width / mix_cost

        orders = {
            "random": rng.permutation(m_indices),
            "volume_greedy": m_indices[np.argsort(-C_arr[m_indices], kind="mergesort")],
            "width_greedy": m_indices[np.argsort(-lev_per_case, kind="mergesort")],
            "dpi": m_indices[np.argsort(-dpi_sc, kind="mergesort")],
            "outbreak_width": m_indices[np.argsort(-ob_width, kind="mergesort")],
            "mixed_leverage": m_indices[np.argsort(-mix_sc, kind="mergesort")],
        }

        # 1. Per-case cost model
        for pol in case_pols:
            res_case[pol][0]["worst_det"].append(w0)
            res_case[pol][0]["delta5_det"].append(d50)
            res_case[pol][0]["overlap_sum"].append(100.0 * ov0)
            res_case[pol][0]["spent"].append(0.0)
            res_case[pol][0]["lev_pp"].append(0.0)
            ord_p = orders[pol]
            for b in case_budgets:
                chosen, spent = _take_budget_filling(ord_p, C_arr, b)
                H_obs = H_obs0.copy(); U_obs = U_obs0.copy(); K_obs = K_obs0.copy()
                if len(chosen) > 0:
                    H_obs[chosen] = H_true[chosen]; U_obs[chosen] = 0.0; K_obs[chosen] = C_arr[chosen]
                    lev = float(np.mean(100.0 * w_vec[c_idx[chosen]] / C_cell[g_idx[chosen], c_idx[chosen]]))
                else:
                    lev = 0.0
                wd, d5d, ov = eval_table(make_table(H_obs, U_obs, K_obs))
                res_case[pol][b]["worst_det"].append(wd)
                res_case[pol][b]["delta5_det"].append(d5d)
                res_case[pol][b]["overlap_sum"].append(100.0 * ov)
                res_case[pol][b]["spent"].append(spent)
                res_case[pol][b]["lev_pp"].append(lev)

                if b in sens_budgets:
                    Hp = H_obs0.copy(); Up = U_obs0.copy(); Kp = K_obs0.copy()
                    for idx in chosen:
                        c_i = int(C_arr[idx]); h_i = int(H_true[idx])
                        k_rec_i = int(rng.binomial(c_i, 0.70))
                        h_rec_i = int(rng.hypergeometric(h_i, c_i - h_i, k_rec_i)) if k_rec_i > 0 else 0
                        Hp[idx] = h_rec_i; Kp[idx] = k_rec_i; Up[idx] = c_i - k_rec_i
                    wd_p, _, _ = eval_table(make_table(Hp, Up, Kp))
                    partial_res[pol][b].append(wd_p)

        # 2. Per-outbreak cost model (budget counted in outbreaks)
        for pol in ob_pols:
            res_ob[pol][0]["worst_det"].append(w0)
            res_ob[pol][0]["spent_ob"].append(0.0)
            res_ob[pol][0]["spent_cases"].append(0.0)
            ord_p = orders[pol]
            for b in outbreak_budgets:
                chosen = ord_p[:b]
                H_obs = H_obs0.copy(); U_obs = U_obs0.copy(); K_obs = K_obs0.copy()
                if len(chosen) > 0:
                    H_obs[chosen] = H_true[chosen]; U_obs[chosen] = 0.0; K_obs[chosen] = C_arr[chosen]
                wd, _, _ = eval_table(make_table(H_obs, U_obs, K_obs))
                res_ob[pol][b]["worst_det"].append(wd)
                res_ob[pol][b]["spent_ob"].append(float(len(chosen)))
                res_ob[pol][b]["spent_cases"].append(float(C_arr[chosen].sum()))

        # 3. Mixed cost model (20 + C_i case-equivalents)
        mix_cost_full = fixed_eq_cases + C_arr
        for pol in mix_pols:
            res_mix[pol][0]["worst_det"].append(w0)
            res_mix[pol][0]["spent"].append(0.0)
            ord_p = orders[pol]
            for b in mixed_budgets:
                chosen, spent = _take_budget_filling(ord_p, mix_cost_full, b)
                H_obs = H_obs0.copy(); U_obs = U_obs0.copy(); K_obs = K_obs0.copy()
                if len(chosen) > 0:
                    H_obs[chosen] = H_true[chosen]; U_obs[chosen] = 0.0; K_obs[chosen] = C_arr[chosen]
                wd, _, _ = eval_table(make_table(H_obs, U_obs, K_obs))
                res_mix[pol][b]["worst_det"].append(wd)
                res_mix[pol][b]["spent"].append(spent)

        # Adaptive DPI re-scoring in 250-case tranches
        for target_b in sens_budgets:
            Ha = H_obs0.copy(); Ua = U_obs0.copy(); Ka = K_obs0.copy()
            rem_list = list(m_indices.tolist())
            spent_b = 0.0
            while rem_list:
                cur_tb = make_table(Ha, Ua, Ka)
                sc = disambiguation_scores(cur_tb, eps=0.015)
                rem_arr = np.array(rem_list, dtype=int)
                cand_order = rem_arr[np.argsort(-sc[g_idx[rem_arr], c_idx[rem_arr]], kind="mergesort")]
                tranche_taken = False; tranche_c = 0.0
                for idx in cand_order:
                    c_i = float(C_arr[idx])
                    if spent_b + c_i <= target_b + 1e-12:
                        Ha[idx] = H_true[idx]; Ua[idx] = 0.0; Ka[idx] = C_arr[idx]
                        spent_b += c_i; tranche_c += c_i
                        rem_list.remove(int(idx))
                        tranche_taken = True
                        if tranche_c >= 250.0:
                            break
                if not tranche_taken:
                    break
            wd_a, _, _ = eval_table(make_table(Ha, Ua, Ka))
            adaptive_res[target_b].append(wd_a)

        # Epsilon sensitivity for DPI
        for ep in eps_vals:
            sc_ep = disambiguation_scores(tb0, eps=ep)[g_idx[m_indices], c_idx[m_indices]]
            ord_ep = m_indices[np.argsort(-sc_ep, kind="mergesort")]
            for b in (1000, 2000):
                ch_ep, _ = _take_budget_filling(ord_ep, C_arr, b)
                He = H_obs0.copy(); Ue = U_obs0.copy(); Ke = K_obs0.copy()
                if len(ch_ep) > 0:
                    He[ch_ep] = H_true[ch_ep]; Ue[ch_ep] = 0.0; Ke[ch_ep] = C_arr[ch_ep]
                wd_e, _, _ = eval_table(make_table(He, Ue, Ke))
                eps_res[str(ep)][b].append(wd_e)

    def pack_summary(res_dict, pols, b_list, extra_keys=()):
        out = OrderedDict()
        for pol in pols:
            out[pol] = OrderedDict()
            for b in b_list:
                mw, sw = summarize_array(res_dict[pol][b]["worst_det"])
                ent = OrderedDict(
                    mean_worst_det=mw, sd_worst_det=sw,
                    ci95_worst_det=[mw - 1.96 * sw / math.sqrt(n_trials), mw + 1.96 * sw / math.sqrt(n_trials)],
                )
                for ek in extra_keys:
                    me, _ = summarize_array(res_dict[pol][b][ek])
                    ent[f"mean_{ek}"] = me
                if "delta5_det" in res_dict[pol][b]:
                    md5, sd5 = summarize_array(res_dict[pol][b]["delta5_det"])
                    mo, so = summarize_array(res_dict[pol][b]["overlap_sum"])
                    ent.update(mean_delta5_det=md5, sd_delta5_det=sd5, mean_overlap_pp=mo, sd_overlap_pp=so)
                out[pol][str(b)] = ent
        return out

    summary = pack_summary(res_case, case_pols, (0,) + case_budgets, extra_keys=("spent", "lev_pp"))
    for pol in case_pols:
        for b_s, d in summary[pol].items():
            d["mean_cases_spent"] = d.pop("mean_spent")
            d["mean_leverage_pp"] = d.pop("mean_lev_pp")

    ob_summary = pack_summary(res_ob, ob_pols, (0,) + outbreak_budgets, extra_keys=("spent_ob", "spent_cases"))
    mix_summary = pack_summary(res_mix, mix_pols, (0,) + mixed_budgets, extra_keys=("spent",))

    def pair_diff_arr(a_list, b_list):
        d = np.asarray(a_list, dtype=float) - np.asarray(b_list, dtype=float)
        se = float(d.std(ddof=1) / math.sqrt(n_trials))
        return float(d.mean()), float(d.std(ddof=1)), [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)], d

    paired_diffs = OrderedDict()
    for b in case_budgets:
        m_dw, sd_dw, ci_dw, arr_dw = pair_diff_arr(res_case["dpi"][b]["worst_det"], res_case["width_greedy"][b]["worst_det"])
        m_wr, _, ci_wr, _ = pair_diff_arr(res_case["width_greedy"][b]["worst_det"], res_case["random"][b]["worst_det"])
        m_wv, _, ci_wv, _ = pair_diff_arr(res_case["width_greedy"][b]["worst_det"], res_case["volume_greedy"][b]["worst_det"])
        paired_diffs[str(b)] = OrderedDict(
            dpi_minus_wg_mean=m_dw, dpi_minus_wg_sd=sd_dw, dpi_minus_wg_ci95=ci_dw,
            share_dpi_gt_wg=float(np.mean(arr_dw > 0)),
            share_dpi_eq_wg=float(np.mean(arr_dw == 0)),
            share_dpi_lt_wg=float(np.mean(arr_dw < 0)),
            wg_minus_random_mean=m_wr, wg_minus_random_ci95=ci_wr,
            wg_minus_volume_mean=m_wv, wg_minus_volume_ci95=ci_wv,
        )

    paired_diffs_outbreak = OrderedDict()
    for b in outbreak_budgets:
        m_ov, _, ci_ov, _ = pair_diff_arr(res_ob["outbreak_width"][b]["worst_det"], res_ob["volume_greedy"][b]["worst_det"])
        m_vr, _, ci_vr, _ = pair_diff_arr(res_ob["volume_greedy"][b]["worst_det"], res_ob["random"][b]["worst_det"])
        m_vw, _, ci_vw, _ = pair_diff_arr(res_ob["volume_greedy"][b]["worst_det"], res_ob["width_greedy"][b]["worst_det"])
        paired_diffs_outbreak[str(b)] = OrderedDict(
            outbreak_width_minus_volume_mean=m_ov, outbreak_width_minus_volume_ci95=ci_ov,
            volume_minus_random_mean=m_vr, volume_minus_random_ci95=ci_vr,
            volume_minus_wg_mean=m_vw, volume_minus_wg_ci95=ci_vw,
        )

    def run_extra_regime(regime_name: str, seed_off: int):
        rr = np.random.default_rng(seed + seed_off)
        out = {pol: {b: [] for b in (0,) + sens_budgets} for pol in case_pols}
        for _ in range(n_trials):
            if regime_name == "mnar_outcome_size":
                p_out = np.clip(mask_probs_mcar * size_factor, 0.08, 0.70)
                part_flag = rr.random(len(complete_df)) < p_out
                for gi in range(G):
                    for ci in range(K):
                        idx_gc = np.where((g_idx == gi) & (c_idx == ci))[0]
                        if np.all(part_flag[idx_gc]):
                            part_flag[rr.choice(idx_gc)] = False
                H_obs0 = H_true.copy(); U_obs0 = np.zeros_like(C_arr); K_obs0 = C_arr.copy()
                for idx in np.where(part_flag)[0]:
                    h_i = int(H_true[idx]); nh_i = int(C_arr[idx] - H_true[idx])
                    mh = int(rr.binomial(h_i, 0.25)) if h_i > 0 else 0
                    mnh = int(rr.binomial(nh_i, 0.55)) if nh_i > 0 else 0
                    if mh + mnh == h_i + nh_i:
                        if h_i > 0 and mh > 0:
                            mh -= 1
                        elif nh_i > 0 and mnh > 0:
                            mnh -= 1
                    H_obs0[idx] = h_i - mh; U_obs0[idx] = mh + mnh; K_obs0[idx] = (h_i - mh) + (nh_i - mnh)
                m_idx = np.where(U_obs0 > 0)[0]; cost_arr = U_obs0
            else:
                masked_s = rr.random(len(complete_df)) < mask_probs_state
                for gi in range(G):
                    for ci in range(K):
                        idx_gc = np.where((g_idx == gi) & (c_idx == ci))[0]
                        if np.all(masked_s[idx_gc]):
                            masked_s[rr.choice(idx_gc)] = False
                H_obs0 = np.where(masked_s, 0.0, H_true)
                U_obs0 = np.where(masked_s, C_arr, 0.0)
                K_obs0 = np.where(masked_s, 0.0, C_arr)
                m_idx = np.where(masked_s)[0]; cost_arr = C_arr

            tb_x = make_table(H_obs0, U_obs0, K_obs0)
            w0_x, _, _ = eval_table(tb_x)
            for pol in case_pols:
                out[pol][0].append(w0_x)
                if pol == "random":
                    ord_p = rr.permutation(m_idx)
                elif pol == "volume_greedy":
                    ord_p = m_idx[np.argsort(-cost_arr[m_idx], kind="mergesort")]
                elif pol == "width_greedy":
                    sc = w_vec[c_idx[m_idx]] / C_cell[g_idx[m_idx], c_idx[m_idx]]
                    ord_p = m_idx[np.argsort(-sc, kind="mergesort")]
                else:
                    sc_tbl = disambiguation_scores(tb_x, eps=0.015)
                    sc = sc_tbl[g_idx[m_idx], c_idx[m_idx]]
                    ord_p = m_idx[np.argsort(-sc, kind="mergesort")]
                for b in sens_budgets:
                    ch, _ = _take_budget_filling(ord_p, cost_arr, b)
                    H_obs = H_obs0.copy(); U_obs = U_obs0.copy(); K_obs = K_obs0.copy()
                    if len(ch) > 0:
                        H_obs[ch] = H_true[ch]; U_obs[ch] = 0.0; K_obs[ch] = C_arr[ch]
                    wd, _, _ = eval_table(make_table(H_obs, U_obs, K_obs))
                    out[pol][b].append(wd)

        res_s = OrderedDict()
        for pol in case_pols:
            res_s[pol] = OrderedDict()
            for b in (0,) + sens_budgets:
                m_v, s_v = summarize_array(out[pol][b])
                res_s[pol][str(b)] = OrderedDict(mean_worst_det=m_v, sd_worst_det=s_v)
        return res_s

    mnar_summary = run_extra_regime("mnar_outcome_size", 101)
    state_mask_summary = run_extra_regime("state_clustered", 202)
    partial_summary = OrderedDict(
        (pol, OrderedDict((str(b), OrderedDict(zip(("mean_worst_det", "sd_worst_det"), summarize_array(partial_res[pol][b]))))
                          for b in sens_budgets))
        for pol in case_pols
    )
    adaptive_summary = OrderedDict(
        (str(b), OrderedDict(zip(("mean_worst_det", "sd_worst_det"), summarize_array(adaptive_res[b]))))
        for b in sens_budgets
    )
    eps_summary = OrderedDict(
        (k, OrderedDict((str(b), OrderedDict(zip(("mean_worst_det", "sd_worst_det"), summarize_array(eps_res[k][b]))))
                        for b in (1000, 2000)))
        for k in eps_res
    )

    real_scores = disambiguation_scores(t_main, eps=0.015)
    real_top = []
    for gi, g in enumerate(t_main.groups):
        for ci, c in enumerate(t_main.cells):
            unk = float(t_main.U[gi, ci])
            if unk > 0:
                real_top.append(OrderedDict(
                    group=g, cell=c, unrecorded_cases=unk,
                    total_cases=float(t_main.C[gi, ci]), weight=float(t_main.w[ci]),
                    width_leverage_pp=100.0 * float(t_main.w[ci] / t_main.C[gi, ci]),
                    dpi_per_case_pp=100.0 * float(real_scores[gi, ci]),
                ))
    real_top.sort(key=lambda r: -r["dpi_per_case_pp"])

    return OrderedDict(
        n_trials=n_trials, seed=seed, budgets_cases=list((0,) + case_budgets),
        budgets_outbreaks=list((0,) + outbreak_budgets),
        budgets_mixed=list((0,) + mixed_budgets),
        mixed_fixed_case_equiv=fixed_eq_cases,
        budget_rule="budget_filling",
        n_complete_outbreaks=int(len(complete_df)), total_complete_cases=int(C_arr.sum()),
        true_rates=true_rates, summary=summary, paired_differences=paired_diffs,
        outbreak_budget_summary=ob_summary, outbreak_paired_differences=paired_diffs_outbreak,
        mixed_cost_summary=mix_summary,
        sensitivity_regimes=OrderedDict(
            mnar_outcome_size=mnar_summary,
            state_clustered=state_mask_summary,
            partial_yield_70pct=partial_summary,
            adaptive_rescoring_dpi=adaptive_summary,
        ),
        eps_sensitivity=eps_summary,
        real_data_top_strata=real_top[:12],
    )


def run_all(main_df: pd.DataFrame, t_main: S.Table, main_res: dict, seed: int = 20261125,
            boot_block: dict | None = None) -> OrderedDict:
    """Run v4.0 30-hypothesis two-sided inference, state clustering, WCB, non-Gaussian bootstrap,
    bootstrap-world calibration, unconfounded missingness diagnostics, and cost-conditional follow-up."""
    sorted_p = sorted(
        main_res["pairs"].items(),
        key=lambda kv: -(np.inf if kv[1]["delta_star_pp"] is None else kv[1]["delta_star_pp"]),
    )
    gi_map = {g: i for i, g in enumerate(t_main.groups)}
    ordered_15 = [(gi_map[p["higher_at_delta0"]], gi_map[p["lower_at_delta0"]]) for _, p in sorted_p]
    ordered_30 = ordered_15 + [(l, h) for h, l in ordered_15]
    P15, P30 = len(ordered_15), len(ordered_30)

    inf = StackedInference(main_df, t_main, ordered_30)
    G, K = inf.G, inf.K
    off = 3 * G
    n_sim = 200000
    Z_std_30 = np.random.default_rng(seed).standard_normal((n_sim, P30))

    # Verify closed-form IF vs central-difference at worst case, delta = 5, and asymmetric (-15/+5)
    th_fd_wc, V_fd_wc, _ = inf.sandwich_cov(1.0, hc1=True)
    th_hc1, V_hc1, V_hc0, V_st_wc, st_scores_wc = inf.analytic_covariances(1.0, return_state_scores=True)
    diff_if_wc = float(np.max(np.abs(V_fd_wc - V_hc1)))

    _, V_fd_d5, _ = inf.sandwich_cov(0.05, hc1=True)
    th_d5, V_d5_hc1, _, V_st_d5, st_scores_d5 = inf.analytic_covariances(0.05, return_state_scores=True)
    diff_if_d5 = float(np.max(np.abs(V_fd_d5 - V_d5_hc1)))

    _, V_fd_asym, _ = inf.sandwich_cov(0.15, hc1=True, d_up=0.05)
    th_asym, V_asym_hc1, _, V_asym_st, _ = inf.analytic_covariances(0.15, return_state_scores=True, d_up=0.05)
    diff_if_asym = float(np.max(np.abs(V_fd_asym - V_asym_hc1)))

    # Kink cells check at delta = 0.05
    p_valid = t_main.p_obs[~np.isnan(t_main.p_obs)]
    n_kinks_d5 = int(np.sum((np.abs(p_valid - 0.05) < 1e-6) | (np.abs(p_valid - 0.95) < 1e-6)))

    # Full 30 two-sided Max-T FWER (Outbreak-HC1 & 53-State) + 15-directed check
    cov_dL_wc = V_hc1[off:off + P30, off:off + P30]
    dL_wc = th_hc1[off:off + P30]; dU_wc = th_hc1[off + P30:]
    seL_wc = np.sqrt(np.maximum(np.diag(cov_dL_wc), 1e-20))
    seU_wc = np.sqrt(np.maximum(np.diag(V_hc1[off + P30:, off + P30:]), 1e-20))
    p30_wc, mcse30_wc, c30_wc = max_t_stepdown(dL_wc / seL_wc, cov_dL_wc, 0.05, Z_std=Z_std_30)
    _, _, c15_wc = max_t_stepdown(dL_wc[:P15] / seL_wc[:P15], cov_dL_wc[:P15, :P15], 0.05, Z_std=Z_std_30[:, :P15])

    cov_dL_d5 = V_d5_hc1[off:off + P30, off:off + P30]
    dL_d5 = th_d5[off:off + P30]
    seL_d5 = np.sqrt(np.maximum(np.diag(cov_dL_d5), 1e-20))
    p30_d5, mcse30_d5, c30_d5 = max_t_stepdown(dL_d5 / seL_d5, cov_dL_d5, 0.05, Z_std=Z_std_30)
    _, _, c15_d5 = max_t_stepdown(dL_d5[:P15] / seL_d5[:P15], cov_dL_d5[:P15, :P15], 0.05, Z_std=Z_std_30[:, :P15])

    cov_st_wc = V_st_wc[off:off + P30, off:off + P30]
    seL_st_wc = np.sqrt(np.maximum(np.diag(cov_st_wc), 1e-20))
    seU_st_wc = np.sqrt(np.maximum(np.diag(V_st_wc[off + P30:, off + P30:]), 1e-20))
    p30_st_wc, _, c30_st_wc = max_t_stepdown(dL_wc / seL_st_wc, cov_st_wc, 0.05, Z_std=Z_std_30)
    _, _, c15_st_wc = max_t_stepdown(dL_wc[:P15] / seL_st_wc[:P15], cov_st_wc[:P15, :P15], 0.05, Z_std=Z_std_30[:, :P15])

    cov_st_d5 = V_st_d5[off:off + P30, off:off + P30]
    seL_st_d5 = np.sqrt(np.maximum(np.diag(cov_st_d5), 1e-20))
    p30_st_d5, _, c30_st_d5 = max_t_stepdown(dL_d5 / seL_st_d5, cov_st_d5, 0.05, Z_std=Z_std_30)

    # Asymmetric (-15/+5 pp) 30-hypothesis inference
    cov_asym_hc1 = V_asym_hc1[off:off + P30, off:off + P30]
    dL_asym = th_asym[off:off + P30]
    seL_asym = np.sqrt(np.maximum(np.diag(cov_asym_hc1), 1e-20))
    p30_asym_hc1, _, c30_asym_hc1 = max_t_stepdown(dL_asym / seL_asym, cov_asym_hc1, 0.05, Z_std=Z_std_30)

    cov_asym_st = V_asym_st[off:off + P30, off:off + P30]
    seL_asym_st = np.sqrt(np.maximum(np.diag(cov_asym_st), 1e-20))
    p30_asym_st, _, c30_asym_st = max_t_stepdown(dL_asym / seL_asym_st, cov_asym_st, 0.05, Z_std=Z_std_30)

    # Wild Cluster Bootstrap (all 30 hypotheses)
    wcb_wc = wild_cluster_bootstrap_fwer(dL_wc, st_scores_wc[:, off:off + P30], 0.05, 9999, seed=seed)
    wcb_d5 = wild_cluster_bootstrap_fwer(dL_d5, st_scores_d5[:, off:off + P30], 0.05, 9999, seed=seed + 1)

    # Dynamic c_sim_30(delta) grids
    grid_d, grid_c_hc1 = dynamic_csim_grid(inf, "hc1", Z_std_30, alpha=0.05, n_grid=21)
    _, grid_c_st = dynamic_csim_grid(inf, "state", Z_std_30, alpha=0.05, n_grid=21)

    # Fixed-scale standardized bootstrap (from bootstrap.py) & true studentized bootstrap-t (B=2000)
    sb_blk = (boot_block or {}).get("standardized_boot_max_t") or (boot_block or {}).get("nongaussian_max_t", {})
    sb_wc = sb_blk.get("worst", {})
    sb_d5 = sb_blk.get("delta5", {})
    p_sb_wc = sb_wc.get("p_fwer", [None] * P30)
    p_sb_d5 = sb_d5.get("p_fwer", [None] * P30)
    sd_sb_wc = np.asarray(sb_wc["sd_boot_pp"], dtype=float) / 100.0 if "sd_boot_pp" in sb_wc else None
    sd_sb_d5 = np.asarray(sb_d5["sd_boot_pp"], dtype=float) / 100.0 if "sd_boot_pp" in sb_d5 else None

    bt_true = run_studentized_bootstrap_t(inf, B_bt=2000, seed=seed)
    bt_wc = bt_true["worst"]; bt_d5 = bt_true["delta5"]
    c_bt_wc = float(bt_wc["c_bt"]); c_bt_d5 = float(bt_d5["c_bt"])
    bt_grid_d = np.asarray(bt_true["c_bt_grid_pp"], dtype=float) / 100.0
    bt_grid_c = np.asarray(bt_true["c_bt_grid_vals"], dtype=float)

    # Exposure-state cluster share of score variance (Carter-Schnepel-Steigerwald 2017; MNW 2023) & Multistate checks
    def cluster_var_share(scores_dL: np.ndarray):
        l2_sq = np.sum(scores_dL ** 2, axis=1)
        tot = float(l2_sq.sum())
        shares = l2_sq / tot if tot > 0 else np.zeros_like(l2_sq)
        idx_max = int(np.argmax(shares))
        g_eff = float(1.0 / np.sum(shares ** 2)) if np.sum(shares ** 2) > 0 else 0.0
        return OrderedDict(
            n_clusters=int(len(shares)),
            max_variance_share=float(shares[idx_max]),
            max_variance_cluster=inf.unique_states[idx_max],
            effective_clusters_g_eff=g_eff,
        )

    cvs_wc = cluster_var_share(st_scores_wc[:, off:off + P30])
    cvs_d5 = cluster_var_share(st_scores_d5[:, off:off + P30])

    # Multistate split as singleton clusters (52 jurisdictions + 219 singleton Multistate outbreaks)
    def eval_multistate_singleton_cov(delta_val: float):
        multi_idx = inf.state_map.get("Multistate")
        n_sing = 0
        for ki in range(G):
            for ci in range(K):
                n_sing += int(np.sum(inf.cell_states[(ki, ci)] == multi_idx)) if multi_idx is not None else 0
        other_states = [s for s in inf.unique_states if s != "Multistate"]
        omap = {inf.state_map[s]: idx for idx, s in enumerate(other_states)}
        n_cl = len(other_states) + n_sing
        _, V_h1, _, _ = inf.analytic_covariances(delta_val)
        # Recompute cluster scores cleanly
        t = inf.t_main; C_tot = float(t.C.sum()); w = t.w
        undef = np.isnan(t.p_obs); pp = np.where(undef, 0.0, t.p_obs)
        qL = np.zeros((G, K)) if delta_val >= 1.0 else np.where(undef, 0.0, np.maximum(0.0, pp - delta_val))
        qU = np.ones((G, K)) if delta_val >= 1.0 else np.where(undef, 1.0, np.minimum(1.0, pp + delta_val))
        dL_dp = np.zeros((G, K)) if delta_val >= 1.0 else np.where((~undef) & (pp > delta_val), 1.0, 0.0)
        dU_dp = np.zeros((G, K)) if delta_val >= 1.0 else np.where((~undef) & (pp < 1.0 - delta_val), 1.0, 0.0)
        rL = (t.H + t.U * qL) / t.C; rU = (t.H + t.U * qU) / t.C
        L_b = rL @ w; U_b = rU @ w
        ph = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
        pl = np.array([l for _, l in inf.ordered_pairs], dtype=int)
        scores_sg = np.zeros((n_cl, P30))
        sing_ptr = len(other_states)
        for ki in range(G):
            for ci in range(K):
                Z = inf.cell_arrays[(ki, ci)]; st_arr = inf.cell_states[(ki, ci)]
                m = Z.shape[0]; Zc = Z - inf.mu[ki, ci]
                Cc, Hc, Uc, Hrc, Krc = Zc[:, 0], Zc[:, 1], Zc[:, 2], Zc[:, 3], Zc[:, 4]
                psiL = Cc[:, None] * ((rL - L_b[:, None]) / C_tot)[:, ci][None, :]
                psiU = Cc[:, None] * ((rU - U_b[:, None]) / C_tot)[:, ci][None, :]
                k_rec = float(t.K_rec[ki, ci])
                p_tilde = (Hrc - pp[ki, ci] * Krc) / k_rec if k_rec > 0.0 else np.zeros(m)
                fac = w[ci] / float(t.C[ki, ci])
                psiL[:, ki] += fac * (Hc + qL[ki, ci] * Uc + dL_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rL[ki, ci] * Cc)
                psiU[:, ki] += fac * (Hc + qU[ki, ci] * Uc + dU_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rU[ki, ci] * Cc)
                psi_dL = psiL[:, ph] - psiU[:, pl]
                for row_i in range(m):
                    if st_arr[row_i] == multi_idx:
                        scores_sg[sing_ptr] += psi_dL[row_i]
                        sing_ptr += 1
                    else:
                        scores_sg[omap[int(st_arr[row_i])]] += psi_dL[row_i]
        return (n_cl / (n_cl - 1.0)) * (scores_sg.T @ scores_sg), n_cl

    V_sg_wc, n_cl_sg = eval_multistate_singleton_cov(1.0)
    se_sg_wc = np.sqrt(np.maximum(np.diag(V_sg_wc), 1e-20))
    p30_sg_wc, _, c30_sg_wc = max_t_stepdown(dL_wc / se_sg_wc, V_sg_wc, 0.05, Z_std=Z_std_30)
    V_sg_d5, _ = eval_multistate_singleton_cov(0.05)
    se_sg_d5 = np.sqrt(np.maximum(np.diag(V_sg_d5), 1e-20))
    p30_sg_d5, _, c30_sg_d5 = max_t_stepdown(dL_d5 / se_sg_d5, V_sg_d5, 0.05, Z_std=Z_std_30)

    # Single-state-only (R11) 52-state clustered inference
    r11_df = main_df[main_df["state"].astype(str) != "Multistate"].copy()
    t_r11 = S.build(r11_df, R.GROUPS, "setting_period", threshold=3)
    inf_r11 = StackedInference(r11_df, t_r11, ordered_30)
    th_r11_wc, _, _, V_r11_st_wc = inf_r11.analytic_covariances(1.0)
    th_r11_d5, _, _, V_r11_st_d5 = inf_r11.analytic_covariances(0.05)
    cV_r11_wc = V_r11_st_wc[off:off + P30, off:off + P30]
    p30_r11_wc, _, c30_r11_wc = max_t_stepdown(th_r11_wc[off:off + P30] / np.sqrt(np.maximum(np.diag(cV_r11_wc), 1e-20)), cV_r11_wc, 0.05, Z_std=Z_std_30)
    cV_r11_d5 = V_r11_st_d5[off:off + P30, off:off + P30]
    p30_r11_d5, _, c30_r11_d5 = max_t_stepdown(th_r11_d5[off:off + P30] / np.sqrt(np.maximum(np.diag(cV_r11_d5), 1e-20)), cV_r11_d5, 0.05, Z_std=Z_std_30)

    # Post-selection rank sets under R4a (threshold 2) and R4b (threshold 5), plus coarser R2 & equal-outbreak R10 inference
    r4_rank_sets = OrderedDict()
    for rcode, thr in (("R4a_threshold_2", 2), ("R4b_threshold_5", 5)):
        t_r = S.build(main_df, R.GROUPS, "setting_period", threshold=thr)
        inf_r = StackedInference(main_df, t_r, ordered_30)
        th_rw, V_rw, _, _ = inf_r.analytic_covariances(1.0)
        p_rw, _, _ = max_t_stepdown(th_rw[off:off + P30] / np.sqrt(np.maximum(np.diag(V_rw[off:off + P30, off:off + P30]), 1e-20)), V_rw[off:off + P30, off:off + P30], 0.05, Z_std=Z_std_30)
        th_r5, V_r5, _, _ = inf_r.analytic_covariances(0.05)
        p_r5, _, _ = max_t_stepdown(th_r5[off:off + P30] / np.sqrt(np.maximum(np.diag(V_r5[off:off + P30, off:off + P30]), 1e-20)), V_r5[off:off + P30, off:off + P30], 0.05, Z_std=Z_std_30)
        bt_r = run_studentized_bootstrap_t(inf_r, B_bt=1000, seed=seed, grid_pp=(5.0, 100.0))
        r4_rank_sets[rcode] = OrderedDict(
            worst=rank_confidence_sets(inf.groups, ordered_30, p_rw, 0.05),
            delta5=rank_confidence_sets(inf.groups, ordered_30, p_r5, 0.05),
            worst_boot_t=bt_r["worst"]["rank_confidence_sets"],
            delta5_boot_t=bt_r["delta5"]["rank_confidence_sets"],
        )

    def summarize_alt_spec(alt_df: pd.DataFrame, alt_table: S.Table, fixed_w_alt: np.ndarray | None = None) -> OrderedDict:
        inf_a = StackedInference(alt_df, alt_table, ordered_30, fixed_w=fixed_w_alt)
        th_aw, V_aw_hc1, _, V_aw_st = inf_a.analytic_covariances(1.0)
        th_a5, V_a5_hc1, _, V_a5_st = inf_a.analytic_covariances(0.05)
        dLw_a = th_aw[off:off + P30]; dL5_a = th_a5[off:off + P30]
        seLw_hc1 = np.sqrt(np.maximum(np.diag(V_aw_hc1[off:off + P30, off:off + P30]), 1e-20))
        seL5_hc1 = np.sqrt(np.maximum(np.diag(V_a5_hc1[off:off + P30, off:off + P30]), 1e-20))
        seLw_st = np.sqrt(np.maximum(np.diag(V_aw_st[off:off + P30, off:off + P30]), 1e-20))
        seL5_st = np.sqrt(np.maximum(np.diag(V_a5_st[off:off + P30, off:off + P30]), 1e-20))
        pw_hc1, _, cw_hc1 = max_t_stepdown(dLw_a / seLw_hc1, V_aw_hc1[off:off + P30, off:off + P30], 0.05, Z_std=Z_std_30)
        p5_hc1, _, c5_hc1 = max_t_stepdown(dL5_a / seL5_hc1, V_a5_hc1[off:off + P30, off:off + P30], 0.05, Z_std=Z_std_30)
        pw_st, _, cw_st = max_t_stepdown(dLw_a / seLw_st, V_aw_st[off:off + P30, off:off + P30], 0.05, Z_std=Z_std_30)
        p5_st, _, c5_st = max_t_stepdown(dL5_a / seL5_st, V_a5_st[off:off + P30, off:off + P30], 0.05, Z_std=Z_std_30)
        bt_a = run_studentized_bootstrap_t(inf_a, B_bt=2000, seed=seed, grid_pp=(5.0, 100.0))
        return OrderedDict(
            n_cells=int(alt_table.n_cells if hasattr(alt_table, "n_cells") else len(alt_table.cells)),
            min_cell_outbreaks=int(inf_a.n.min()),
            n_sig_worst=OrderedDict(hc1=int(sum(x <= 0.05 for x in pw_hc1)),
                                    state=int(sum(x <= 0.05 for x in pw_st)),
                                    boot_t=int(bt_a["worst"]["n_sig"])),
            n_sig_delta5=OrderedDict(hc1=int(sum(x <= 0.05 for x in p5_hc1)),
                                     state=int(sum(x <= 0.05 for x in p5_st)),
                                     boot_t=int(bt_a["delta5"]["n_sig"])),
            c_bt_worst=float(bt_a["worst"]["c_bt"]),
            c_bt_delta5=float(bt_a["delta5"]["c_bt"]),
            stec_nov_gii_p_worst=OrderedDict(hc1=float(pw_hc1[3]), state=float(pw_st[3]), boot_t=float(bt_a["worst"]["p_fwer"][3])),
            ranks=OrderedDict(
                worst_hc1=rank_confidence_sets(inf.groups, ordered_30, pw_hc1, 0.05),
                worst_state=rank_confidence_sets(inf.groups, ordered_30, pw_st, 0.05),
                worst_boot_t=bt_a["worst"]["rank_confidence_sets"],
                delta5_hc1=rank_confidence_sets(inf.groups, ordered_30, p5_hc1, 0.05),
                delta5_boot_t=bt_a["delta5"]["rank_confidence_sets"],
            ),
        )

    r2_df = main_df.assign(cell=main_df["period"]).copy()
    t_r2 = S.build(main_df, R.GROUPS, "period", threshold=3)
    res_r2_inf = summarize_alt_spec(r2_df, t_r2, fixed_w_alt=None)

    r10_df = main_df[main_df["cell"].isin(t_main.cells)].copy()
    r10_df["H"] = r10_df["H"] / r10_df["C"]
    r10_df["U"] = r10_df["U"] / r10_df["C"]
    r10_df["C"] = 1.0
    t_r10 = S.build(r10_df, R.GROUPS, "setting_period", fixed_cells=t_main.cells)
    t_r10.w = t_main.w.copy()
    res_r10_inf = summarize_alt_spec(r10_df, t_r10, fixed_w_alt=t_main.w)

    coarser_and_weighting_inf = OrderedDict(
        R2_period_only=res_r2_inf,
        R10_outbreak_weighted=res_r10_inf,
        R4a_threshold_2=r4_rank_sets["R4a_threshold_2"],
        R4b_threshold_5=r4_rank_sets["R4b_threshold_5"],
    )

    # Leave-one-outbreak-out (LOO) across all 3,176 common-support outbreaks + leave-largest-outbreak-out (LLO) per group
    sup_loo_df = main_df[main_df["cell"].isin(t_main.cells)].copy().reset_index(drop=True)
    ci_lookup = {c: i for i, c in enumerate(t_main.cells)}
    gi_stec = gi_map["STEC"]; gi_sal = gi_map["SAL"]
    wc_four_idx = [gi_map["CAMP"], gi_map["CPERF"], gi_map["NOV_GI"], gi_map["NOV_GII"]]
    wc_det_counts = []
    d5_det_counts = []
    stec_sal_ds_vals = []
    min_wc_margins = {t_main.groups[l]: float("inf") for l in wc_four_idx}
    worst_drop_ob = None
    min_margin_seen = float("inf")
    pairs_15_idx = [(a, b) for a in range(G) for b in range(a + 1, G)]

    for _, row in sup_loo_df.iterrows():
        gi = gi_map[row["group"]]; ci = ci_lookup[row["cell"]]
        C_l = t_main.C.copy(); H_l = t_main.H.copy(); U_l = t_main.U.copy()
        Hr_l = t_main.H_rec.copy(); Kr_l = t_main.K_rec.copy(); n_l = t_main.n.copy()
        C_l[gi, ci] -= float(row["C"]); H_l[gi, ci] -= float(row["H"]); U_l[gi, ci] -= float(row["U"])
        Hr_l[gi, ci] -= float(row["H_rec"]); Kr_l[gi, ci] -= float(row["K_rec"]); n_l[gi, ci] -= 1.0
        tb_l = S.Table(groups=list(t_main.groups), cells=list(t_main.cells),
                       C=C_l, H=H_l, U=U_l, H_rec=Hr_l, K_rec=Kr_l, n=n_l)
        tb_l.w = S.weights(tb_l, "pooled_cases")
        Lw_l = np.array([B.worst_bounds(tb_l, a)[0] for a in range(G)])
        Uw_l = np.array([B.worst_bounds(tb_l, a)[1] for a in range(G)])
        L5_l = np.array([B.group_bounds(tb_l, a, 0.05)[0] for a in range(G)])
        U5_l = np.array([B.group_bounds(tb_l, a, 0.05)[1] for a in range(G)])
        wc_det_counts.append(sum((Lw_l[a] > Uw_l[b]) or (Lw_l[b] > Uw_l[a]) for a, b in pairs_15_idx))
        d5_det_counts.append(sum((L5_l[a] > U5_l[b]) or (L5_l[b] > U5_l[a]) for a, b in pairs_15_idx))
        for l_idx in wc_four_idx:
            marg = 100.0 * float(Lw_l[gi_stec] - Uw_l[l_idx])
            g_lo = t_main.groups[l_idx]
            if marg < min_wc_margins[g_lo]:
                min_wc_margins[g_lo] = marg
            if marg < min_margin_seen:
                min_margin_seen = marg
                worst_drop_ob = OrderedDict(group=str(row["group"]), cell=str(row["cell"]), state=str(row["state"]),
                                            cases=float(row["C"]), pair=f"STEC>{g_lo}", min_margin_pp=marg)
        ds_ex, _ = B.delta_star(tb_l, gi_stec, gi_sal, bisect=False)
        if ds_ex is not None:
            stec_sal_ds_vals.append(100.0 * float(ds_ex))

    # Leave-largest-outbreak-out (one largest outbreak removed from each of the 6 groups)
    C_llo = t_main.C.copy(); H_llo = t_main.H.copy(); U_llo = t_main.U.copy()
    Hr_llo = t_main.H_rec.copy(); Kr_llo = t_main.K_rec.copy(); n_llo = t_main.n.copy()
    llo_removed = []
    for g in R.GROUPS:
        sub_g = sup_loo_df[sup_loo_df["group"] == g]
        top_row = sub_g.sort_values(["C", "H"], ascending=[False, False]).iloc[0]
        gi = gi_map[g]; ci = ci_lookup[top_row["cell"]]
        C_llo[gi, ci] -= float(top_row["C"]); H_llo[gi, ci] -= float(top_row["H"]); U_llo[gi, ci] -= float(top_row["U"])
        Hr_llo[gi, ci] -= float(top_row["H_rec"]); Kr_llo[gi, ci] -= float(top_row["K_rec"]); n_llo[gi, ci] -= 1.0
        llo_removed.append(OrderedDict(group=g, cell=str(top_row["cell"]), state=str(top_row["state"]), cases=float(top_row["C"])))
    tb_llo = S.Table(groups=list(t_main.groups), cells=list(t_main.cells),
                     C=C_llo, H=H_llo, U=U_llo, H_rec=Hr_llo, K_rec=Kr_llo, n=n_llo)
    tb_llo.w = S.weights(tb_llo, "pooled_cases")
    Lw_llo = [B.worst_bounds(tb_llo, a)[0] for a in range(G)]
    Uw_llo = [B.worst_bounds(tb_llo, a)[1] for a in range(G)]
    L5_llo = [B.group_bounds(tb_llo, a, 0.05)[0] for a in range(G)]
    U5_llo = [B.group_bounds(tb_llo, a, 0.05)[1] for a in range(G)]
    ds_llo_stec_sal, _ = B.delta_star(tb_llo, gi_stec, gi_sal, bisect=False)

    loo_influence = OrderedDict(
        n_outbreaks_checked=int(len(sup_loo_df)),
        wc_determined_range=[int(min(wc_det_counts)), int(max(wc_det_counts))],
        delta5_determined_range=[int(min(d5_det_counts)), int(max(d5_det_counts))],
        min_wc_lower_margin_pp=OrderedDict((k, float(v)) for k, v in min_wc_margins.items()),
        stec_gt_sal_delta_star_range_pp=[float(min(stec_sal_ds_vals)), float(max(stec_sal_ds_vals))],
        most_influential_single_outbreak=worst_drop_ob,
        leave_largest_per_group=OrderedDict(
            removed=llo_removed,
            wc_determined=int(sum((Lw_llo[a] > Uw_llo[b]) or (Lw_llo[b] > Uw_llo[a]) for a, b in pairs_15_idx)),
            delta5_determined=int(sum((L5_llo[a] > U5_llo[b]) or (L5_llo[b] > U5_llo[a]) for a, b in pairs_15_idx)),
            stec_gt_sal_delta_star_pp=(None if ds_llo_stec_sal is None else 100.0 * float(ds_llo_stec_sal)),
        ),
    )

    z_pw = float(norm.ppf(0.95))
    pairs = OrderedDict()
    max_dyn_err = 0.0
    kink_dists = []
    bps_all = np.concatenate([p_valid, 1.0 - p_valid])
    c_bt_unif = float(np.max(bt_grid_c))
    for idx, (key, p) in enumerate(sorted_p):
        c_im = imbens_manski_crit(float(dU_wc[idx] - dL_wc[idx]), float(seL_wc[idx]), float(seU_wc[idx]), 0.05)
        ci_lo = float(dL_wc[idx] - c_im * seL_wc[idx])
        ci_hi = float(dU_wc[idx] + c_im * seU_wc[idx])

        c_im_st = imbens_manski_crit(float(dU_wc[idx] - dL_wc[idx]), float(seL_st_wc[idx]), float(seU_st_wc[idx]), 0.05)
        ci_lo_st = float(dL_wc[idx] - c_im_st * seL_st_wc[idx])
        ci_hi_st = float(dU_wc[idx] + c_im_st * seU_st_wc[idx])

        ds_pw, _ = confidence_tipping_point_dynamic(inf, idx, "hc1", fixed_c=z_pw)
        ds_sim, c_interp = confidence_tipping_point_dynamic(inf, idx, "hc1", grid_d=grid_d, grid_c=grid_c_hc1)
        if ds_sim is not None:
            _, V_ex, _, _ = inf.analytic_covariances(ds_sim / 100.0)
            _, _, c_exact = max_t_stepdown(np.zeros(P30), V_ex[off:off + P30, off:off + P30], 0.05, Z_std=Z_std_30)
            max_dyn_err = max(max_dyn_err, abs(c_exact - c_interp))
            kink_dists.append(float(np.min(np.abs(bps_all - ds_sim / 100.0))) * 100.0)
        ds_sim_st, _ = confidence_tipping_point_dynamic(inf, idx, "state", grid_d=grid_d, grid_c=grid_c_st)
        ds_bt, _ = confidence_tipping_point_dynamic(inf, idx, "hc1", grid_d=bt_grid_d, grid_c=bt_grid_c)
        ds_bt_unif, _ = confidence_tipping_point_dynamic(inf, idx, "hc1", fixed_c=c_bt_unif)

        pairs[key] = OrderedDict(
            higher_at_delta0=p["higher_at_delta0"],
            lower_at_delta0=p["lower_at_delta0"],
            dL_worst=float(dL_wc[idx]),
            dU_worst=float(dU_wc[idx]),
            se_dL_hc1=float(seL_wc[idx]),
            se_dU_hc1=float(seU_wc[idx]),
            se_dL_state=float(seL_st_wc[idx]),
            se_dU_state=float(seU_st_wc[idx]),
            state_vs_hc1_se_ratio=float(seL_st_wc[idx] / seL_wc[idx]),
            z_dL_hc1=float(dL_wc[idx] / seL_wc[idx]),
            z_dL_state=float(dL_wc[idx] / seL_st_wc[idx]),
            c_im=float(c_im),
            ci95_im_lo=ci_lo,
            ci95_im_hi=ci_hi,
            im_excludes_zero=bool(ci_lo > 0.0),
            ci95_im_state_lo=ci_lo_st,
            ci95_im_state_hi=ci_hi_st,
            im_state_excludes_zero=bool(ci_lo_st > 0.0),
            p_fwer_worst=float(p30_wc[idx]),
            mc_se_fwer_worst=float(mcse30_wc[idx]),
            p_fwer_delta5=float(p30_d5[idx]),
            mc_se_fwer_delta5=float(mcse30_d5[idx]),
            p_fwer_state_worst=float(p30_st_wc[idx]),
            p_fwer_state_delta5=float(p30_st_d5[idx]),
            p_fwer_wcb_worst=float(wcb_wc["p_fwer_wcb"][idx]),
            p_fwer_wcb_delta5=float(wcb_d5["p_fwer_wcb"][idx]),
            p_fwer_asym_15_5_hc1=float(p30_asym_hc1[idx]),
            p_fwer_asym_15_5_state=float(p30_asym_st[idx]),
            p_fwer_boot_std_worst=(None if p_sb_wc[idx] is None else float(p_sb_wc[idx])),
            p_fwer_boot_std_delta5=(None if p_sb_d5[idx] is None else float(p_sb_d5[idx])),
            p_fwer_boot_t_worst=float(bt_wc["p_fwer"][idx]),
            mc_se_boot_t_worst=float(bt_wc["mc_se_p"][idx]),
            p_fwer_boot_t_delta5=float(bt_d5["p_fwer"][idx]),
            mc_se_boot_t_delta5=float(bt_d5["mc_se_p"][idx]),
            p_fwer_singleton_worst=float(p30_sg_wc[idx]),
            p_fwer_singleton_delta5=float(p30_sg_d5[idx]),
            p_fwer_r11_state_worst=float(p30_r11_wc[idx]),
            p_fwer_r11_state_delta5=float(p30_r11_d5[idx]),
            delta_star_pt_pp=p["delta_star_pp"],
            delta_star_95_pw_pp=ds_pw,
            delta_star_95_sim_pp=ds_sim,
            delta_star_95_sim_state_pp=ds_sim_st,
            delta_star_95_bt_pp=ds_bt,
            delta_star_95_bt_unif_pp=ds_bt_unif,
        )

    # Compute exact simultaneous rank breakdown points for upper-tier rank intervals
    def rank_bp(crit_field: str):
        val = lambda k: 100.0 if pairs[k][crit_field] is None else pairs[k][crit_field]
        s_val = min(val(f"SAL|STEC"), val("STEC|CAMP"), val("STEC|CPERF"), val("STEC|NOV_GI"), val("STEC|NOV_GII"))
        a_val = min(val("SAL|STEC"), val("SAL|CAMP"), val("SAL|CPERF"), val("SAL|NOV_GI"), val("SAL|NOV_GII"))
        c_val = min(val("SAL|CAMP"), val("STEC|CAMP"), val("CAMP|CPERF"), val("CAMP|NOV_GI"), val("CAMP|NOV_GII"))
        return OrderedDict(
            STEC_1_1=s_val,
            SAL_2_2=a_val,
            CAMP_3_3=c_val,
            joint_upper_tier_min=min(s_val, a_val, c_val),
        )

    rank_breakdown = OrderedDict(
        hc1=rank_bp("delta_star_95_sim_pp"),
        state=rank_bp("delta_star_95_sim_state_pp"),
        bootstrap_t=rank_bp("delta_star_95_bt_pp"),
        bootstrap_t_unif=rank_bp("delta_star_95_bt_unif_pp"),
    )

    per_group = OrderedDict()
    for gi, g in enumerate(inf.groups):
        per_group[g] = OrderedDict(
            L_worst=float(th_hc1[gi]),
            U_worst=float(th_hc1[G + gi]),
            se_L_hc1=float(math.sqrt(max(V_hc1[gi, gi], 0.0))),
            se_U_hc1=float(math.sqrt(max(V_hc1[G + gi, G + gi], 0.0))),
            se_L_hc0=float(math.sqrt(max(V_hc0[gi, gi], 0.0))),
            se_U_hc0=float(math.sqrt(max(V_hc0[G + gi, G + gi], 0.0))),
            se_L_state=float(math.sqrt(max(V_st_wc[gi, gi], 0.0))),
            se_U_state=float(math.sqrt(max(V_st_wc[G + gi, G + gi], 0.0))),
        )

    ranks = OrderedDict(
        worst_hc1=rank_confidence_sets(inf.groups, ordered_30, p30_wc, 0.05),
        worst_state=rank_confidence_sets(inf.groups, ordered_30, p30_st_wc, 0.05),
        worst_wcb=rank_confidence_sets(inf.groups, ordered_30, wcb_wc["p_fwer_wcb"], 0.05),
        worst_boot_std=sb_wc.get("rank_confidence_sets"),
        worst_boot_t=bt_wc["rank_confidence_sets"],
        delta5_hc1=rank_confidence_sets(inf.groups, ordered_30, p30_d5, 0.05),
        delta5_state=rank_confidence_sets(inf.groups, ordered_30, p30_st_d5, 0.05),
        delta5_wcb=rank_confidence_sets(inf.groups, ordered_30, wcb_d5["p_fwer_wcb"], 0.05),
        delta5_boot_std=sb_d5.get("rank_confidence_sets"),
        delta5_boot_t=bt_d5["rank_confidence_sets"],
        asym_15_5_hc1=rank_confidence_sets(inf.groups, ordered_30, p30_asym_hc1, 0.05),
        asym_15_5_state=rank_confidence_sets(inf.groups, ordered_30, p30_asym_st, 0.05),
        r4a_r4b=r4_rank_sets,
    )

    clust = cluster_diagnostics(main_df, t_main)
    clust["cluster_variance_share"] = OrderedDict(worst=cvs_wc, delta5=cvs_d5)
    clust["multistate_sensitivity"] = OrderedDict(
        n_singleton_clusters=int(n_cl_sg),
        c_sim_singleton_worst=float(c30_sg_wc),
        c_sim_singleton_delta5=float(c30_sg_d5),
        n_sig_singleton_worst=int(sum(x <= 0.05 for x in p30_sg_wc)),
        n_sig_singleton_delta5=int(sum(x <= 0.05 for x in p30_sg_d5)),
        n_r11_state_clusters=int(inf_r11.n_states),
        c_sim_r11_state_worst=float(c30_r11_wc),
        c_sim_r11_state_delta5=float(c30_r11_d5),
        n_sig_r11_state_worst=int(sum(x <= 0.05 for x in p30_r11_wc)),
        n_sig_r11_state_delta5=int(sum(x <= 0.05 for x in p30_r11_d5)),
    )
    small_cal = run_small_sample_calibration(inf, n_mc=400, B_inner=399, seed=seed)
    emp_cal = empirical_delta_calibration(main_df, t_main, ordered_15)
    followup = run_followup_masking(main_df, t_main, n_trials=200, seed=seed)

    ratios_st = [v["state_vs_hc1_se_ratio"] for v in pairs.values()]

    stacked = OrderedDict(
        alpha=0.05,
        n_hypotheses=int(P30),
        n_multiplier_draws=n_sim,
        c_sim_worst=float(c30_wc),
        c_sim_delta5=float(c30_d5),
        c_sim_state_worst=float(c30_st_wc),
        c_sim_state_delta5=float(c30_st_d5),
        c_sim_asym_15_5_hc1=float(c30_asym_hc1),
        c_sim_asym_15_5_state=float(c30_asym_st),
        c_sim_15_worst=float(c15_wc),
        c_sim_15_delta5=float(c15_d5),
        c_sim_15_state_worst=float(c15_st_wc),
        c_boot_std_worst=sb_wc.get("c_boot"),
        c_boot_std_delta5=sb_d5.get("c_boot"),
        c_bt_worst=float(c_bt_wc),
        c_bt_delta5=float(c_bt_d5),
        c_bonferroni_two_sided_30=float(norm.ppf(1.0 - 0.05 / P30)),
        n_im_excludes_zero=sum(v["im_excludes_zero"] for v in pairs.values()),
        n_im_state_excludes_zero=sum(v["im_state_excludes_zero"] for v in pairs.values()),
        n_fwer_worst_sig=int(sum(x <= 0.05 for x in p30_wc)),
        n_fwer_delta5_sig=int(sum(x <= 0.05 for x in p30_d5)),
        n_fwer_state_worst_sig=int(sum(x <= 0.05 for x in p30_st_wc)),
        n_fwer_state_delta5_sig=int(sum(x <= 0.05 for x in p30_st_d5)),
        n_fwer_wcb_worst_sig=int(sum(x <= 0.05 for x in wcb_wc["p_fwer_wcb"])),
        n_fwer_wcb_delta5_sig=int(sum(x <= 0.05 for x in wcb_d5["p_fwer_wcb"])),
        n_fwer_asym_15_5_hc1_sig=int(sum(x <= 0.05 for x in p30_asym_hc1)),
        n_fwer_asym_15_5_state_sig=int(sum(x <= 0.05 for x in p30_asym_st)),
        n_fwer_boot_std_worst_sig=sb_wc.get("n_sig"),
        n_fwer_boot_std_delta5_sig=sb_d5.get("n_sig"),
        n_fwer_boot_t_worst_sig=int(bt_wc["n_sig"]),
        n_fwer_boot_t_delta5_sig=int(bt_d5["n_sig"]),
        studentized_bootstrap_t=bt_true,
        state_se_ratio_min=float(min(ratios_st)),
        state_se_ratio_max=float(max(ratios_st)),
        wild_cluster_bootstrap=OrderedDict(worst=wcb_wc, delta5=wcb_d5),
        dynamic_csim_grid=OrderedDict(delta_pp=(100.0 * grid_d).tolist(),
                                      hc1=grid_c_hc1.tolist(),
                                      state=grid_c_st.tolist()),
        rank_confidence_sets=ranks,
        rank_breakdown_points_pp=rank_breakdown,
        coarser_and_weighting_inference=coarser_and_weighting_inf,
        loo_influence=loo_influence,
        per_group=per_group,
        pairs=pairs,
        checks=OrderedDict(
            max_abs_diff_analytic_if_vs_sandwich_worst=diff_if_wc,
            max_abs_diff_analytic_if_vs_sandwich_delta5=diff_if_d5,
            max_abs_diff_analytic_if_vs_sandwich_asym=diff_if_asym,
            max_abs_diff_analytic_if_vs_sandwich=max(diff_if_wc, diff_if_d5, diff_if_asym),
            max_abs_diff_dynamic_vs_exact_csim=float(max_dyn_err),
            max_abs_diff_theta_hat=float(np.max(np.abs(th_fd_wc - th_hc1))),
            n_kink_cells_delta5=n_kinks_d5,
            min_dist_to_kink_roots_pp=(float(min(kink_dists)) if kink_dists else None),
            n_strata=int(G * K),
            n_state_clusters=int(inf.n_states),
        ),
    )

    return OrderedDict(
        stacked_inference=stacked,
        cluster_diagnostics=clust,
        small_sample_calibration=small_cal,
        empirical_calibration=emp_cal,
        followup_masking=followup,
    )
