"""Stacked M-estimation inference, state clustering, calibration, rank sets, and follow-up priority.

Implements the v3.0 inferential and sensitivity suite for the NORS partial-identification study:
1. Stratified outbreak-cluster M-estimator with closed-form influence functions at any delta in [0, 1]
   (coupling shared pooled-case weight variation and within-stratum rate/completeness variation),
   cross-checked against central-difference Jacobians (HC0 and HC1).
2. Jurisdiction-clustered (reporting state) sandwich covariance and Cameron-Gelbach-Miller (2008)
   Rademacher wild cluster bootstrap across states.
3. Imbens-Manski (2004) uniformly valid 95% confidence intervals, 200,000-draw Gaussian multiplier
   Max-T stepdown FWER p-values with Monte Carlo SEs, dynamic c_sim(delta) 95% confidence tipping
   points, bootstrap-covariance FWER check, and simultaneous 95% confidence sets for ranks.
4. Stratified small-sample Monte Carlo calibration assessing boundary coverage and empirical FWER
   in sparse strata (N_eff ~ 2).
5. Empirical within-data delta calibration (fully vs partially recorded outbreaks; high-completeness
   states), asymmetric bands, and odds-ratio pattern-mixture sensitivity (Gamma).
6. Disambiguation Priority Index (DPI) vs width-greedy (w_s / C_js) follow-up allocation evaluated
   under budget-filling retrospective masking across baseline MCAR, MNAR outcome/size-dependent,
   state-clustered, 70%-yield, and adaptive re-scoring protocols.
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

    def __init__(self, main_df: pd.DataFrame, t_main: S.Table, ordered_pairs: list[tuple[int, int]]):
        self.groups = list(t_main.groups)
        self.cells = list(t_main.cells)
        self.G = len(self.groups)
        self.K = len(self.cells)
        self.P = len(ordered_pairs)
        self.t_main = t_main
        self.ordered_pairs = list(ordered_pairs)

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

    def analytic_covariances(self, delta: float, return_state_scores: bool = False):
        """Fast closed-form influence covariance at any delta in [0, 1].

        Returns (theta_hat, V_hc1, V_hc0, V_state[, state_scores]).
        Includes weight uncertainty d(w_s)/d(C_{ks}), cell rate variation, and
        within-cell recorded-rate p_{js} variation when delta < 1.
        """
        t = self.t_main
        G, K, P = self.G, self.K, self.P
        q = 3 * G + 2 * P
        C_tot = float(t.C.sum())
        w = t.w
        p_obs = t.p_obs
        undef = np.isnan(p_obs)
        pp = np.where(undef, 0.0, p_obs)

        if delta >= 1.0:
            qL = np.where(undef, 0.0, 0.0)
            qU = np.where(undef, 1.0, 1.0)
            dL_dp = np.zeros((G, K))
            dU_dp = np.zeros((G, K))
        else:
            qL = np.where(undef, 0.0, np.maximum(0.0, pp - delta))
            qU = np.where(undef, 1.0, np.minimum(1.0, pp + delta))
            dL_dp = np.where((~undef) & (pp > delta), 1.0, 0.0)
            dU_dp = np.where((~undef) & (pp < 1.0 - delta), 1.0, 0.0)
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

        # Precompute weight-channel sensitivity vectors per cell s: (3*G,)
        w_L_coeff = (rL - L_bar[:, None]) / C_tot  # (G, K)
        w_U_coeff = (rU - U_bar[:, None]) / C_tot
        w_0_coeff = (r0 - T0_bar[:, None]) / C_tot

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
    P = len(t_stats)
    sd = np.sqrt(np.maximum(np.diag(cov_L), 1e-20))
    corr = cov_L / np.outer(sd, sd)
    corr = 0.5 * (corr + corr.T)
    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 0.0)
    L_chol = vecs @ np.diag(np.sqrt(vals))

    if Z_std is None:
        rng = np.random.default_rng(seed)
        Z_std = rng.standard_normal((n_sim, P))
    Z = Z_std @ L_chol.T
    M = Z.shape[0]
    c_sim_all = float(np.quantile(Z.max(axis=1), 1.0 - alpha))

    order = np.argsort(-t_stats, kind="mergesort")
    adj_p = np.zeros(P)
    mc_se = np.zeros(P)
    prev_p = 0.0
    prev_se = 0.0
    for step in range(P):
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
    G_st, P = scores_dL.shape
    scale = math.sqrt(G_st / (G_st - 1.0)) if G_st > 1 else 1.0
    se_obs = np.sqrt(np.maximum(scale * scale * np.sum(scores_dL ** 2, axis=0), 1e-20))
    t_obs = theta_dL / se_obs

    rng = np.random.default_rng(seed)
    rad = rng.choice(np.array([-1.0, 1.0]), size=(B_wcb, G_st))
    boot_diff = scale * (rad @ scores_dL)  # (B_wcb, P)

    # Studentize each wild replicate with its own cluster-centered variance across states
    boot_se = np.zeros((B_wcb, P))
    for p in range(P):
        s_gp = rad * scores_dL[:, p][None, :]  # (B_wcb, G_st)
        s_gp_c = s_gp - s_gp.mean(axis=1, keepdims=True)
        boot_se[:, p] = np.sqrt(np.maximum(scale * scale * np.sum(s_gp_c ** 2, axis=1), 1e-20))
    T_boot = boot_diff / boot_se
    c_wcb = float(np.quantile(T_boot.max(axis=1), 1.0 - alpha))

    order = np.argsort(-t_obs, kind="mergesort")
    adj_p = np.zeros(P)
    prev_p = 0.0
    for step in range(P):
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
    """Compute Max-T critical value curve c_sim(delta) on delta in [0, 1]."""
    grid = np.linspace(0.0, 1.0, n_grid)
    c_vals = np.zeros(n_grid)
    off = 3 * inf.G
    P = inf.P
    dummy_t = np.zeros(P)
    for k, d in enumerate(grid):
        _, V_hc1, _, V_st = inf.analytic_covariances(float(d))
        V = V_hc1 if cov_kind == "hc1" else V_st
        cov_dL = V[off:off + P, off:off + P]
        _, _, c_vals[k] = max_t_stepdown(dummy_t, cov_dL, alpha=alpha, Z_std=Z_std)
    return grid, c_vals


def confidence_tipping_point_dynamic(inf: StackedInference, pair_idx: int, cov_kind: str,
                                     grid_d: np.ndarray | None = None, grid_c: np.ndarray | None = None,
                                     fixed_c: float | None = None) -> tuple[float | None, float | None]:
    """Smallest delta (pp) where dL(delta) - c(delta) * SE(dL(delta)) <= 0.

    Returns (tipping_pp, c_at_tipping). If fixed_c is None, uses interpolated c_sim(delta).
    """
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
                         p_fwer: np.ndarray, alpha: float = 0.05) -> OrderedDict:
    """Simultaneous (1 - alpha) confidence set for each pathogen's rank via joint pairwise tests."""
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
        # Top-3 state share of unrecorded cases for group g inside common support
        sub_g = sup_df[sup_df["group"] == g]
        u_by_st = sub_g.groupby("state")["U"].sum().sort_values(ascending=False)
        u_tot_g = float(sub_g["U"].sum())
        top3_share = float(u_by_st.iloc[:3].sum() / u_tot_g) if u_tot_g > 0 else 0.0
        per_group[g] = OrderedDict(
            min_n=int(min(ns)), median_n=float(np.median(ns)), max_n=int(max(ns)),
            mean_hhi=float(np.mean(hhis)), max_hhi=float(max(hhis)),
            min_neff=float(min(neffs)), max_single_share=float(max(mshs)),
            most_concentrated_cell=max_cell["cell"],
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


def run_small_sample_calibration(inf: StackedInference, n_mc: int = 500,
                                 seed: int = 20261125) -> OrderedDict:
    """Stratified Monte Carlo calibration of Imbens-Manski 95% CIs and Max-T FWER (N_eff ~ 2).

    Resamples outbreaks with replacement within each (group, cell) stratum (fixed n_js),
    treating the empirical sample bounds [Delta_L0, Delta_U0] as the true population parameters.
    """
    G, K, P = inf.G, inf.K, inf.P
    rng = np.random.default_rng(seed + 17)
    Z_std_mc = np.random.default_rng(seed + 19).standard_normal((10000, P))
    off = 3 * G

    th0_wc, _, _, _ = inf.analytic_covariances(1.0)
    dL0_wc, dU0_wc = th0_wc[off:off + P], th0_wc[off + P:]
    L0_grp, U0_grp = th0_wc[:G], th0_wc[G:2 * G]

    th0_d5, _, _, _ = inf.analytic_covariances(0.05)
    dL0_d5 = th0_d5[off:off + P]

    # Fast helper evaluating analytic cov directly from cell arrays & t_table
    def eval_resampled(res_arrays, delta_val: float):
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
        dL_hat = np.array([L_bar[h] - U_bar[l] for h, l in inf.ordered_pairs])
        dU_hat = np.array([U_bar[h] - L_bar[l] for h, l in inf.ordered_pairs])
        V_hc0 = np.zeros((q, q))
        V_hc1 = np.zeros((q, q))
        pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
        pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
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
                psi[:, 3 * G:3 * G + P] = psi[:, pair_h] - psi[:, G + pair_l]
                psi[:, 3 * G + P:] = psi[:, G + pair_h] - psi[:, pair_l]
                gram = psi.T @ psi
                V_hc0 += gram
                V_hc1 += ((m / (m - 1.0)) if m > 1 else 1.0) * gram
        return L_bar, U_bar, dL_hat, dU_hat, V_hc1, V_hc0

    im_cov_lo_hc1 = np.zeros(P); im_cov_hi_hc1 = np.zeros(P)
    im_cov_lo_hc0 = np.zeros(P); im_cov_hi_hc0 = np.zeros(P)
    fwer_rej_wc_hc1 = 0; fwer_rej_wc_hc0 = 0
    fwer_rej_d5_hc1 = 0; fwer_rej_d5_hc0 = 0

    L_draws = np.zeros((n_mc, G)); U_draws = np.zeros((n_mc, G))
    seL_draws_hc1 = np.zeros((n_mc, G)); seU_draws_hc1 = np.zeros((n_mc, G))
    dummy_t = np.zeros(P)

    for r_idx in range(n_mc):
        res_arrs = {}
        for gi in range(G):
            for ci in range(K):
                arr = inf.cell_arrays[(gi, ci)]
                m = arr.shape[0]
                idx = rng.integers(0, m, size=m)
                res_arrs[(gi, ci)] = arr[idx]

        L_r, U_r, dL_r, dU_r, V1_wc, V0_wc = eval_resampled(res_arrs, 1.0)
        L_draws[r_idx] = L_r; U_draws[r_idx] = U_r
        seL_draws_hc1[r_idx] = np.sqrt(np.maximum(np.diag(V1_wc[:G, :G]), 0.0))
        seU_draws_hc1[r_idx] = np.sqrt(np.maximum(np.diag(V1_wc[G:2 * G, G:2 * G]), 0.0))

        seL1 = np.sqrt(np.maximum(np.diag(V1_wc[off:off + P, off:off + P]), 1e-20))
        seU1 = np.sqrt(np.maximum(np.diag(V1_wc[off + P:, off + P:]), 1e-20))
        seL0 = np.sqrt(np.maximum(np.diag(V0_wc[off:off + P, off:off + P]), 1e-20))
        seU0 = np.sqrt(np.maximum(np.diag(V0_wc[off + P:, off + P:]), 1e-20))

        for p in range(P):
            c1 = imbens_manski_crit(float(dU_r[p] - dL_r[p]), float(seL1[p]), float(seU1[p]), 0.05)
            im_cov_lo_hc1[p] += ((dL_r[p] - c1 * seL1[p]) <= dL0_wc[p] <= (dU_r[p] + c1 * seU1[p]))
            im_cov_hi_hc1[p] += ((dL_r[p] - c1 * seL1[p]) <= dU0_wc[p] <= (dU_r[p] + c1 * seU1[p]))
            c0 = imbens_manski_crit(float(dU_r[p] - dL_r[p]), float(seL0[p]), float(seU0[p]), 0.05)
            im_cov_lo_hc0[p] += ((dL_r[p] - c0 * seL0[p]) <= dL0_wc[p] <= (dU_r[p] + c0 * seU0[p]))
            im_cov_hi_hc0[p] += ((dL_r[p] - c0 * seL0[p]) <= dU0_wc[p] <= (dU_r[p] + c0 * seU0[p]))

        _, _, cmax1 = max_t_stepdown(dummy_t, V1_wc[off:off + P, off:off + P], 0.05, Z_std=Z_std_mc)
        _, _, cmax0 = max_t_stepdown(dummy_t, V0_wc[off:off + P, off:off + P], 0.05, Z_std=Z_std_mc)
        if np.max((dL_r - dL0_wc) / seL1) > cmax1:
            fwer_rej_wc_hc1 += 1
        if np.max((dL_r - dL0_wc) / seL0) > cmax0:
            fwer_rej_wc_hc0 += 1

        _, _, dL_5r, _, V1_5, V0_5 = eval_resampled(res_arrs, 0.05)
        seL1_5 = np.sqrt(np.maximum(np.diag(V1_5[off:off + P, off:off + P]), 1e-20))
        seL0_5 = np.sqrt(np.maximum(np.diag(V0_5[off:off + P, off:off + P]), 1e-20))
        _, _, cmax1_5 = max_t_stepdown(dummy_t, V1_5[off:off + P, off:off + P], 0.05, Z_std=Z_std_mc)
        _, _, cmax0_5 = max_t_stepdown(dummy_t, V0_5[off:off + P, off:off + P], 0.05, Z_std=Z_std_mc)
        if np.max((dL_5r - dL0_d5) / seL1_5) > cmax1_5:
            fwer_rej_d5_hc1 += 1
        if np.max((dL_5r - dL0_d5) / seL0_5) > cmax0_5:
            fwer_rej_d5_hc0 += 1

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

    return OrderedDict(
        n_mc=int(n_mc),
        seed=int(seed + 17),
        nominal_alpha=0.05,
        im95_min_endpoint_cov_hc1=float(cov_worst_hc1.min()),
        im95_median_endpoint_cov_hc1=float(np.median(cov_worst_hc1)),
        im95_mean_endpoint_cov_hc1=float(cov_worst_hc1.mean()),
        im95_min_endpoint_cov_hc0=float(cov_worst_hc0.min()),
        im95_median_endpoint_cov_hc0=float(np.median(cov_worst_hc0)),
        im95_per_pair_min_cov_hc1=[float(x) for x in cov_worst_hc1],
        fwer_worst_hc1=float(fwer_rej_wc_hc1 / n_mc),
        fwer_worst_hc0=float(fwer_rej_wc_hc0 / n_mc),
        fwer_delta5_hc1=float(fwer_rej_d5_hc1 / n_mc),
        fwer_delta5_hc0=float(fwer_rej_d5_hc0 / n_mc),
        per_group=per_grp_cal,
    )


def empirical_delta_calibration(main_df: pd.DataFrame, t_main: S.Table,
                                ordered_pairs: list[tuple[int, int]]) -> OrderedDict:
    """Within-data delta calibration, asymmetric sensitivity, and odds-ratio Gamma bounds."""
    sup_df = main_df[main_df["cell"].isin(t_main.cells)].copy()
    full_df = sup_df[sup_df["U"] == 0]
    part_df = sup_df[(sup_df["U"] > 0) & (sup_df["K_rec"] > 0)]

    per_group = OrderedDict()
    down_calib = {}
    for i, g in enumerate(t_main.groups):
        fg = full_df[full_df["group"] == g]
        pg = part_df[part_df["group"] == g]
        r_full_pool = float(fg["H_rec"].sum() / fg["K_rec"].sum())
        r_part_pool = float(pg["H_rec"].sum() / pg["K_rec"].sum()) if len(pg) else r_full_pool

        # Cell-standardized comparison restricted to cells where both complete & partial exist
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

        p_obs_g = t_main.p_obs[i]
        r_all_std = float(np.sum(t_main.w * p_obs_g))
        # Implied downward departure if unrecorded cases had the fully-recorded rate
        gap_pool = max(0.0, r_part_pool - r_full_pool)
        gap_obs_vs_full = max(0.0, r_all_std - r_full_std)
        down_calib[g] = max(0.05, gap_pool)
        per_group[g] = OrderedDict(
            n_full_outbreaks=int(len(fg)),
            cases_full=float(fg["C"].sum()),
            rate_full_pooled=r_full_pool,
            n_partial_outbreaks=int(len(pg)),
            known_cases_partial=float(pg["K_rec"].sum()),
            rate_partial_known_pooled=r_part_pool,
            full_vs_partial_diff_pp=100.0 * (r_part_pool - r_full_pool),
            rate_full_std=r_full_std,
            rate_partial_known_std=r_part_std,
            std_obs_minus_full_pp=100.0 * gap_obs_vs_full,
            calibrated_down_pp=100.0 * down_calib[g],
        )

    # High-completeness jurisdiction benchmark (< 5% unrecorded, >= 200 cases)
    st_grp = sup_df.groupby("state").agg(C=("C", "sum"), U=("U", "sum"))
    hi_states = st_grp[(st_grp["C"] >= 200) & (st_grp["U"] / st_grp["C"] < 0.05)].index.tolist()
    hi_df = sup_df[sup_df["state"].isin(hi_states)]
    hi_rates = OrderedDict()
    for i, g in enumerate(t_main.groups):
        hg = hi_df[hi_df["group"] == g]
        kr = float(hg["K_rec"].sum())
        hi_rates[g] = OrderedDict(
            cases=float(hg["C"].sum()),
            unrecorded_share=float(hg["U"].sum() / hg["C"].sum()) if hg["C"].sum() > 0 else None,
            recorded_hosp_rate=float(hg["H_rec"].sum() / kr) if kr > 0 else None,
        )

    # Asymmetric & Group-Calibrated Box Verdicts
    asym_scenarios = [
        ("asym_10_5", "Asymmetric (-10 / +5 pp)", 0.10, 0.05),
        ("asym_15_5", "Asymmetric (-15 / +5 pp)", 0.15, 0.05),
        ("asym_20_5", "Asymmetric (-20 / +5 pp)", 0.20, 0.05),
    ]
    asym_results = OrderedDict()
    for code, label, dd, du in asym_scenarios:
        det = 0
        verdicts = OrderedDict()
        for h, l in ordered_pairs:
            key = f"{t_main.groups[min(h, l)]}|{t_main.groups[max(h, l)]}"
            Lh, Uh = B.group_bounds_asym(t_main, h, dd, du)
            Ll, Ul = B.group_bounds_asym(t_main, l, dd, du)
            v = B.verdict(Lh, Uh, Ll, Ul, t_main.groups[h], t_main.groups[l])
            if v != B.UNDET:
                det += 1
            verdicts[key] = v
        asym_results[code] = OrderedDict(label=label, down_pp=100.0 * dd, up_pp=100.0 * du,
                                        determined=int(det), verdicts=verdicts)

    # Group-calibrated asymmetric box: downward = max(5 pp, full-vs-partial diff), upward = 5 pp
    det_cal = 0
    verdicts_cal = OrderedDict()
    for h, l in ordered_pairs:
        gh, gl = t_main.groups[h], t_main.groups[l]
        key = f"{t_main.groups[min(h, l)]}|{t_main.groups[max(h, l)]}"
        Lh, Uh = B.group_bounds_asym(t_main, h, down_calib[gh], 0.05)
        Ll, Ul = B.group_bounds_asym(t_main, l, down_calib[gl], 0.05)
        v = B.verdict(Lh, Uh, Ll, Ul, gh, gl)
        if v != B.UNDET:
            det_cal += 1
        verdicts_cal[key] = v
    asym_results["calibrated_group_box"] = OrderedDict(
        label="Group-calibrated (-max(5, full-vs-partial) / +5 pp)",
        down_pp_by_group={g: 100.0 * down_calib[g] for g in t_main.groups},
        up_pp=5.0,
        determined=int(det_cal),
        verdicts=verdicts_cal,
    )

    # Odds-ratio pattern-mixture sensitivity (Gamma >= 1)
    n_zero_rec = int(np.sum((~np.isnan(t_main.p_obs)) & (t_main.p_obs == 0.0)))
    or_gammas = [1.25, 1.5, 2.0, 3.0, 5.0]
    or_by_gamma = OrderedDict()
    for gm in or_gammas:
        det_g = 0
        vmap = OrderedDict()
        for h, l in ordered_pairs:
            gh, gl = t_main.groups[h], t_main.groups[l]
            key = f"{t_main.groups[min(h, l)]}|{t_main.groups[max(h, l)]}"
            Lh, _ = B.group_bounds_or(t_main, h, gm)
            _, Ul = B.group_bounds_or(t_main, l, gm)
            v = gh if Lh > Ul else B.UNDET
            if v != B.UNDET:
                det_g += 1
            vmap[key] = v
        or_by_gamma[str(gm)] = OrderedDict(gamma=gm, determined=int(det_g), verdicts=vmap)

    gamma_star_pairs = OrderedDict()
    for h, l in ordered_pairs:
        gh, gl = t_main.groups[h], t_main.groups[l]
        key = f"{t_main.groups[min(h, l)]}|{t_main.groups[max(h, l)]}"
        gamma_star_pairs[key] = B.gamma_star(t_main, h, l)

    return OrderedDict(
        per_group=per_group,
        high_completeness_states=OrderedDict(
            n_states=int(len(hi_states)), states=hi_states, per_group=hi_rates,
        ),
        asymmetric=asym_results,
        odds_ratio=OrderedDict(
            n_cells_zero_recorded_rate=n_zero_rec,
            by_gamma=or_by_gamma,
            gamma_star_pairs=gamma_star_pairs,
        ),
    )


def disambiguation_scores(tb: S.Table, eps: float = 0.015) -> np.ndarray:
    """Marginal disambiguation priority index (DPI) per case followed up across unresolved pairs."""
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


# Backward-compatible alias
voi_scores_for_table = disambiguation_scores


def _take_budget_filling(order: np.ndarray, C_arr: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
    """Budget-filling selection: scan priority-ranked list, skipping outbreaks exceeding remaining budget."""
    chosen = []
    spent = 0.0
    for idx in order:
        c = float(C_arr[idx])
        if spent + c <= budget + 1e-12:
            chosen.append(int(idx))
            spent += c
    return np.asarray(chosen, dtype=int), spent


def run_followup_masking(main_df: pd.DataFrame, t_main: S.Table,
                         budgets=(250, 500, 1000, 1500, 2000, 3500, 5000),
                         n_trials: int = 200, seed: int = 20261125) -> OrderedDict:
    """Budget-filling retrospective masking benchmark across MCAR, MNAR, state-clustered,
    partial-yield, and adaptive re-scoring follow-up protocols."""
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

    # State-level incomplete probability for state-clustered masking scenario
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
    policies = ("random", "volume_greedy", "width_greedy", "dpi")
    all_b = (0,) + tuple(budgets)

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

    def rank_policy(pol: str, tb0: S.Table, m_indices: np.ndarray, rng_obj, eps: float = 0.015):
        if pol == "random":
            return rng_obj.permutation(m_indices)
        if pol == "volume_greedy":
            return m_indices[np.argsort(-C_arr[m_indices], kind="mergesort")]
        if pol == "width_greedy":
            score = w_vec[c_idx[m_indices]] / C_cell[g_idx[m_indices], c_idx[m_indices]]
            return m_indices[np.argsort(-score, kind="mergesort")]
        cell_val = disambiguation_scores(tb0, eps=eps)
        score = cell_val[g_idx[m_indices], c_idx[m_indices]]
        return m_indices[np.argsort(-score, kind="mergesort")]

    def summarize_array(arr):
        a = np.asarray(arr, dtype=float)
        return float(a.mean()), float(a.std(ddof=1))

    # --- 1. Main MCAR Budget-Filling Benchmark + Paired Diffs + Partial Yield + Adaptive + Eps ---
    results = {pol: {b: {"worst_det": [], "overlap_sum": [], "delta5_det": [], "spent": [], "lev_pp": []}
                     for b in all_b}
               for pol in policies}

    sens_budgets = (500, 1000, 2000, 5000)
    partial_res = {pol: {b: [] for b in sens_budgets} for pol in policies}
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

        for pol in policies:
            results[pol][0]["worst_det"].append(w0)
            results[pol][0]["delta5_det"].append(d50)
            results[pol][0]["overlap_sum"].append(100.0 * ov0)
            results[pol][0]["spent"].append(0.0)
            results[pol][0]["lev_pp"].append(0.0)

            order = rank_policy(pol, tb0, m_indices, rng, eps=0.015)
            for b in budgets:
                chosen, spent = _take_budget_filling(order, C_arr, b)
                H_obs = H_obs0.copy(); U_obs = U_obs0.copy(); K_obs = K_obs0.copy()
                if len(chosen) > 0:
                    H_obs[chosen] = H_true[chosen]
                    U_obs[chosen] = 0.0
                    K_obs[chosen] = C_arr[chosen]
                    lev = float(np.mean(100.0 * w_vec[c_idx[chosen]] / C_cell[g_idx[chosen], c_idx[chosen]]))
                else:
                    lev = 0.0
                wd, d5d, ov = eval_table(make_table(H_obs, U_obs, K_obs))
                results[pol][b]["worst_det"].append(wd)
                results[pol][b]["delta5_det"].append(d5d)
                results[pol][b]["overlap_sum"].append(100.0 * ov)
                results[pol][b]["spent"].append(spent)
                results[pol][b]["lev_pp"].append(lev)

                if b in sens_budgets:
                    # Partial follow-up yield (70% of cases resolved per selected outbreak)
                    Hp = H_obs0.copy(); Up = U_obs0.copy(); Kp = K_obs0.copy()
                    for idx in chosen:
                        c_i = int(C_arr[idx]); h_i = int(H_true[idx])
                        k_rec_i = int(rng.binomial(c_i, 0.70))
                        h_rec_i = int(rng.hypergeometric(h_i, c_i - h_i, k_rec_i)) if k_rec_i > 0 else 0
                        Hp[idx] = h_rec_i; Kp[idx] = k_rec_i; Up[idx] = c_i - k_rec_i
                    wd_p, _, _ = eval_table(make_table(Hp, Up, Kp))
                    partial_res[pol][b].append(wd_p)

        # Adaptive tranche re-scoring for DPI (250-case steps) at checkpoint budgets
        for target_b in sens_budgets:
            Ha = H_obs0.copy(); Ua = U_obs0.copy(); Ka = K_obs0.copy()
            rem_list = list(m_indices.tolist())
            spent_b = 0.0
            while rem_list:
                cur_tb = make_table(Ha, Ua, Ka)
                sc = disambiguation_scores(cur_tb, eps=0.015)
                rem_arr = np.array(rem_list, dtype=int)
                cand_order = rem_arr[np.argsort(-sc[g_idx[rem_arr], c_idx[rem_arr]], kind="mergesort")]
                tranche_taken = False
                tranche_c = 0.0
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

        # Epsilon sensitivity at 1000 and 2000 cases
        for ep in eps_vals:
            ord_ep = rank_policy("dpi", tb0, m_indices, rng, eps=ep)
            for b in (1000, 2000):
                ch_ep, _ = _take_budget_filling(ord_ep, C_arr, b)
                He = H_obs0.copy(); Ue = U_obs0.copy(); Ke = K_obs0.copy()
                if len(ch_ep) > 0:
                    He[ch_ep] = H_true[ch_ep]; Ue[ch_ep] = 0.0; Ke[ch_ep] = C_arr[ch_ep]
                wd_e, _, _ = eval_table(make_table(He, Ue, Ke))
                eps_res[str(ep)][b].append(wd_e)

    # Summarize MCAR baseline & paired differences
    summary = OrderedDict()
    for pol in policies:
        summary[pol] = OrderedDict()
        for b in all_b:
            mw, sw = summarize_array(results[pol][b]["worst_det"])
            md5, sd5 = summarize_array(results[pol][b]["delta5_det"])
            mo, so = summarize_array(results[pol][b]["overlap_sum"])
            msp, _ = summarize_array(results[pol][b]["spent"])
            mlev, _ = summarize_array(results[pol][b]["lev_pp"])
            summary[pol][str(b)] = OrderedDict(
                mean_worst_det=mw, sd_worst_det=sw,
                ci95_worst_det=[mw - 1.96 * sw / math.sqrt(n_trials), mw + 1.96 * sw / math.sqrt(n_trials)],
                mean_delta5_det=md5, sd_delta5_det=sd5,
                mean_overlap_pp=mo, sd_overlap_pp=so,
                mean_cases_spent=msp, mean_leverage_pp=mlev,
            )

    paired_diffs = OrderedDict()
    for b in budgets:
        dpi_w = np.asarray(results["dpi"][b]["worst_det"], dtype=float)
        wg_w = np.asarray(results["width_greedy"][b]["worst_det"], dtype=float)
        rnd_w = np.asarray(results["random"][b]["worst_det"], dtype=float)
        vol_w = np.asarray(results["volume_greedy"][b]["worst_det"], dtype=float)
        diff_dw = dpi_w - wg_w
        se_dw = float(diff_dw.std(ddof=1) / math.sqrt(n_trials))
        diff_wr = wg_w - rnd_w
        se_wr = float(diff_wr.std(ddof=1) / math.sqrt(n_trials))
        diff_wv = wg_w - vol_w
        se_wv = float(diff_wv.std(ddof=1) / math.sqrt(n_trials))
        paired_diffs[str(b)] = OrderedDict(
            dpi_minus_wg_mean=float(diff_dw.mean()),
            dpi_minus_wg_sd=float(diff_dw.std(ddof=1)),
            dpi_minus_wg_ci95=[float(diff_dw.mean() - 1.96 * se_dw), float(diff_dw.mean() + 1.96 * se_dw)],
            share_dpi_gt_wg=float(np.mean(diff_dw > 0)),
            share_dpi_eq_wg=float(np.mean(diff_dw == 0)),
            share_dpi_lt_wg=float(np.mean(diff_dw < 0)),
            wg_minus_random_mean=float(diff_wr.mean()),
            wg_minus_random_ci95=[float(diff_wr.mean() - 1.96 * se_wr), float(diff_wr.mean() + 1.96 * se_wr)],
            wg_minus_volume_mean=float(diff_wv.mean()),
            wg_minus_volume_ci95=[float(diff_wv.mean() - 1.96 * se_wv), float(diff_wv.mean() + 1.96 * se_wv)],
        )

    # --- 2. Additional Masking Regimes: MNAR outcome+size, State-clustered ---
    def run_extra_regime(regime_name: str, seed_off: int):
        rr = np.random.default_rng(seed + seed_off)
        out = {pol: {b: [] for b in (0,) + sens_budgets} for pol in policies}
        for _ in range(n_trials):
            if regime_name == "mnar_outcome_size":
                p_out = np.clip(mask_probs_mcar * size_factor, 0.08, 0.70)
                part_flag = rr.random(len(complete_df)) < p_out
                # Guarantee >=1 fully/partially known case in every (g, c)
                for gi in range(G):
                    for ci in range(K):
                        idx_gc = np.where((g_idx == gi) & (c_idx == ci))[0]
                        if np.all(part_flag[idx_gc]):
                            part_flag[rr.choice(idx_gc)] = False
                H_obs0 = H_true.copy(); U_obs0 = np.zeros_like(C_arr); K_obs0 = C_arr.copy()
                for idx in np.where(part_flag)[0]:
                    h_i = int(H_true[idx]); nh_i = int(C_arr[idx] - H_true[idx])
                    # Outcome-dependent: non-hospitalized masked at ~2x rate of hospitalized
                    mh = int(rr.binomial(h_i, 0.25)) if h_i > 0 else 0
                    mnh = int(rr.binomial(nh_i, 0.55)) if nh_i > 0 else 0
                    if mh + mnh == h_i + nh_i:
                        # leave at least 1 case known if total >= 1
                        if h_i > 0 and mh > 0:
                            mh -= 1
                        elif nh_i > 0 and mnh > 0:
                            mnh -= 1
                    H_obs0[idx] = h_i - mh
                    U_obs0[idx] = mh + mnh
                    K_obs0[idx] = (h_i - mh) + (nh_i - mnh)
                m_idx = np.where(U_obs0 > 0)[0]
                cost_arr = U_obs0
            else:
                # state_clustered outbreak-level masking
                masked = rr.random(len(complete_df)) < mask_probs_state
                for gi in range(G):
                    for ci in range(K):
                        idx_gc = np.where((g_idx == gi) & (c_idx == ci))[0]
                        if np.all(masked[idx_gc]):
                            masked[rr.choice(idx_gc)] = False
                H_obs0 = np.where(masked, 0.0, H_true)
                U_obs0 = np.where(masked, C_arr, 0.0)
                K_obs0 = np.where(masked, 0.0, C_arr)
                m_idx = np.where(masked)[0]
                cost_arr = C_arr

            tb0 = make_table(H_obs0, U_obs0, K_obs0)
            w0, _, _ = eval_table(tb0)
            for pol in policies:
                out[pol][0].append(w0)
                if pol == "random":
                    ord_p = rr.permutation(m_idx)
                elif pol == "volume_greedy":
                    ord_p = m_idx[np.argsort(-cost_arr[m_idx], kind="mergesort")]
                elif pol == "width_greedy":
                    sc = w_vec[c_idx[m_idx]] / C_cell[g_idx[m_idx], c_idx[m_idx]]
                    ord_p = m_idx[np.argsort(-sc, kind="mergesort")]
                else:
                    sc_tbl = disambiguation_scores(tb0, eps=0.015)
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
        for pol in policies:
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
        for pol in policies
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

    # Real-data top priority strata on the actual NORS incomplete records (all 12+ ranked)
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
        n_trials=n_trials, seed=seed, budgets_cases=list(all_b),
        budget_rule="budget_filling",
        n_complete_outbreaks=int(len(complete_df)), total_complete_cases=int(C_arr.sum()),
        true_rates=true_rates, summary=summary, paired_differences=paired_diffs,
        sensitivity_regimes=OrderedDict(
            mnar_outcome_size=mnar_summary,
            state_clustered=state_mask_summary,
            partial_yield_70pct=partial_summary,
            adaptive_rescoring_dpi=adaptive_summary,
        ),
        eps_sensitivity=eps_summary,
        real_data_top_strata=real_top[:12],
    )


def run_all(main_df: pd.DataFrame, t_main: S.Table, main_res: dict, seed: int = 20261125) -> OrderedDict:
    """Run v3.0 inference, state clustering, small-sample calibration, empirical delta calibration,
    and follow-up masking experiments."""
    sorted_p = sorted(
        main_res["pairs"].items(),
        key=lambda kv: -(np.inf if kv[1]["delta_star_pp"] is None else kv[1]["delta_star_pp"]),
    )
    gi_map = {g: i for i, g in enumerate(t_main.groups)}
    ordered_pairs = [(gi_map[p["higher_at_delta0"]], gi_map[p["lower_at_delta0"]]) for _, p in sorted_p]

    inf = StackedInference(main_df, t_main, ordered_pairs)
    G, K, P = inf.G, inf.K, inf.P
    off = 3 * G
    n_sim = 200000
    Z_std = np.random.default_rng(seed).standard_normal((n_sim, P))

    # Cross-check closed-form influence covariance vs central-difference at delta = 1.0 and 0.05
    th_fd_wc, V_fd_wc, _ = inf.sandwich_cov(1.0, hc1=True)
    th_hc1, V_hc1, V_hc0, V_st_wc, st_scores_wc = inf.analytic_covariances(1.0, return_state_scores=True)
    diff_if_wc = float(np.max(np.abs(V_fd_wc - V_hc1)))

    _, V_fd_d5, _ = inf.sandwich_cov(0.05, hc1=True)
    th_d5, V_d5_hc1, _, V_st_d5, st_scores_d5 = inf.analytic_covariances(0.05, return_state_scores=True)
    diff_if_d5 = float(np.max(np.abs(V_fd_d5 - V_d5_hc1)))

    # Outbreak-HC1 worst-case & delta=5 Max-T FWER (200,000 draws)
    cov_dL_wc = V_hc1[off:off + P, off:off + P]
    dL_wc = th_hc1[off:off + P]; dU_wc = th_hc1[off + P:]
    seL_wc = np.sqrt(np.maximum(np.diag(cov_dL_wc), 1e-20))
    seU_wc = np.sqrt(np.maximum(np.diag(V_hc1[off + P:, off + P:]), 1e-20))
    p_fwer_wc, mcse_wc, c_sim_wc = max_t_stepdown(dL_wc / seL_wc, cov_dL_wc, 0.05, Z_std=Z_std)

    cov_dL_d5 = V_d5_hc1[off:off + P, off:off + P]
    dL_d5 = th_d5[off:off + P]
    seL_d5 = np.sqrt(np.maximum(np.diag(cov_dL_d5), 1e-20))
    p_fwer_d5, mcse_d5, c_sim_d5 = max_t_stepdown(dL_d5 / seL_d5, cov_dL_d5, 0.05, Z_std=Z_std)

    # State-clustered worst-case & delta=5 Max-T FWER + Wild Cluster Bootstrap
    cov_st_wc = V_st_wc[off:off + P, off:off + P]
    seL_st_wc = np.sqrt(np.maximum(np.diag(cov_st_wc), 1e-20))
    seU_st_wc = np.sqrt(np.maximum(np.diag(V_st_wc[off + P:, off + P:]), 1e-20))
    p_fwer_st_wc, mcse_st_wc, c_sim_st_wc = max_t_stepdown(dL_wc / seL_st_wc, cov_st_wc, 0.05, Z_std=Z_std)

    cov_st_d5 = V_st_d5[off:off + P, off:off + P]
    seL_st_d5 = np.sqrt(np.maximum(np.diag(cov_st_d5), 1e-20))
    p_fwer_st_d5, mcse_st_d5, c_sim_st_d5 = max_t_stepdown(dL_d5 / seL_st_d5, cov_st_d5, 0.05, Z_std=Z_std)

    wcb_wc = wild_cluster_bootstrap_fwer(dL_wc, st_scores_wc[:, off:off + P], 0.05, 9999, seed=seed)
    wcb_d5 = wild_cluster_bootstrap_fwer(dL_d5, st_scores_d5[:, off:off + P], 0.05, 9999, seed=seed + 1)

    # Dynamic c_sim(delta) grids for outbreak-HC1 and state-clustered tipping points
    grid_d, grid_c_hc1 = dynamic_csim_grid(inf, "hc1", Z_std, alpha=0.05, n_grid=21)
    _, grid_c_st = dynamic_csim_grid(inf, "state", Z_std, alpha=0.05, n_grid=21)

    z_pw = float(norm.ppf(0.95))
    pairs = OrderedDict()
    max_dyn_err = 0.0
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
            _, _, c_exact = max_t_stepdown(np.zeros(P), V_ex[off:off + P, off:off + P], 0.05, Z_std=Z_std)
            max_dyn_err = max(max_dyn_err, abs(c_exact - c_interp))

        ds_sim_st, _ = confidence_tipping_point_dynamic(inf, idx, "state", grid_d=grid_d, grid_c=grid_c_st)

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
            p_fwer_worst=float(p_fwer_wc[idx]),
            mc_se_fwer_worst=float(mcse_wc[idx]),
            p_fwer_delta5=float(p_fwer_d5[idx]),
            mc_se_fwer_delta5=float(mcse_d5[idx]),
            p_fwer_state_worst=float(p_fwer_st_wc[idx]),
            p_fwer_state_delta5=float(p_fwer_st_d5[idx]),
            p_fwer_wcb_worst=float(wcb_wc["p_fwer_wcb"][idx]),
            p_fwer_wcb_delta5=float(wcb_d5["p_fwer_wcb"][idx]),
            delta_star_pt_pp=p["delta_star_pp"],
            delta_star_95_pw_pp=ds_pw,
            delta_star_95_sim_pp=ds_sim,
            delta_star_95_sim_state_pp=ds_sim_st,
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
        worst_hc1=rank_confidence_sets(inf.groups, ordered_pairs, p_fwer_wc, 0.05),
        delta5_hc1=rank_confidence_sets(inf.groups, ordered_pairs, p_fwer_d5, 0.05),
        worst_state=rank_confidence_sets(inf.groups, ordered_pairs, p_fwer_st_wc, 0.05),
        delta5_state=rank_confidence_sets(inf.groups, ordered_pairs, p_fwer_st_d5, 0.05),
    )

    clust = cluster_diagnostics(main_df, t_main)
    small_cal = run_small_sample_calibration(inf, n_mc=500, seed=seed)
    emp_cal = empirical_delta_calibration(main_df, t_main, ordered_pairs)
    followup = run_followup_masking(main_df, t_main, n_trials=200, seed=seed)

    ratios_st = [v["state_vs_hc1_se_ratio"] for v in pairs.values()]

    stacked = OrderedDict(
        alpha=0.05,
        n_multiplier_draws=n_sim,
        c_sim_worst=float(c_sim_wc),
        c_sim_delta5=float(c_sim_d5),
        c_sim_state_worst=float(c_sim_st_wc),
        c_sim_state_delta5=float(c_sim_st_d5),
        c_bonferroni_one_sided=float(norm.ppf(1.0 - 0.05 / P)),
        n_im_excludes_zero=sum(v["im_excludes_zero"] for v in pairs.values()),
        n_im_state_excludes_zero=sum(v["im_state_excludes_zero"] for v in pairs.values()),
        n_fwer_worst_sig=sum(v["p_fwer_worst"] <= 0.05 for v in pairs.values()),
        n_fwer_delta5_sig=sum(v["p_fwer_delta5"] <= 0.05 for v in pairs.values()),
        n_fwer_state_worst_sig=sum(v["p_fwer_state_worst"] <= 0.05 for v in pairs.values()),
        n_fwer_state_delta5_sig=sum(v["p_fwer_state_delta5"] <= 0.05 for v in pairs.values()),
        n_fwer_wcb_worst_sig=sum(v["p_fwer_wcb_worst"] <= 0.05 for v in pairs.values()),
        n_fwer_wcb_delta5_sig=sum(v["p_fwer_wcb_delta5"] <= 0.05 for v in pairs.values()),
        state_se_ratio_min=float(min(ratios_st)),
        state_se_ratio_max=float(max(ratios_st)),
        wild_cluster_bootstrap=OrderedDict(worst=wcb_wc, delta5=wcb_d5),
        dynamic_csim_grid=OrderedDict(delta_pp=(100.0 * grid_d).tolist(),
                                      hc1=grid_c_hc1.tolist(),
                                      state=grid_c_st.tolist()),
        rank_confidence_sets=ranks,
        per_group=per_group,
        pairs=pairs,
        checks=OrderedDict(
            max_abs_diff_analytic_if_vs_sandwich_worst=diff_if_wc,
            max_abs_diff_analytic_if_vs_sandwich_delta5=diff_if_d5,
            max_abs_diff_analytic_if_vs_sandwich=max(diff_if_wc, diff_if_d5),
            max_abs_diff_dynamic_vs_exact_csim=float(max_dyn_err),
            max_abs_diff_theta_hat=float(np.max(np.abs(th_fd_wc - th_hc1))),
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
