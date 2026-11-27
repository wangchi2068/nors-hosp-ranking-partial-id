"""Stacked M-estimation, 30-hypothesis one-sided inference, cluster jackknife/bootstrap, and sensitivity suite (v8.0).

Implements the v8.0 methodological suite:
1. Closed-form outbreak-level influence functions for arbitrary symmetric, asymmetric, OR (Gamma >= 1),
   and explicit plausibility-interval (P1/P2) bands, verified against central-difference Jacobians.
2. Cross-platform deterministic Cholesky Max-T stepdown (30 hypotheses), outbreak-HC1, 53-state CV1,
   leave-one-cluster-out CV3 jackknife, unrestricted Rademacher WCB, Bayesian 53-cluster bootstrap-t,
   and primary stratified studentized bootstrap-t (with conditional K_rec sensitivity).
3. Smallest-root breakdown points with local-bracket uninterpolated c_bt(delta) refinement, OR-scale
   inference (Gamma*_0.95), implied unrecorded rates at breakdown thresholds, and simultaneous 95% ranks.
4. Three double-bootstrap evaluations: stratified 400 x 499 outbreak (including CV3), Bayesian cluster
   300 x 399 across 53/271/52 clusterings, and parametric log-normal heavy-tail 300 x 399 calibration.
5. Descriptive cohort characteristics, 72-cell sparsity metrics, and bootstrap-CI missingness diagnostics.
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
        self.state_cell_sums = np.zeros((self.n_states, self.G, self.K, 5), dtype=float)
        self.state_cell_n = np.zeros((self.n_states, self.G, self.K), dtype=float)

        for gi, g in enumerate(self.groups):
            for ci, c in enumerate(self.cells):
                sub = sup_df[(sup_df["group"] == g) & (sup_df["cell"] == c)]
                Z = sub[["C", "H", "U", "H_rec", "K_rec"]].to_numpy(dtype=float)
                m = Z.shape[0]
                self.n[gi, ci] = m
                self.cell_arrays[(gi, ci)] = Z
                st_idx = np.array([self.state_map[str(s)] for s in sub["state"]], dtype=int)
                self.cell_states[(gi, ci)] = st_idx
                np.add.at(self.state_cell_sums[:, gi, ci, :], st_idx, Z)
                np.add.at(self.state_cell_n[:, gi, ci], st_idx, 1.0)
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

                # Exact uncentered cluster Gateaux weight derivative (own-cell ratio terms are degree-0 homogeneous)
                psi_cl = psi.copy()
                dC = Z[:, 0] - Cc
                psi_cl[:, :G] += dC[:, None] * w_L_coeff[:, ci][None, :]
                psi_cl[:, G:2 * G] += dC[:, None] * w_U_coeff[:, ci][None, :]
                psi_cl[:, 2 * G:3 * G] += dC[:, None] * w_0_coeff[:, ci][None, :]
                psi_cl[:, 3 * G:3 * G + P] = psi_cl[:, pair_h] - psi_cl[:, G + pair_l]
                psi_cl[:, 3 * G + P:] = psi_cl[:, G + pair_h] - psi_cl[:, pair_l]
                np.add.at(state_scores, self.cell_states[(ki, ci)], psi_cl)

        st_scale = (self.n_states / (self.n_states - 1.0)) if self.n_states > 1 else 1.0
        V_state = st_scale * (state_scores.T @ state_scores)

        if return_state_scores:
            return theta_hat, V_hc1, V_hc0, V_state, state_scores
        return theta_hat, V_hc1, V_hc0, V_state

    def analytic_covariances_or(self, gamma: float, floor: float = 0.001, return_state_scores: bool = False):
        """Closed-form influence covariance under odds-ratio factor Gamma >= 1 (with optional zero-cell floor)."""
        t = self.t_main
        G, K, P = self.G, self.K, self.P
        q = 3 * G + 2 * P
        C_tot = float(t.C.sum())
        w = t.w
        p_obs = t.p_obs
        undef = np.isnan(p_obs)
        pp = np.where(undef, 0.0, p_obs)
        if floor > 0.0:
            pp_eff = np.where(undef, 0.0, np.maximum(pp, floor))
            dp_eff = np.where((~undef) & (pp > floor), 1.0, 0.0)
        else:
            pp_eff = pp
            dp_eff = np.where(undef, 0.0, 1.0)

        denom_lo = pp_eff + gamma * (1.0 - pp_eff)
        denom_hi = gamma * pp_eff + (1.0 - pp_eff)
        qL = np.where(undef, 0.0, pp_eff / np.maximum(denom_lo, 1e-15))
        qU = np.where(undef, 1.0, (gamma * pp_eff) / np.maximum(denom_hi, 1e-15))
        dL_dp = dp_eff * (gamma / np.maximum(denom_lo * denom_lo, 1e-15))
        dU_dp = dp_eff * (gamma / np.maximum(denom_hi * denom_hi, 1e-15))

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
                psi[:, ki] += fac * (Hc + qL[ki, ci] * Uc + dL_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rL[ki, ci] * Cc)
                psi[:, G + ki] += fac * (Hc + qU[ki, ci] * Uc + dU_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rU[ki, ci] * Cc)
                psi[:, 2 * G + ki] += fac * (Hc + q0[ki, ci] * Uc + d0_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - r0[ki, ci] * Cc)
                psi[:, 3 * G:3 * G + P] = psi[:, pair_h] - psi[:, G + pair_l]
                psi[:, 3 * G + P:] = psi[:, G + pair_h] - psi[:, pair_l]
                gram = psi.T @ psi
                V_hc0 += gram
                V_hc1 += ((m / (m - 1.0)) if m > 1 else 1.0) * gram
                psi_cl = psi.copy()
                dC = Z[:, 0] - Cc
                psi_cl[:, :G] += dC[:, None] * w_L_coeff[:, ci][None, :]
                psi_cl[:, G:2 * G] += dC[:, None] * w_U_coeff[:, ci][None, :]
                psi_cl[:, 2 * G:3 * G] += dC[:, None] * w_0_coeff[:, ci][None, :]
                psi_cl[:, 3 * G:3 * G + P] = psi_cl[:, pair_h] - psi_cl[:, G + pair_l]
                psi_cl[:, 3 * G + P:] = psi_cl[:, G + pair_h] - psi_cl[:, pair_l]
                np.add.at(state_scores, self.cell_states[(ki, ci)], psi_cl)

        st_scale = (self.n_states / (self.n_states - 1.0)) if self.n_states > 1 else 1.0
        V_state = st_scale * (state_scores.T @ state_scores)
        if return_state_scores:
            return theta_hat, V_hc1, V_hc0, V_state, state_scores
        return theta_hat, V_hc1, V_hc0, V_state

    def analytic_covariances_group_asym(self, d_down_vec: np.ndarray, d_up_vec: np.ndarray):
        """Closed-form influence covariance for group-specific asymmetric deviations (G,)."""
        t = self.t_main
        G, K, P = self.G, self.K, self.P
        q = 3 * G + 2 * P
        C_tot = float(t.C.sum())
        w = t.w
        p_obs = t.p_obs
        undef = np.isnan(p_obs)
        pp = np.where(undef, 0.0, p_obs)
        dd = np.asarray(d_down_vec, dtype=float)[:, None]
        du = np.asarray(d_up_vec, dtype=float)[:, None]
        qL = np.where(undef, 0.0, np.where(dd >= 1.0, 0.0, np.maximum(0.0, pp - dd)))
        qU = np.where(undef, 1.0, np.where(du >= 1.0, 1.0, np.minimum(1.0, pp + du)))
        dL_dp = np.where((~undef) & (dd < 1.0) & (pp > dd), 1.0, 0.0)
        dU_dp = np.where((~undef) & (du < 1.0) & (pp < 1.0 - du), 1.0, 0.0)
        q0 = np.where(undef, 0.5, pp)
        d0_dp = np.where(~undef, 1.0, 0.0)
        rL = (t.H + t.U * qL) / t.C
        rU = (t.H + t.U * qU) / t.C
        r0 = (t.H + t.U * q0) / t.C
        L_bar = rL @ w; U_bar = rU @ w; T0_bar = r0 @ w
        dL_hat = np.array([L_bar[h] - U_bar[l] for h, l in self.ordered_pairs])
        dU_hat = np.array([U_bar[h] - L_bar[l] for h, l in self.ordered_pairs])
        theta_hat = np.concatenate([L_bar, U_bar, T0_bar, dL_hat, dU_hat])
        V_hc1 = np.zeros((q, q))
        state_scores = np.zeros((self.n_states, q))
        pair_h = np.array([h for h, _ in self.ordered_pairs], dtype=int)
        pair_l = np.array([l for _, l in self.ordered_pairs], dtype=int)
        w_L_coeff = np.zeros((G, K)) if self.fixed_w is not None else (rL - L_bar[:, None]) / C_tot
        w_U_coeff = np.zeros((G, K)) if self.fixed_w is not None else (rU - U_bar[:, None]) / C_tot
        w_0_coeff = np.zeros((G, K)) if self.fixed_w is not None else (r0 - T0_bar[:, None]) / C_tot
        for ki in range(G):
            for ci in range(K):
                Z = self.cell_arrays[(ki, ci)]; m = Z.shape[0]; Zc = Z - self.mu[ki, ci]
                Cc, Hc, Uc, Hrc, Krc = Zc[:, 0], Zc[:, 1], Zc[:, 2], Zc[:, 3], Zc[:, 4]
                psi = np.zeros((m, q))
                psi[:, :G] = Cc[:, None] * w_L_coeff[:, ci][None, :]
                psi[:, G:2 * G] = Cc[:, None] * w_U_coeff[:, ci][None, :]
                psi[:, 2 * G:3 * G] = Cc[:, None] * w_0_coeff[:, ci][None, :]
                k_rec = float(t.K_rec[ki, ci])
                p_tilde = (Hrc - pp[ki, ci] * Krc) / k_rec if k_rec > 0.0 else np.zeros(m)
                fac = w[ci] / float(t.C[ki, ci])
                psi[:, ki] += fac * (Hc + qL[ki, ci] * Uc + dL_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rL[ki, ci] * Cc)
                psi[:, G + ki] += fac * (Hc + qU[ki, ci] * Uc + dU_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rU[ki, ci] * Cc)
                psi[:, 2 * G + ki] += fac * (Hc + q0[ki, ci] * Uc + d0_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - r0[ki, ci] * Cc)
                psi[:, 3 * G:3 * G + P] = psi[:, pair_h] - psi[:, G + pair_l]
                psi[:, 3 * G + P:] = psi[:, G + pair_h] - psi[:, pair_l]
                V_hc1 += ((m / (m - 1.0)) if m > 1 else 1.0) * (psi.T @ psi)
                psi_cl = psi.copy()
                dC = Z[:, 0] - Cc
                psi_cl[:, :G] += dC[:, None] * w_L_coeff[:, ci][None, :]
                psi_cl[:, G:2 * G] += dC[:, None] * w_U_coeff[:, ci][None, :]
                psi_cl[:, 2 * G:3 * G] += dC[:, None] * w_0_coeff[:, ci][None, :]
                psi_cl[:, 3 * G:3 * G + P] = psi_cl[:, pair_h] - psi_cl[:, G + pair_l]
                psi_cl[:, 3 * G + P:] = psi_cl[:, G + pair_h] - psi_cl[:, pair_l]
                np.add.at(state_scores, self.cell_states[(ki, ci)], psi_cl)
        st_scale = (self.n_states / (self.n_states - 1.0)) if self.n_states > 1 else 1.0
        return theta_hat, V_hc1, st_scale * (state_scores.T @ state_scores)

    def cv3_cluster_jackknife_cov(self, delta: float | None = 1.0, d_up: float | None = None,
                                  or_gamma: float | None = None, floor: float = 0.001,
                                  d_down_vec: np.ndarray | None = None,
                                  d_up_vec: np.ndarray | None = None):
        """MacKinnon-Nielsen-Webb (2023 §4.3.1/§7) leave-one-cluster-out CV3 variance for 30 dL_hl."""
        G_st, G, K, P = self.n_states, self.G, self.K, self.P
        totals = np.sum(self.state_cell_sums, axis=0)  # (G, K, 5)
        pair_h = np.array([h for h, _ in self.ordered_pairs], dtype=int)
        pair_l = np.array([l for _, l in self.ordered_pairs], dtype=int)
        dL_loo = []
        skipped = []
        for g in range(G_st):
            sg = totals - self.state_cell_sums[g]
            C_g = sg[:, :, 0]
            if np.any(C_g <= 0.0):
                skipped.append(self.unique_states[g])
                continue
            H_g, U_g, Hr_g, Kr_g = sg[:, :, 1], sg[:, :, 2], sg[:, :, 3], sg[:, :, 4]
            if self.fixed_w is not None:
                w_g = self.fixed_w
            else:
                cs = C_g.sum(axis=0)
                w_g = cs / cs.sum()
            undef = Kr_g <= 0.0
            pp = np.where(undef, 0.0, np.divide(Hr_g, np.maximum(Kr_g, 1e-15)))
            if or_gamma is not None:
                pp_e = np.where(undef, 0.0, np.maximum(pp, floor)) if floor > 0.0 else pp
                qL = np.where(undef, 0.0, pp_e / np.maximum(pp_e + or_gamma * (1.0 - pp_e), 1e-15))
                qU = np.where(undef, 1.0, (or_gamma * pp_e) / np.maximum(or_gamma * pp_e + (1.0 - pp_e), 1e-15))
            elif d_down_vec is not None and d_up_vec is not None:
                dd_a = np.asarray(d_down_vec, dtype=float)[:, None]
                du_a = np.asarray(d_up_vec, dtype=float)[:, None]
                qL = np.where(undef, 0.0, np.where(dd_a >= 1.0, 0.0, np.maximum(0.0, pp - dd_a)))
                qU = np.where(undef, 1.0, np.where(du_a >= 1.0, 1.0, np.minimum(1.0, pp + du_a)))
            else:
                du = delta if d_up is None else d_up
                qL = np.zeros((G, K)) if delta >= 1.0 else np.where(undef, 0.0, np.maximum(0.0, pp - delta))
                qU = np.ones((G, K)) if du >= 1.0 else np.where(undef, 1.0, np.minimum(1.0, pp + du))
            rL = (H_g + U_g * qL) / C_g
            rU = (H_g + U_g * qU) / C_g
            Lb = rL @ w_g; Ub = rU @ w_g
            dL_loo.append(Lb[pair_h] - Ub[pair_l])
        mat = np.asarray(dL_loo, dtype=float)
        n_val = mat.shape[0]
        centered = mat - mat.mean(axis=0, keepdims=True)
        V_cv3 = ((n_val - 1.0) / n_val) * (centered.T @ centered)
        return V_cv3, n_val, skipped


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
                   n_sim: int = 200000, seed: int = 20261125, Z_std: np.ndarray | None = None,
                   df: float | None = None):
    """Gaussian or multivariate-t(df) multiplier stepdown FWER p-values, MC SEs, and Max-T critical value via deterministic Cholesky."""
    M_hyp = len(t_stats)
    sd = np.sqrt(np.maximum(np.diag(cov_L), 1e-20))
    corr = cov_L / np.outer(sd, sd)
    corr = 0.5 * (corr + corr.T)
    try:
        L_chol = np.linalg.cholesky(corr + 1e-10 * np.eye(M_hyp))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(corr)
        vals = np.maximum(vals, 0.0)
        for j in range(M_hyp):
            if vecs[int(np.argmax(np.abs(vecs[:, j]))), j] < 0.0:
                vecs[:, j] *= -1.0
        L_chol = vecs @ np.diag(np.sqrt(vals))

    if Z_std is None:
        rng = np.random.default_rng(seed)
        Z_std = rng.standard_normal((n_sim, M_hyp))
    if df is not None:
        rng_t = np.random.default_rng(seed + int(round(df)))
        t_scale = np.sqrt(float(df) / rng_t.chisquare(float(df), size=(Z_std.shape[0], 1)))
        Z = (Z_std @ L_chol.T) * t_scale
    else:
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
    """Unrestricted Rademacher score/multiplier wild cluster bootstrap (WCB; weaker than WCR under MNW 2023)."""
    G_st, M_hyp = scores_dL.shape
    scale = math.sqrt(G_st / (G_st - 1.0)) if G_st > 1 else 1.0
    se_obs = np.sqrt(np.maximum(scale * scale * np.sum(scores_dL ** 2, axis=0), 1e-20))
    t_obs = theta_dL / se_obs

    rng = np.random.default_rng(seed)
    rad = rng.choice(np.array([-1.0, 1.0]), size=(B_wcb, G_st))
    boot_diff = rad @ scores_dL

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
        if cov_kind == "cv3":
            cov_dL, _, _ = inf.cv3_cluster_jackknife_cov(float(d))
        else:
            _, V_hc1, _, V_st = inf.analytic_covariances(float(d))
            V = V_hc1 if cov_kind == "hc1" else V_st
            cov_dL = V[off:off + M_hyp, off:off + M_hyp]
        _, _, c_vals[k] = max_t_stepdown(dummy_t, cov_dL, alpha=alpha, Z_std=Z_std)
    return grid, c_vals


def confidence_tipping_point_dynamic(inf: StackedInference, pair_idx: int, cov_kind: str,
                                     grid_d: np.ndarray | None = None, grid_c: np.ndarray | None = None,
                                     fixed_c: float | None = None,
                                     cov_cache: dict[float, tuple[np.ndarray, np.ndarray]] | None = None) -> tuple[float | None, float | None, bool, int]:
    """Smallest delta (pp) where dL(delta) - c(delta) * SE(dL(delta)) <= 0, checking monotonicity on 201 points."""
    off = 3 * inf.G
    P = inf.P

    def crit(d: float) -> float:
        if fixed_c is not None:
            return fixed_c
        return float(np.interp(d, grid_d, grid_c))

    def get_vecs(d: float) -> tuple[np.ndarray, np.ndarray]:
        dk = round(d, 10)
        if cov_cache is not None and dk in cov_cache:
            return cov_cache[dk]
        th, V_hc1, _, V_st = inf.analytic_covariances(d)
        dL_v = th[off:off + P]
        if cov_kind == "cv3":
            V_cv3, _, _ = inf.cv3_cluster_jackknife_cov(d)
            seL_v = np.sqrt(np.maximum(np.diag(V_cv3), 0.0))
        else:
            V = V_hc1 if cov_kind == "hc1" else V_st
            seL_v = np.sqrt(np.maximum(np.diag(V[off:off + P, off:off + P]), 0.0))
        if cov_cache is not None:
            cov_cache[dk] = (dL_v, seL_v)
        return dL_v, seL_v

    def margin(d: float) -> float:
        dL_v, seL_v = get_vecs(d)
        return float(dL_v[pair_idx] - crit(d) * seL_v[pair_idx])

    d_scan = np.linspace(0.0, 1.0, 201)
    m_scan = np.array([margin(float(x)) for x in d_scan])
    is_mono = bool(np.all(np.diff(m_scan) <= 1e-12))
    pos = m_scan > 0.0
    crossings = int(np.sum(pos[:-1] & (~pos[1:])))

    if m_scan[0] <= 0.0:
        return 0.0, crit(0.0), is_mono, crossings
    idx_cross = np.where(pos[:-1] & (~pos[1:]))[0]
    if len(idx_cross) == 0:
        if m_scan[-1] > 0.0:
            return None, crit(1.0), is_mono, crossings
        return 100.0, crit(1.0), is_mono, crossings
    k0 = int(idx_cross[0])
    root = float(brentq(margin, float(d_scan[k0]), float(d_scan[k0 + 1]), xtol=1e-6))
    return 100.0 * root, crit(root), is_mono, crossings


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
                unrecorded_share=float(sub["U"].sum() / sub["C"].sum()) if sub["C"].sum() > 0 else 0.0,
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
                         fixed_indices: dict[tuple[int, int], np.ndarray] | None = None,
                         asym_vecs: tuple[np.ndarray, np.ndarray] | None = None) -> dict[float | str, tuple[np.ndarray, np.ndarray]]:
    """Vectorized HC1 dL and se(dL) across B stratified resamples via exact 5-moment algebraic expansion."""
    G, K, P = inf.G, inf.K, inf.P
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    idx_h = [np.where(pair_h == ki)[0] for ki in range(G)]
    idx_l = [np.where(pair_l == ki)[0] for ki in range(G)]
    fixed_w = inf.fixed_w

    specs: list[tuple[float | str, float | None, np.ndarray | None, np.ndarray | None]] = [
        (float(d), float(d), None, None) for d in delta_vals
    ]
    if asym_vecs is not None:
        dd_a = np.asarray(asym_vecs[0], dtype=float)[None, :, None]
        du_a = np.asarray(asym_vecs[1], dtype=float)[None, :, None]
        specs.append(("asym", None, dd_a, du_a))

    out = {k: (np.zeros((B, P)), np.zeros((B, P))) for k, _, _, _ in specs}

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

        for key_spec, delta_val, dd_a, du_a in specs:
            if dd_a is not None:
                qL = np.where(undef, 0.0, np.where(dd_a >= 1.0, 0.0, np.maximum(0.0, pp - dd_a)))
                qU = np.where(undef, 1.0, np.where(du_a >= 1.0, 1.0, np.minimum(1.0, pp + du_a)))
                dL_dp = np.where((~undef) & (dd_a < 1.0) & (pp > dd_a), 1.0, 0.0)
                dU_dp = np.where((~undef) & (du_a < 1.0) & (pp < 1.0 - du_a), 1.0, 0.0)
            elif delta_val >= 1.0:
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

            out[key_spec][0][start:end] = dL_hat
            out[key_spec][1][start:end] = np.sqrt(np.maximum(var_dL, 0.0))

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

    undef_mask = np.zeros(B_bt, dtype=bool)
    for gi in range(G):
        for ci in range(K):
            kr_col = inf.cell_arrays[(gi, ci)][:, 4]
            undef_mask |= (kr_col[fixed_indices[(gi, ci)]].sum(axis=1) <= 0.0)
    valid_krec = ~undef_mask

    dL_bw, se_bw = evals[1.0]
    n_degen_wc = int(np.sum(se_bw < 1e-12))
    T_wc = (dL_bw - dL_hat_wc[None, :]) / np.maximum(se_bw, 1e-12)

    dL_b5, se_b5 = evals[0.05]
    n_degen_d5 = int(np.sum(se_b5 < 1e-12))
    T_d5 = (dL_b5 - dL_hat_d5[None, :]) / np.maximum(se_b5, 1e-12)

    def summarize_bt(t_obs, T_mat):
        B_loc = int(T_mat.shape[0])
        c_crit = float(np.quantile(T_mat.max(axis=1), 0.95))
        order = np.argsort(-t_obs, kind="mergesort")
        adj_p = np.zeros(P); mc_se = np.zeros(P)
        prev_p = 0.0; prev_se = 0.0
        for step in range(P):
            active = order[step:]
            max_t = T_mat[:, active].max(axis=1)
            p_raw = float((1.0 + np.sum(max_t >= t_obs[order[step]])) / (B_loc + 1.0))
            se_raw = float(math.sqrt(max(p_raw * (1.0 - p_raw), 0.0) / B_loc))
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

    cond_w = summarize_bt(t_obs_wc, T_wc[valid_krec])
    cond_d5 = summarize_bt(t_obs_d5, T_d5[valid_krec])

    c_bt_grid_vals = []
    c_bt_cond_grid_vals = []
    for pp in grid_pp:
        dv = round(pp / 100.0, 6)
        th_d, _, _, _ = inf.analytic_covariances(dv)
        dL_hat_d = th_d[off:off + P]
        dL_bd, se_bd = evals[dv]
        T_d = (dL_bd - dL_hat_d[None, :]) / np.maximum(se_bd, 1e-12)
        c_bt_grid_vals.append(float(np.quantile(T_d.max(axis=1), 0.95)))
        c_bt_cond_grid_vals.append(float(np.quantile(T_d[valid_krec].max(axis=1), 0.95)))

    return OrderedDict(
        B_bt=int(B_bt),
        n_replicates_undef_krec=int(undef_mask.sum()),
        cond_valid_B=int(valid_krec.sum()),
        worst=w_res, delta5=d_res,
        cond_worst=cond_w, cond_delta5=cond_d5,
        c_bt_grid_pp=list(grid_pp), c_bt_grid_vals=c_bt_grid_vals,
        c_bt_cond_grid_vals=c_bt_cond_grid_vals,
        fixed_indices=fixed_indices,
    )


def _batched_boot_t_or_dL(inf: StackedInference, cell_arrays: dict, gamma_vals: list[float],
                            B: int, fixed_indices: dict[tuple[int, int], np.ndarray],
                            floor: float = 0.001, batch_size: int = 200) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Vectorized HC1 dL and se(dL) under OR scale Gamma >= 1 via exact 5-moment algebraic expansion."""
    G, K, P = inf.G, inf.K, inf.P
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    idx_h = [np.where(pair_h == ki)[0] for ki in range(G)]
    idx_l = [np.where(pair_l == ki)[0] for ki in range(G)]
    out = {float(g): (np.zeros((B, P)), np.zeros((B, P))) for g in gamma_vals}

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        bs = end - start
        cell_Z = {}
        sums = np.zeros((bs, G, K, 5))
        mu_r = np.zeros((bs, G, K, 5))
        for gi in range(G):
            for ci in range(K):
                arr = cell_arrays[(gi, ci)]
                idx = fixed_indices[(gi, ci)][start:end]
                Z = arr[idx]
                cell_Z[(gi, ci)] = Z
                sums[:, gi, ci, :] = Z.sum(axis=1)
                mu_r[:, gi, ci, :] = Z.mean(axis=1)

        C_mat, H_mat, U_mat, Hr_mat, Kr_mat = sums[:, :, :, 0], sums[:, :, :, 1], sums[:, :, :, 2], sums[:, :, :, 3], sums[:, :, :, 4]
        C_cell_sum = C_mat.sum(axis=1)
        C_tot = np.maximum(C_cell_sum.sum(axis=1), 1e-15)
        w = C_cell_sum / C_tot[:, None]
        undef = Kr_mat <= 0.0
        pp = np.where(undef, 0.0, np.divide(Hr_mat, np.maximum(Kr_mat, 1e-15)))
        if floor > 0.0:
            pp_eff = np.where(undef, 0.0, np.maximum(pp, floor))
            dp_eff = np.where((~undef) & (pp > floor), 1.0, 0.0)
        else:
            pp_eff = pp
            dp_eff = np.where(undef, 0.0, 1.0)

        for g_raw in gamma_vals:
            gv = float(g_raw)
            dlo = np.maximum(pp_eff + gv * (1.0 - pp_eff), 1e-15)
            dhi = np.maximum(gv * pp_eff + (1.0 - pp_eff), 1e-15)
            qL = np.where(undef, 0.0, pp_eff / dlo)
            qU = np.where(undef, 1.0, (gv * pp_eff) / dhi)
            dL_dp = dp_eff * (gv / (dlo * dlo))
            dU_dp = dp_eff * (gv / (dhi * dhi))
            C_safe = np.maximum(C_mat, 1e-15)
            rL = (H_mat + U_mat * qL) / C_safe
            rU = (H_mat + U_mat * qU) / C_safe
            L_bar = np.einsum("bsk,bk->bs", rL, w)
            U_bar = np.einsum("bsk,bk->bs", rU, w)
            dL_hat = L_bar[:, pair_h] - U_bar[:, pair_l]
            w_pair_coeff = ((rL - L_bar[:, :, None]) / C_tot[:, None, None])[:, pair_h, :] - ((rU - U_bar[:, :, None]) / C_tot[:, None, None])[:, pair_l, :]

            var_dL = np.zeros((bs, P))
            for ki in range(G):
                ih = idx_h[ki]; il = idx_l[ki]
                for ci in range(K):
                    Z = cell_Z[(ki, ci)]; m = Z.shape[1]
                    Zc = Z - mu_r[:, ki, ci, :][:, None, :]
                    Cc, Hc, Uc, Hrc, Krc = Zc[:, :, 0], Zc[:, :, 1], Zc[:, :, 2], Zc[:, :, 3], Zc[:, :, 4]
                    k_rec = Kr_mat[:, ki, ci][:, None]
                    p_tilde = np.where(k_rec > 0.0, (Hrc - pp[:, ki, ci][:, None] * Krc) / np.maximum(k_rec, 1e-15), 0.0)
                    fac = (w[:, ci] / C_safe[:, ki, ci])[:, None]
                    own_L = fac * (Hc + qL[:, ki, ci][:, None] * Uc + (dL_dp[:, ki, ci] * U_mat[:, ki, ci])[:, None] * p_tilde - rL[:, ki, ci][:, None] * Cc)
                    own_U = fac * (Hc + qU[:, ki, ci][:, None] * Uc + (dU_dp[:, ki, ci] * U_mat[:, ki, ci])[:, None] * p_tilde - rU[:, ki, ci][:, None] * Cc)
                    scale = (m / (m - 1.0)) if m > 1 else 1.0
                    s_cc = np.sum(Cc * Cc, axis=1)
                    s_cL = np.sum(Cc * own_L, axis=1); s_LL = np.sum(own_L * own_L, axis=1)
                    s_cU = np.sum(Cc * own_U, axis=1); s_UU = np.sum(own_U * own_U, axis=1)
                    a = w_pair_coeff[:, :, ci]
                    var_dL += scale * (a * a) * s_cc[:, None]
                    if len(ih):
                        var_dL[:, ih] += scale * (2.0 * a[:, ih] * s_cL[:, None] + s_LL[:, None])
                    if len(il):
                        var_dL[:, il] += scale * (-2.0 * a[:, il] * s_cU[:, None] + s_UU[:, None])
            out[gv][0][start:end] = dL_hat
            out[gv][1][start:end] = np.sqrt(np.maximum(var_dL, 0.0))
    return out


def _batched_cluster_boot_t_dL(inf: StackedInference, state_sums: np.ndarray, state_n: np.ndarray,
                                 delta_vals: list[float], B: int, rng: np.random.Generator,
                                 batch_size: int = 200) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Bayesian cluster bootstrap-t via i.i.d. Exp(1) weights across exposure-state clusters (preserving within-state & Multistate dependence)."""
    G_st, G, K, _ = state_sums.shape
    P = inf.P
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    out = {float(d): (np.zeros((B, P)), np.zeros((B, P))) for d in delta_vals}
    st_scale = (G_st / (G_st - 1.0)) if G_st > 1 else 1.0

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        bs = end - start
        W_b = rng.exponential(1.0, size=(bs, G_st))
        W_b = W_b / np.mean(W_b, axis=1, keepdims=True)  # normalize mean cluster weight = 1
        sums_b = np.einsum("bg,gskv->bskv", W_b, state_sums)   # (bs, G, K, 5)

        C_mat, H_mat, U_mat, Hr_mat, Kr_mat = sums_b[:, :, :, 0], sums_b[:, :, :, 1], sums_b[:, :, :, 2], sums_b[:, :, :, 3], sums_b[:, :, :, 4]
        C_cell = C_mat.sum(axis=1)
        C_tot = np.maximum(C_cell.sum(axis=1), 1e-15)
        w = C_cell / C_tot[:, None]
        undef = Kr_mat <= 0.0
        pp = np.where(undef, 0.0, np.divide(Hr_mat, np.maximum(Kr_mat, 1e-15)))
        C_safe = np.maximum(C_mat, 1e-15)
        Kt_safe = np.maximum(Kr_mat, 1e-15)

        # Uncentered cluster sums give the exact Gateaux derivative matching reweighted totals W_b @ state_sums
        Cc = state_sums[None, :, :, :, 0]; Hc = state_sums[None, :, :, :, 1]; Uc = state_sums[None, :, :, :, 2]
        Hrc = state_sums[None, :, :, :, 3]; Krc = state_sums[None, :, :, :, 4]
        p_tilde_g = np.where(Kr_mat[:, None, :, :] > 0.0, (Hrc - pp[:, None, :, :] * Krc) / Kt_safe[:, None, :, :], 0.0)
        fac = w[:, None, None, :] / C_safe[:, None, :, :]

        for d_raw in delta_vals:
            dv = float(d_raw)
            if dv >= 1.0:
                qL = np.zeros((bs, G, K)); qU = np.ones((bs, G, K))
                dL_dp = np.zeros((bs, G, K)); dU_dp = np.zeros((bs, G, K))
            else:
                qL = np.where(undef, 0.0, np.maximum(0.0, pp - dv))
                qU = np.where(undef, 1.0, np.minimum(1.0, pp + dv))
                dL_dp = np.where((~undef) & (pp > dv), 1.0, 0.0)
                dU_dp = np.where((~undef) & (pp < 1.0 - dv), 1.0, 0.0)
            rL = (H_mat + U_mat * qL) / C_safe
            rU = (H_mat + U_mat * qU) / C_safe
            L_bar = np.einsum("bsk,bk->bs", rL, w)
            U_bar = np.einsum("bsk,bk->bs", rU, w)
            dL_hat = L_bar[:, pair_h] - U_bar[:, pair_l]

            w_L_coeff = (rL - L_bar[:, :, None]) / C_tot[:, None, None]
            w_U_coeff = (rU - U_bar[:, :, None]) / C_tot[:, None, None]
            # Cluster-level influence on L_j and U_j: (bs, G_st, G)
            psiL_g = np.einsum("bgks,bjs->bgj", Cc, w_L_coeff) + np.sum(fac * (
                Hc + qL[:, None, :, :] * Uc + (dL_dp * U_mat)[:, None, :, :] * p_tilde_g - rL[:, None, :, :] * Cc
            ), axis=3)
            psiU_g = np.einsum("bgks,bjs->bgj", Cc, w_U_coeff) + np.sum(fac * (
                Hc + qU[:, None, :, :] * Uc + (dU_dp * U_mat)[:, None, :, :] * p_tilde_g - rU[:, None, :, :] * Cc
            ), axis=3)
            psi_raw = psiL_g[:, :, pair_h] - psiU_g[:, :, pair_l]  # (bs, G_st, P)
            var_st = st_scale * np.sum(W_b[:, :, None] * (psi_raw * psi_raw), axis=1)
            out[dv][0][start:end] = dL_hat
            out[dv][1][start:end] = np.sqrt(np.maximum(var_st, 0.0))
    return out


def run_small_sample_calibration(inf: StackedInference,
                                 n_mc: int = 400,
                                 B_inner: int = 499,
                                 seed: int = 20261125) -> OrderedDict:
    """Non-circular stratified double bootstrap (Beran 1988) under least-favourable boundary configuration.

    Inside every outer bootstrap world, all 7 procedures re-estimate their own critical value:
      (1) HC0 Gaussian, (2) HC1 Gaussian, (3) Fixed-scale standardized bootstrap (B_inner draws),
      (4) Studentized bootstrap-t (B_inner draws with analytic inner HC1 SEs),
      (5) 53-exposure-state Gaussian (CV1), (6) Unrestricted Rademacher WCB (B_inner draws),
      (7) CV3 leave-one-state-out jackknife.
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
        boot_diff = rad @ state_scores_dL
        sum_sq = np.sum(state_scores_dL ** 2, axis=0)[None, :]
        mean_term = (rad @ state_scores_dL) / G_st
        boot_se = scale * np.sqrt(np.maximum(sum_sq - G_st * (mean_term ** 2), 1e-24))
        c_loc = float(np.quantile((boot_diff / boot_se).max(axis=1), 0.95))
        return bool(t_max_obs > c_loc)

    im_cov_lo_hc1 = np.zeros(P15); im_cov_hi_hc1 = np.zeros(P15)
    im_cov_lo_hc0 = np.zeros(P15); im_cov_hi_hc0 = np.zeros(P15)
    keys_m = ("hc0", "hc1", "fixed_scale_boot", "bootstrap_t", "state", "cv3_jackknife", "wcb")
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
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    G_st = inf.n_states

    def resampled_cv3_cov(res_arrs, res_sts, dv: float) -> np.ndarray | None:
        st_sums = np.zeros((G_st, G, K, 5), dtype=float)
        for gi in range(G):
            for ci in range(K):
                np.add.at(st_sums[:, gi, ci, :], res_sts[(gi, ci)], res_arrs[(gi, ci)])
        tot = st_sums.sum(axis=0)
        loo = tot[None, ...] - st_sums
        valid = np.all(loo[..., 0] > 1e-12, axis=(1, 2))
        if int(valid.sum()) < 3:
            return None
        sl = loo[valid]
        Cm, Hm, Um, Hrm, Krm = sl[..., 0], sl[..., 1], sl[..., 2], sl[..., 3], sl[..., 4]
        Cs = Cm.sum(axis=-2); Ct = np.maximum(Cs.sum(axis=-1), 1e-15); w_a = Cs / Ct[..., None]
        und = Krm <= 0.0; pp_a = np.where(und, 0.0, np.divide(Hrm, np.maximum(Krm, 1e-15)))
        if dv >= 1.0:
            qL_a = np.zeros_like(pp_a); qU_a = np.ones_like(pp_a)
        else:
            qL_a = np.where(und, 0.0, np.maximum(0.0, pp_a - dv))
            qU_a = np.where(und, 1.0, np.minimum(1.0, pp_a + dv))
        rL_a = (Hm + Um * qL_a) / Cm; rU_a = (Hm + Um * qU_a) / Cm
        Lb = np.einsum("vks,vs->vk", rL_a, w_a); Ub = np.einsum("vks,vs->vk", rU_a, w_a)
        dL_l = Lb[:, pair_h] - Ub[:, pair_l]
        nv = dL_l.shape[0]
        cent = dL_l - dL_l.mean(axis=0, keepdims=True)
        return ((nv - 1.0) / nv) * (cent.T @ cent)

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
        V_c3w = resampled_cv3_cov(res_arrs, res_sts, 1.0)
        if V_c3w is not None:
            _, _, cc3w = max_t_stepdown(dummy_t, V_c3w, 0.05, Z_std=Z_std_mc)
            sec3w = np.sqrt(np.maximum(np.diag(V_c3w), 1e-20))
            fwer_wc["cv3_jackknife"] += int(float(np.max((dL_r - dL0_wc) / sec3w)) > cc3w)

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
        V_c35 = resampled_cv3_cov(res_arrs, res_sts, 0.05)
        if V_c35 is not None:
            _, _, cc35 = max_t_stepdown(dummy_t, V_c35, 0.05, Z_std=Z_std_mc)
            sec35 = np.sqrt(np.maximum(np.diag(V_c35), 1e-20))
            fwer_d5["cv3_jackknife"] += int(float(np.max((dL_5r - dL0_d5) / sec35)) > cc35)

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


def _fit_cell_betabinom_params(inf: StackedInference):
    """Estimate per-cell empirical recording/hospitalization proportions and Beta-Binomial concentrations."""
    G, K = inf.G, inf.K
    params = {}
    for gi in range(G):
        for ci in range(K):
            arr = inf.cell_arrays[(gi, ci)]
            C_v = np.maximum(arr[:, 0], 1.0)
            H_v, Hr_v, Kr_v = arr[:, 1], arr[:, 3], arr[:, 4]
            pk = float(np.clip(Kr_v.sum() / max(C_v.sum(), 1.0), 0.02, 0.98))
            ph = float(np.clip(Hr_v.sum() / max(Kr_v.sum(), 1.0), 0.005, 0.98)) if Kr_v.sum() > 0 else 0.01
            # Empirical Pearson dispersion chi2/df on K|C and H|K
            vk = np.maximum(C_v * pk * (1.0 - pk), 1e-6)
            phi_k = float(np.clip(np.mean(((Kr_v - C_v * pk) ** 2) / vk), 1.5, 45.0))
            kr_pos = Kr_v > 0.0
            if np.sum(kr_pos) > 1:
                vh = np.maximum(Kr_v[kr_pos] * ph * (1.0 - ph), 1e-6)
                phi_h = float(np.clip(np.mean(((Hr_v[kr_pos] - Kr_v[kr_pos] * ph) ** 2) / vh), 1.2, 25.0))
            else:
                phi_h = 3.0
            c_bar = max(float(C_v.mean()), 2.0)
            k_bar = max(float(Kr_v[kr_pos].mean()) if np.any(kr_pos) else c_bar * pk, 2.0)
            mk = float(np.clip((c_bar - phi_k) / max(phi_k - 1.0, 0.1), 0.4, 25.0))
            mh = float(np.clip((k_bar - phi_h) / max(phi_h - 1.0, 0.1), 0.8, 40.0))
            params[(gi, ci)] = (arr, inf.cell_states[(gi, ci)], pk, ph, mk, mh)
    return params


def _beta_mean_preserving(rng: np.random.Generator, p, conc: float, size=None):
    """Draw Beta(c*p, c*(1-p)) while scaling concentration instead of asymmetric shape flooring."""
    p_arr = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    c_eff = np.maximum(float(conc), 1e-4 / np.minimum(p_arr, 1.0 - p_arr))
    return rng.beta(c_eff * p_arr, c_eff * (1.0 - p_arr), size=size)


def run_cluster_double_bootstrap(inf: StackedInference, n_mc: int = 300,
                                 B_inner: int = 399, seed: int = 20261125,
                                 include_cbt: bool = True) -> OrderedDict:
    """Valid unweighted synthetic cluster random-effects + Beta-Binomial calibration across G_st clusters.

    Preserves exact outbreak-to-cluster mapping (state_53, singleton_271, or r11_52), draws cluster
    random effects on recording & hospitalization logits plus empirically overdispersed Beta-Binomial
    outcomes, and evaluates UNWEIGHTED CV1, UNWEIGHTED leave-one-cluster-out CV3 jackknife, WCB, and
    linear-weight Bayesian cluster bootstrap-t with ZERO W_g^2 SE inflation.
    """
    G_st, G, K, P = inf.n_states, inf.G, inf.K, inf.P
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    st_scale = (G_st / (G_st - 1.0)) if G_st > 1 else 1.0
    Z_std = np.random.default_rng(seed + 109).standard_normal((10000, P))
    rng_ref = np.random.default_rng(seed + 97)
    rng_out = np.random.default_rng(seed + 101)
    rng_in = np.random.default_rng(seed + 103)
    rng_wcb = np.random.default_rng(seed + 107)
    dummy_t = np.zeros(P)
    bb_params = _fit_cell_betabinom_params(inf)
    sigma_uk, sigma_uh = 0.65, 0.45

    def logit(p):
        pc = np.clip(p, 1e-4, 1.0 - 1e-4)
        return np.log(pc / (1.0 - pc))

    def expit(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -8.0, 8.0)))

    def sample_cluster_world(rng_loc: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        uk = rng_loc.normal(0.0, sigma_uk, size=G_st)
        uh = rng_loc.normal(0.0, sigma_uh, size=G_st)
        st_sums = np.zeros((G_st, G, K, 5), dtype=float)
        st_n = np.zeros((G_st, G, K), dtype=float)
        for gi in range(G):
            for ci in range(K):
                arr, cl_idx, pk0, ph0, mk, mh = bb_params[(gi, ci)]
                m = arr.shape[0]
                # Preserve exact observed outbreak sizes per cluster so Multistate dominance is maintained
                C_s = arr[:, 0]
                pk_i = expit(logit(pk0) + uk[cl_idx])
                ph_i = expit(logit(ph0) + uh[cl_idx])
                p_kd = _beta_mean_preserving(rng_loc, pk_i, mk)
                K_s = rng_loc.binomial(C_s.astype(int), p_kd).astype(float)
                p_hd = _beta_mean_preserving(rng_loc, ph_i, mh)
                H_s = rng_loc.binomial(K_s.astype(int), p_hd).astype(float)
                U_s = C_s - K_s
                Z_s = np.stack([C_s, H_s, U_s, H_s, K_s], axis=1)
                np.add.at(st_sums[:, gi, ci, :], cl_idx, Z_s)
                np.add.at(st_n[:, gi, ci], cl_idx, 1.0)
        return st_sums, st_n

    # Compute exact population boundary expectation over n_ref = 3000 synthetic worlds
    n_ref = 3000
    tot_ref = np.zeros((G, K, 5), dtype=float)
    for _ in range(n_ref):
        ss_r, _ = sample_cluster_world(rng_ref)
        tot_ref += ss_r.sum(axis=0)
    mean_tot = tot_ref / float(n_ref)

    def eval_table_dL(sums_arr, dv: float):
        Cm, Hm, Um, Hrm, Krm = sums_arr[..., 0], sums_arr[..., 1], sums_arr[..., 2], sums_arr[..., 3], sums_arr[..., 4]
        Cs = Cm.sum(axis=-2); Ct = np.maximum(Cs.sum(axis=-1), 1e-15); w_a = Cs / Ct[..., None]
        und = Krm <= 0.0; pp_a = np.where(und, 0.0, np.divide(Hrm, np.maximum(Krm, 1e-15)))
        if dv >= 1.0:
            qL_a = np.zeros_like(pp_a); qU_a = np.ones_like(pp_a)
        else:
            qL_a = np.where(und, 0.0, np.maximum(0.0, pp_a - dv))
            qU_a = np.where(und, 1.0, np.minimum(1.0, pp_a + dv))
        Csf = np.maximum(Cm, 1e-15)
        rL_a = (Hm + Um * qL_a) / Csf; rU_a = (Hm + Um * qU_a) / Csf
        Lb = np.einsum("...ks,...s->...k", rL_a, w_a); Ub = np.einsum("...ks,...s->...k", rU_a, w_a)
        return Lb[..., pair_h] - Ub[..., pair_l], rL_a, rU_a, Lb, Ub, w_a, Ct, pp_a, und, qL_a, qU_a, Csf, Krm

    dL0_wc = eval_table_dL(mean_tot, 1.0)[0]
    dL0_d5 = eval_table_dL(mean_tot, 0.05)[0]

    def eval_unweighted_cluster(st_sums: np.ndarray, st_n: np.ndarray, dv: float):
        sums_tot = st_sums.sum(axis=0)
        dL_out, rL, rU, Lb, Ub, w, Ct, pp, und, qL, qU, Csf, Kr = eval_table_dL(sums_tot, dv)
        dL_dp = np.zeros_like(pp) if dv >= 1.0 else np.where((~und) & (pp > dv), 1.0, 0.0)
        dU_dp = np.zeros_like(pp) if dv >= 1.0 else np.where((~und) & (pp < 1.0 - dv), 1.0, 0.0)
        Kt_safe = np.maximum(Kr, 1e-15)
        p_tilde_g = np.where(Kr[None, ...] > 0.0, (st_sums[..., 3] - pp[None, ...] * st_sums[..., 4]) / Kt_safe[None, ...], 0.0)
        fac = w[None, None, :] / Csf[None, ...]
        wLc = (rL - Lb[:, None]) / Ct; wUc = (rU - Ub[:, None]) / Ct
        psiL_g = np.einsum("gks,js->gj", st_sums[..., 0], wLc) + np.sum(fac * (
            st_sums[..., 1] + qL[None, ...] * st_sums[..., 2] + (dL_dp * sums_tot[..., 2])[None, ...] * p_tilde_g - rL[None, ...] * st_sums[..., 0]
        ), axis=2)
        psiU_g = np.einsum("gks,js->gj", st_sums[..., 0], wUc) + np.sum(fac * (
            st_sums[..., 1] + qU[None, ...] * st_sums[..., 2] + (dU_dp * sums_tot[..., 2])[None, ...] * p_tilde_g - rU[None, ...] * st_sums[..., 0]
        ), axis=2)
        sc_g = psiL_g[:, pair_h] - psiU_g[:, pair_l]  # (G_st, P) exact uncentered cluster Gateaux score
        V_cv1 = st_scale * (sc_g.T @ sc_g)

        sums_loo = sums_tot[None, ...] - st_sums
        valid = np.all(sums_loo[..., 0] > 1e-12, axis=(1, 2))
        dL_loo = eval_table_dL(sums_loo[valid], dv)[0]
        nv = int(dL_loo.shape[0])
        cent = dL_loo - dL_loo.mean(axis=0, keepdims=True)
        V_cv3 = ((nv - 1.0) / nv) * (cent.T @ cent)
        return dL_out, V_cv1, V_cv3, sc_g

    df_cv3 = float(max(G_st - 1, 1))
    Z_t = Z_std * np.sqrt(df_cv3 / np.random.default_rng(seed + 111).chisquare(df_cv3, size=(Z_std.shape[0], 1)))
    keys = ("cv1_state", "cv3_jackknife", "cv3_t_df", "wcb", "cluster_boot_t") if include_cbt else ("cv1_state", "cv3_jackknife", "cv3_t_df", "wcb")
    fwer_wc = {k: 0 for k in keys}; fwer_d5 = {k: 0 for k in keys}
    dLw_draws = np.zeros((n_mc, P)); se1w_draws = np.zeros((n_mc, P)); se3w_draws = np.zeros((n_mc, P))
    c_cbt_in_wc = []; c_cbt_in_d5 = []

    for mc_i in range(n_mc):
        st_sums, st_n = sample_cluster_world(rng_out)
        dLw, V1w, V3w, scw = eval_unweighted_cluster(st_sums, st_n, 1.0)
        dL5, V15, V35, sc5 = eval_unweighted_cluster(st_sums, st_n, 0.05)

        se1w = np.sqrt(np.maximum(np.diag(V1w), 1e-20)); se3w = np.sqrt(np.maximum(np.diag(V3w), 1e-20))
        se15 = np.sqrt(np.maximum(np.diag(V15), 1e-20)); se35 = np.sqrt(np.maximum(np.diag(V35), 1e-20))
        dLw_draws[mc_i] = dLw; se1w_draws[mc_i] = se1w; se3w_draws[mc_i] = se3w

        _, _, c1w = max_t_stepdown(dummy_t, V1w, 0.05, Z_std=Z_std)
        _, _, c3w = max_t_stepdown(dummy_t, V3w, 0.05, Z_std=Z_std)
        _, _, c3w_t = max_t_stepdown(dummy_t, V3w, 0.05, Z_std=Z_t)
        _, _, c15 = max_t_stepdown(dummy_t, V15, 0.05, Z_std=Z_std)
        _, _, c35 = max_t_stepdown(dummy_t, V35, 0.05, Z_std=Z_std)
        _, _, c35_t = max_t_stepdown(dummy_t, V35, 0.05, Z_std=Z_t)

        tmax_1w = float(np.max((dLw - dL0_wc) / se1w))
        tmax_15 = float(np.max((dL5 - dL0_d5) / se15))
        tmax_3w = float(np.max((dLw - dL0_wc) / se3w))
        tmax_35 = float(np.max((dL5 - dL0_d5) / se35))
        fwer_wc["cv1_state"] += int(tmax_1w > c1w)
        fwer_wc["cv3_jackknife"] += int(tmax_3w > c3w)
        fwer_wc["cv3_t_df"] += int(tmax_3w > c3w_t)
        fwer_d5["cv1_state"] += int(tmax_15 > c15)
        fwer_d5["cv3_jackknife"] += int(tmax_35 > c35)
        fwer_d5["cv3_t_df"] += int(tmax_35 > c35_t)

        for fdict, diff_v, sc_m in ((fwer_wc, dLw - dL0_wc, scw), (fwer_d5, dL5 - dL0_d5, sc5)):
            t_wcb = float(np.max(diff_v / (math.sqrt(st_scale) * np.sqrt(np.maximum(np.sum(sc_m ** 2, axis=0), 1e-20)))))
            rad = rng_wcb.choice(np.array([-1.0, 1.0]), size=(B_inner, G_st))
            bd = rad @ sc_m
            ss = np.sum(sc_m ** 2, axis=0)[None, :]
            mt = (rad @ sc_m) / G_st
            bse = math.sqrt(st_scale) * np.sqrt(np.maximum(ss - G_st * (mt ** 2), 1e-24))
            fdict["wcb"] += int(t_wcb > float(np.quantile((bd / bse).max(axis=1), 0.95)))

        if include_cbt:
            in_c = _batched_cluster_boot_t_dL(inf, st_sums, st_n, [1.0, 0.05], B_inner, rng=rng_in, batch_size=B_inner)
            cw_in = float(np.quantile(np.max((in_c[1.0][0] - dLw[None, :]) / np.maximum(in_c[1.0][1], 1e-12), axis=1), 0.95))
            c_cbt_in_wc.append(cw_in)
            fwer_wc["cluster_boot_t"] += int(tmax_1w > cw_in)

            c5_in = float(np.quantile(np.max((in_c[0.05][0] - dL5[None, :]) / np.maximum(in_c[0.05][1], 1e-12), axis=1), 0.95))
            c_cbt_in_d5.append(c5_in)
            fwer_d5["cluster_boot_t"] += int(tmax_15 > c5_in)

    mc_sd_w = np.maximum(np.std(dLw_draws, axis=0, ddof=1), 1e-15)
    fw = OrderedDict((k, float(v / n_mc)) for k, v in fwer_wc.items())
    f5 = OrderedDict((k, float(v / n_mc)) for k, v in fwer_d5.items())
    sew = OrderedDict((k, float(math.sqrt(max(p * (1.0 - p), 0.0) / n_mc))) for k, p in fw.items())
    se5 = OrderedDict((k, float(math.sqrt(max(p * (1.0 - p), 0.0) / n_mc))) for k, p in f5.items())
    out = OrderedDict(
        n_mc=int(n_mc), B_inner=int(B_inner), n_clusters=int(G_st), seed=int(seed + 101),
        design=(f"Fixed-size cluster random-effects + Beta-Binomial DGP ({n_mc} outer worlds x {B_inner} inner draws "
                f"preserving {G_st} clusters and observed outbreak sizes C_s)"),
        median_se_ratio_cv1=float(np.median(se1w_draws.mean(axis=0) / mc_sd_w)),
        median_se_ratio_cv3=float(np.median(se3w_draws.mean(axis=0) / mc_sd_w)),
        fwer_worst=fw, fwer_worst_mc_se=sew,
        fwer_delta5=f5, fwer_delta5_mc_se=se5,
    )
    if include_cbt:
        out["inner_c_cbt_worst_summary"] = OrderedDict(
            mean=float(np.mean(c_cbt_in_wc)), p5=float(np.quantile(c_cbt_in_wc, 0.05)),
            p50=float(np.quantile(c_cbt_in_wc, 0.50)), p95=float(np.quantile(c_cbt_in_wc, 0.95)),
        )
        out["inner_c_cbt_delta5_summary"] = OrderedDict(
            mean=float(np.mean(c_cbt_in_d5)), p5=float(np.quantile(c_cbt_in_d5, 0.05)),
            p50=float(np.quantile(c_cbt_in_d5, 0.50)), p95=float(np.quantile(c_cbt_in_d5, 0.95)),
        )
    return out


def run_parametric_heavy_tail_calibration(inf: StackedInference, n_mc: int = 300,
                                          B_inner: int = 399, seed: int = 20261125) -> OrderedDict:
    """Overdispersed Beta-Binomial + LogNormal heavy-tail double bootstrap (M3c): matches empirical
    recording/hospitalization overdispersion so synthetic HC1 SEs match observed NORS scale."""
    G, K, P = inf.G, inf.K, inf.P
    off = 3 * G
    rng_out = np.random.default_rng(seed + 311)
    rng_in = np.random.default_rng(seed + 317)
    Z_std = np.random.default_rng(seed + 331).standard_normal((10000, P))
    dummy_t = np.zeros(P)
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    _, V_obs_wc, _, _ = inf.analytic_covariances(1.0)
    se_obs_wc = np.sqrt(np.maximum(np.diag(V_obs_wc[off:off + P, off:off + P]), 1e-20))

    bb_params = _fit_cell_betabinom_params(inf)
    cell_params = {}
    exp_totals = np.zeros((G, K, 5), dtype=float)
    for gi in range(G):
        for ci in range(K):
            arr, _, pk, ph, mk, mh = bb_params[(gi, ci)]
            m = arr.shape[0]
            logC = np.log(np.maximum(arr[:, 0], 1.0))
            mu_l = float(np.mean(logC))
            sd_l = max(float(np.std(logC, ddof=1)) if m > 1 else 0.5, 0.25)
            draws_ref = np.maximum(1.0, np.round(np.exp(np.clip(
                np.random.default_rng(seed + 500 + gi * K + ci).normal(mu_l, sd_l, size=100000), 0.0, 9.5
            ))))
            mean_c = float(draws_ref.mean())
            cell_params[(gi, ci)] = (m, mu_l, sd_l, pk, ph, mk, mh)
            exp_totals[gi, ci, 0] = m * mean_c
            exp_totals[gi, ci, 1] = m * mean_c * pk * ph
            exp_totals[gi, ci, 2] = m * mean_c * (1.0 - pk)
            exp_totals[gi, ci, 3] = m * mean_c * pk * ph
            exp_totals[gi, ci, 4] = m * mean_c * pk

    def eval_pop_dL(dv: float) -> np.ndarray:
        tb_p = S.Table(groups=inf.groups, cells=inf.cells, C=exp_totals[:, :, 0], H=exp_totals[:, :, 1],
                       U=exp_totals[:, :, 2], H_rec=exp_totals[:, :, 3], K_rec=exp_totals[:, :, 4],
                       n=inf.n.astype(float))
        tb_p.w = S.weights(tb_p, "pooled_cases")
        Lb = np.zeros(G); Ub = np.zeros(G)
        for i in range(G):
            Lb[i], Ub[i] = B.worst_bounds(tb_p, i) if dv >= 1.0 else B.group_bounds(tb_p, i, dv)
        return Lb[pair_h] - Ub[pair_l]

    dL0_wc = eval_pop_dL(1.0)
    dL0_d5 = eval_pop_dL(0.05)

    keys = ("hc1", "bootstrap_t")
    fwer_wc = {k: 0 for k in keys}
    fwer_d5 = {k: 0 for k in keys}
    dL_syn_wc = np.zeros((n_mc, P))
    se_syn_wc = np.zeros((n_mc, P))

    for mc_i in range(n_mc):
        syn_arrs = {}
        for gi in range(G):
            for ci in range(K):
                m, mu_l, sd_l, pk, ph, mk, mh = cell_params[(gi, ci)]
                C_s = np.maximum(1.0, np.round(np.exp(np.clip(rng_out.normal(mu_l, sd_l, size=m), 0.0, 9.5))))
                p_kd = _beta_mean_preserving(rng_out, pk, mk, size=m)
                K_s = rng_out.binomial(C_s.astype(int), p_kd).astype(float)
                p_hd = _beta_mean_preserving(rng_out, ph, mh, size=m)
                H_s = rng_out.binomial(K_s.astype(int), p_hd).astype(float)
                U_s = C_s - K_s
                syn_arrs[(gi, ci)] = np.stack([C_s, H_s, U_s, H_s, K_s], axis=1)

        _, _, dL_rw, _, V1_w, _ = _eval_resampled_table(inf, syn_arrs, 1.0, None)
        _, _, dL_r5, _, V1_5, _ = _eval_resampled_table(inf, syn_arrs, 0.05, None)
        se1w = np.sqrt(np.maximum(np.diag(V1_w[off:off + P, off:off + P]), 1e-20))
        se15 = np.sqrt(np.maximum(np.diag(V1_5[off:off + P, off:off + P]), 1e-20))
        dL_syn_wc[mc_i] = dL_rw; se_syn_wc[mc_i] = se1w
        _, _, c1w = max_t_stepdown(dummy_t, V1_w[off:off + P, off:off + P], 0.05, Z_std=Z_std)
        _, _, c15 = max_t_stepdown(dummy_t, V1_5[off:off + P, off:off + P], 0.05, Z_std=Z_std)
        tmax_w = float(np.max((dL_rw - dL0_wc) / se1w))
        tmax_5 = float(np.max((dL_r5 - dL0_d5) / se15))
        fwer_wc["hc1"] += int(tmax_w > c1w)
        fwer_d5["hc1"] += int(tmax_5 > c15)

        in_ev = _batched_boot_t_dL(inf, syn_arrs, [1.0, 0.05], B_inner, rng=rng_in, batch_size=B_inner)
        cbt_w = float(np.quantile(np.max((in_ev[1.0][0] - dL_rw[None, :]) / np.maximum(in_ev[1.0][1], 1e-12), axis=1), 0.95))
        cbt_5 = float(np.quantile(np.max((in_ev[0.05][0] - dL_r5[None, :]) / np.maximum(in_ev[0.05][1], 1e-12), axis=1), 0.95))
        fwer_wc["bootstrap_t"] += int(tmax_w > cbt_w)
        fwer_d5["bootstrap_t"] += int(tmax_5 > cbt_5)

    fw = OrderedDict((k, float(v / n_mc)) for k, v in fwer_wc.items())
    f5 = OrderedDict((k, float(v / n_mc)) for k, v in fwer_d5.items())
    sew = OrderedDict((k, float(math.sqrt(max(p * (1.0 - p), 0.0) / n_mc))) for k, p in fw.items())
    se5 = OrderedDict((k, float(math.sqrt(max(p * (1.0 - p), 0.0) / n_mc))) for k, p in f5.items())
    mc_sd = np.maximum(np.std(dL_syn_wc, axis=0, ddof=1), 1e-15)
    return OrderedDict(
        n_mc=int(n_mc), B_inner=int(B_inner), seed=int(seed + 311),
        design=(f"Overdispersed parametric heavy-tail calibration ({n_mc} LogNormal + Beta-Binomial worlds "
                f"with {B_inner} inner bootstrap-t resamples preserving n_js; no cluster correlation simulated)"),
        median_synthetic_vs_observed_hc1_se_ratio=float(np.median(se_syn_wc.mean(axis=0) / se_obs_wc)),
        median_se_hc1_to_mc_sd=float(np.median(se_syn_wc.mean(axis=0) / mc_sd)),
        max_null_bias_in_sd=float(np.max(np.abs(dL_syn_wc.mean(axis=0) - dL0_wc) / mc_sd)),
        fwer_worst=fw, fwer_worst_mc_se=sew,
        fwer_delta5=f5, fwer_delta5_mc_se=se5,
    )


def _is_confirmed_o157_tok(sg: str, es: str) -> bool:
    sg_p = [p.strip() for p in str(sg).split(";")]
    es_p = [p.strip() for p in str(es).split(";")]
    if len(sg_p) == len(es_p) and len(sg_p) > 0:
        return any(("O157" in s.upper()) and (e.lower() == "confirmed") for s, e in zip(sg_p, es_p))
    return ("O157" in str(sg).upper()) and ("confirmed" in str(es).lower())


def empirical_delta_calibration(main_df: pd.DataFrame, t_main: S.Table,
                                ordered_pairs_15: list[tuple[int, int]],
                                fixed_indices_bt: dict[tuple[int, int], np.ndarray] | None = None) -> OrderedDict:
    """Decompose fully vs partially recorded comparisons with bootstrap 95% CIs (M2d),
    scenario stress tests, and OR (pure & 0.1%-floor) sensitivity."""
    sup_df = main_df[main_df["cell"].isin(t_main.cells)].copy()
    sup_df["is_multi"] = sup_df["state"].astype(str) == "Multistate"
    full_df = sup_df[sup_df["U"] == 0]
    part_df = sup_df[(sup_df["U"] > 0) & (sup_df["K_rec"] > 0)]

    # Fast outbreak-level bootstrap 95% CIs for missingness contrast metrics using fixed_indices_bt
    ci_diffs = {g: {k: None for k in ("within_cell_diff_ci95_pp", "std_obs_minus_full_ci95_pp",
                                      "single_state_diff_ci95_pp", "multistate_diff_ci95_pp")} for g in t_main.groups}
    if fixed_indices_bt is not None:
        G_n, K_n = len(t_main.groups), len(t_main.cells)
        B_n = next(iter(fixed_indices_bt.values())).shape[0]
        cell_data = {}
        for gi, g in enumerate(t_main.groups):
            for ci, c in enumerate(t_main.cells):
                sub = sup_df[(sup_df["group"] == g) & (sup_df["cell"] == c)]
                Hr = sub["H_rec"].to_numpy(dtype=float)
                Kr = sub["K_rec"].to_numpy(dtype=float)
                C_a = sub["C"].to_numpy(dtype=float)
                U_a = sub["U"].to_numpy(dtype=float)
                is_f = (U_a == 0.0)
                is_p = (U_a > 0.0) & (Kr > 0.0)
                is_m = (sub["state"].astype(str) == "Multistate").to_numpy()
                cell_data[(gi, ci)] = np.stack([
                    C_a, Hr, Kr,
                    np.where(is_f, Hr, 0.0), np.where(is_f, Kr, 0.0),
                    np.where(is_p, Hr, 0.0), np.where(is_p, Kr, 0.0),
                    np.where(is_f & (~is_m), Hr, 0.0), np.where(is_f & (~is_m), Kr, 0.0),
                    np.where(is_p & (~is_m), Hr, 0.0), np.where(is_p & (~is_m), Kr, 0.0),
                    np.where(is_f & is_m, Hr, 0.0), np.where(is_f & is_m, Kr, 0.0),
                    np.where(is_p & is_m, Hr, 0.0), np.where(is_p & is_m, Kr, 0.0),
                ], axis=1)  # (m, 15)
        sums_b = np.zeros((B_n, G_n, K_n, 15), dtype=float)
        for gi in range(G_n):
            for ci in range(K_n):
                idx = fixed_indices_bt[(gi, ci)]
                sums_b[:, gi, ci, :] = cell_data[(gi, ci)][idx].sum(axis=1)
        C_cell = sums_b[:, :, :, 0].sum(axis=1)
        w_b = C_cell / np.maximum(C_cell.sum(axis=1, keepdims=True), 1e-15)
        Hr_b, Kr_b = sums_b[..., 1], sums_b[..., 2]
        Hrf_b, Krf_b = sums_b[..., 3], sums_b[..., 4]
        Hrp_b, Krp_b = sums_b[..., 5], sums_b[..., 6]
        valid_c = (Krf_b > 0.0) & (Krp_b > 0.0) & (Kr_b > 0.0)
        rf_b = np.where(valid_c, np.divide(Hrf_b, np.maximum(Krf_b, 1e-15)), 0.0)
        rp_b = np.where(valid_c, np.divide(Hrp_b, np.maximum(Krp_b, 1e-15)), 0.0)
        ra_b = np.where(valid_c, np.divide(Hr_b, np.maximum(Kr_b, 1e-15)), 0.0)
        wt_v = np.where(valid_c, w_b[:, None, :], 0.0)
        den_v = wt_v.sum(axis=2)
        pool_rf = np.divide(Hrf_b.sum(axis=2), np.maximum(Krf_b.sum(axis=2), 1e-15))
        pool_rp = np.divide(Hrp_b.sum(axis=2), np.maximum(Krp_b.sum(axis=2), 1e-15))
        std_f = np.where(den_v > 0.0, np.divide(np.sum(wt_v * rf_b, axis=2), np.maximum(den_v, 1e-15)), pool_rf)
        std_p = np.where(den_v > 0.0, np.divide(np.sum(wt_v * rp_b, axis=2), np.maximum(den_v, 1e-15)), pool_rp)
        std_a = np.where(den_v > 0.0, np.divide(np.sum(wt_v * ra_b, axis=2), np.maximum(den_v, 1e-15)), 0.0)
        wc_boot = 100.0 * (std_p - std_f)
        obs_boot = 100.0 * (std_a - std_f)
        fs_r = np.divide(sums_b[..., 7].sum(axis=2), np.maximum(sums_b[..., 8].sum(axis=2), 1e-15))
        ps_r = np.divide(sums_b[..., 9].sum(axis=2), np.maximum(sums_b[..., 10].sum(axis=2), 1e-15))
        ss_valid = (sums_b[..., 8].sum(axis=2) > 0.0) & (sums_b[..., 10].sum(axis=2) > 0.0)
        fm_r = np.divide(sums_b[..., 11].sum(axis=2), np.maximum(sums_b[..., 12].sum(axis=2), 1e-15))
        pm_r = np.divide(sums_b[..., 13].sum(axis=2), np.maximum(sums_b[..., 14].sum(axis=2), 1e-15))
        ms_valid = (sums_b[..., 12].sum(axis=2) > 0.0) & (sums_b[..., 14].sum(axis=2) > 0.0)
        q2 = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
        for gi, g in enumerate(t_main.groups):
            ci_diffs[g]["within_cell_diff_ci95_pp"] = q2(wc_boot[:, gi])
            ci_diffs[g]["std_obs_minus_full_ci95_pp"] = q2(obs_boot[:, gi])
            if np.any(ss_valid[:, gi]):
                ci_diffs[g]["single_state_diff_ci95_pp"] = q2(100.0 * (ps_r[ss_valid[:, gi], gi] - fs_r[ss_valid[:, gi], gi]))
            if np.sum(ms_valid[:, gi]) >= 20:
                ci_diffs[g]["multistate_diff_ci95_pp"] = q2(100.0 * (pm_r[ms_valid[:, gi], gi] - fm_r[ms_valid[:, gi], gi]))

    per_group = OrderedDict()
    for i, g in enumerate(t_main.groups):
        fg = full_df[full_df["group"] == g]
        pg = part_df[part_df["group"] == g]
        r_full_pool = float(fg["H_rec"].sum() / fg["K_rec"].sum())
        r_part_pool = float(pg["H_rec"].sum() / pg["K_rec"].sum()) if len(pg) else r_full_pool

        num_f = 0.0; num_p = 0.0; num_all_cc = 0.0; den = 0.0; n_cc = 0
        for ci, c in enumerate(t_main.cells):
            fc = fg[fg["cell"] == c]
            pc = pg[pg["cell"] == c]
            if len(fc) > 0 and len(pc) > 0 and fc["K_rec"].sum() > 0 and pc["K_rec"].sum() > 0:
                num_f += t_main.w[ci] * float(fc["H_rec"].sum() / fc["K_rec"].sum())
                num_p += t_main.w[ci] * float(pc["H_rec"].sum() / pc["K_rec"].sum())
                num_all_cc += t_main.w[ci] * float(t_main.p_obs[i, ci])
                den += t_main.w[ci]
                n_cc += 1
        r_full_std = num_f / den if den > 0 else r_full_pool
        r_part_std = num_p / den if den > 0 else r_part_pool
        r_all_common_std = num_all_cc / den if den > 0 else float(B.point_delta0(t_main, i)[0])
        r_all_std = r_all_common_std

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
            n_common_cells_contrast=int(n_cc),
            common_cells_weight_share=float(den),
            rate_full_std=r_full_std,
            rate_partial_known_std=r_part_std,
            rate_all_common_std=r_all_std,
            within_cell_diff_pp=100.0 * (r_part_std - r_full_std),
            within_cell_diff_ci95_pp=ci_diffs[g]["within_cell_diff_ci95_pp"],
            std_obs_minus_full_pp=100.0 * (r_all_std - r_full_std),
            std_obs_minus_full_ci95_pp=ci_diffs[g]["std_obs_minus_full_ci95_pp"],
            single_state_full_rate=r_fs,
            single_state_partial_rate=r_ps,
            single_state_diff_pp=(100.0 * (r_ps - r_fs)) if (r_fs is not None and r_ps is not None) else None,
            single_state_diff_ci95_pp=ci_diffs[g]["single_state_diff_ci95_pp"],
            multistate_full_rate=r_fm,
            multistate_partial_rate=r_pm,
            multistate_diff_pp=(100.0 * (r_pm - r_fm)) if (r_fm is not None and r_pm is not None) else None,
            multistate_diff_ci95_pp=ci_diffs[g]["multistate_diff_ci95_pp"],
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

    # Diagnostic-informed group-specific asymmetric band (SAL -10/+5, CAMP -7/+5, others -5/+5 pp)
    diag_down_map = {"SAL": 0.10, "CAMP": 0.07, "STEC": 0.05, "CPERF": 0.05, "NOV_GI": 0.05, "NOV_GII": 0.05}
    diag_up_map = {g: 0.05 for g in t_main.groups}
    diag_verdicts = OrderedDict()
    diag_det = 0
    for h, l in ordered_pairs_15:
        gh, gl = t_main.groups[h], t_main.groups[l]
        key = f"{t_main.groups[min(h, l)]}|{t_main.groups[max(h, l)]}"
        Lh, _ = B.group_bounds_asym(t_main, h, diag_down_map[gh], diag_up_map[gh])
        _, Ul = B.group_bounds_asym(t_main, l, diag_down_map[gl], diag_up_map[gl])
        v = gh if Lh > Ul else B.UNDET
        if v != B.UNDET:
            diag_det += 1
        diag_verdicts[key] = v
    asym_results["diag_asymmetric"] = OrderedDict(
        label="Diagnostic-informed asymmetric (SAL -10/+5, CAMP -7/+5, others +-5 pp)",
        down_by_group_pp=OrderedDict((g, 100.0 * diag_down_map[g]) for g in t_main.groups),
        up_by_group_pp=OrderedDict((g, 100.0 * diag_up_map[g]) for g in t_main.groups),
        determined=int(diag_det),
        verdicts=diag_verdicts,
    )

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
    o157_m = pd.Series(
        [_is_confirmed_o157_tok(sg, es) for sg, es in zip(stec_sup["serotype_or_genotype"], stec_sup["etiology_status"])],
        index=stec_sup.index,
    ) if "serotype_or_genotype" in stec_sup.columns else pd.Series(False, index=stec_sup.index)
    stec_o157_cases = float(stec_sup.loc[o157_m, "C"].sum())
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


def _plausible_q_rules(inf: StackedInference, lower_max: float, upper_mult_dn: float = 0.5, upper_mult_up: float = 1.5):
    """Per-cell qL, qU, dL_dp, dU_dp under explicit plausibility interval assumptions (M2b)."""
    t = inf.t_main
    G, K = inf.G, inf.K
    undef = np.isnan(t.p_obs)
    pp = np.where(undef, 0.0, t.p_obs)
    qL = np.zeros((G, K)); qU = np.ones((G, K))
    dL_dp = np.zeros((G, K)); dU_dp = np.zeros((G, K))
    lower_set = {"CPERF", "NOV_GI", "NOV_GII"}
    for gi, g in enumerate(inf.groups):
        if g in lower_set:
            qL[gi, :] = 0.0
            qU[gi, :] = lower_max
            dL_dp[gi, :] = 0.0
            dU_dp[gi, :] = 0.0
        else:
            qL[gi, :] = np.where(undef[gi], 0.0, np.clip(upper_mult_dn * pp[gi], 0.0, 1.0))
            qU[gi, :] = np.where(undef[gi], 1.0, np.clip(upper_mult_up * pp[gi], 0.0, 1.0))
            dL_dp[gi, :] = np.where((~undef[gi]) & (upper_mult_dn * pp[gi] > 0.0) & (upper_mult_dn * pp[gi] < 1.0), upper_mult_dn, 0.0)
            dU_dp[gi, :] = np.where((~undef[gi]) & (upper_mult_up * pp[gi] > 0.0) & (upper_mult_up * pp[gi] < 1.0), upper_mult_up, 0.0)
    return qL, qU, dL_dp, dU_dp


def run_plausible_rate_scenarios(inf: StackedInference, fixed_indices_bt: dict[tuple[int, int], np.ndarray],
                                 Z_std_30: np.ndarray) -> OrderedDict:
    """Evaluate explicit unrecorded-rate plausibility intervals P1/P2 (M2b) with exact IFs, CV3, and boot-t."""
    t = inf.t_main
    G, K, P = inf.G, inf.K, inf.P
    C_tot = float(t.C.sum()); w = t.w
    undef = np.isnan(t.p_obs); pp = np.where(undef, 0.0, t.p_obs)
    pair_h = np.array([h for h, _ in inf.ordered_pairs], dtype=int)
    pair_l = np.array([l for _, l in inf.ordered_pairs], dtype=int)
    idx_h = [np.where(pair_h == ki)[0] for ki in range(G)]
    idx_l = [np.where(pair_l == ki)[0] for ki in range(G)]
    lower_set = {"CPERF", "NOV_GI", "NOV_GII"}
    lower_mask = np.array([g in lower_set for g in inf.groups], dtype=bool)[:, None]

    B_bt = next(iter(fixed_indices_bt.values())).shape[0]
    scenarios = [
        ("P1_lower5_upper50pct", "Lower tier [0, 5%]; upper tier [0.5p, min(1, 1.5p)]", 0.05, 0.5, 1.5),
        ("P2_lower10_upper50pct", "Lower tier [0, 10%]; upper tier [0.5p, min(1, 1.5p)]", 0.10, 0.5, 1.5),
    ]
    out = OrderedDict()
    for code, label, lmax, mdn, mup in scenarios:
        qL, qU, dL_dp, dU_dp = _plausible_q_rules(inf, lmax, mdn, mup)
        rL = (t.H + t.U * qL) / t.C
        rU = (t.H + t.U * qU) / t.C
        L_bar = rL @ w; U_bar = rU @ w
        dL_hat = L_bar[pair_h] - U_bar[pair_l]

        # Sample intervals & determined count (/15)
        det_sample = sum(1 for a in range(G) for b in range(a + 1, G) if (L_bar[a] > U_bar[b]) or (L_bar[b] > U_bar[a]))

        # Analytical HC1 covariance on dL_hat (P, P)
        wLc = (rL - L_bar[:, None]) / C_tot
        wUc = (rU - U_bar[:, None]) / C_tot
        V_hc1 = np.zeros((P, P))
        for ki in range(G):
            for ci in range(K):
                Z = inf.cell_arrays[(ki, ci)]; m = Z.shape[0]; Zc = Z - inf.mu[ki, ci]
                Cc, Hc, Uc, Hrc, Krc = Zc[:, 0], Zc[:, 1], Zc[:, 2], Zc[:, 3], Zc[:, 4]
                k_rec = float(t.K_rec[ki, ci])
                p_tilde = (Hrc - pp[ki, ci] * Krc) / k_rec if k_rec > 0.0 else np.zeros(m)
                fac = w[ci] / float(t.C[ki, ci])
                psiL = Cc[:, None] * wLc[:, ci][None, :]
                psiU = Cc[:, None] * wUc[:, ci][None, :]
                psiL[:, ki] += fac * (Hc + qL[ki, ci] * Uc + dL_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rL[ki, ci] * Cc)
                psiU[:, ki] += fac * (Hc + qU[ki, ci] * Uc + dU_dp[ki, ci] * float(t.U[ki, ci]) * p_tilde - rU[ki, ci] * Cc)
                psi_d = psiL[:, pair_h] - psiU[:, pair_l]
                V_hc1 += ((m / (m - 1.0)) if m > 1 else 1.0) * (psi_d.T @ psi_d)

        # CV3 leave-one-state-out jackknife covariance on dL_hat
        totals = np.sum(inf.state_cell_sums, axis=0)
        dL_loo = []
        for g_st in range(inf.n_states):
            sg = totals - inf.state_cell_sums[g_st]
            Cg = sg[:, :, 0]
            if np.any(Cg <= 0.0):
                continue
            cs = Cg.sum(axis=0); wg = cs / cs.sum()
            Hg, Ug, Hrg, Krg = sg[:, :, 1], sg[:, :, 2], sg[:, :, 3], sg[:, :, 4]
            ug = Krg <= 0.0; pg = np.where(ug, 0.0, np.divide(Hrg, np.maximum(Krg, 1e-15)))
            qLg = np.where(lower_mask, 0.0, np.where(ug, 0.0, np.clip(mdn * pg, 0.0, 1.0)))
            qUg = np.where(lower_mask, lmax, np.where(ug, 1.0, np.clip(mup * pg, 0.0, 1.0)))
            Lg = ((Hg + Ug * qLg) / Cg) @ wg
            Ug_b = ((Hg + Ug * qUg) / Cg) @ wg
            dL_loo.append(Lg[pair_h] - Ug_b[pair_l])
        mat_c = np.asarray(dL_loo, dtype=float)
        cent_c = mat_c - mat_c.mean(axis=0, keepdims=True)
        V_cv3 = ((mat_c.shape[0] - 1.0) / mat_c.shape[0]) * (cent_c.T @ cent_c)
        se_cv3 = np.sqrt(np.maximum(np.diag(V_cv3), 1e-20))
        p_cv3, _, c_cv3 = max_t_stepdown(dL_hat / se_cv3, V_cv3, 0.05, Z_std=Z_std_30)

        # Vectorized studentized bootstrap-t across fixed_indices_bt
        dL_b = np.zeros((B_bt, P)); se_b = np.zeros((B_bt, P))
        for start in range(0, B_bt, 100):
            end = min(start + 100, B_bt); bs = end - start
            cell_Z = {}; sums = np.zeros((bs, G, K, 5)); mu_r = np.zeros((bs, G, K, 5))
            for gi in range(G):
                for ci in range(K):
                    Z = inf.cell_arrays[(gi, ci)][fixed_indices_bt[(gi, ci)][start:end]]
                    cell_Z[(gi, ci)] = Z; sums[:, gi, ci, :] = Z.sum(axis=1); mu_r[:, gi, ci, :] = Z.mean(axis=1)
            Cm, Hm, Um, Hrm, Krm = sums[..., 0], sums[..., 1], sums[..., 2], sums[..., 3], sums[..., 4]
            Cs = Cm.sum(axis=1); Ct = np.maximum(Cs.sum(axis=1), 1e-15); wb = Cs / Ct[:, None]
            und_b = Krm <= 0.0; pp_b = np.where(und_b, 0.0, np.divide(Hrm, np.maximum(Krm, 1e-15)))
            lm_b = lower_mask[None, :, :]
            qL_b = np.where(lm_b, 0.0, np.where(und_b, 0.0, np.clip(mdn * pp_b, 0.0, 1.0)))
            qU_b = np.where(lm_b, lmax, np.where(und_b, 1.0, np.clip(mup * pp_b, 0.0, 1.0)))
            dLp_b = np.where(lm_b, 0.0, np.where((~und_b) & (mdn * pp_b > 0.0) & (mdn * pp_b < 1.0), mdn, 0.0))
            dUp_b = np.where(lm_b, 0.0, np.where((~und_b) & (mup * pp_b > 0.0) & (mup * pp_b < 1.0), mup, 0.0))
            Csf = np.maximum(Cm, 1e-15)
            rLb = (Hm + Um * qL_b) / Csf; rUb = (Hm + Um * qU_b) / Csf
            Lb_b = np.einsum("bsk,bk->bs", rLb, wb); Ub_b = np.einsum("bsk,bk->bs", rUb, wb)
            dL_b[start:end] = Lb_b[:, pair_h] - Ub_b[:, pair_l]
            wpc = ((rLb - Lb_b[:, :, None]) / Ct[:, None, None])[:, pair_h, :] - ((rUb - Ub_b[:, :, None]) / Ct[:, None, None])[:, pair_l, :]
            vdl = np.zeros((bs, P))
            for ki in range(G):
                ih, il = idx_h[ki], idx_l[ki]
                for ci in range(K):
                    Z = cell_Z[(ki, ci)]; m = Z.shape[1]; Zc = Z - mu_r[:, ki, ci, :][:, None, :]
                    Cc, Hc, Uc, Hrc, Krc = Zc[..., 0], Zc[..., 1], Zc[..., 2], Zc[..., 3], Zc[..., 4]
                    kr = Krm[:, ki, ci][:, None]
                    pt = np.where(kr > 0.0, (Hrc - pp_b[:, ki, ci][:, None] * Krc) / np.maximum(kr, 1e-15), 0.0)
                    fac = (wb[:, ci] / Csf[:, ki, ci])[:, None]
                    oL = fac * (Hc + qL_b[:, ki, ci][:, None] * Uc + (dLp_b[:, ki, ci] * Um[:, ki, ci])[:, None] * pt - rLb[:, ki, ci][:, None] * Cc)
                    oU = fac * (Hc + qU_b[:, ki, ci][:, None] * Uc + (dUp_b[:, ki, ci] * Um[:, ki, ci])[:, None] * pt - rUb[:, ki, ci][:, None] * Cc)
                    sc = (m / (m - 1.0)) if m > 1 else 1.0
                    scc = np.sum(Cc * Cc, axis=1); scL = np.sum(Cc * oL, axis=1); scU = np.sum(Cc * oU, axis=1)
                    sLL = np.sum(oL * oL, axis=1); sUU = np.sum(oU * oU, axis=1); a = wpc[:, :, ci]
                    vdl += sc * (a * a) * scc[:, None]
                    if len(ih): vdl[:, ih] += sc * (2.0 * a[:, ih] * scL[:, None] + sLL[:, None])
                    if len(il): vdl[:, il] += sc * (-2.0 * a[:, il] * scU[:, None] + sUU[:, None])
            se_b[start:end] = np.sqrt(np.maximum(vdl, 0.0))

        se_hc1 = np.sqrt(np.maximum(np.diag(V_hc1), 1e-20))
        t_obs = dL_hat / se_hc1
        p_len = G * K * 5
        fmu = inf.mu.reshape(-1)
        J_fd = np.zeros((P, p_len))
        for idx_p in range(p_len):
            st_p = 1e-6 * max(1.0, abs(fmu[idx_p]))
            vals_2 = []
            for sgn in (1.0, -1.0):
                mp = fmu.copy(); mp[idx_p] += sgn * st_p
                tb_p = inf.table_from_mu(mp.reshape(G, K, 5))
                und_p = np.isnan(tb_p.p_obs); pp_p = np.where(und_p, 0.0, tb_p.p_obs)
                qL_p = np.where(lower_mask, 0.0, np.where(und_p, 0.0, np.clip(mdn * pp_p, 0.0, 1.0)))
                qU_p = np.where(lower_mask, lmax, np.where(und_p, 1.0, np.clip(mup * pp_p, 0.0, 1.0)))
                Lp = ((tb_p.H + tb_p.U * qL_p) / tb_p.C) @ tb_p.w
                Up = ((tb_p.H + tb_p.U * qU_p) / tb_p.C) @ tb_p.w
                vals_2.append(Lp[pair_h] - Up[pair_l])
            J_fd[:, idx_p] = (vals_2[0] - vals_2[1]) / (2.0 * st_p)
        V_fd = np.zeros((P, P))
        for ki in range(G):
            for ci in range(K):
                off_p = (ki * K + ci) * 5; Jb = J_fd[:, off_p:off_p + 5]
                V_fd += Jb @ inf.omega_hc1[ki, ci] @ Jb.T
        diff_if_fd = float(np.max(np.abs(V_fd - V_hc1)))
        T_b = (dL_b - dL_hat[None, :]) / np.maximum(se_b, 1e-12)
        c_bt = float(np.quantile(T_b.max(axis=1), 0.95))
        order = np.argsort(-t_obs, kind="mergesort")
        adj_p = np.zeros(P); prev_p = 0.0
        for step in range(P):
            p_raw = float((1.0 + np.sum(T_b[:, order[step:]].max(axis=1) >= t_obs[order[step]])) / (B_bt + 1.0))
            prev_p = max(prev_p, p_raw); adj_p[order[step]] = prev_p

        out[code] = OrderedDict(
            label=label, lower_max_rate=lmax, upper_mult_range=[mdn, mup],
            per_group=OrderedDict((g, OrderedDict(L=float(L_bar[i]), U=float(U_bar[i]))) for i, g in enumerate(inf.groups)),
            sample_determined=int(det_sample),
            max_abs_diff_if_vs_fd=diff_if_fd,
            c_cv3=float(c_cv3), n_sig_cv3=int(sum(x <= 0.05 for x in p_cv3)),
            ranks_cv3=rank_confidence_sets(inf.groups, inf.ordered_pairs, p_cv3, 0.05),
            c_bt=c_bt, n_sig_boot_t=int(sum(x <= 0.05 for x in adj_p)),
            ranks_boot_t=rank_confidence_sets(inf.groups, inf.ordered_pairs, adj_p, 0.05),
        )
    return out


def build_descriptive_cohort(main_df: pd.DataFrame, t_main: S.Table) -> OrderedDict:
    """Per-pathogen descriptive characteristics across common-support outbreaks (STROBE 14a)."""
    sup_df = main_df[main_df["cell"].isin(t_main.cells)].copy()
    out = OrderedDict()
    for g in t_main.groups:
        sub = sup_df[sup_df["group"] == g]
        c_tot = float(sub["C"].sum())
        kr = float(sub["K_rec"].sum())
        out[g] = OrderedDict(
            outbreaks=int(len(sub)),
            n_blank_k_outbreaks=int((sub["audit_category"] == "single_missing_K").sum()),
            cases=c_tot,
            median_cases=float(sub["C"].median()),
            iqr_cases=[float(sub["C"].quantile(0.25)), float(sub["C"].quantile(0.75))],
            pct_restaurant_cases=float(100.0 * sub.loc[sub["setting_group"] == "Restaurant", "C"].sum() / c_tot),
            pct_institution_cases=float(100.0 * sub.loc[sub["setting_group"] == "Institution", "C"].sum() / c_tot),
            pct_unknown_setting_cases=float(100.0 * sub.loc[sub["setting_group"] == "Unknown", "C"].sum() / c_tot),
            pct_multistate_outbreaks=float(100.0 * (sub["state"].astype(str) == "Multistate").mean()),
            pct_multistate_cases=float(100.0 * sub.loc[sub["state"].astype(str) == "Multistate", "C"].sum() / c_tot),
            pct_2009_2018_cases=float(100.0 * sub.loc[sub["period"].isin(["2009-2013", "2014-2018"]), "C"].sum() / c_tot),
            unrecorded_share=float(sub["U"].sum() / c_tot),
            recorded_hosp_rate=float(sub["H_rec"].sum() / kr) if kr > 0 else None,
        )
    return out


def run_all(main_df: pd.DataFrame, t_main: S.Table, main_res: dict, seed: int = 20261125,
            boot_block: dict | None = None) -> OrderedDict:
    """Run v8.1 30-hypothesis simultaneous inference, cluster Gateaux covariance, CV3 (Gaussian & t_df), WCB,
    studentized bootstrap-t, multi-world calibrations, OR/plausibility bounds, and diagnostics."""
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
    _, grid_c_cv3 = dynamic_csim_grid(inf, "cv3", Z_std_30, alpha=0.05, n_grid=21)

    # CV3 cluster jackknife (MacKinnon-Nielsen-Webb 2023) at worst-case, delta=5%, and asym (-15/+5)
    V_cv3_wc, cv3_nval_wc, cv3_skip_wc = inf.cv3_cluster_jackknife_cov(1.0)
    seL_cv3_wc = np.sqrt(np.maximum(np.diag(V_cv3_wc), 1e-20))
    p30_cv3_wc, _, c30_cv3_wc = max_t_stepdown(dL_wc / seL_cv3_wc, V_cv3_wc, 0.05, Z_std=Z_std_30)
    p30_cv3_t_wc, _, c30_cv3_t_wc = max_t_stepdown(dL_wc / seL_cv3_wc, V_cv3_wc, 0.05, Z_std=Z_std_30, df=float(max(inf.n_states - 1, 1)))
    _, _, c15_cv3_wc = max_t_stepdown(dL_wc[:P15] / seL_cv3_wc[:P15], V_cv3_wc[:P15, :P15], 0.05, Z_std=Z_std_30[:, :P15])

    V_cv3_d5, cv3_nval_d5, cv3_skip_d5 = inf.cv3_cluster_jackknife_cov(0.05)
    seL_cv3_d5 = np.sqrt(np.maximum(np.diag(V_cv3_d5), 1e-20))
    p30_cv3_d5, _, c30_cv3_d5 = max_t_stepdown(dL_d5 / seL_cv3_d5, V_cv3_d5, 0.05, Z_std=Z_std_30)
    p30_cv3_t_d5, _, c30_cv3_t_d5 = max_t_stepdown(dL_d5 / seL_cv3_d5, V_cv3_d5, 0.05, Z_std=Z_std_30, df=float(max(inf.n_states - 1, 1)))

    V_cv3_asym, _, _ = inf.cv3_cluster_jackknife_cov(0.15, d_up=0.05)
    seL_cv3_asym = np.sqrt(np.maximum(np.diag(V_cv3_asym), 1e-20))
    p30_cv3_asym, _, c30_cv3_asym = max_t_stepdown(dL_asym / seL_cv3_asym, V_cv3_asym, 0.05, Z_std=Z_std_30)

    # Fixed-scale standardized bootstrap (from bootstrap.py) & true studentized bootstrap-t (B=2000)
    sb_blk = (boot_block or {}).get("standardized_boot_max_t") or (boot_block or {}).get("nongaussian_max_t", {})
    sb_wc = sb_blk.get("worst", {})
    sb_d5 = sb_blk.get("delta5", {})
    p_sb_wc = sb_wc.get("p_fwer", [None] * P30)
    p_sb_d5 = sb_d5.get("p_fwer", [None] * P30)

    bt_true = run_studentized_bootstrap_t(inf, B_bt=2000, seed=seed)
    fixed_indices_bt = bt_true.pop("fixed_indices")
    bt_wc = bt_true["worst"]; bt_d5 = bt_true["delta5"]
    c_bt_wc = float(bt_wc["c_bt"]); c_bt_d5 = float(bt_d5["c_bt"])
    bt_grid_d = np.asarray(bt_true["c_bt_grid_pp"], dtype=float) / 100.0
    bt_grid_c = np.asarray(bt_true["c_bt_grid_vals"], dtype=float)

    # 53-cluster Bayesian bootstrap-t (Exp(1) weights across exposure-state clusters)
    def summarize_custom_bt(t_obs: np.ndarray, T_mat: np.ndarray, B_val: int) -> OrderedDict:
        c_crit = float(np.quantile(T_mat.max(axis=1), 0.95))
        order = np.argsort(-t_obs, kind="mergesort")
        adj_p = np.zeros(P30); mc_se = np.zeros(P30)
        prev_p = 0.0; prev_se = 0.0
        for step in range(P30):
            active = order[step:]
            max_t = T_mat[:, active].max(axis=1)
            p_raw = float((1.0 + np.sum(max_t >= t_obs[order[step]])) / (B_val + 1.0))
            se_raw = float(math.sqrt(max(p_raw * (1.0 - p_raw), 0.0) / B_val))
            if p_raw >= prev_p:
                adj_p[order[step]] = p_raw; mc_se[order[step]] = se_raw
                prev_p = p_raw; prev_se = se_raw
            else:
                adj_p[order[step]] = prev_p; mc_se[order[step]] = prev_se
        return OrderedDict(
            c_bt=c_crit, p_fwer=adj_p.tolist(), mc_se_p=mc_se.tolist(),
            n_sig=int(sum(x <= 0.05 for x in adj_p)),
            rank_confidence_sets=rank_confidence_sets(inf.groups, ordered_30, adj_p, 0.05),
        )

    B_cbt = 2000
    cbt_eval = _batched_cluster_boot_t_dL(
        inf, inf.state_cell_sums, inf.state_cell_n, [1.0, 0.05], B_cbt,
        np.random.default_rng(seed + 77), batch_size=100,
    )
    dL_cbw, se_cbw = cbt_eval[1.0]
    T_cbw = (dL_cbw - dL_wc[None, :]) / np.maximum(se_cbw, 1e-12)
    cbt_wc = summarize_custom_bt(dL_wc / seL_st_wc, T_cbw, B_cbt)
    dL_cb5, se_cb5 = cbt_eval[0.05]
    T_cb5 = (dL_cb5 - dL_d5[None, :]) / np.maximum(se_cb5, 1e-12)
    cbt_d5 = summarize_custom_bt(dL_d5 / seL_st_d5, T_cb5, B_cbt)
    cluster_boot_t = OrderedDict(B=B_cbt, worst=cbt_wc, delta5=cbt_d5)

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
                m = Z.shape[0]
                Cc, Hc, Uc, Hrc, Krc = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3], Z[:, 4]
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

    # Single-state-only (R11) full 52-state inference (HC1, CV1, CV3, WCB, boot-t, breakdown)
    r11_df = main_df[main_df["state"].astype(str) != "Multistate"].copy()
    t_r11 = S.build(r11_df, R.GROUPS, "setting_period", threshold=3)
    inf_r11 = StackedInference(r11_df, t_r11, ordered_30)
    th_r11_wc, V_r11_hc1_wc, _, V_r11_st_wc, st_r11_sc_wc = inf_r11.analytic_covariances(1.0, return_state_scores=True)
    th_r11_d5, V_r11_hc1_d5, _, V_r11_st_d5, st_r11_sc_d5 = inf_r11.analytic_covariances(0.05, return_state_scores=True)
    dL_r11_wc = th_r11_wc[off:off + P30]; dL_r11_d5 = th_r11_d5[off:off + P30]
    hV_r11_wc = V_r11_hc1_wc[off:off + P30, off:off + P30]; hV_r11_d5 = V_r11_hc1_d5[off:off + P30, off:off + P30]
    cV_r11_wc = V_r11_st_wc[off:off + P30, off:off + P30]; cV_r11_d5 = V_r11_st_d5[off:off + P30, off:off + P30]
    p30_r11_hc1_wc, _, c30_r11_hc1_wc = max_t_stepdown(dL_r11_wc / np.sqrt(np.maximum(np.diag(hV_r11_wc), 1e-20)), hV_r11_wc, 0.05, Z_std=Z_std_30)
    p30_r11_hc1_d5, _, c30_r11_hc1_d5 = max_t_stepdown(dL_r11_d5 / np.sqrt(np.maximum(np.diag(hV_r11_d5), 1e-20)), hV_r11_d5, 0.05, Z_std=Z_std_30)
    p30_r11_wc, _, c30_r11_wc = max_t_stepdown(dL_r11_wc / np.sqrt(np.maximum(np.diag(cV_r11_wc), 1e-20)), cV_r11_wc, 0.05, Z_std=Z_std_30)
    p30_r11_d5, _, c30_r11_d5 = max_t_stepdown(dL_r11_d5 / np.sqrt(np.maximum(np.diag(cV_r11_d5), 1e-20)), cV_r11_d5, 0.05, Z_std=Z_std_30)
    V_r11_cv3_wc, r11_cv3_nval, r11_cv3_skip = inf_r11.cv3_cluster_jackknife_cov(1.0)
    V_r11_cv3_d5, _, _ = inf_r11.cv3_cluster_jackknife_cov(0.05)
    p30_r11_cv3_wc, _, c30_r11_cv3_wc = max_t_stepdown(dL_r11_wc / np.sqrt(np.maximum(np.diag(V_r11_cv3_wc), 1e-20)), V_r11_cv3_wc, 0.05, Z_std=Z_std_30)
    p30_r11_cv3_d5, _, c30_r11_cv3_d5 = max_t_stepdown(dL_r11_d5 / np.sqrt(np.maximum(np.diag(V_r11_cv3_d5), 1e-20)), V_r11_cv3_d5, 0.05, Z_std=Z_std_30)
    wcb_r11_wc = wild_cluster_bootstrap_fwer(dL_r11_wc, st_r11_sc_wc[:, off:off + P30], 0.05, 9999, seed=seed + 3)
    wcb_r11_d5 = wild_cluster_bootstrap_fwer(dL_r11_d5, st_r11_sc_d5[:, off:off + P30], 0.05, 9999, seed=seed + 4)
    bt_r11 = run_studentized_bootstrap_t(inf_r11, B_bt=2000, seed=seed + 5)
    bt_r11.pop("fixed_indices", None)
    _, grid_c_r11_hc1 = dynamic_csim_grid(inf_r11, "hc1", Z_std_30, alpha=0.05, n_grid=21)
    _, grid_c_r11_st = dynamic_csim_grid(inf_r11, "state", Z_std_30, alpha=0.05, n_grid=21)
    _, grid_c_r11_cv3 = dynamic_csim_grid(inf_r11, "cv3", Z_std_30, alpha=0.05, n_grid=21)
    r11_bt_gd = np.asarray(bt_r11["c_bt_grid_pp"], dtype=float) / 100.0
    r11_bt_gc = np.asarray(bt_r11["c_bt_grid_vals"], dtype=float)
    r11_caches = {"hc1": {}, "state": {}, "cv3": {}}
    r11_pair_ds = OrderedDict()
    for idx_r, (k_r, _) in enumerate(sorted_p):
        r_hc1, _, _, _ = confidence_tipping_point_dynamic(inf_r11, idx_r, "hc1", grid_d=grid_d, grid_c=grid_c_r11_hc1, cov_cache=r11_caches["hc1"])
        r_st, _, _, _ = confidence_tipping_point_dynamic(inf_r11, idx_r, "state", grid_d=grid_d, grid_c=grid_c_r11_st, cov_cache=r11_caches["state"])
        r_cv3, _, _, _ = confidence_tipping_point_dynamic(inf_r11, idx_r, "cv3", grid_d=grid_d, grid_c=grid_c_r11_cv3, cov_cache=r11_caches["cv3"])
        r_bt, _, _, _ = confidence_tipping_point_dynamic(inf_r11, idx_r, "hc1", grid_d=r11_bt_gd, grid_c=r11_bt_gc, cov_cache=r11_caches["hc1"])
        r11_pair_ds[k_r] = OrderedDict(hc1=r_hc1, state=r_st, cv3=r_cv3, bootstrap_t=r_bt)

    def summarize_rank_bp(pair_map: dict, field_fn) -> OrderedDict:
        def best_of(keys_chk):
            best_k, best_v = None, 999.0
            for kk in keys_chk:
                vv = field_fn(pair_map[kk])
                vn = 100.0 if vv is None else float(vv)
                if vn < best_v:
                    best_v = vn; best_k = kk
            return (None if best_v >= 100.0 and field_fn(pair_map[best_k]) is None else best_v), best_k

        stec_v, stec_p = best_of(["SAL|STEC", "STEC|CAMP", "STEC|CPERF", "STEC|NOV_GI", "STEC|NOV_GII"])
        sal_v, sal_p = best_of(["SAL|STEC", "SAL|CAMP", "SAL|CPERF", "SAL|NOV_GI", "SAL|NOV_GII"])
        camp_v, camp_p = best_of(["SAL|CAMP", "STEC|CAMP", "CAMP|CPERF", "CAMP|NOV_GI", "CAMP|NOV_GII"])
        within_v, within_p = best_of(["SAL|STEC", "SAL|CAMP", "STEC|CAMP"])
        cross_v, cross_p = best_of([
            f"{hi}|{lo}" for hi in ("STEC", "SAL", "CAMP") for lo in ("CPERF", "NOV_GI", "NOV_GII")
            if f"{hi}|{lo}" in pair_map
        ] + [
            f"{lo}|{hi}" for hi in ("STEC", "SAL", "CAMP") for lo in ("CPERF", "NOV_GI", "NOV_GII")
            if f"{lo}|{hi}" in pair_map
        ])
        joint_v, joint_p = best_of([stec_p, sal_p, camp_p])
        return OrderedDict(
            STEC_1_1=stec_v, STEC_1_1_binding_pair=stec_p,
            SAL_2_2=sal_v, SAL_2_2_binding_pair=sal_p,
            CAMP_3_3=camp_v, CAMP_3_3_binding_pair=camp_p,
            within_upper_tier_min=within_v, within_upper_tier_binding_pair=within_p,
            upper_vs_lower_tier_min=cross_v, upper_vs_lower_tier_binding_pair=cross_p,
            joint_upper_tier_min=joint_v, joint_upper_tier_binding_pair=joint_p,
        )

    r11_inference = OrderedDict(
        n_states=int(inf_r11.n_states),
        n_outbreaks=int(len(r11_df[r11_df["cell"].isin(t_r11.cells)])),
        n_cells=int(len(t_r11.cells)),
        cv3_n_valid=int(r11_cv3_nval),
        cv3_skipped=r11_cv3_skip,
        c_sim_worst=OrderedDict(hc1=float(c30_r11_hc1_wc), state=float(c30_r11_wc), cv3=float(c30_r11_cv3_wc),
                                wcb=float(wcb_r11_wc["c_wcb_sim"]), boot_t=float(bt_r11["worst"]["c_bt"])),
        c_sim_delta5=OrderedDict(hc1=float(c30_r11_hc1_d5), state=float(c30_r11_d5), cv3=float(c30_r11_cv3_d5),
                                 wcb=float(wcb_r11_d5["c_wcb_sim"]), boot_t=float(bt_r11["delta5"]["c_bt"])),
        n_sig_worst=OrderedDict(
            hc1=int(sum(x <= 0.05 for x in p30_r11_hc1_wc)),
            state=int(sum(x <= 0.05 for x in p30_r11_wc)),
            cv3=int(sum(x <= 0.05 for x in p30_r11_cv3_wc)),
            wcb=int(sum(x <= 0.05 for x in wcb_r11_wc["p_fwer_wcb"])),
            boot_t=int(bt_r11["worst"]["n_sig"]),
        ),
        n_sig_delta5=OrderedDict(
            hc1=int(sum(x <= 0.05 for x in p30_r11_hc1_d5)),
            state=int(sum(x <= 0.05 for x in p30_r11_d5)),
            cv3=int(sum(x <= 0.05 for x in p30_r11_cv3_d5)),
            wcb=int(sum(x <= 0.05 for x in wcb_r11_d5["p_fwer_wcb"])),
            boot_t=int(bt_r11["delta5"]["n_sig"]),
        ),
        ranks=OrderedDict(
            worst_hc1=rank_confidence_sets(inf_r11.groups, ordered_30, p30_r11_hc1_wc, 0.05),
            worst_state=rank_confidence_sets(inf_r11.groups, ordered_30, p30_r11_wc, 0.05),
            worst_cv3=rank_confidence_sets(inf_r11.groups, ordered_30, p30_r11_cv3_wc, 0.05),
            worst_wcb=rank_confidence_sets(inf_r11.groups, ordered_30, wcb_r11_wc["p_fwer_wcb"], 0.05),
            worst_boot_t=bt_r11["worst"]["rank_confidence_sets"],
            delta5_hc1=rank_confidence_sets(inf_r11.groups, ordered_30, p30_r11_hc1_d5, 0.05),
            delta5_state=rank_confidence_sets(inf_r11.groups, ordered_30, p30_r11_d5, 0.05),
            delta5_cv3=rank_confidence_sets(inf_r11.groups, ordered_30, p30_r11_cv3_d5, 0.05),
            delta5_wcb=rank_confidence_sets(inf_r11.groups, ordered_30, wcb_r11_d5["p_fwer_wcb"], 0.05),
            delta5_boot_t=bt_r11["delta5"]["rank_confidence_sets"],
        ),
        rank_breakdown_pp=OrderedDict(
            hc1=summarize_rank_bp(r11_pair_ds, lambda x: x["hc1"]),
            state=summarize_rank_bp(r11_pair_ds, lambda x: x["state"]),
            cv3=summarize_rank_bp(r11_pair_ds, lambda x: x["cv3"]),
            bootstrap_t=summarize_rank_bp(r11_pair_ds, lambda x: x["bootstrap_t"]),
        ),
        pair_breakdown_pp=r11_pair_ds,
    )

    # Diagnostic-informed group-specific asymmetric band (SAL -10/+5, CAMP -7/+5, others +-5 pp) formal inference
    diag_dd = np.array([0.10 if g == "SAL" else (0.07 if g == "CAMP" else 0.05) for g in inf.groups], dtype=float)
    diag_du = np.full(G, 0.05, dtype=float)
    th_da, V_da_hc1, V_da_st = inf.analytic_covariances_group_asym(diag_dd, diag_du)
    dL_da = th_da[off:off + P30]
    cov_da_hc1 = V_da_hc1[off:off + P30, off:off + P30]
    cov_da_st = V_da_st[off:off + P30, off:off + P30]
    V_da_cv3, _, _ = inf.cv3_cluster_jackknife_cov(d_down_vec=diag_dd, d_up_vec=diag_du)
    seL_da_hc1 = np.sqrt(np.maximum(np.diag(cov_da_hc1), 1e-20))
    seL_da_st = np.sqrt(np.maximum(np.diag(cov_da_st), 1e-20))
    seL_da_cv3 = np.sqrt(np.maximum(np.diag(V_da_cv3), 1e-20))
    p30_da_hc1, _, c30_da_hc1 = max_t_stepdown(dL_da / seL_da_hc1, cov_da_hc1, 0.05, Z_std=Z_std_30)
    p30_da_st, _, c30_da_st = max_t_stepdown(dL_da / seL_da_st, cov_da_st, 0.05, Z_std=Z_std_30)
    p30_da_cv3, _, c30_da_cv3 = max_t_stepdown(dL_da / seL_da_cv3, V_da_cv3, 0.05, Z_std=Z_std_30)
    dL_da_bt, se_da_bt = _batched_boot_t_dL(
        inf, inf.cell_arrays, [], bt_true["B_bt"], batch_size=100,
        fixed_indices=fixed_indices_bt, asym_vecs=(diag_dd, diag_du),
    )["asym"]
    da_bt = summarize_custom_bt(dL_da / seL_da_hc1, (dL_da_bt - dL_da[None, :]) / np.maximum(se_da_bt, 1e-12), bt_true["B_bt"])
    diag_asymmetric_inference = OrderedDict(
        down_by_group_pp=OrderedDict((g, 100.0 * float(diag_dd[i])) for i, g in enumerate(inf.groups)),
        up_by_group_pp=OrderedDict((g, 100.0 * float(diag_du[i])) for i, g in enumerate(inf.groups)),
        c_sim=OrderedDict(hc1=float(c30_da_hc1), state=float(c30_da_st), cv3=float(c30_da_cv3), boot_t=float(da_bt["c_bt"])),
        n_sig=OrderedDict(
            hc1=int(sum(x <= 0.05 for x in p30_da_hc1)),
            state=int(sum(x <= 0.05 for x in p30_da_st)),
            cv3=int(sum(x <= 0.05 for x in p30_da_cv3)),
            boot_t=int(da_bt["n_sig"]),
        ),
        ranks=OrderedDict(
            hc1=rank_confidence_sets(inf.groups, ordered_30, p30_da_hc1, 0.05),
            state=rank_confidence_sets(inf.groups, ordered_30, p30_da_st, 0.05),
            cv3=rank_confidence_sets(inf.groups, ordered_30, p30_da_cv3, 0.05),
            boot_t=da_bt["rank_confidence_sets"],
        ),
    )

    # OR-scale (floor=0.001) full 30-hypothesis inference (Gamma=1.5, 2, 3, 5 + Gamma*_0.95)
    or_gammas_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 15.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, 10000.0]
    or_bt_eval = _batched_boot_t_or_dL(inf, inf.cell_arrays, or_gammas_grid, bt_true["B_bt"], fixed_indices_bt, floor=0.001, batch_size=100)
    or_c_bt_grid = []
    or_by_gamma = OrderedDict()
    or_grid_c_hc1 = []; or_grid_c_st = []; or_grid_c_cv3 = []
    for gv in or_gammas_grid:
        th_g, V_g_hc1, _, V_g_st = inf.analytic_covariances_or(gv, floor=0.001)
        dL_g = th_g[off:off + P30]
        cV_g_hc1 = V_g_hc1[off:off + P30, off:off + P30]
        cV_g_st = V_g_st[off:off + P30, off:off + P30]
        V_g_cv3, _, _ = inf.cv3_cluster_jackknife_cov(or_gamma=gv, floor=0.001)
        se_g_hc1 = np.sqrt(np.maximum(np.diag(cV_g_hc1), 1e-20))
        se_g_st = np.sqrt(np.maximum(np.diag(cV_g_st), 1e-20))
        se_g_cv3 = np.sqrt(np.maximum(np.diag(V_g_cv3), 1e-20))
        pg_hc1, _, cg_hc1 = max_t_stepdown(dL_g / se_g_hc1, cV_g_hc1, 0.05, Z_std=Z_std_30)
        pg_st, _, cg_st = max_t_stepdown(dL_g / se_g_st, cV_g_st, 0.05, Z_std=Z_std_30)
        pg_cv3, _, cg_cv3 = max_t_stepdown(dL_g / se_g_cv3, V_g_cv3, 0.05, Z_std=Z_std_30)
        dL_bg, se_bg = or_bt_eval[gv]
        T_bg = (dL_bg - dL_g[None, :]) / np.maximum(se_bg, 1e-12)
        bt_g = summarize_custom_bt(dL_g / se_g_hc1, T_bg, bt_true["B_bt"])
        or_c_bt_grid.append(float(bt_g["c_bt"]))
        or_grid_c_hc1.append(float(cg_hc1)); or_grid_c_st.append(float(cg_st)); or_grid_c_cv3.append(float(cg_cv3))
        if gv in (1.5, 2.0, 3.0, 5.0):
            or_by_gamma[str(gv)] = OrderedDict(
                gamma=float(gv),
                c_sim=OrderedDict(hc1=float(cg_hc1), state=float(cg_st), cv3=float(cg_cv3), boot_t=float(bt_g["c_bt"])),
                n_sig=OrderedDict(
                    hc1=int(sum(x <= 0.05 for x in pg_hc1)),
                    state=int(sum(x <= 0.05 for x in pg_st)),
                    cv3=int(sum(x <= 0.05 for x in pg_cv3)),
                    boot_t=int(bt_g["n_sig"]),
                ),
                ranks=OrderedDict(
                    hc1=rank_confidence_sets(inf.groups, ordered_30, pg_hc1, 0.05),
                    state=rank_confidence_sets(inf.groups, ordered_30, pg_st, 0.05),
                    cv3=rank_confidence_sets(inf.groups, ordered_30, pg_cv3, 0.05),
                    boot_t=bt_g["rank_confidence_sets"],
                ),
                pairs_p_fwer=OrderedDict((k_p, OrderedDict(hc1=float(pg_hc1[i_p]), state=float(pg_st[i_p]), cv3=float(pg_cv3[i_p]), boot_t=float(bt_g["p_fwer"][i_p])))
                                          for i_p, (k_p, _) in enumerate(sorted_p)),
            )

    # Pre-scan 121 log-spaced Gammas in [1, 10000] + [1e4, 1e7] tail check for Gamma*_0.95 across all 15 ordered pairs
    g_scan = np.geomspace(1.0, 10000.0, 121)
    or_scan_dL = np.zeros((len(g_scan), P15))
    or_scan_se = {m: np.zeros((len(g_scan), P15)) for m in ("hc1", "state", "cv3")}
    for gi_s, gv_s in enumerate(g_scan):
        th_s, Vh_s, _, Vs_s = inf.analytic_covariances_or(float(gv_s), floor=0.001)
        Vc_s, _, _ = inf.cv3_cluster_jackknife_cov(or_gamma=float(gv_s), floor=0.001)
        or_scan_dL[gi_s] = th_s[off:off + P15]
        or_scan_se["hc1"][gi_s] = np.sqrt(np.maximum(np.diag(Vh_s[off:off + P15, off:off + P15]), 0.0))
        or_scan_se["state"][gi_s] = np.sqrt(np.maximum(np.diag(Vs_s[off:off + P15, off:off + P15]), 0.0))
        or_scan_se["cv3"][gi_s] = np.sqrt(np.maximum(np.diag(Vc_s[:P15, :P15]), 0.0))

    or_grid_arr = np.asarray(or_gammas_grid, dtype=float)
    def or_gamma_star(pair_idx: int, se_kind: str, crit_mode: str) -> float | None:
        def crit_g(gv: float) -> float:
            if crit_mode == "pw":
                return float(norm.ppf(0.95))
            if crit_mode == "bt":
                return float(np.interp(min(gv, or_grid_arr[-1]), or_grid_arr, or_c_bt_grid))
            if crit_mode == "hc1":
                return float(np.interp(min(gv, or_grid_arr[-1]), or_grid_arr, or_grid_c_hc1))
            if crit_mode == "state":
                return float(np.interp(min(gv, or_grid_arr[-1]), or_grid_arr, or_grid_c_st))
            return float(np.interp(min(gv, or_grid_arr[-1]), or_grid_arr, or_grid_c_cv3))

        def margin_g(gv: float) -> float:
            th_q, Vh_q, _, Vs_q = inf.analytic_covariances_or(float(gv), floor=0.001)
            dL_q = float(th_q[off + pair_idx])
            if se_kind == "cv3":
                Vc_q, _, _ = inf.cv3_cluster_jackknife_cov(or_gamma=float(gv), floor=0.001)
                se_q = math.sqrt(max(float(Vc_q[pair_idx, pair_idx]), 0.0))
            elif se_kind == "state":
                se_q = math.sqrt(max(float(Vs_q[off + pair_idx, off + pair_idx]), 0.0))
            else:
                se_q = math.sqrt(max(float(Vh_q[off + pair_idx, off + pair_idx]), 0.0))
            return dL_q - crit_g(gv) * se_q

        m_arr = np.array([or_scan_dL[k_g, pair_idx] - crit_g(float(g_scan[k_g])) * or_scan_se[se_kind][k_g, pair_idx]
                          for k_g in range(len(g_scan))])
        if m_arr[0] <= 0.0:
            return 1.0
        idx_c = np.where((m_arr[:-1] > 0.0) & (m_arr[1:] <= 0.0))[0]
        if len(idx_c) == 0:
            if margin_g(1e7) <= 0.0:
                return float(brentq(margin_g, float(g_scan[-1]), 1e7, xtol=1e-4))
            return None
        k0 = int(idx_c[0])
        return float(brentq(margin_g, float(g_scan[k0]), float(g_scan[k0 + 1]), xtol=1e-4))

    or_gamma_star_95 = OrderedDict()
    for idx_o, (k_o, p_o) in enumerate(sorted_p):
        or_gamma_star_95[k_o] = OrderedDict(
            higher=p_o["higher_at_delta0"],
            lower=p_o["lower_at_delta0"],
            gamma_star_pt=B.gamma_star(t_main, ordered_15[idx_o][0], ordered_15[idx_o][1], floor=0.001),
            gamma_star_95_pw=or_gamma_star(idx_o, "hc1", "pw"),
            gamma_star_95_hc1=or_gamma_star(idx_o, "hc1", "hc1"),
            gamma_star_95_state=or_gamma_star(idx_o, "state", "state"),
            gamma_star_95_cv3=or_gamma_star(idx_o, "cv3", "cv3"),
            gamma_star_95_bt=or_gamma_star(idx_o, "hc1", "bt"),
        )
    or_inference = OrderedDict(floor=0.001, by_gamma=or_by_gamma, gamma_star_95=or_gamma_star_95)

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
        bt_r.pop("fixed_indices", None)
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
        V_aw_cv3, _, _ = inf_a.cv3_cluster_jackknife_cov(1.0)
        V_a5_cv3, _, _ = inf_a.cv3_cluster_jackknife_cov(0.05)
        seLw_cv3 = np.sqrt(np.maximum(np.diag(V_aw_cv3), 1e-20))
        seL5_cv3 = np.sqrt(np.maximum(np.diag(V_a5_cv3), 1e-20))
        pw_cv3, _, _ = max_t_stepdown(dLw_a / seLw_cv3, V_aw_cv3, 0.05, Z_std=Z_std_30)
        p5_cv3, _, _ = max_t_stepdown(dL5_a / seL5_cv3, V_a5_cv3, 0.05, Z_std=Z_std_30)
        bt_a = run_studentized_bootstrap_t(inf_a, B_bt=2000, seed=seed, grid_pp=(5.0, 100.0))
        bt_a.pop("fixed_indices", None)
        return OrderedDict(
            n_cells=int(alt_table.n_cells if hasattr(alt_table, "n_cells") else len(alt_table.cells)),
            min_cell_outbreaks=int(inf_a.n.min()),
            fixed_weights=bool(fixed_w_alt is not None),
            n_sig_worst=OrderedDict(hc1=int(sum(x <= 0.05 for x in pw_hc1)),
                                    state=int(sum(x <= 0.05 for x in pw_st)),
                                    cv3=int(sum(x <= 0.05 for x in pw_cv3)),
                                    boot_t=int(bt_a["worst"]["n_sig"])),
            n_sig_delta5=OrderedDict(hc1=int(sum(x <= 0.05 for x in p5_hc1)),
                                     state=int(sum(x <= 0.05 for x in p5_st)),
                                     cv3=int(sum(x <= 0.05 for x in p5_cv3)),
                                     boot_t=int(bt_a["delta5"]["n_sig"])),
            c_bt_worst=float(bt_a["worst"]["c_bt"]),
            c_bt_delta5=float(bt_a["delta5"]["c_bt"]),
            stec_nov_gii_p_worst=OrderedDict(hc1=float(pw_hc1[3]), state=float(pw_st[3]), cv3=float(pw_cv3[3]), boot_t=float(bt_a["worst"]["p_fwer"][3])),
            ranks=OrderedDict(
                worst_hc1=rank_confidence_sets(inf.groups, ordered_30, pw_hc1, 0.05),
                worst_state=rank_confidence_sets(inf.groups, ordered_30, pw_st, 0.05),
                worst_cv3=rank_confidence_sets(inf.groups, ordered_30, pw_cv3, 0.05),
                worst_boot_t=bt_a["worst"]["rank_confidence_sets"],
                delta5_hc1=rank_confidence_sets(inf.groups, ordered_30, p5_hc1, 0.05),
                delta5_cv3=rank_confidence_sets(inf.groups, ordered_30, p5_cv3, 0.05),
                delta5_boot_t=bt_a["delta5"]["rank_confidence_sets"],
            ),
        )

    r2_df = main_df.assign(cell=main_df["period"]).copy()
    t_r2 = S.build(main_df, R.GROUPS, "period", threshold=3)
    res_r2_inf = summarize_alt_spec(r2_df, t_r2, fixed_w_alt=None)

    r10_df = main_df[main_df["cell"].isin(t_main.cells)].copy()
    kr_pos_r10 = r10_df["K_rec"] > 0
    r10_df.loc[kr_pos_r10, "H_rec"] = r10_df.loc[kr_pos_r10, "H_rec"] / r10_df.loc[kr_pos_r10, "K_rec"]
    r10_df.loc[kr_pos_r10, "K_rec"] = 1.0
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
    max_dyn_err_cv3 = 0.0
    kink_dists_by_method = {"sim_hc1": [], "sim_state": [], "sim_cv3": [], "boot_t": []}
    bps_all = np.concatenate([p_valid, 1.0 - p_valid])
    caches = {"hc1": {}, "state": {}, "cv3": {}}

    def implied_rates_at_delta(delta_pp_val: float | None, h_idx: int, l_idx: int):
        if delta_pp_val is None or delta_pp_val <= 0.0 or delta_pp_val >= 100.0:
            return None
        d_frac = delta_pp_val / 100.0
        undef_t = np.isnan(t_main.p_obs); pp_t = np.where(undef_t, 0.0, t_main.p_obs)
        qL_h = np.where(undef_t[h_idx], 0.0, np.maximum(0.0, pp_t[h_idx] - d_frac))
        qU_l = np.where(undef_t[l_idx], 1.0, np.minimum(1.0, pp_t[l_idx] + d_frac))
        L_h, _ = B.group_bounds(t_main, h_idx, d_frac)
        _, U_l = B.group_bounds(t_main, l_idx, d_frac)
        wt_h = t_main.w * (t_main.U[h_idx] / np.maximum(t_main.C[h_idx], 1e-15))
        wt_l = t_main.w * (t_main.U[l_idx] / np.maximum(t_main.C[l_idx], 1e-15))
        qL_h_mean = float(np.sum(wt_h * qL_h) / np.sum(wt_h)) if np.sum(wt_h) > 0 else 0.0
        qU_l_mean = float(np.sum(wt_l * qU_l) / np.sum(wt_l)) if np.sum(wt_l) > 0 else 0.0
        return OrderedDict(
            L_high_pct=100.0 * float(L_h), U_low_pct=100.0 * float(U_l),
            implied_unrec_high_pct=100.0 * qL_h_mean, implied_unrec_low_pct=100.0 * qU_l_mean,
        )

    def refine_bt_root(pair_idx: int, ds_bt_grid: float | None) -> tuple[float | None, float | None, float | None]:
        """Exact uninterpolated c_bt(delta) verification & local root refinement for finite bootstrap-t roots."""
        if ds_bt_grid is None or ds_bt_grid <= 0.0 or ds_bt_grid >= 100.0:
            return ds_bt_grid, ds_bt_grid, None
        d0 = ds_bt_grid / 100.0
        lo = max(0.0, d0 - 0.005); hi = min(1.0, d0 + 0.005)
        loc_deltas = sorted({round(lo, 6), round(d0, 6), round(hi, 6)})
        ev = _batched_boot_t_dL(inf, inf.cell_arrays, loc_deltas, bt_true["B_bt"], batch_size=100, fixed_indices=fixed_indices_bt)
        c_loc = []
        for dv in loc_deltas:
            th_d, _, _, _ = inf.analytic_covariances(dv)
            dL_b, se_b = ev[dv]
            T_d = (dL_b - th_d[off:off + P30][None, :]) / np.maximum(se_b, 1e-12)
            c_loc.append(float(np.quantile(T_d.max(axis=1), 0.95)))
        def m_loc(d: float) -> float:
            th_d, Vh_d, _, _ = inf.analytic_covariances(d)
            se_d = math.sqrt(max(float(Vh_d[off + pair_idx, off + pair_idx]), 0.0))
            c_val = float(np.interp(d, loc_deltas, c_loc))
            return float(th_d[off + pair_idx] - c_val * se_d)
        if m_loc(lo) > 0.0 >= m_loc(hi):
            d_ex = float(brentq(m_loc, lo, hi, xtol=1e-6))
        else:
            d_ex = d0
        ev_ex = _batched_boot_t_dL(inf, inf.cell_arrays, [round(d_ex, 6)], bt_true["B_bt"], batch_size=100, fixed_indices=fixed_indices_bt)[round(d_ex, 6)]
        th_ex, Vh_ex, _, _ = inf.analytic_covariances(round(d_ex, 6))
        T_ex = (ev_ex[0] - th_ex[off:off + P30][None, :]) / np.maximum(ev_ex[1], 1e-12)
        c_ex = float(np.quantile(T_ex.max(axis=1), 0.95))
        marg_ex = float(th_ex[off + pair_idx] - c_ex * math.sqrt(max(float(Vh_ex[off + pair_idx, off + pair_idx]), 0.0)))
        return 100.0 * d_ex, ds_bt_grid, marg_ex

    for idx, (key, p) in enumerate(sorted_p):
        h_idx, l_idx = ordered_15[idx]
        c_im = imbens_manski_crit(float(dU_wc[idx] - dL_wc[idx]), float(seL_wc[idx]), float(seU_wc[idx]), 0.05)
        ci_lo = float(dL_wc[idx] - c_im * seL_wc[idx])
        ci_hi = float(dU_wc[idx] + c_im * seU_wc[idx])

        c_im_st = imbens_manski_crit(float(dU_wc[idx] - dL_wc[idx]), float(seL_st_wc[idx]), float(seU_st_wc[idx]), 0.05)
        ci_lo_st = float(dL_wc[idx] - c_im_st * seL_st_wc[idx])
        ci_hi_st = float(dU_wc[idx] + c_im_st * seU_st_wc[idx])

        ds_pw, _, pw_mono, pw_cross = confidence_tipping_point_dynamic(inf, idx, "hc1", fixed_c=z_pw, cov_cache=caches["hc1"])
        ds_sim, c_interp, hc1_mono, hc1_cross = confidence_tipping_point_dynamic(inf, idx, "hc1", grid_d=grid_d, grid_c=grid_c_hc1, cov_cache=caches["hc1"])
        if ds_sim is not None:
            _, V_ex, _, _ = inf.analytic_covariances(ds_sim / 100.0)
            _, _, c_exact = max_t_stepdown(np.zeros(P30), V_ex[off:off + P30, off:off + P30], 0.05, Z_std=Z_std_30)
            max_dyn_err = max(max_dyn_err, abs(c_exact - c_interp))
        ds_sim_st, _, st_mono, st_cross = confidence_tipping_point_dynamic(inf, idx, "state", grid_d=grid_d, grid_c=grid_c_st, cov_cache=caches["state"])
        ds_sim_cv3, c_cv3_interp, cv3_mono, cv3_cross = confidence_tipping_point_dynamic(inf, idx, "cv3", grid_d=grid_d, grid_c=grid_c_cv3, cov_cache=caches["cv3"])
        if ds_sim_cv3 is not None:
            V_cv3_ex, _, _ = inf.cv3_cluster_jackknife_cov(ds_sim_cv3 / 100.0)
            _, _, c_cv3_ex = max_t_stepdown(np.zeros(P30), V_cv3_ex, 0.05, Z_std=Z_std_30)
            max_dyn_err_cv3 = max(max_dyn_err_cv3, abs(c_cv3_ex - c_cv3_interp))
        ds_bt_grid_val, _, bt_mono, bt_cross = confidence_tipping_point_dynamic(inf, idx, "hc1", grid_d=bt_grid_d, grid_c=bt_grid_c, cov_cache=caches["hc1"])
        ds_bt_exact, _, bt_exact_marg = refine_bt_root(idx, ds_bt_grid_val)

        kd_map = OrderedDict()
        for m_name, d_root in (("sim_hc1", ds_sim), ("sim_state", ds_sim_st), ("sim_cv3", ds_sim_cv3), ("boot_t", ds_bt_exact)):
            if d_root is not None and 0.0 < d_root < 100.0:
                kd = float(np.min(np.abs(bps_all - d_root / 100.0))) * 100.0
                kd_map[m_name] = kd
                kink_dists_by_method[m_name].append(kd)
            else:
                kd_map[m_name] = None

        pairs[key] = OrderedDict(
            higher_at_delta0=p["higher_at_delta0"],
            lower_at_delta0=p["lower_at_delta0"],
            dL_worst=float(dL_wc[idx]),
            dU_worst=float(dU_wc[idx]),
            se_dL_hc1=float(seL_wc[idx]),
            se_dU_hc1=float(seU_wc[idx]),
            se_dL_state=float(seL_st_wc[idx]),
            se_dU_state=float(seU_st_wc[idx]),
            se_dL_cv3=float(seL_cv3_wc[idx]),
            se_dL_cv3_delta5=float(seL_cv3_d5[idx]),
            state_vs_hc1_se_ratio=float(seL_st_wc[idx] / seL_wc[idx]),
            cv3_vs_hc1_se_ratio=float(seL_cv3_wc[idx] / seL_wc[idx]),
            cv3_vs_state_se_ratio=float(seL_cv3_wc[idx] / seL_st_wc[idx]),
            z_dL_hc1=float(dL_wc[idx] / seL_wc[idx]),
            z_dL_state=float(dL_wc[idx] / seL_st_wc[idx]),
            z_dL_cv3=float(dL_wc[idx] / seL_cv3_wc[idx]),
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
            p_fwer_cv3_worst=float(p30_cv3_wc[idx]),
            p_fwer_cv3_delta5=float(p30_cv3_d5[idx]),
            p_fwer_wcb_worst=float(wcb_wc["p_fwer_wcb"][idx]),
            p_fwer_wcb_delta5=float(wcb_d5["p_fwer_wcb"][idx]),
            p_fwer_cluster_bt_worst=float(cbt_wc["p_fwer"][idx]),
            p_fwer_cluster_bt_delta5=float(cbt_d5["p_fwer"][idx]),
            p_fwer_asym_15_5_hc1=float(p30_asym_hc1[idx]),
            p_fwer_asym_15_5_state=float(p30_asym_st[idx]),
            p_fwer_asym_15_5_cv3=float(p30_cv3_asym[idx]),
            p_fwer_diag_asym_hc1=float(p30_da_hc1[idx]),
            p_fwer_diag_asym_state=float(p30_da_st[idx]),
            p_fwer_diag_asym_cv3=float(p30_da_cv3[idx]),
            p_fwer_diag_asym_boot_t=float(da_bt["p_fwer"][idx]),
            p_fwer_boot_std_worst=(None if p_sb_wc[idx] is None else float(p_sb_wc[idx])),
            p_fwer_boot_std_delta5=(None if p_sb_d5[idx] is None else float(p_sb_d5[idx])),
            p_fwer_boot_t_worst=float(bt_wc["p_fwer"][idx]),
            mc_se_boot_t_worst=float(bt_wc["mc_se_p"][idx]),
            p_fwer_boot_t_delta5=float(bt_d5["p_fwer"][idx]),
            mc_se_boot_t_delta5=float(bt_d5["mc_se_p"][idx]),
            p_fwer_singleton_worst=float(p30_sg_wc[idx]),
            p_fwer_singleton_delta5=float(p30_sg_d5[idx]),
            p_fwer_r11_hc1_worst=float(p30_r11_hc1_wc[idx]),
            p_fwer_r11_state_worst=float(p30_r11_wc[idx]),
            p_fwer_r11_cv3_worst=float(p30_r11_cv3_wc[idx]),
            p_fwer_r11_state_delta5=float(p30_r11_d5[idx]),
            delta_star_pt_pp=p["delta_star_pp"],
            delta_star_95_pw_pp=ds_pw,
            delta_star_95_sim_pp=ds_sim,
            delta_star_95_sim_state_pp=ds_sim_st,
            delta_star_95_cv3_pp=ds_sim_cv3,
            delta_star_95_bt_pp=ds_bt_exact,
            delta_star_95_bt_grid_pp=ds_bt_grid_val,
            bt_exact_margin_at_root=bt_exact_marg,
            root_monotonic=OrderedDict(pw=pw_mono, hc1=hc1_mono, state=st_mono, cv3=cv3_mono, boot_t=bt_mono),
            root_crossings=OrderedDict(pw=pw_cross, hc1=hc1_cross, state=st_cross, cv3=cv3_cross, boot_t=bt_cross),
            kink_dist_pp=kd_map,
            implied_at_delta_star_95=OrderedDict(
                hc1=implied_rates_at_delta(ds_sim, h_idx, l_idx),
                state=implied_rates_at_delta(ds_sim_st, h_idx, l_idx),
                cv3=implied_rates_at_delta(ds_sim_cv3, h_idx, l_idx),
                boot_t=implied_rates_at_delta(ds_bt_exact, h_idx, l_idx),
            ),
        )

    # Compute exact simultaneous rank breakdown points & explicit binding pairs for upper-tier rank intervals
    # Also compute exact stepdown CV3 breakdown point for SAL|CAMP (M6 transparency)
    idx_sal_camp = [k for k, _ in sorted_p].index("SAL|CAMP")
    cv3_sd_scan = np.linspace(0.03, 0.10, 15)
    p_sc_sd = []
    for dv_s in cv3_sd_scan:
        th_s, _, _, _ = inf.analytic_covariances(float(dv_s))
        Vc_s, _, _ = inf.cv3_cluster_jackknife_cov(float(dv_s))
        p_s, _, _ = max_t_stepdown(th_s[off:off + P30] / np.sqrt(np.maximum(np.diag(Vc_s), 1e-20)), Vc_s, 0.05, Z_std=Z_std_30)
        p_sc_sd.append(float(p_s[idx_sal_camp]))
    sal_camp_cv3_stepdown_pp = None
    for ks in range(len(cv3_sd_scan) - 1):
        if p_sc_sd[ks] <= 0.05 < p_sc_sd[ks + 1]:
            def _sd_margin(dv_q: float) -> float:
                th_q, _, _, _ = inf.analytic_covariances(float(dv_q))
                Vc_q, _, _ = inf.cv3_cluster_jackknife_cov(float(dv_q))
                pq, _, _ = max_t_stepdown(th_q[off:off + P30] / np.sqrt(np.maximum(np.diag(Vc_q), 1e-20)), Vc_q, 0.05, Z_std=Z_std_30)
                return float(0.05 - pq[idx_sal_camp])
            sal_camp_cv3_stepdown_pp = 100.0 * float(brentq(_sd_margin, float(cv3_sd_scan[ks]), float(cv3_sd_scan[ks + 1]), xtol=1e-4))
            break

    rank_breakdown = OrderedDict(
        hc1=summarize_rank_bp(pairs, lambda x: x["delta_star_95_sim_pp"]),
        state=summarize_rank_bp(pairs, lambda x: x["delta_star_95_sim_state_pp"]),
        cv3=summarize_rank_bp(pairs, lambda x: x["delta_star_95_cv3_pp"]),
        cv3_sal_camp_stepdown_pp=sal_camp_cv3_stepdown_pp,
        bootstrap_t=summarize_rank_bp(pairs, lambda x: x["delta_star_95_bt_pp"]),
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
        worst_cv3=rank_confidence_sets(inf.groups, ordered_30, p30_cv3_wc, 0.05),
        worst_wcb=rank_confidence_sets(inf.groups, ordered_30, wcb_wc["p_fwer_wcb"], 0.05),
        worst_cluster_bt=cbt_wc["rank_confidence_sets"],
        worst_boot_std=sb_wc.get("rank_confidence_sets"),
        worst_boot_t=bt_wc["rank_confidence_sets"],
        delta5_hc1=rank_confidence_sets(inf.groups, ordered_30, p30_d5, 0.05),
        delta5_state=rank_confidence_sets(inf.groups, ordered_30, p30_st_d5, 0.05),
        delta5_cv3=rank_confidence_sets(inf.groups, ordered_30, p30_cv3_d5, 0.05),
        delta5_wcb=rank_confidence_sets(inf.groups, ordered_30, wcb_d5["p_fwer_wcb"], 0.05),
        delta5_cluster_bt=cbt_d5["rank_confidence_sets"],
        delta5_boot_std=sb_d5.get("rank_confidence_sets"),
        delta5_boot_t=bt_d5["rank_confidence_sets"],
        asym_15_5_hc1=rank_confidence_sets(inf.groups, ordered_30, p30_asym_hc1, 0.05),
        asym_15_5_state=rank_confidence_sets(inf.groups, ordered_30, p30_asym_st, 0.05),
        asym_15_5_cv3=rank_confidence_sets(inf.groups, ordered_30, p30_cv3_asym, 0.05),
        diag_asym_hc1=diag_asymmetric_inference["ranks"]["hc1"],
        diag_asym_state=diag_asymmetric_inference["ranks"]["state"],
        diag_asym_cv3=diag_asymmetric_inference["ranks"]["cv3"],
        diag_asym_boot_t=diag_asymmetric_inference["ranks"]["boot_t"],
        r11=r11_inference["ranks"],
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
        r11_full=r11_inference,
    )
    small_cal = run_small_sample_calibration(inf, n_mc=400, B_inner=499, seed=seed)
    clust_cal_53 = run_cluster_double_bootstrap(inf, n_mc=300, B_inner=399, seed=seed, include_cbt=True)
    # 271-singleton clustering (M4): treat each of the 219 Multistate outbreaks as its own singleton cluster
    df_271 = main_df.copy()
    is_ms = df_271["state"].astype(str) == "Multistate"
    df_271.loc[is_ms, "state"] = "Multi_" + df_271.loc[is_ms, "outbreak_id"].astype(str)
    inf_271 = StackedInference(df_271, t_main, ordered_30)
    clust_cal_271 = run_cluster_double_bootstrap(inf_271, n_mc=300, B_inner=399, seed=seed + 201, include_cbt=False)
    clust_cal_r11 = run_cluster_double_bootstrap(inf_r11, n_mc=300, B_inner=399, seed=seed + 202, include_cbt=False)
    clust_cal_by_def = OrderedDict(state_53=clust_cal_53, singleton_271=clust_cal_271, r11_52=clust_cal_r11)
    param_cal = run_parametric_heavy_tail_calibration(inf, n_mc=300, B_inner=399, seed=seed)
    plausible = run_plausible_rate_scenarios(inf, fixed_indices_bt, Z_std_30)
    desc_cohort = build_descriptive_cohort(main_df, t_main)
    emp_cal = empirical_delta_calibration(main_df, t_main, ordered_15, fixed_indices_bt=fixed_indices_bt)

    # Also verify OR & group-asym IF vs finite-diff sandwich directly in checks (Audit Item a.1)
    diff_if_or = 0.0
    for gv_chk in (1.5, 2.0, 3.0, 5.0):
        _, V_an_chk, _, _ = inf.analytic_covariances_or(gv_chk, floor=0.001)
        # Finite diff over cell means under OR rule
        p_len = G * K * 5; q_len = 3 * G + 2 * P30; J_chk = np.zeros((q_len, p_len))
        fmu = inf.mu.reshape(-1)
        ph_i = np.array([h for h, _ in ordered_30], dtype=int); pl_i = np.array([l for _, l in ordered_30], dtype=int)
        for idx_p in range(p_len):
            st_p = 1e-6 * max(1.0, abs(fmu[idx_p]))
            vals_2 = []
            for sgn in (1.0, -1.0):
                mp = fmu.copy(); mp[idx_p] += sgn * st_p
                tb_p = inf.table_from_mu(mp.reshape(G, K, 5))
                undef_p = np.isnan(tb_p.p_obs); pp_p = np.where(undef_p, 0.0, tb_p.p_obs)
                pp_e = np.where(undef_p, 0.0, np.maximum(pp_p, 0.001))
                qLp = np.where(undef_p, 0.0, pp_e / np.maximum(pp_e + gv_chk * (1.0 - pp_e), 1e-15))
                qUp = np.where(undef_p, 1.0, (gv_chk * pp_e) / np.maximum(gv_chk * pp_e + (1.0 - pp_e), 1e-15))
                q0p = np.where(undef_p, 0.5, pp_p)
                Lp = ((tb_p.H + tb_p.U * qLp) / tb_p.C) @ tb_p.w
                Up = ((tb_p.H + tb_p.U * qUp) / tb_p.C) @ tb_p.w
                T0p = ((tb_p.H + tb_p.U * q0p) / tb_p.C) @ tb_p.w
                vals_2.append(np.concatenate([Lp, Up, T0p, Lp[ph_i] - Up[pl_i], Up[ph_i] - Lp[pl_i]]))
            J_chk[:, idx_p] = (vals_2[0] - vals_2[1]) / (2.0 * st_p)
        V_fd_o = np.zeros((q_len, q_len))
        for ki in range(G):
            for ci in range(K):
                off_p = (ki * K + ci) * 5; Jb = J_chk[:, off_p:off_p + 5]
                V_fd_o += Jb @ inf.omega_hc1[ki, ci] @ Jb.T
        diff_if_or = max(diff_if_or, float(np.max(np.abs(V_fd_o - V_an_chk))))

    # Verify group-specific diagnostic asymmetric IF vs finite-diff sandwich directly in checks
    J_da = np.zeros((q_len, G * K * 5))
    fmu_da = inf.mu.reshape(-1)
    ph_da = np.array([h for h, _ in ordered_30], dtype=int); pl_da = np.array([l for _, l in ordered_30], dtype=int)
    for idx_p in range(G * K * 5):
        st_p = 1e-6 * max(1.0, abs(fmu_da[idx_p]))
        vals_2 = []
        for sgn in (1.0, -1.0):
            mp = fmu_da.copy(); mp[idx_p] += sgn * st_p
            tb_p = inf.table_from_mu(mp.reshape(G, K, 5))
            undef_p = np.isnan(tb_p.p_obs); pp_p = np.where(undef_p, 0.0, tb_p.p_obs)
            qLp = np.where(undef_p, 0.0, np.maximum(0.0, pp_p - diag_dd[:, None]))
            qUp = np.where(undef_p, 1.0, np.minimum(1.0, pp_p + diag_du[:, None]))
            q0p = np.where(undef_p, 0.5, pp_p)
            Lp = ((tb_p.H + tb_p.U * qLp) / tb_p.C) @ tb_p.w
            Up = ((tb_p.H + tb_p.U * qUp) / tb_p.C) @ tb_p.w
            T0p = ((tb_p.H + tb_p.U * q0p) / tb_p.C) @ tb_p.w
            vals_2.append(np.concatenate([Lp, Up, T0p, Lp[ph_da] - Up[pl_da], Up[ph_da] - Lp[pl_da]]))
        J_da[:, idx_p] = (vals_2[0] - vals_2[1]) / (2.0 * st_p)
    V_fd_da = np.zeros((q_len, q_len))
    for ki in range(G):
        for ci in range(K):
            off_p = (ki * K + ci) * 5; Jb = J_da[:, off_p:off_p + 5]
            V_fd_da += Jb @ inf.omega_hc1[ki, ci] @ Jb.T
    diff_if_da = float(np.max(np.abs(V_fd_da - V_da_hc1)))
    diff_if_plausible = float(max(v["max_abs_diff_if_vs_fd"] for v in plausible.values()))

    # Verify exact cluster Gateaux IF (state_scores) vs central FD of cluster weights W_g = 1 +- h
    def cluster_fd_check(dv_c: float, st_sc_an: np.ndarray) -> float:
        ph_c = np.array([h for h, _ in ordered_30], dtype=int)
        pl_c = np.array([l for _, l in ordered_30], dtype=int)
        G_st = inf.n_states
        h_step = 1e-6
        sc_fd = np.zeros((G_st, 3 * G + 2 * P30))
        tot_s = np.sum(inf.state_cell_sums, axis=0)
        for g_idx in range(G_st):
            vals_2 = []
            for sgn in (1.0, -1.0):
                sg = tot_s + (sgn * h_step) * inf.state_cell_sums[g_idx]
                tb_c = inf.table_from_mu(sg / inf.n[:, :, None])
                ud = np.isnan(tb_c.p_obs); pp_c = np.where(ud, 0.0, tb_c.p_obs)
                qL_c = np.zeros((G, K)) if dv_c >= 1.0 else np.where(ud, 0.0, np.maximum(0.0, pp_c - dv_c))
                qU_c = np.ones((G, K)) if dv_c >= 1.0 else np.where(ud, 1.0, np.minimum(1.0, pp_c + dv_c))
                q0_c = np.where(ud, 0.5, pp_c)
                L_c = ((tb_c.H + tb_c.U * qL_c) / tb_c.C) @ tb_c.w
                U_c = ((tb_c.H + tb_c.U * qU_c) / tb_c.C) @ tb_c.w
                T0_c = ((tb_c.H + tb_c.U * q0_c) / tb_c.C) @ tb_c.w
                vals_2.append(np.concatenate([L_c, U_c, T0_c, L_c[ph_c] - U_c[pl_c], U_c[ph_c] - L_c[pl_c]]))
            sc_fd[g_idx] = (vals_2[0] - vals_2[1]) / (2.0 * h_step)
        return float(np.max(np.abs(sc_fd - st_sc_an)))

    diff_clust_wc = cluster_fd_check(1.0, st_scores_wc)
    diff_clust_d5 = cluster_fd_check(0.05, st_scores_d5)

    ratios_st = [v["state_vs_hc1_se_ratio"] for v in pairs.values()]
    ratios_cv3_hc1 = [v["cv3_vs_hc1_se_ratio"] for v in pairs.values()]
    ratios_cv3_st = [v["cv3_vs_state_se_ratio"] for v in pairs.values()]
    all_kink_dists = [d for vals in kink_dists_by_method.values() for d in vals]

    stacked = OrderedDict(
        alpha=0.05,
        n_hypotheses=int(P30),
        n_multiplier_draws=n_sim,
        c_sim_worst=float(c30_wc),
        c_sim_delta5=float(c30_d5),
        c_sim_state_worst=float(c30_st_wc),
        c_sim_state_delta5=float(c30_st_d5),
        c_sim_cv3_worst=float(c30_cv3_wc),
        c_sim_cv3_delta5=float(c30_cv3_d5),
        c_sim_cv3_t_worst=float(c30_cv3_t_wc),
        c_sim_cv3_t_delta5=float(c30_cv3_t_d5),
        c_sim_asym_15_5_hc1=float(c30_asym_hc1),
        c_sim_asym_15_5_state=float(c30_asym_st),
        c_sim_asym_15_5_cv3=float(c30_cv3_asym),
        c_sim_15_worst=float(c15_wc),
        c_sim_15_delta5=float(c15_d5),
        c_sim_15_state_worst=float(c15_st_wc),
        c_sim_15_cv3_worst=float(c15_cv3_wc),
        c_boot_std_worst=sb_wc.get("c_boot"),
        c_boot_std_delta5=sb_d5.get("c_boot"),
        c_bt_worst=float(c_bt_wc),
        c_bt_delta5=float(c_bt_d5),
        c_cbt_worst=float(cbt_wc["c_bt"]),
        c_cbt_delta5=float(cbt_d5["c_bt"]),
        c_bonferroni_two_sided_30=float(norm.ppf(1.0 - 0.05 / P30)),
        n_im_excludes_zero=sum(v["im_excludes_zero"] for v in pairs.values()),
        n_im_state_excludes_zero=sum(v["im_state_excludes_zero"] for v in pairs.values()),
        n_fwer_worst_sig=int(sum(x <= 0.05 for x in p30_wc)),
        n_fwer_delta5_sig=int(sum(x <= 0.05 for x in p30_d5)),
        n_fwer_state_worst_sig=int(sum(x <= 0.05 for x in p30_st_wc)),
        n_fwer_state_delta5_sig=int(sum(x <= 0.05 for x in p30_st_d5)),
        n_fwer_cv3_worst_sig=int(sum(x <= 0.05 for x in p30_cv3_wc)),
        n_fwer_cv3_delta5_sig=int(sum(x <= 0.05 for x in p30_cv3_d5)),
        n_fwer_cv3_t_worst_sig=int(sum(x <= 0.05 for x in p30_cv3_t_wc)),
        n_fwer_cv3_t_delta5_sig=int(sum(x <= 0.05 for x in p30_cv3_t_d5)),
        stec_nov_gii_p_cv3_t_worst=float(p30_cv3_t_wc[3]),
        n_fwer_wcb_worst_sig=int(sum(x <= 0.05 for x in wcb_wc["p_fwer_wcb"])),
        n_fwer_wcb_delta5_sig=int(sum(x <= 0.05 for x in wcb_d5["p_fwer_wcb"])),
        n_fwer_cluster_bt_worst_sig=int(cbt_wc["n_sig"]),
        n_fwer_cluster_bt_delta5_sig=int(cbt_d5["n_sig"]),
        n_fwer_asym_15_5_hc1_sig=int(sum(x <= 0.05 for x in p30_asym_hc1)),
        n_fwer_asym_15_5_state_sig=int(sum(x <= 0.05 for x in p30_asym_st)),
        n_fwer_asym_15_5_cv3_sig=int(sum(x <= 0.05 for x in p30_cv3_asym)),
        n_fwer_boot_std_worst_sig=sb_wc.get("n_sig"),
        n_fwer_boot_std_delta5_sig=sb_d5.get("n_sig"),
        n_fwer_boot_t_worst_sig=int(bt_wc["n_sig"]),
        n_fwer_boot_t_delta5_sig=int(bt_d5["n_sig"]),
        studentized_bootstrap_t=bt_true,
        cluster_boot_t=cluster_boot_t,
        state_se_ratio_min=float(min(ratios_st)),
        state_se_ratio_max=float(max(ratios_st)),
        cv3_vs_hc1_se_ratio_min=float(min(ratios_cv3_hc1)),
        cv3_vs_hc1_se_ratio_max=float(max(ratios_cv3_hc1)),
        cv3_vs_state_se_ratio_min=float(min(ratios_cv3_st)),
        cv3_vs_state_se_ratio_max=float(max(ratios_cv3_st)),
        cv3_n_valid_worst=int(cv3_nval_wc),
        cv3_skipped_worst=cv3_skip_wc,
        wild_cluster_bootstrap=OrderedDict(worst=wcb_wc, delta5=wcb_d5),
        dynamic_csim_grid=OrderedDict(delta_pp=(100.0 * grid_d).tolist(),
                                      hc1=grid_c_hc1.tolist(),
                                      state=grid_c_st.tolist(),
                                      cv3=grid_c_cv3.tolist()),
        diag_asymmetric_inference=diag_asymmetric_inference,
        or_inference=or_inference,
        plausible_scenarios=plausible,
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
            max_abs_diff_analytic_if_vs_sandwich_diag_asym=diff_if_da,
            max_abs_diff_analytic_if_vs_sandwich_or=diff_if_or,
            max_abs_diff_analytic_if_vs_sandwich_plausible=diff_if_plausible,
            max_abs_diff_analytic_if_vs_sandwich=max(diff_if_wc, diff_if_d5, diff_if_asym, diff_if_da, diff_if_or, diff_if_plausible),
            max_abs_diff_cluster_if_vs_fd_worst=diff_clust_wc,
            max_abs_diff_cluster_if_vs_fd_delta5=diff_clust_d5,
            max_abs_diff_dynamic_vs_exact_csim=float(max_dyn_err),
            max_abs_diff_dynamic_vs_exact_csim_cv3=float(max_dyn_err_cv3),
            max_abs_diff_theta_hat=float(np.max(np.abs(th_fd_wc - th_hc1))),
            n_kink_cells_delta5=n_kinks_d5,
            min_dist_to_kink_roots_pp_by_method=OrderedDict((m, (float(min(v)) if v else None)) for m, v in kink_dists_by_method.items()),
            min_dist_to_kink_roots_pp=(float(min(all_kink_dists)) if all_kink_dists else None),
            all_roots_unique_crossing=bool(all(
                v in (0, 1) for p_d in pairs.values() for v in p_d["root_crossings"].values()
            )),
            all_roots_monotone=bool(all(
                all(p_d["root_monotonic"].values()) for p_d in pairs.values()
            )),
            n_strata=int(G * K),
            n_state_clusters=int(inf.n_states),
        ),
    )

    return OrderedDict(
        descriptive_cohort=desc_cohort,
        stacked_inference=stacked,
        cluster_diagnostics=clust,
        small_sample_calibration=small_cal,
        cluster_double_bootstrap=clust_cal_53,
        cluster_double_bootstrap_by_def=clust_cal_by_def,
        parametric_calibration=param_cal,
        plausible_scenarios=plausible,
        empirical_calibration=emp_cal,
    )
