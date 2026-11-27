"""Worst-case and delta bounds, pairwise verdicts, critical deviation delta*, and OR/asymmetric bands.

Per group j and cell s (C = cases, H = hospitalized, U = cases with unrecorded status,
p = H_rec / K_rec recorded rate):
  worst case:  L_js = H/C,                    U_js = (H + U)/C
  delta band:  L_js(d) = (H + U max(0,p-d))/C, U_js(d) = (H + U min(1,p+d))/C
Standardized: L_j = sum_s w_s L_js. A cell without a recorded rate (K_rec = 0) keeps its
worst-case bounds at every d (never occurs in the main analysis).
delta* for a pair (h = higher point value at d = 0, l = other) = inf{d >= 0: L_h(d) <= U_l(d)};
None (= infinity) if L_h > U_l under the worst case; 0 if not guaranteed at d = 0.
"""
from __future__ import annotations

import numpy as np

UNDET = "undetermined"


def group_bounds(t, i: int, d: float):
    """Standardized (L, U) of group index i at deviation d (fraction; d >= 1 = worst case)."""
    C, H, U, p, w = t.C[i], t.H[i], t.U[i], t.p_obs[i], t.w
    undefined = np.isnan(p)
    pp = np.where(undefined, 0.0, p)
    lo_rate = np.where(undefined, 0.0, np.maximum(0.0, pp - d))
    hi_rate = np.where(undefined, 1.0, np.minimum(1.0, pp + d))
    L = float(np.sum(w * (H + U * lo_rate) / C))
    Uu = float(np.sum(w * (H + U * hi_rate) / C))
    return L, Uu


def worst_bounds(t, i: int):
    C, H, U, w = t.C[i], t.H[i], t.U[i], t.w
    return float(np.sum(w * H / C)), float(np.sum(w * (H + U) / C))


def lower_at(t, i, d):
    return group_bounds(t, i, d)[0]


def upper_at(t, i, d):
    return group_bounds(t, i, d)[1]


def point_delta0(t, i):
    """(L, U) at d = 0; a point unless some cell lacks a recorded rate."""
    return group_bounds(t, i, 0.0)


def direction_delta0(t, a: int, b: int):
    """Index of the group with the higher value at d = 0 (midpoint if the d = 0 bound is an
    interval), and the other index."""
    la, ua = point_delta0(t, a)
    lb, ub = point_delta0(t, b)
    ma, mb = (la + ua) / 2.0, (lb + ub) / 2.0
    return (a, b) if ma >= mb else (b, a)


def verdict(La, Ua, Lb, Ub, a_name, b_name):
    if La > Ub:
        return a_name
    if Lb > Ua:
        return b_name
    return UNDET


def _breakpoints(t, idx):
    bps = {0.0, 1.0}
    for i in idx:
        for p in t.p_obs[i]:
            if not np.isnan(p):
                for v in (p, 1.0 - p):
                    if 0.0 < v < 1.0:
                        bps.add(float(v))
    return sorted(bps)


def gap(t, h, l, d):
    return lower_at(t, h, d) - upper_at(t, l, d)


def delta_star(t, h: int, l: int, bisect: bool = True):
    """Return (exact, bisection) delta* as fractions (None = holds under the worst case)."""
    if gap(t, h, l, 1.0) > 0:
        return None, None
    if gap(t, h, l, 0.0) <= 0:
        return 0.0, (0.0 if bisect else None)
    bps = _breakpoints(t, (h, l))
    prev_d, prev_g = 0.0, gap(t, h, l, 0.0)
    exact = None
    for d in bps[1:]:
        g = gap(t, h, l, d)
        if g <= 0:
            exact = prev_d + prev_g * (d - prev_d) / (prev_g - g)
            break
        prev_d, prev_g = d, g
    bis = None
    if bisect:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if mid <= lo or mid >= hi:
                break
            if gap(t, h, l, mid) > 0:
                lo = mid
            else:
                hi = mid
        bis = hi
    return exact, bis


def lo_boundary(t, h: int, l: int, d_hi: float):
    """2-D region: largest deviation of the lower group l keeping L_h(d_hi) > U_l(d_lo).
    Returns inf{d_lo >= 0: U_l(d_lo) >= L_h(d_hi)} (fraction), None if it holds for every d_lo,
    0.0 if it fails already at d_lo = 0."""
    target = lower_at(t, h, d_hi)
    if upper_at(t, l, 1.0) < target:
        return None
    if upper_at(t, l, 0.0) >= target:
        return 0.0
    bps = _breakpoints(t, (l,))
    prev_d, prev_v = 0.0, upper_at(t, l, 0.0) - target
    for d in bps[1:]:
        v = upper_at(t, l, d) - target
        if v >= 0:
            return prev_d + (-prev_v) * (d - prev_d) / (v - prev_v)
        prev_d, prev_v = d, v
    return None


def group_bounds_asym(t, i: int, d_down: float, d_up: float) -> tuple[float, float]:
    """Standardized (L, U) of group index i with asymmetric downward/upward deviations (fractions)."""
    C, H, U, p, w = t.C[i], t.H[i], t.U[i], t.p_obs[i], t.w
    undefined = np.isnan(p)
    pp = np.where(undefined, 0.0, p)
    lo_rate = np.where(undefined, 0.0, np.maximum(0.0, pp - d_down))
    hi_rate = np.where(undefined, 1.0, np.minimum(1.0, pp + d_up))
    L = float(np.sum(w * (H + U * lo_rate) / C))
    Uu = float(np.sum(w * (H + U * hi_rate) / C))
    return L, Uu


def group_bounds_or(t, i: int, gamma: float, floor: float = 0.0) -> tuple[float, float]:
    """Standardized (L, U) of group index i under odds-ratio pattern-mixture factor Gamma >= 1.

    Unrecorded-status odds in cell s lie within [odds(p_js) / Gamma, Gamma * odds(p_js)].
    When floor == 0.0, cells with p_js == 0 keep 0; when floor > 0, recorded rates are floored continuously at max(p_js, floor).
    """
    C, H, U, p, w = t.C[i], t.H[i], t.U[i], t.p_obs[i], t.w
    undefined = np.isnan(p)
    pp = np.where(undefined, 0.0, p)
    if floor > 0.0:
        pp = np.where(undefined, 0.0, np.maximum(pp, floor))
    denom_lo = pp + gamma * (1.0 - pp)
    denom_hi = gamma * pp + (1.0 - pp)
    lo_rate = np.where(undefined, 0.0, np.where(denom_lo > 0.0, pp / denom_lo, 0.0))
    hi_rate = np.where(undefined, 1.0, np.where(denom_hi > 0.0, (gamma * pp) / denom_hi, 0.0))
    L = float(np.sum(w * (H + U * lo_rate) / C))
    Uu = float(np.sum(w * (H + U * hi_rate) / C))
    return L, Uu


def gamma_star(t, h: int, l: int, max_gamma: float = 1e6, floor: float = 0.0) -> float | None:
    """Smallest odds-ratio sensitivity parameter Gamma >= 1 where L_h^OR(Gamma) <= U_l^OR(Gamma).

    Returns None if L_h^OR(max_gamma) > U_l^OR(max_gamma) (including pairs that hold even under
    worst-case bounds or when cell p_ls == 0 caps OR upward movement when floor == 0.0).
    """
    from scipy.optimize import brentq

    def gap_or(g: float) -> float:
        lh, _ = group_bounds_or(t, h, g, floor=floor)
        _, ul = group_bounds_or(t, l, g, floor=floor)
        return lh - ul

    if gap_or(1.0) <= 0.0:
        return 1.0
    if gap_or(max_gamma) > 0.0:
        return None
    log_g = brentq(lambda lg: gap_or(float(np.exp(lg))), 0.0, float(np.log(max_gamma)), xtol=1e-7)
    return float(np.exp(log_g))


def lo_boundary_asym(t, h: int, l: int, d_hi_down: float):
    """Exact max upward deviation of group l keeping L_h(d_hi_down) > U_l(d_lo_up)."""
    return lo_boundary(t, h, l, d_hi_down)
