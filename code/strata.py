"""Strata, common support and reference weights (rebuilt 2026-11-25).

A Table holds, for a list of groups and a list of cells, the per-group x cell sums
C (cases), H (hospitalized), U (cases with unrecorded status), H_rec / K_rec (from records
with both H and K recorded), the number of outbreaks, and one weight vector w shared by all
groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nors_rules import PERIODS, SETTING_GROUPS

ALL_CELLS = [f"{s} | {p}" for s in SETTING_GROUPS for p in PERIODS]


@dataclass
class Table:
    groups: list
    cells: list
    C: np.ndarray
    H: np.ndarray
    U: np.ndarray
    H_rec: np.ndarray
    K_rec: np.ndarray
    n: np.ndarray
    w: np.ndarray = field(default=None)

    @property
    def p_obs(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.K_rec > 0, self.H_rec / np.where(self.K_rec > 0, self.K_rec, 1.0), np.nan)


def cell_column(kind: str) -> str:
    return {"setting_period": "cell", "period": "period", "all": "cell_all"}[kind]


def cell_order(kind: str) -> list:
    if kind == "setting_period":
        return list(ALL_CELLS)
    if kind == "period":
        return list(PERIODS)
    return ["all"]


def aggregate(cohort: pd.DataFrame, groups: list, kind: str = "setting_period",
              group_col: str = "group") -> Table:
    """Sum the audited counts by group x cell (all cells of the given kind, in fixed order)."""
    df = cohort.copy()
    if kind == "all":
        df["cell_all"] = "all"
    col = cell_column(kind)
    cells = cell_order(kind)
    shape = (len(groups), len(cells))
    arr = {k: np.zeros(shape) for k in ("C", "H", "U", "H_rec", "K_rec", "n")}
    gi = {g: i for i, g in enumerate(groups)}
    ci = {c: j for j, c in enumerate(cells)}
    sums = df.groupby([group_col, col], sort=False)[["C", "H", "U", "H_rec", "K_rec"]].sum()
    counts = df.groupby([group_col, col], sort=False).size()
    for (g, c), row in sums.iterrows():
        if g in gi and c in ci:
            i, j = gi[g], ci[c]
            for k in ("C", "H", "U", "H_rec", "K_rec"):
                arr[k][i, j] = row[k]
            arr["n"][i, j] = counts[(g, c)]
    return Table(groups=list(groups), cells=cells, **arr)


def common_support(t: Table, threshold: int = 3) -> list:
    """Cells in which every group has >= threshold outbreaks (cell order preserved)."""
    keep = np.all(t.n >= threshold, axis=0)
    return [c for c, k in zip(t.cells, keep) if k]


def restrict(t: Table, cells: list) -> Table:
    idx = [t.cells.index(c) for c in cells]
    return Table(groups=t.groups, cells=list(cells), C=t.C[:, idx], H=t.H[:, idx], U=t.U[:, idx],
                 H_rec=t.H_rec[:, idx], K_rec=t.K_rec[:, idx], n=t.n[:, idx],
                 w=None if t.w is None else t.w[idx])


def weights(t: Table, scheme: str = "pooled_cases") -> np.ndarray:
    """One reference weight vector over the table's cells, shared by all groups."""
    if scheme == "pooled_cases":
        tot = t.C.sum(axis=0)
        return tot / tot.sum()
    if scheme == "pooled_outbreaks":
        tot = t.n.sum(axis=0)
        return tot / tot.sum()
    if scheme == "group_balanced":
        shares = t.C / t.C.sum(axis=1, keepdims=True)
        return shares.mean(axis=0)
    if scheme == "uniform":
        return np.full(len(t.cells), 1.0 / len(t.cells))
    raise ValueError(scheme)


def build(cohort: pd.DataFrame, groups: list, kind: str = "setting_period", threshold: int = 3,
          weight_scheme: str = "pooled_cases", drop_settings=(), group_col: str = "group",
          fixed_cells=None) -> Table:
    """Aggregate, restrict to the common support (or fixed cells) and attach weights."""
    t = aggregate(cohort, groups, kind, group_col=group_col)
    if fixed_cells is not None:
        cells = list(fixed_cells)
    else:
        cells = common_support(t, threshold)
    if drop_settings:
        cells = [c for c in cells if c.split(" | ")[0] not in drop_settings]
    t = restrict(t, cells)
    t.w = weights(t, weight_scheme)
    return t
