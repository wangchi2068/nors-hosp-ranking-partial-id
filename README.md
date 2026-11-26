# Hospitalization Orderings among Six Pathogen Groups in Foodborne Outbreaks with Incompletely Recorded Outcomes, United States, 2009–2023

Complete manuscript source, standalone Supplementary Material (with full 22-item RECORD checklist), raw CDC NORS data snapshot, and deterministic replication pipeline (`analysis_version: 6.0`).

## Quick Reproduction

From the repository root (tested with Python 3.11.2 on Windows and Linux):

```bash
pip install -r requirements.txt
python reproduce.py --render-outputs
```

Or on Linux/macOS from `experiment/`:

```bash
bash code/run_all.sh
```

Both commands verify the raw snapshot SHA-256, rebuild all derived cohorts, run all main, breakdown-frontier, common-mechanism, composition-adjusted, 22-variant sensitivity, 2,000-replicate bootstrap (fixed-scale, true replicate-studentized bootstrap-$t$, and $\delta$-specific $c_{\text{bt}}(\delta)$ grid), 30-hypothesis Gaussian Max-$T$ FWER, 9,999-replicate unrestricted wild cluster bootstrap, $400 \times 399$ non-circular stratified double-bootstrap FWER evaluation, coarser `R2`/equal-outbreak `R10` inference, 3,176 leave-one-outbreak-out checks, rank breakdown points, and supplementary follow-up benchmarks into `experiment/rerun/`, and verify equality against `experiment/analysis/results.json` (`1e-9` for deterministic leaves; `1e-5` for cross-platform simulation/root-finding blocks) and all 14 CSV tables in `experiment/data/analysis_tables/`. To rebuild PDFs (`paper/paper.pdf` and `paper/supplementary.pdf`), run `cd paper && bash compile.sh` (requires XeLaTeX + BibTeX).

## Key findings (v6.0)

1. **Without unrecorded-outcome assumptions:** For reported foodborne outbreak cases (distinct from intrinsic virulence), 4 of 15 sample pairs separate (STEC above *Campylobacter* by `5.5` pp, *C. perfringens* by `5.0` pp, norovirus GI by `6.0` pp, and norovirus GII by `7.5` pp). Worst-case significance is fragile and method-dependent: nominal 30-hypothesis Gaussian tests place STEC above norovirus GII (`p = 0.024` outbreak-HC1; `p = 0.033` 53-cluster exposure-state; `p = 0.031` unrestricted WCB), whereas pre-specified primary **studentized bootstrap-$t$** inference (`c_bt = 4.519`) rejects `0/30` one-sided hypotheses (`p = 0.249`, MC SE `0.0097`; all 95% ranks `[1,6]`).
2. **Non-circular `400 × 399` double-bootstrap size check:** Re-estimating every procedure's critical value inside every outer bootstrap world (preserving `N_eff ~ 2` strata), empirical 30-test FWER at nominal 5% worst case is `33.2% ± 2.4%` (HC0), `30.5% ± 2.3%` (HC1), `24.8% ± 2.2%` (fixed-scale standardized bootstrap), `20.5% ± 2.0%` (53-cluster Gaussian), and `20.5% ± 2.0%` (unrestricted WCB), versus **`6.5% ± 1.2%`** (`6.8% ± 1.3%` at `δ = 5` pp) for studentized bootstrap-$t$.
3. **Externally anchored breakdown frontiers & missingness diagnostics:** Within-cell (`+0.05` to `+9.02` pp) and single-state (`+0.28` to `+8.37` pp) contrasts are positive across all groups, whereas `Std obs − Full` is negative for STEC (`-2.53` pp) and *Campylobacter* (`-0.27` pp) and the Multistate STEC contrast is `-1.28` pp, indicating mixed missingness direction for STEC. Under bounded departures, simultaneous 95% rank sets isolate STEC `[1,1]`, *Salmonella* `[2,2]`, and *Campylobacter* `[3,3]` above `[4,6]` up to a **joint $\delta$-specific bootstrap-$t$ breakdown deviation of `11.6` pp** (`11.6` pp STEC/*Salmonella*, `13.5` pp *Campylobacter*; `7.9` pp under worst-case `c_bt(100) = 4.52`; `18.5` pp under HC1) and OR factors `Γ <= 3.0` (`14/15` pairs), substantial relative to lower-tier base rates (`0.5%–1.6%`; Scallan Walter et al. 2025 benchmarks). Lower-tier orderings fail below `5` pp.
4. **Exposure-state clustering, coarser strata & outbreak influence:** `Multistate` holds `84.8%` of unrecorded STEC cases and `48.3%–66.8%` of cluster score variance share (`G_eff = 4.04 / 2.21`). Splitting Multistate into 219 singletons (`271` clusters), single-state `R11`, and coarser period-only `R2` (`min n = 24`) all yield `0` bootstrap-$t$ worst-case and `12` `δ = 5` rejections; equal outbreak weighting (`R10`) yields `5` worst-case and `12` `δ = 5` bootstrap-$t$ rejections. Across all `3,176` leave-one-outbreak-out deletions (and simultaneous 6-group LLO), worst-case determinations stay `[4, 4]` and `δ = 5` stays `[12, 13]`.
5. **Sensitivity variants (22 specifications):** Across all 20 six-group variants (including O157 STEC `R12`, outbreak-level MAR `R13`, and denominator-contamination checks `R14a` [`25%` extrapolated non-cases] and `R14b` [`<=50%` unrecorded]), `4–12` pairs separate without outcome assumptions and `12–14` at `δ = 5` pp with zero primary reversals.

## File guide

| File | Purpose |
|---|---|
| `reproduce.py` | Cross-platform one-command reproduction and verification script |
| `RESPONSE_TO_REFEREE.md` | Point-by-point responses to Rounds 1–4 referee reports |
| `requirements.txt` | Exact Python dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`) |
| `paper/paper.pdf` | Main manuscript (19 pages, 5 figures, 3 tables) |
| `paper/supplementary.pdf` | Standalone Supplementary Material (11 pages, Sections S1–S4, Tables S1–S12, Figures S1–S4, full RECORD checklist) |
| `experiment/analysis/results.json` | Canonical source of all reported metrics (`analysis_version: 6.0`) |
| `experiment/data/raw/nors_soda_full_20261125T040830Z.csv` | Full CDC NORS snapshot (`66,713` rows; SHA-256 verified) |
| `experiment/data/derived/` | Confirmed, suspected & unspecified-norovirus cohorts, audit tables, excluded summaries, and `data_manifest.json` |
| `experiment/data/analysis_tables/` | All 14 CSV tables (`table_1`–`table_14`) matching MS Tables 1–3 and Supp Tables S1–S11 |
| `experiment/code/` | Modular pipeline (`nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference_voi.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`, `run_all.sh`) |
| `experiment/rebuild_check/` | Verification reports (`rebuild_check.md`, `rebuild_check.json`, `table_check.json`) |

## License

Analysis code is released under the [MIT License](LICENSE). Public CDC NORS surveillance data remain subject to CDC open-data terms.
