# Hospitalization Orderings among Reported Outbreak Cases across Six Foodborne Pathogen Groups with Incompletely Recorded Outcomes, United States, 2009–2023

Complete manuscript source, standalone Supplementary Material (with full 22-item RECORD checklist), raw CDC NORS data snapshot, and deterministic replication pipeline (`analysis_version: 7.0`).

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

Both commands verify the raw snapshot SHA-256, rebuild all derived cohorts, run all main, breakdown-frontier, common-mechanism, composition-adjusted, 23-variant sensitivity (including non-typhoidal *Salmonella* `R12b`), 2,000-replicate bootstrap (fixed-scale, true replicate-studentized bootstrap-$t$, and $\delta$-specific $c_{\text{bt}}(\delta)$ grid), 30-hypothesis HC1, 53-state CV1, leave-one-cluster-out CV3 jackknife, Bayesian 53-cluster bootstrap-$t$, and 9,999-replicate unrestricted wild cluster bootstrap FWER tests, OR-scale 30-test inference ($\Gamma^*_{0.95}$), both $400 \times 499$ outbreak-stratified and $300 \times 399$ 53-cluster Bayesian double-bootstrap FWER evaluations, coarser `R2`/equal-outbreak `R10`/full single-state `R11` inference, 3,176 leave-one-outbreak-out checks, rank breakdown points, and supplementary follow-up benchmarks into `experiment/rerun/`, and verify equality against `experiment/analysis/results.json` (`1e-9` for deterministic leaves; `1e-5` for cross-platform simulation/root-finding blocks) and all 14 CSV tables in `experiment/data/analysis_tables/`. To rebuild PDFs (`paper/main.pdf` and `paper/supplementary.pdf`), run `cd paper && bash compile.sh` (requires XeLaTeX + BibTeX).

## Key findings (v7.0)

1. **Without unrecorded-outcome assumptions:** For reported foodborne outbreak cases (distinct from intrinsic virulence), 4 of 15 sample pairs separate (STEC above *Campylobacter* by `5.5` pp, *C. perfringens* by `5.0` pp, norovirus GI by `6.0` pp, and norovirus GII by `7.5` pp). Worst-case significance is fragile and method-dependent: nominal 30-hypothesis Gaussian HC1 (`p = 0.024`), 53-state CV1 (`p = 0.033`), and unrestricted WCB (`p = 0.031`) flag STEC above norovirus GII, whereas **CV3 cluster jackknife** (`p = 0.241`), **Bayesian 53-cluster bootstrap-$t$** (`p = 0.176`), and **outbreak-stratified studentized bootstrap-$t$** (`p = 0.249`, `c_bt = 4.519`) reject `0/30` one-sided hypotheses (all 95% ranks `[1,6]`).
2. **Double-bootstrap size checks (`400 × 499` outbreak & `300 × 399` 53-cluster):** Under the least-favourable boundary null, empirical 30-test FWER in the `400 × 499` outbreak world is `33.2% ± 2.4%` (HC0), `30.5% ± 2.3%` (HC1), `25.2% ± 2.2%` (fixed-scale), `20.5% ± 2.0%` (53-state CV1), and `21.0% ± 2.0%` (WCB), versus **`7.2% ± 1.3%`** (`7.8% ± 1.3%` at `δ = 5` pp) for studentized bootstrap-$t$. Under `Exp(1)` state-cluster weights (`300 × 399`), CV1 (`17.0%`/`16.0%`) and WCB (`19.3%`/`17.7%`) over-reject due to Multistate leverage, Bayesian cluster bootstrap-$t$ drops to `9.3%`/`10.0%`, and **CV3 cluster jackknife controls FWER at `7.3% ± 1.5%` (`4.7% ± 1.2%` at `δ = 5` pp)**.
3. **Externally anchored upper-vs.-lower vs. precision-limited within-upper-tier frontiers:** Within-cell (`+0.05` to `+9.02` pp), single-state (`+0.28` to `+8.37` pp), and common-cell standardized observed-versus-full contrasts (`+0.07` to `+2.86` pp; STEC `+0.88` pp) are positive across all six groups, while STEC shows a negative multistate contrast (`-1.28` pp). Under bounded additive departures, simultaneous 95% ranks fix STEC `[1,1]`, *Salmonella* `[2,2]`, and *Campylobacter* `[3,3]` above `[4,6]`. Upper-vs.-lower separation (*Campylobacter* > *C. perfringens*) holds up to **`13.5` pp** under $\delta$-specific bootstrap-$t$ (`12.3` pp CV3; `34.5–79.2` pp for *Salmonella*/STEC vs. lower tier)---substantial relative to external `0.03%–<1%` lower-tier benchmarks (Scallan 2011; Scallan Walter 2025)---whereas within-upper-tier `STEC > Salmonella` breaks down at **`11.5` pp** strictly from sampling precision (implied unrecorded rates `20.4%` vs. `29.6%`; `4.4` pp single-step CV3 on *Salmonella* > *Campylobacter*, stepdown `p = 0.040` at `δ = 5`). Lower-tier orderings fail below `5` pp on the additive scale, while OR scaling separates norovirus GII above GI (`Γ*_0.95 = 1.94` boot-$t$, `13/30` at `Γ = 2.0`).
4. **Exposure-state clustering, coarser strata & outbreak influence:** `Multistate` holds `84.8%` of unrecorded STEC cases and `48.3%–66.8%` of cluster score variance share (`G_eff = 4.04 / 2.21`). Splitting Multistate into 219 singletons (`271` clusters), single-state `R11` (`52/52` valid CV3, `0/12` sig worst case across all 5 methods; `7.4–14.8` pp within-upper vs `13.1–19.5` pp upper-vs-lower), and coarser period-only `R2` (`min n = 24`) all yield `0` bootstrap-$t$ worst-case and `12` `δ = 5` rejections; normalized equal outbreak weighting (`R10`, `H_rec/K_rec` fixed) yields `5` worst-case and `12` `δ = 5` bootstrap-$t$ rejections. Across all `3,176` LOO deletions (and 6-group LLO), worst-case determinations stay `[4, 4]` and `δ = 5` stays `[12, 13]`.
5. **Sensitivity variants (23 specifications):** Across all 21 six-group variants (including confirmed O157 STEC `R12` [279/388], excluding 6 *Salmonella* Typhi `R12b` [87 cases], outbreak-level MAR `R13`, and `R14a` [`25%` extrapolated non-cases] / selection-sensitivity `R14b` [`<=50%` unrecorded]), `4–12` pairs separate without outcome assumptions and `12–14` at `δ = 5` pp with zero primary reversals.

## File guide

| File | Purpose |
|---|---|
| `reproduce.py` | Cross-platform one-command reproduction and verification script |
| `RESPONSE_TO_REFEREE.md` | Point-by-point responses to Rounds 1–5 referee reports |
| `requirements.txt` | Exact Python dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`) |
| `paper/paper.pdf` | Main manuscript (20 pages, 5 figures, 3 tables) |
| `paper/supplementary.pdf` | Standalone Supplementary Material (12 pages, Sections S1–S4, Tables S1–S12, Figures S1–S4, full RECORD checklist) |
| `experiment/analysis/results.json` | Canonical source of all reported metrics (`analysis_version: 7.0`) |
| `experiment/data/raw/nors_soda_full_20261125T040830Z.csv` | Full CDC NORS snapshot (`66,713` rows; SHA-256 verified) |
| `experiment/data/derived/` | Confirmed, suspected & unspecified-norovirus cohorts, audit tables, excluded summaries, and `data_manifest.json` |
| `experiment/data/analysis_tables/` | All 14 CSV tables (`table_1`–`table_14`) matching MS Tables 1–3 and Supp Tables S1–S11 |
| `experiment/code/` | Modular pipeline (`nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference_voi.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`, `run_all.sh`) |
| `experiment/rebuild_check/` | Verification reports (`rebuild_check.md`, `rebuild_check.json`, `table_check.json`) |

## License

Analysis code is released under the [MIT License](LICENSE). Public CDC NORS surveillance data remain subject to CDC open-data terms.
