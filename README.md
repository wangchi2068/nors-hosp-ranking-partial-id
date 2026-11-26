# Hospitalization Orderings among Six Pathogen Groups in Foodborne Outbreaks with Incompletely Recorded Outcomes, United States, 2009–2023

Complete manuscript source, standalone Supplementary Material (with full 22-item RECORD checklist), raw CDC NORS data snapshot, and deterministic replication pipeline (`analysis_version: 5.0`).

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

Both commands verify the raw snapshot SHA-256, rebuild all derived cohorts, run all main, breakdown-frontier, common-mechanism, composition-adjusted, 20-variant sensitivity, 2,000-replicate bootstrap (fixed-scale and true replicate-studentized bootstrap-$t$), 30-hypothesis Gaussian Max-$T$ FWER, 9,999-replicate wild cluster bootstrap, 500-replicate 6-method bootstrap-world calibration, rank breakdown points, and 200-trial cost-conditional follow-up benchmarks into `experiment/rerun/`, and verify equality against `experiment/analysis/results.json` (`1e-9` for deterministic leaves; `1e-5` for cross-platform simulation/root-finding blocks) and all 14 CSV tables in `experiment/data/analysis_tables/`. To rebuild PDFs (`paper/paper.pdf` and `paper/supplementary.pdf`), run `cd paper && bash compile.sh` (requires XeLaTeX + BibTeX).

## Key findings (v5.0)

1. **Without unrecorded-outcome assumptions:** 4 of 15 sample pairs separate (STEC above *Campylobacter* by `5.5` pp, *C. perfringens* by `5.0` pp, norovirus GI by `6.0` pp, and norovirus GII by `7.5` pp). Nominal 30-hypothesis one-sided FWER inference places STEC above norovirus GII (`p = 0.024` outbreak-HC1; `p = 0.033` 53-cluster exposure-state; nominal 95% ranks: STEC `[1,5]`, norovirus GII `[2,6]`, others `[1,6]`), whereas **calibrated studentized bootstrap-$t$** (`c_bt = 4.519`) yields no worst-case rejection (`p = 0.249`, MC SE `0.0097`; calibrated worst-case ranks `[1,6]` for all groups).
2. **Aligned 6-method bootstrap-world calibration:** In 500 stratified resamples preserving `N_eff ~ 2` strata, nominal Gaussian Max-$T$ procedures over-reject (`29.8%` HC1, `21.4%` 53-cluster, `20.6%` WCB at nominal 5% worst case), whereas matched fixed-scale standardized bootstrap (`5.0%` worst case, `4.4%` at `δ = 5`) and studentized bootstrap-$t$ (`3.8%` worst case, `5.8%` at `δ = 5`) preserve nominal size.
3. **Breakdown frontiers & missingness diagnostics:** Within-cell and single-state hospitalization rates are uniformly higher in partially recorded than fully recorded outbreaks across all six groups (`+0.19` to `+9.02` pp within cells), compatible with outcome-dependent recording or outbreak-size/multistate heterogeneity. Across bounded departures, all 12 upper-tier orderings hold jointly and lock simultaneous 95% ranks for STEC `[1,1]`, *Salmonella* `[2,2]`, and *Campylobacter* `[3,3]` above `[4,6]` up to **rank breakdown points** of `15.0–15.9` pp (calibrated bootstrap-$t$) and `18.5–19.1` pp (HC1), under common-mechanism proportional scaling (`α >= 0.50`), and under odds-ratio factors `Γ <= 3.0` (`14/15` pairs), while lower-tier orderings break below `4.3` pp.
4. **Exposure-state & Multistate clustering:** NORS `state` records state of exposure, with the CDC-coordinated `Multistate` category holding `84.8%` of unrecorded STEC cases and `48.3%–66.8%` of cluster score leverage (MNW `G_eff = 4.04` worst case, `2.21` at `δ = 5`). Splitting Multistate into 219 singletons (`271` clusters) or clustering strictly within 52 single states (`R11`) yields `0` worst-case and `12` `δ = 5` significant pairs.
5. **Sensitivity variants (20 specifications):** Across all 18 six-group specifications (including O157 STEC `R12`, outbreak-level MAR `R13`, single-state `R11`, healthcare exclusion `R9`, equal outbreak weighting `R10`, and suspected etiologies `R8`), `4–7` pairs separate without outcome assumptions and `12–14` at `δ = 5` pp with zero primary reversals; combined norovirus variants `R7` and `R7b` (adding 304 unspecified confirmed outbreaks) both yield `3` worst-case and `9` `δ = 5` orderings (of 10 pairs).
6. **Exploratory follow-up rules:** Performance depends on cost structure—stratum leverage `w_s/C_js` leads under per-case budgets (`6.29` pairs at 1,000 cases), per-outbreak width `w_s U_i/C_js` leads under per-outbreak budgets (`9.12` at 100 outbreaks), and per-case `w_s/C_js` exceeds mixed targeting at 5,000 mixed units (`9.72` vs. `9.22`).

## File guide

| File | Purpose |
|---|---|
| `reproduce.py` | Cross-platform one-command reproduction and verification script |
| `RESPONSE_TO_REFEREE.md` | Point-by-point responses to Rounds 1, 2 & 3 referee reports |
| `requirements.txt` | Exact Python dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`) |
| `paper/paper.pdf` | Main manuscript (18 pages, 5 figures, 3 tables) |
| `paper/supplementary.pdf` | Standalone Supplementary Material (11 pages, Sections S1–S4, Tables S1–S12, Figures S1–S3, full RECORD checklist) |
| `experiment/analysis/results.json` | Canonical source of all reported metrics (`analysis_version: 5.0`) |
| `experiment/data/raw/nors_soda_full_20261125T040830Z.csv` | Full CDC NORS snapshot (`66,713` rows; SHA-256 verified) |
| `experiment/data/derived/` | Confirmed, suspected & unspecified-norovirus cohorts, audit tables, excluded summaries, and `data_manifest.json` |
| `experiment/data/analysis_tables/` | All 14 CSV tables (`table_1`–`table_14`) matching MS Tables 1–3 and Supp Tables S1–S11 |
| `experiment/code/` | Modular pipeline (`nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference_voi.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`, `run_all.sh`) |
| `experiment/rebuild_check/` | Verification reports (`rebuild_check.md`, `rebuild_check.json`, `table_check.json`) |

## License

Analysis code is released under the [MIT License](LICENSE). Public CDC NORS surveillance data remain subject to CDC open-data terms.
