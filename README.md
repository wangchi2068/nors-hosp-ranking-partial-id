# Hospitalization Orderings among Six Pathogen Groups in Foodborne Outbreaks with Incompletely Recorded Outcomes, United States, 2009–2023

Complete manuscript source, standalone Supplementary Material (with RECORD checklist), raw CDC NORS data snapshot, and deterministic replication pipeline (`analysis_version: 3.0`).

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

Both commands verify the raw snapshot SHA-256, rebuild the audited confirmed and suspected cohorts, run all main, empirical-calibration, asymmetric/OR-sensitivity, 16-variant robustness, 2,000-replicate bootstrap, outbreak-HC1 and 53-state-clustered M-estimation FWER, 9,999-replicate wild cluster bootstrap, 500-replicate small-sample calibration, rank confidence set, and 200-trial budget-filling follow-up masking experiments into `experiment/rerun/`, and verify equality against `experiment/analysis/results.json` (`1e-9` for deterministic leaves; `1e-5` for cross-platform simulation/root-finding sections) and all 14 CSV tables in `experiment/data/analysis_tables/`. To rebuild PDFs (`paper/paper.pdf` and `paper/supplementary.pdf`), run `cd paper && bash compile.sh` (requires XeLaTeX + BibTeX).

## Key findings (v3.0)

1. **Sample partial identification & empirical missingness calibration:** Without assumptions on unrecorded hospitalization outcomes, 4 of 15 pairwise orderings are determined in the sample (STEC above *Campylobacter* by `5.5` pp, *C. perfringens* by `5.0` pp, norovirus GI by `6.0` pp, and norovirus GII by `7.5` pp). Within the common support, known-status cases in partially recorded outbreaks had substantially higher hospitalization than fully recorded outbreaks (*Salmonella*: `25.9%` vs. `14.1%`; STEC: `36.2%` vs. `25.3%`), indicating likely downward departures; nonetheless, all 12 upper-tier orderings hold under illustrative `±5` pp symmetric bounds, calibrated asymmetric bands up to `-20/+5` pp, and odds-ratio factors `Γ <= 3.0` (`14/15` pairs), while the three bottom pairs fail at `0.19`, `3.7`, and `4.2` pp.
2. **Joint M-estimation, state clustering, wild cluster bootstrap & rank sets:** Across 200,000 multiplier draws, STEC above norovirus GII remains significant across all 15 tests without unrecorded-outcome assumptions under outbreak-HC1 (`p_FWER = 0.0179`, `c = 2.487`), 2,000-replicate bootstrap covariance (`p_FWER = 0.0225`, `c = 2.494`), 53-state clustering (`p_FWER = 0.0275`, `c = 2.463`), and 9,999-replicate studentized Rademacher wild cluster bootstrap (`p_WCB = 0.0278`). At `δ = 5` pp, all 12 upper-tier pairs are jointly significant (`p_FWER < 0.001`), yielding simultaneous 95% rank confidence sets `[1,1]` for STEC, `[2,2]` for *Salmonella*, `[3,3]` for *Campylobacter*, and `[4,6]` for the lower tier (without assumptions: `[1,5]` STEC, `[2,6]` norovirus GII, `[1,6]` others). Dynamic simultaneous 95% confidence tipping points are `19.2` pp to `∞` (upper tier) and `<= 1.5` pp (bottom tier). Stratified 500-replicate MC calibration documents finite-sample HC1 under-coverage (`76.4%–87.4%` empirical IM boundary coverage in `N_eff ~ 2` strata), motivating the wild cluster and bootstrap checks.
3. **Follow-up leverage (`w_s/C_js`) vs. DPI & largest-outbreak targeting:** Under a budget-filling selection rule across 200 masking trials (`2,414` fully recorded outbreaks, `45,249` cases), targeting largest outbreaks performs worse than random at 1,000 cases (`2.00` vs. `2.54` pairs; mean leverage `0.0088` pp/case), whereas unweighted stratum-leverage targeting (`w_s/C_js`) achieves `6.29` pairs (`+3.75` over random, 95% CI `[+3.58, +3.92]`; leverage `0.0573` pp/case). Pair-urgency weighting (`DPI`) adds minimal single-shot gain (`6.36` pairs; paired diff `+0.07`, 95% CI `[-0.07, +0.21]`) but reaches `7.31` pairs when updated adaptively in 250-case tranches; superiority of `w_s/C_js` and DPI persists across MNAR outcome/size-dependent masking, state-clustered missingness, and 70% partial recovery.
4. **Sensitivity & exclusions:** Across all 16 specifications (including suspected etiologies R8, healthcare exclusion R9, and within-cell outbreak weighting R10), `4–7` pairs hold without unrecorded-outcome assumptions and `12–14` at `δ = 5` pp with zero primary reversals.

## File guide

| File | Purpose |
|---|---|
| `reproduce.py` | Cross-platform one-command reproduction and verification script |
| `RESPONSE_TO_REFEREE.md` | Point-by-point summary of v3.0 revisions addressing all 8 Major and 20 Minor comments |
| `requirements.txt` | Exact Python dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`) |
| `paper/paper.pdf` | Main manuscript (18 pages: 5 figures, 3 tables, 38 references) |
| `paper/supplementary.pdf` | Standalone Supplementary Material (10 pages: Sections S1–S4, Tables S1–S11, Figures S1–S3, RECORD checklist) |
| `experiment/analysis/results.json` | Canonical source of all reported metrics (`analysis_version: 3.0`) |
| `experiment/data/raw/nors_soda_full_20261125T040830Z.csv` | Full CDC NORS snapshot (`66,713` rows; SHA-256 verified) |
| `experiment/data/derived/` | Confirmed & suspected cohorts, audit tables, excluded summaries, and `data_manifest.json` |
| `experiment/data/analysis_tables/` | CSV tables (`table_1`–`table_14`) |
| `experiment/code/` | Modular pipeline (`nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference_voi.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`, `run_all.sh`) |
| `experiment/rebuild_check/` | Verification reports (`rebuild_check.md`, `rebuild_check.json`, `table_check.json`) |

## License

Analysis code is released under the [MIT License](LICENSE). Public CDC NORS surveillance data remain subject to CDC open-data terms.
