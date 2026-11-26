# Hospitalization Orderings among Six Pathogen Groups in Foodborne Outbreaks with Incompletely Recorded Outcomes, United States, 2009–2023

Complete manuscript source, standalone Supplementary Material (with RECORD checklist), raw CDC NORS data snapshot, and deterministic replication pipeline (`analysis_version: 4.0`).

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

Both commands verify the raw snapshot SHA-256, rebuild the audited confirmed and suspected cohorts, run all main, composition-adjusted missingness diagnostics, asymmetric/OR (`0.1%`-floor) sensitivity, 17-variant robustness, 2,000-replicate bootstrap (with 30-hypothesis non-Gaussian studentized Max-$T$), 30-hypothesis outbreak-HC1 and 53-state-clustered M-estimation FWER, 9,999-replicate wild cluster bootstrap, 500-replicate 5-method bootstrap-world calibration, simultaneous rank confidence sets, and 200-trial cost-conditional follow-up masking experiments into `experiment/rerun/`, and verify equality against `experiment/analysis/results.json` (`1e-9` for deterministic leaves; `1e-5` for cross-platform simulation/root-finding blocks) and all 14 CSV tables in `experiment/data/analysis_tables/`. To rebuild PDFs (`paper/paper.pdf` and `paper/supplementary.pdf`), run `cd paper && bash compile.sh` (requires XeLaTeX + BibTeX).

## Key findings (v4.0)

1. **Sample bounds & composition confounding:** Without assumptions on unrecorded hospitalization outcomes, 4 of 15 pairwise orderings are determined in the sample (STEC above *Campylobacter* by `5.5` pp, *C. perfringens* by `5.0` pp, norovirus GI by `6.0` pp, and norovirus GII by `7.5` pp). Crude hospitalization gaps between partially recorded and fully recorded outbreaks (`+10.9` pp STEC; `+11.8` pp *Salmonella*) largely reflect outbreak size and multistate composition (`+0.19` pp within cells for STEC; `Std obs - Full = -3.22` pp), so residual missingness direction is uncertain. Nevertheless, all 12 upper-tier orderings remain separated under illustrative `±5` pp symmetric bounds, asymmetric `-15/+5` to `-20/+5` pp stress tests, and odds-ratio factors `Γ <= 3.0` (`14/15` pairs under both pure OR scaling and a `0.1%` zero-cell continuity floor), while bottom-tier orderings fail at `0.19–4.2` pp.
2. **30-hypothesis FWER, non-Gaussian bootstrap, state clustering & rank sets:** Testing all 30 ordered two-sided hypotheses without unrecorded-outcome assumptions confirms **STEC above norovirus GII** across all five procedures: outbreak-HC1 (`p = 0.0245`, `c = 2.611`), 2,000-replicate bootstrap covariance (`p = 0.0315`, `c = 2.617`), 53-state clustering (`p = 0.0332`, `c = 2.536`), wild cluster bootstrap (`p = 0.0306`), and studentized non-Gaussian bootstrap Max-$T$ (`p = 0.0430`, `c = 2.725`). At `δ = 5` pp and under `-15/+5` pp, all 12 upper-tier orderings are jointly significant (`p <= 0.0020` non-Gaussian bootstrap; `p < 0.001` HC1/state), yielding simultaneous 95% rank confidence sets `[1,1]` for STEC, `[2,2]` for *Salmonella*, `[3,3]` for *Campylobacter*, and `[4,6]` for the lower tier (without outcome assumptions: `[1,5]` STEC, `[2,6]` norovirus GII, `[1,6]` others).
3. **Bootstrap-world calibration:** In 500 stratified resamples preserving sparse `N_eff ~ 2` strata, empirical 30-test Max-$T$ FWER decreases from `32.8%` (HC0) and `29.8%` (HC1) to `25.6%` (non-Gaussian bootstrap critical value), `21.4%` (53-state) and `20.6%` (WCB) at worst case (`20.8%`, `18.0%`, `12.6%`, `8.8%` and `8.0%` at `δ = 5` pp).
4. **Cost-conditional follow-up targeting:** Expected bound reduction per unit cost (`w_s U_i / (C_js * cost_i)`) dominates across all cost metrics: per-case leverage (`w_s/C_js`) resolves `6.29` pairs at 1,000 cases vs. `2.54` random and `2.00` largest-first; under per-outbreak costs, largest-first (`7.90` pairs at 100 outbreaks) outperforms random (`3.27`) and per-case `w_s/C_js` (`7.38`), while per-outbreak width (`w_s U_i/C_js`) leads at `9.12` pairs.
5. **Sensitivity variants (17 specifications):** Across all 17 variants (including suspected etiologies R8, healthcare exclusion R9, equal outbreak weighting R10, and single-state restriction R11), `4–7` pairs hold without unrecorded-outcome assumptions and `12–14` at `δ = 5` pp with zero primary reversals.

## File guide

| File | Purpose |
|---|---|
| `reproduce.py` | Cross-platform one-command reproduction and verification script |
| `RESPONSE_TO_REFEREE.md` | Point-by-point responses to Rounds 1 & 2 referee reports |
| `requirements.txt` | Exact Python dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`) |
| `paper/paper.pdf` | Main manuscript (17 pages: 5 figures, 3 tables, 41 references) |
| `paper/supplementary.pdf` | Standalone Supplementary Material (11 pages: Sections S1–S4, Tables S1–S12, Figures S1–S3, RECORD checklist) |
| `experiment/analysis/results.json` | Canonical source of all reported metrics (`analysis_version: 4.0`) |
| `experiment/data/raw/nors_soda_full_20261125T040830Z.csv` | Full CDC NORS snapshot (`66,713` rows; SHA-256 verified) |
| `experiment/data/derived/` | Confirmed & suspected cohorts, audit tables, excluded summaries, and `data_manifest.json` |
| `experiment/data/analysis_tables/` | CSV tables (`table_1`–`table_14`) corresponding to MS Tables 1–3 and Supp Tables S1–S11 |
| `experiment/code/` | Modular pipeline (`nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference_voi.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`, `run_all.sh`) |
| `experiment/rebuild_check/` | Verification reports (`rebuild_check.md`, `rebuild_check.json`, `table_check.json`) |

## License

Analysis code is released under the [MIT License](LICENSE). Public CDC NORS surveillance data remain subject to CDC open-data terms.
