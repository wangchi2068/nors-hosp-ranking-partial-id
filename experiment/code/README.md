# Experiment Pipeline (`experiment/code/`)

Deterministic and seeded computational pipeline (`analysis_version: 5.0`) for the NORS partial-identification hospitalization ranking study.

## Scripts

- `nors_rules.py`: Locked CDC NORS group mapping, healthcare setting identification, setting/period harmonization, and arithmetic record-audit rules.
- `prepare_data.py`: Verifies the raw CSV SHA-256 against `experiment/data/raw/nors_soda_full_20261125T040830Z.json` and builds `cohort_outbreaks.csv`, `suspected_cohort_outbreaks.csv`, `excluded_outbreaks_summary.json`, `record_audit.csv`, mapping tables, `strata_counts.csv`, and `data_manifest.json`.
- `strata.py`: Group-by-cell aggregation, common-support restriction, and reference weights.
- `bounds.py`: Sample worst-case bounds, symmetric/asymmetric additive bands, multiplicative odds-ratio bounds (`Gamma` and critical `Gamma*`), piecewise-linear `delta*`, and 2D boundaries.
- `bootstrap.py`: Seeded stratified outbreak-level bootstrap (`B = 2000`, `seed = 20261125`) with true SDs and full `15 x 15` pairwise covariance.
- `inference_voi.py`:
  - Closed-form outbreak-level influence functions for arbitrary `delta in [0, 1]` (verified against central-difference Jacobians).
  - Outbreak-HC1 and 53-state-clustered sandwich covariances + 9,999-replicate studentized Rademacher wild cluster bootstrap.
  - Imbens–Manski 95% CIs, 200,000-draw Gaussian multiplier Max-$T$ FWER stepdown with MC SEs, dynamic `c_sim(delta)` tipping points, and simultaneous 95% rank confidence sets.
  - 500-replicate stratified small-sample calibration simulation, within-data empirical `delta` calibration, and budget-filling follow-up benchmarks (baseline MCAR, MNAR outcome+size, state-clustered, 70% partial yield, adaptive tranche re-scoring).
- `analyze.py`: End-to-end orchestrator writing `experiment/analysis/results.json`.
- `make_outputs.py`: Renders `experiment/analysis/statistical_summary.json`, all 14 CSV/LaTeX tables (`table_1`–`table_14`), and all generated figures.
- `compare_results.py`: Leaf-by-leaf verifier (`1e-9` deterministic tolerance; `1e-5` cross-platform tolerance for Monte Carlo / root-finding blocks) + bootstrap checks.
- `check_tables.py`: Normalized line-ending byte-identity check across all 14 CSV tables.
- `run_all.sh`: Shell wrapper running the full pipeline into `experiment/rerun/`.
