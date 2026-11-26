# Experiment Pipeline (`experiment/code/`)

Deterministic and seeded computational pipeline (`analysis_version: 7.0`) for the NORS partial-identification hospitalization ranking study.

## Scripts

- `nors_rules.py`: Locked CDC NORS group mapping, healthcare setting identification, setting/period harmonization, and arithmetic record-audit rules.
- `prepare_data.py`: Verifies the raw CSV SHA-256 against `experiment/data/raw/nors_soda_full_20261125T040830Z.json` and builds `cohort_outbreaks.csv`, `suspected_cohort_outbreaks.csv`, `excluded_outbreaks_summary.json`, `record_audit.csv`, mapping tables, `strata_counts.csv`, and `data_manifest.json`.
- `strata.py`: Group-by-cell aggregation, common-support restriction, and reference weights.
- `bounds.py`: Sample worst-case bounds, symmetric/asymmetric additive bands, multiplicative odds-ratio bounds (`Gamma` and critical `Gamma*`), piecewise-linear `delta*`, and 2D boundaries.
- `bootstrap.py`: Seeded stratified outbreak-level bootstrap (`B = 2000`, `seed = 20261125`) with true SDs and full `15 x 15` pairwise covariance.
- `inference_voi.py`:
  - Closed-form influence functions for arbitrary `delta in [0, 1]`, group-specific asymmetric shifts, and odds-ratio (`Gamma >= 1`, `floor=0.001`) scaling (verified against central-difference Jacobians).
  - Outbreak-HC1, 53-state CV1, leave-one-cluster-out **CV3 jackknife**, 53-cluster Bayesian bootstrap-$t$ (`Exp(1)` weights), and 9,999-replicate unrestricted studentized Rademacher wild cluster bootstrap + cluster score variance shares (`G_eff`).
  - Pointwise Imbens–Manski 95% CIs, 200,000-draw Gaussian multiplier Max-$T$ stepdown, $\delta$-specific studentized bootstrap-$t$ (`c_bt(delta)` grid with uninterpolated root refinement), OR 30-test inference ($\Gamma^*_{0.95}$), coarser `R2`/normalized equal-outbreak `R10`/full single-state `R11` inference, `LOO`/`LLO` influence checks, and simultaneous 95% rank confidence sets.
  - Non-circular `400 x 499` stratified outbreak double bootstrap (6 methods) AND `300 x 399` Bayesian cluster double bootstrap (4 cluster methods), common-cell empirical `delta` calibration, and supplementary budget-filling follow-up benchmarks.
- `analyze.py`: End-to-end orchestrator writing `experiment/analysis/results.json`.
- `make_outputs.py`: Renders `experiment/analysis/statistical_summary.json`, all 14 CSV/LaTeX tables (`table_1`–`table_14`), and all generated figures.
- `compare_results.py`: Leaf-by-leaf verifier (`1e-9` deterministic tolerance; `1e-5` cross-platform tolerance for Monte Carlo / root-finding blocks) + bootstrap checks.
- `check_tables.py`: Normalized line-ending byte-identity check across all 14 CSV tables.
- `run_all.sh`: Shell wrapper running the full pipeline into `experiment/rerun/`.
