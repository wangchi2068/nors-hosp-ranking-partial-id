# Partial Identification, Completeness Thresholds, and Simultaneous Rank Inference for Pathogen Hospitalization Comparisons among Reported Foodborne Outbreak Cases, United States, 2009–2023

Clean-slate `v11.0` repository containing complete manuscript source (`paper/paper.pdf`, `paper/supplementary.pdf`), raw CDC NORS data snapshot and portal metadata (`:id` retained for revision checks), deterministic data-preparation pipeline, partial-identification & completeness/power prescription suite, conventional/empirical-Bayes benchmarks, and cross-platform reproduction script (`analysis_version: 11.0`).

## Quick Reproduction

From the repository root (Python 3.11+):

```bash
pip install -r requirements.txt
python reproduce.py --render-outputs
```

Full end-to-end execution takes approximately **14–18 minutes** on CPU (`--workers 2` default; pass `--workers 4` to speed up the initial 2,000-replicate stratified bootstrap, or `--B 200` for a quick smoke check). The script verifies the raw CDC snapshot SHA-256, rebuilds derived cohorts into `rerun/`, executes `code/analyze.py` (`B = 2,000`), verifies exact equality on all deterministic leaves (`1e-9`), identical 30-hypothesis decision sets across worst-case, $\delta=5\text{ pp}$, and $\delta=0$ specifications, all 19 rendered CSV tables (`all_non_bootstrap_cells_identical = True`), manuscript text numeric consistency (`code/verify_text_numbers.py`), and Monte Carlo simulation tolerances (`<= 1e-3`, with a cross-platform BLAS warning threshold at `0.025`). All JSON and CSV outputs use Unix `LF` (`\n`) line endings.

## Key Empirical & Methodological Findings (`v11.0`)

1. **Step-by-step non-identification along the canonical chain:** Conventional Wilson `H/K`, Laird–Louis (1989) empirical-Bayes Beta-Binomial baselines, and $\delta = 0$ dual testing (`14/30`) all certify `[1,1]`, `[2,2]`, `[3,3]`, `[4,4]`, `[5,6]`, `[5,6]`. Without missingness assumptions (12.7%--22.2% unrecorded), bounds span 12--19 pp and **0 of 5 adjacent pairs** along the chain separate (overlaps `4.8–18.4 pp`); all 4/15 separated pairs skip tiers (`STEC > CAMP/CPERF/GI/GII`), where case-weighted **dual-criterion** (`boot-t ∩ CV3`) testing rejects `0/30` vs. `5–6/30` under equal outbreak weighting (**R10**).
2. **Surveillance completeness & 80%-power prescriptions (`Table S15`, `Fig. 3b`):** Sample expansion shrinks SEs but cannot close bound overlap; restoring assumption-free separation along upper/cross-tier adjacent steps requires uniform $M/C \le 9.8\%$ (`STEC > SAL`), $\le 8.4\%$ (`SAL > CAMP`), and $\le 6.5\%$ (`CAMP > NoV GII`, completeness $\ge 93.5\%$). At current missingness, reaching 80% robust power on `STEC > NoV GII` (currently **9.2%** empirical power, vs. **75.2%** HC1) requires **`3.4–3.5×`** independent units (`~10,860–11,143` retained outbreaks; `6.9–30.8×` on the other three cross-tier pairs).
3. **Principal scope ceiling & conditional sensitivity:** Because public NORS extracts omit an indicator separating estimated illness counts from uninterviewed cases (treating 25%--50% of $M$ as non-cases shifts worst-case separation from `4/15` to `6–9/15`), conclusions condition on $C$ as a case count. Conditioned on $\delta = 5\text{ pp}$ or plausibility regimes (`P_ext`, `P1/P2`), upper-vs.-lower separation holds to **12.3–13.5 pp** while `SAL > CAMP` breaks at **4.5 pp** single-step (`6.6 pp` stepdown). Global-null and partial-null stepdown FWER across five worlds appear in Tables S10–S11.

## Directory Guide

- `code/`: `nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`, `verify_text_numbers.py`
- `data/raw/` & `data/derived/`: Pinned CDC NORS snapshot, portal view metadata, audited cohorts, `etiology_mapping.csv`, `data_manifest.json`
- `analysis/`: Canonical `results.json` (`analysis_version: 11.0`), `statistical_summary.json`, and 19 CSV tables (`analysis/tables/`)
- `paper/`: Standalone `main.tex` (`paper.pdf`) & `supplementary.tex` (`supplementary.pdf`), sections, tables, figures, and `bibliography.bib`
- `reproduce.py`: One-command end-to-end verification runner
