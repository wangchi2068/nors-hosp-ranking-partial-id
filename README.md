# Partial Identification and Simultaneous Rank Inference for Pathogen Hospitalization Comparisons among Reported Foodborne Outbreak Cases, United States, 2009–2023

Clean-slate `v8.0` repository containing complete manuscript source (`paper/paper.pdf`, `paper/supplementary.pdf`), raw CDC NORS data snapshot, deterministic data-preparation pipeline, partial-identification & inference suite, and cross-platform reproduction script (`analysis_version: 8.0`).

## Quick Reproduction

From the repository root (Python 3.11+):

```bash
pip install -r requirements.txt
python reproduce.py --render-outputs
```

The script verifies the raw CDC snapshot SHA-256, rebuilds derived cohorts into `rerun/`, executes `code/analyze.py` (`B = 2,000`), verifies exact equality on all deterministic leaves (`1e-9`) and all 14 rendered CSV tables (`all_non_bootstrap_cells_identical = True`), and checks Monte Carlo simulation leaves cleanly (`<= 1e-3`, with a cross-platform BLAS warning threshold at `0.025`). All JSON and CSV outputs use Unix `LF` (`\n`) line endings.

## Key Empirical & Methodological Findings (`v8.0`)

1. **Without assumptions on unrecorded outcomes:** Across 4,133 confirmed foodborne outbreaks (3,176 retained in 12 shared setting--period strata; 72,132 cases; 90.4% weight in 2009--2018), 4 of 15 case-weighted sample pairs separate. Primary **studentized bootstrap-$t$** inference (`p = 0.249`, `c_bt = 4.519`) and **CV3 leave-one-state-out jackknife** inference (`p = 0.239`, `c = 2.556`; `t_52` reference `p = 0.249`, `c = 2.632`) reject `0/30` one-sided hypotheses (nominal 95% ranks `[1, 6]`), whereas equal outbreak weighting (**R10**) reduces giant-outbreak leverage and rejects `5/30` (`p = 0.001`; STEC `[1, 2]`) under bootstrap-$t$ and `6/30` (`p < 0.001`) under CV3.
2. **Double-bootstrap & parametric size checks:** Across five evaluation worlds (`400 × 499` outbreak-stratified, `300 × 399` fixed-size cluster random-effects + Beta-Binomial preserving observed outbreak sizes across `53`, `271`, and `52` cluster definitions, and `300 × 399` overdispersed Beta-Binomial + log-normal parametric worlds), unstudentized Gaussian, WCB, and Bayesian cluster bootstrap-$t$ exhibit inflated FWER (`17.3%–45.0%`). Studentized bootstrap-$t$ (`7.2%`/`7.8%` outbreak; `6.3%`/`2.7%` parametric) and CV3 (`4.0%`/`1.0%` outbreak; `18.7%`/`11.7%` 53-state [`15.7%`/`10.7%` `t_52`]; `9.7%–13.3%` across 271/52 clusters) remain closer to nominal 5%; while CV3 achieves `<= 5.0%` in outbreak resampling and bootstrap-$t$ achieves `2.7%` at `delta = 5 pp` in overdispersed log-normal draws, neither primary procedure guarantees `<= 5.0%` worst-case FWER across all five calibration worlds.
3. **Implied unrecorded rates, OR scale & plausibility intervals ($P_1, P_2$):** At additive $\delta = 5\text{ pp}$ or explicit plausibility bounds ($P_1$/$P_2$: lower tier unrecorded rates in `[0, 5%]` or `[0, 10%]`, upper tier `p_js ± 50%`), nominal simultaneous 95% ranks are STEC `[1,1]`, *Salmonella* `[2,2]`, and *Campylobacter* `[3,3]` above `[4,6]`. Upper-vs.-lower separation (`CAMP > CPERF`) holds to **13.5 pp** bootstrap-$t$ (**12.3 pp** CV3), with implied unrecorded hospitalization rates evaluated at the single-step 95% bootstrap-$t$ threshold reaching **0.0%** (*Campylobacter*) and **14.6%** (*C. perfringens*), whereas within-upper-tier orderings break down at **11.5 pp** bootstrap-$t$ (`STEC > SAL`, `20.4%` vs.\ `29.6%` implied unrecorded) and **4.5 pp** single-step CV3 (`SAL > CAMP`, `13.6%` vs.\ `13.2%`, 80% driven by pooled Multistate variance vs.\ `10.3 pp` in R11; stepdown holds to **6.6 pp**, `p = 0.039` at $\delta = 5$). Lower-tier pairs fail below `5 pp` additively, while relative OR scaling separates norovirus GII above GI under stepdown $\Gamma = 2.0$ (`single-step Γ*_0.95 = 1.94` boot-$t$, `1.63` CV3).

## Directory Guide

- `code/`: `nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`
- `data/raw/` & `data/derived/`: Pinned CDC NORS snapshot, audited cohorts, `etiology_mapping.csv`, `data_manifest.json`
- `analysis/`: Canonical `results.json` (`analysis_version: 8.0`), `statistical_summary.json`, and 14 CSV tables (`analysis/tables/`)
- `paper/`: Standalone `main.tex` (`paper.pdf`) & `supplementary.tex` (`supplementary.pdf`), sections, tables, figures, and `bibliography.bib`
- `reproduce.py`: One-command end-to-end verification runner
