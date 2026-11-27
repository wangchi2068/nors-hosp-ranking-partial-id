# Partial Identification and Simultaneous Rank Inference for Pathogen Hospitalization Comparisons among Reported Foodborne Outbreak Cases, United States, 2009–2023

Clean-slate `v9.0` repository containing complete manuscript source (`paper/paper.pdf`, `paper/supplementary.pdf`), raw CDC NORS data snapshot, deterministic data-preparation pipeline, partial-identification & inference suite, and cross-platform reproduction script (`analysis_version: 9.0`).

## Quick Reproduction

From the repository root (Python 3.11+):

```bash
pip install -r requirements.txt
python reproduce.py --render-outputs
```

Full end-to-end execution takes approximately **12–16 minutes** on CPU (2 workers; pass `--workers 4` to speed up the initial 2,000-replicate stratified bootstrap, or `--B 200` for a quick smoke check). The script verifies the raw CDC snapshot SHA-256, rebuilds derived cohorts into `rerun/`, executes `code/analyze.py` (`B = 2,000`), verifies exact equality on all deterministic leaves (`1e-9`), identical 30-hypothesis decision sets across primary procedures, and all 14 rendered CSV tables (`all_non_bootstrap_cells_identical = True`), and checks Monte Carlo simulation leaves cleanly (`<= 1e-3`, with a cross-platform BLAS warning threshold at `0.025`). All JSON and CSV outputs use Unix `LF` (`\n`) line endings.

## Key Empirical & Methodological Findings (`v9.0`)

1. **Without assumptions on unrecorded outcomes (taking reported counts as exact):** Across 4,133 confirmed foodborne outbreaks (3,176 retained in 12 shared setting--period strata; 72,132 cases; 90.4% weight in 2009--2018), worst-case bounds span 12--19 pp (mirroring 12.7%--22.2% unrecorded status) and 4 of 15 case-weighted sample pairs separate. Under our **unified conservative joint rule** requiring both **studentized bootstrap-$t$** (`p = 0.249`, `c_bt = 4.519`) and **CV3 leave-one-state-out jackknife** (`p = 0.239`, `c = 2.556`) to reject at `<= 0.05`, `0/30` one-sided hypotheses reject (nominal 95% ranks `[1, 6]`), whereas equal outbreak weighting (**R10**, targeting a distinct outbreak-average estimand) rejects `5/30` (`boot-t`, `p = 0.001`) or `6/30` (`CV3`, `p < 0.001`).
2. **Full 5-world calibration matrix:** Across five evaluation worlds (`400 × 499` outbreak-stratified, `300 × 399` random-size cluster random-effects + Beta-Binomial across `53`, `271`, and `52` cluster definitions, and `300 × 399` overdispersed Beta-Binomial + log-normal parametric worlds), all primary procedures are evaluated jointly. Our **conservative joint rule** controls worst-case / `delta=5 pp` FWER at **3.0% / 1.0%** (outbreak) and **4.7% / 2.0%** (parametric heavy-tail), and strictly lowers cluster-RE over-rejection (**13.0% / 7.3%** in 53-cluster RE vs.\ **25.0% / 24.0%** standalone boot-$t$ and **17.3% / 11.3%** CV3; **8.0%–8.3%** worst and **3.0%–4.7%** `delta=5` across 271/52 clusters).
3. **Conditional missingness bounds, OR scale & plausibility intervals ($P_1, P_2$):** Conditioned on $\delta = 5\text{ pp}$ or explicit plausibility benchmarks ($P_1$/$P_2$: lower tier unrecorded rates in `[0, 5%]` or `[0, 10%]`, upper tier `p_js ± 50%`), nominal simultaneous 95% ranks are STEC `[1,1]`, *Salmonella* `[2,2]`, and *Campylobacter* `[3,3]` above `[4,6]`. Upper-vs.-lower separation (`CAMP > CPERF`) holds to **13.5 pp** bootstrap-$t$ and **12.3 pp** CV3 (0.0% vs.\ 14.6% implied unrecorded hospitalization), whereas within-upper-tier orderings break down at **11.5 pp** bootstrap-$t$ (`STEC > SAL`, 20.4% vs.\ 29.6%) and **4.5 pp** single-step CV3 (`SAL > CAMP`, 13.6% vs.\ 13.2%, 78.5% driven by pooled Multistate variance vs.\ `10.3 pp` in R11; stepdown holds to **6.6 pp**, `p = 0.039` at $\delta = 5$). On the relative OR scale ($\Gamma = 2.0$), norovirus GII separates above GI **only under stepdown bootstrap-$t$** (`13/30`, single-step `Γ*_0.95 = 1.94`) and **not under CV3 or the conservative joint rule** (`12/30`, single-step `CV3 Γ*_0.95 = 1.63`).

## Directory Guide

- `code/`: `nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`
- `data/raw/` & `data/derived/`: Pinned CDC NORS snapshot, audited cohorts, `etiology_mapping.csv`, `data_manifest.json`
- `analysis/`: Canonical `results.json` (`analysis_version: 9.0`), `statistical_summary.json`, and 14 CSV tables (`analysis/tables/`)
- `paper/`: Standalone `main.tex` (`paper.pdf`) & `supplementary.tex` (`supplementary.pdf`), sections, tables, figures, and `bibliography.bib`
- `reproduce.py`: One-command end-to-end verification runner
