# Partial Identification and Simultaneous Rank Inference for Pathogen Hospitalization Comparisons among Reported Foodborne Outbreak Cases, United States, 2009–2023

Clean-slate `v10.0` repository containing complete manuscript source (`paper/paper.pdf`, `paper/supplementary.pdf`), raw CDC NORS data snapshot and portal metadata (`:id` retained for revision checks), deterministic data-preparation pipeline, partial-identification & inference suite, conventional/empirical-Bayes benchmarks, and cross-platform reproduction script (`analysis_version: 10.0`).

## Quick Reproduction

From the repository root (Python 3.11+):

```bash
pip install -r requirements.txt
python reproduce.py --render-outputs
```

Full end-to-end execution takes approximately **14–18 minutes** on CPU (`--workers 2` default; pass `--workers 4` to speed up the initial 2,000-replicate stratified bootstrap, or `--B 200` for a quick smoke check). The script verifies the raw CDC snapshot SHA-256, rebuilds derived cohorts into `rerun/`, executes `code/analyze.py` (`B = 2,000`), verifies exact equality on all deterministic leaves (`1e-9`), identical 30-hypothesis decision sets across worst-case, $\delta=5\text{ pp}$, and $\delta=0$ specifications, all 18 rendered CSV tables (`all_non_bootstrap_cells_identical = True`), manuscript text numeric consistency (`code/verify_text_numbers.py`), and Monte Carlo simulation tolerances (`<= 1e-3`, with a cross-platform BLAS warning threshold at `0.025`). All JSON and CSV outputs use Unix `LF` (`\n`) line endings.

## Key Empirical & Methodological Findings (`v10.0`)

1. **Conventional MAR baselines vs.\ worst-case identification:** Across 3,176 common-support outbreaks (of 4,133 eligible; 72,132 cases; 90.4% case weight in 2009–2018), marginal Wilson `H/K`, Laird–Louis (1989) empirical-Bayes Beta-Binomial hierarchical rankings (outbreak- and state-level), and simultaneous 30-hypothesis dual testing at $\delta = 0$ (`14/30`) all certify `[1,1]`, `[2,2]`, `[3,3]`, `[4,4]`, `[5,6]`, `[5,6]`. Without missingness assumptions (12.7%--22.2% unrecorded), bounds span 12--19 pp, 4/15 sample pairs separate, and case-weighted **dual-criterion** (`boot-t ∩ CV3`) testing rejects `0/30` (`[1,6]`), whereas equal outbreak weighting (**R10**) rejects `5/30` (`boot-t`, `p = 0.001`) or `6/30` (`CV3`, `p < 0.001`).
2. **Global-null, partial-null stepdown & empirical power calibrations (Tables S10–S11):** Across five global-null worlds, the dual-criterion rule achieves **3.0% / 1.0%** (outbreak) and **4.7% / 2.0%** (parametric heavy-tail) FWER, and reduces cluster-RE over-rejection to **13.0% / 7.3%** (53-cluster) and **8.0%–8.3% / 3.0%–4.7%** (271/52-cluster). Under $\delta = 5\text{ pp}$ partial-null simulations (11 strong empirical separations retained, 19 nulls including `SAL|CAMP`), actual Romano--Wolf stepdown dual FWER is **0.5%** (outbreak) and **6.0%** (53-cluster RE; single-step `5.0%`) with `0.0%` false rejections on `SAL|CAMP`. Empirical worst-case dual power for `STEC > NoV GII` is **9.2%** (outbreak) and **6.7%** (53-cluster RE; vs. **75.2%** HC1), recovering **9.6/12** upper-tier pairs on average at $\delta = 5\text{ pp}$.
3. **Single-step vs.\ stepdown breakdown frontiers & external plausibility anchors (`P_ext`):** Upper-vs.-lower separation (`CAMP > CPERF`) holds to **12.3 pp** CV3 and **13.5 pp** bootstrap-$t$ (0.0% vs.\ 14.6% implied unrecorded hospitalization), whereas `SAL > CAMP` breaks at **4.5 pp** single-step CV3 (`11/30` rejected at $\delta=5$, `[2,3]`) and holds to **6.6 pp** under Romano--Wolf stepdown (`12/30`, `[2,2]/[3,3]`). Externally anchored surveillance intervals (`P_ext`: FoodNet/Scallan upper-tier ranges with `[0, 1%]` lower tier) separate `13/30` pairs (`[1,1], [2,3], [2,3], [4,4], [5,6], [5,6]`).

## Directory Guide

- `code/`: `nors_rules.py`, `prepare_data.py`, `strata.py`, `bounds.py`, `bootstrap.py`, `inference.py`, `analyze.py`, `make_outputs.py`, `compare_results.py`, `check_tables.py`, `verify_text_numbers.py`
- `data/raw/` & `data/derived/`: Pinned CDC NORS snapshot, portal view metadata, audited cohorts, `etiology_mapping.csv`, `data_manifest.json`
- `analysis/`: Canonical `results.json` (`analysis_version: 10.0`), `statistical_summary.json`, and 18 CSV tables (`analysis/tables/`)
- `paper/`: Standalone `main.tex` (`paper.pdf`) & `supplementary.tex` (`supplementary.pdf`), sections, tables, figures, and `bibliography.bib`
- `reproduce.py`: One-command end-to-end verification runner
