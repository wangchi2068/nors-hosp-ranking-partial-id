# Hospitalization Orderings among Six Pathogen Groups in Foodborne Outbreaks with Incompletely Recorded Outcomes, United States, 2009–2023

Complete manuscript source, standalone Supplementary Material, raw CDC NORS data snapshot, and deterministic replication pipeline (`analysis_version: 2.0`).

## Quick Reproduction

From the repository root (tested with Python 3.11.2 on Windows and Linux):

```bash
pip install -r requirements.txt
python reproduce.py
```

Or on Linux/macOS from `experiment/`:

```bash
bash code/run_all.sh
```

Both commands verify the raw snapshot SHA-256, rebuild the audited cohort, run all main, sensitivity, robustness, 2,000-replicate bootstrap, stacked M-estimation FWER, HHI cluster-concentration, and 200-trial Value-of-Information (VoI) masking analyses into `experiment/rerun/`, and verify bit-for-bit equality against `experiment/analysis/results.json` and all 10 tables in `experiment/data/analysis_tables/`. To also rebuild PDFs (`paper/paper.pdf` and `paper/supplementary.pdf`), run `cd paper && bash compile.sh` (requires XeLaTeX + BibTeX).

## Key findings

1. **Sample partial identification:** With no assumption about unrecorded hospitalization outcomes, 4 of 15 pairwise orderings are determined in the sample (STEC above *Campylobacter*, *C. perfringens*, norovirus GI and norovirus GII). Under a 5-point scenario departure ($\delta = 5$), 12 of 15 orderings hold, while the three orderings among *C. perfringens*, norovirus GI and norovirus GII fail at 0.19, 3.7 and 4.2 points. Evidence: `paper/paper.pdf` Figure 2, Table 2; `experiment/analysis/results.json` (`main.pairs`).
2. **Joint M-estimation inference and FWER control:** After accounting for outbreak clustering (`HC1`), shared reference-weight uncertainty, and 15-pair Gaussian multiplier Max-$T$ FWER control ($c = 2.492$ worst case; $2.526$ at $\delta = 5$), STEC above norovirus GII remains unconditionally significant ($p_{\text{FWER}} = 0.0182$), while all 12 upper-tier orderings are jointly significant at $\delta = 5$ ($p_{\text{FWER}} < 0.0001$) with simultaneous 95% confidence tipping points $\delta^{*}_{0.95,\text{sim}}$ of 19.5–81.9 points (STEC above *Salmonella*: 20.4 points). High single-outbreak concentration ($\text{max HHI} \approx 0.50$, $N_{\text{eff}} \approx 2.0$) explains upper-bound sampling variance in smaller cohorts. Evidence: `paper/paper.pdf` Table 3, Figure 4a; `supplementary.pdf` Table S5.
3. **Value-of-Information follow-up:** Across 200 retrospective masking trials on all 2,414 complete-status common-support outbreaks (45,249 cases), targeting follow-up by largest outbreak size performs worse than random follow-up (at 1,000 cases: 2.01 vs. 2.50 resolved pairs), whereas the proposed VoI rule resolves 6.31 pairs with 1,000 follow-up cases—surpassing largest-outbreak follow-up at 5,000 cases (5.95 pairs) and cutting follow-up volume by >80%. Evidence: `paper/paper.pdf` Figure 4b; `supplementary.pdf` Table S7.
4. **Robustness and record audit:** Across 12 pre-specified six-group variants, 4–7 orderings hold without assumptions and 12–14 at 5 points with zero reversals among primary pairs. Retaining all 157 incomplete outbreaks under logical bounds avoids inflating worst-case verdicts (dropping them spuriously raises determined pairs from 4 to 7 in R3b). In 2,000 bootstrap replicates, the four primary worst-case orderings recur in 91.6%–100.0% of replicates. Evidence: `supplementary.pdf` Tables S1, S3, S4, Figure S3.
5. **Data:** Full public CDC NORS snapshot downloaded 2026-11-25 UTC (`66,713` rows, portal updated 2024-12-20): `11,358` foodborne outbreaks in 2009–2023, `4,133` eligible (`103,040` cases, `19.4%` unrecorded status), and `3,176` outbreaks (`72,132` cases) in 12 common-support cells. Evidence: `paper/paper.pdf` Figure 1; `experiment/data/derived/eligibility_flow.json`.

## File guide

| File | Purpose |
|---|---|
| `reproduce.py` | Cross-platform one-command reproduction and verification script |
| `requirements.txt` | Exact Python package dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`) |
| `paper/paper.pdf` | Main manuscript (13 pages: 11 pages body with 4 figures and 3 tables + 2 pages with 29 references) |
| `paper/supplementary.pdf` | Standalone Supplementary Material (8 pages: Sections S1–S3, Tables S1–S7, Figures S1–S3) |
| `paper/main.tex` | Main manuscript LaTeX source (`supplementary.tex` compiles the standalone supplement; `compile.sh` builds both) |
| `experiment/analysis/results.json` | Single canonical source of all reported numbers (`analysis_version: 2.0`) |
| `experiment/analysis/statistical_summary.json` | Compact summary of headline findings |
| `experiment/data/raw/nors_soda_full_20261125T040830Z.csv` | Full NORS snapshot (`66,713` rows; SHA-256 verified against `nors_soda_full_20261125T040830Z.json`) |
| `experiment/data/derived/` | Audited cohort (`cohort_outbreaks.csv`), record audit (`record_audit.csv`), mapping tables, strata counts, and `data_manifest.json` |
| `experiment/data/analysis_tables/` | CSV versions of all 10 tables (`table_1`–`table_10`) |
| `experiment/code/run_all.sh` | Shell script executing the full pipeline into `experiment/rerun/` |
| `experiment/code/nors_rules.py` | Locked eligibility, etiology, setting-harmonization, and audit rules |
| `experiment/code/prepare_data.py` | Raw snapshot → audited cohort, mapping tables, and summaries |
| `experiment/code/strata.py` | Group-by-cell aggregation, common support, and reference weights |
| `experiment/code/bounds.py` | Worst-case bounds, $\delta$-sensitivity intervals, piecewise-linear $\delta^{*}$, and 2D boundaries |
| `experiment/code/bootstrap.py` | Seeded stratified outbreak-level bootstrap (`B = 2000`, `seed = 20261125`) |
| `experiment/code/inference_voi.py` | Stacked M-estimation (`HC1`), Imbens–Manski 95% CIs, Max-$T$ FWER stepdown, HHI diagnostics, and 200-trial VoI masking validation |
| `experiment/code/analyze.py` | Full computational pipeline outputting `results.json` |
| `experiment/code/make_outputs.py` | Renders all 10 LaTeX/CSV tables and all PDF figures from `results.json` |
| `experiment/code/compare_results.py` | Leaf-by-leaf comparison against `experiment/analysis/results.json` |
| `experiment/code/check_tables.py` | Cell-by-cell comparison against `experiment/data/analysis_tables/` |
| `experiment/rebuild_check/` | Verification reports (`rebuild_check.md`, `rebuild_check.json`, `table_check.json`) |

## Reproducibility & Verification

- **Deterministic end-to-end check:** `python reproduce.py` verifies `experiment/data/raw/nors_soda_full_20261125T040830Z.csv` against its SHA-256 (`6005ac026ef124a5a1e5b9bd1f5aafc791184976f4518368bb5cd5aa059e82e9`), rebuilds the audited cohort and strata, runs the full `analysis_version: 2.0` pipeline (`2,000` bootstrap replicates, seed `20261125`; `50,000` Gaussian multiplier draws; `200` masking trials), and confirms `0` discrepancies across all `15,213` non-bootstrap result leaves, all `2,000` bootstrap summaries, and all `10` CSV tables.
- **Data source:** CDC National Outbreak Reporting System (NORS) public open dataset (`data.cdc.gov` ID `5xkq-dg7x`, downloaded 2026-11-25 UTC, portal rows updated 2024-12-20).
- **License:** Analysis code is released under the [MIT License](LICENSE). Public CDC surveillance data remain subject to CDC open-data terms.

