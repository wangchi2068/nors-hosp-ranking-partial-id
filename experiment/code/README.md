# Analysis Code (`experiment/code/`)

Complete, deterministic analysis pipeline (`analysis_version: 2.0`) reproducing all findings, tables, and figures from the raw CDC NORS snapshot (`experiment/data/raw/nors_soda_full_20261125T040830Z.csv`).

## Quick Reproduction

From the repository root (works on Windows, macOS, and Linux):

```bash
pip install -r requirements.txt
python reproduce.py
```

Or on Linux/macOS/Git Bash from `experiment/`:

```bash
bash code/run_all.sh
```

Both commands write regenerated outputs into `experiment/rerun/` without overwriting committed files and verify bit-for-bit equality against `experiment/analysis/results.json` and all 10 CSV tables in `experiment/data/analysis_tables/`.

## Scripts

| File | Role |
|---|---|
| `nors_rules.py` | Etiology mapping, eligibility, setting harmonization, 5-year periods, and record-audit rules |
| `prepare_data.py` | Raw snapshot (SHA-256 verified) → audited cohort, mapping tables, strata counts, common support, audit summaries |
| `strata.py` | Group-by-cell aggregation, common support, and reference weights |
| `bounds.py` | Worst-case bounds, $\delta$-sensitivity intervals, piecewise-linear critical deviations $\delta^{*}$, and 2D boundaries |
| `bootstrap.py` | Seeded stratified outbreak-level bootstrap (`B = 2000`, `seed = 20261125`) |
| `inference_voi.py` | Stratified outbreak-cluster stacked M-estimation (`HC1`), Imbens–Manski 95% CIs, Gaussian multiplier Max-$T$ FWER stepdown, HHI cluster concentration diagnostics, 95% confidence tipping points $\delta^{*}_{0.95}$, and 200-trial retrospective masking VoI validation |
| `analyze.py` | End-to-end execution writing `results.json` (main analysis, fine grid, 13 robustness variants, retention, width decomposition, bootstrap, stacked inference, HHI diagnostics, VoI masking) |
| `make_outputs.py` | Generates all 10 LaTeX/CSV tables and all PDF/PNG figures solely from `results.json` |
| `compare_results.py` | Leaf-by-leaf verification comparing a rerun `results.json` against the canonical `experiment/analysis/results.json` |
| `check_tables.py` | Cell-by-cell verification comparing rendered CSV tables against `experiment/data/analysis_tables/` |
| `requirements.txt` | Exact Python package versions (`numpy`, `pandas`, `scipy`, `matplotlib`) |
