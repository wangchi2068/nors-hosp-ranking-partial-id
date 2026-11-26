# Response to Referee Report (v3.0 Revision Summary)

We thank the referee for an extraordinarily thorough, constructive assessment—including independent code re-implementation, sensitivity probes, and a line-by-line claim audit. Every Major (1–8) and Minor (1–20) comment has been directly addressed in the v3.0 codebase, outputs, manuscript, and Supplementary Material.

---

## Major Comments

### Major Comment 1: Empirical Calibration of $\delta$, Asymmetric Departures, and Odds-Ratio Scale
- **Empirical within-data calibration (`empirical_delta_calibration`, Supp. Table S4):** Partitioned common-support outbreaks into fully recorded ($M_i = 0$) versus partially recorded ($M_i > 0, K_{\text{rec}} > 0$), confirming substantially higher known hospitalization rates in partial records (STEC: 36.2% vs.\ 25.3%, $+10.93$ pp; *Salmonella*: 25.9% vs.\ 14.1%, $+11.76$ pp; *Campylobacter*: 10.8% vs.\ 7.9%, $+2.93$ pp), cross-checked against 11 states with $<5\%$ unrecorded cases (STEC 26.2%, *Salmonella* 13.0%).
- **Asymmetric & group-calibrated sensitivity:** Evaluated $(-10/+5, -15/+5, -20/+5\text{ pp})$ and group-calibrated downward boxes; all determine **12/15** upper-tier pairs.
- **Odds-ratio pattern-mixture scale (`bounds.py`, MS Table 2):** Evaluated multiplicative odds factors $\Gamma \ge 1$ and exact $\Gamma^{*}$ (14/15 determined up to $\Gamma = 3.0$; 13/15 at $\Gamma = 5.0$; STEC > *Salmonella* flips at $\Gamma^{*} = 8.41$).
- **Main-text elevation & reframing:** Added **Main Figure 4 (`fig_04_calibrated_2d.pdf`)** with empirically calibrated downward dashed boxes; reframed $\delta = 5$ pp everywhere as an illustrative scenario threshold.

### Major Comment 2: Superpopulation Sampling Model & Jurisdiction Clustering
- **Sampling model & fixed $n_{js}$:** Explicitly stated the stratified superpopulation cluster sampling model with fixed cell counts $n_{js}$ in MS §2.3 (Minor 13).
- **53-state clustering & Wild Cluster Bootstrap (`inference_voi.py`, MS Table 3, Supp. Table S7):** Implemented closed-form influence functions for arbitrary $\delta \in [0, 1]$ (`max|V_if - V_fd| <= 1.64e-12`), aggregated to 53 reporting jurisdictions (`ratio 0.77–1.73`), and added a 9,999-replicate studentized Rademacher wild cluster bootstrap (`c_sim = 2.463` state worst case; **STEC > Norovirus GII** $p_{\text{FWER}} = 0.0275$, $p_{\text{WCB}} = 0.0278$; all 12 upper-tier pairs $p < 0.001$ at $\delta = 5$ pp).

### Major Comment 3: Small-Sample Calibration & Complete Inferential Reporting
- **Stratified MC calibration (Supp. Table S8):** Over 500 replicates preserving exact $n_{js}$ ($N_{\text{eff}} \approx 2$), HC1 mean SE / MC SD ratios are 0.852–1.024 (lower) and 0.754–0.925 (upper), yielding 76.4%–87.4% empirical IM boundary coverage and elevated Max-$T$ rejection rates. We report this limitation candidly in MS §3.3 & §4.1 and anchor primary worst-case inference on the wild cluster bootstrap and full bootstrap-covariance Max-$T$ check ($p_{\text{FWER}} = 0.0225$, $c = 2.494$).
- **Bootstrap SD fix (`bootstrap.py`, `make_outputs.py`, Supp. Table S7):** Replaced `(p97.5 - p2.5)/3.92` with true empirical bootstrap SDs (`L_sd`, `U_sd`).
- **Inferential disclosures (MS §2.3 & Table 3):** Added Stoye (2009) regularity caveat, increased multiplier draws to `200,000` with explicit MC SEs, clarified Gaussian-multiplier Romano–Wolf variant, and implemented dynamic $c_{\text{sim}}(\delta)$ curves for `95% δ*`.

### Major Comments 4 & 5: Reframing Follow-Up Against `width_greedy`, Budget-Filling & Representative Masking
- **Terminology & comparator:** Replaced "Value of Information" with **Disambiguation Priority Index (DPI)** (citing Ades et al. 2004), elevated unweighted stratum leverage $w_s/C_{js}$ (`width_greedy`) as the primary comparator, and removed the `>80%` claim.
- **Budget-filling rule & mechanism (`inference_voi.py`, MS Table S10 & Fig. 5b):** Fixed budget truncation so every policy spends ~`999.6–999.8` of 1,000 cases; proved the leverage mechanism directly (mean $w_s/C_{js} = 0.0573$ pp/case for width-greedy vs.\ `0.0088` for largest-first).
- **Paired comparisons & expanded protocols (Supp. Table S10 & Fig. S3):** Reported paired trial differences (`DPI - width_greedy = +0.07`, 95% CI `[-0.07, +0.21]` at 1,000 cases), added CI ribbons in Fig. 5b, and added $\epsilon$ sensitivity, **MNAR outcome+size masking**, **state-clustered masking**, **70% partial recovery**, and **adaptive 250-case tranche re-scoring** (reaching `7.31` pairs at 1,000 cases).

### Major Comment 6: Aligning Conclusions, Case Weighting, Exclusions & Burden Context
- Replaced `"unconditionally"` and `"assumption-free"` with `"without assumptions on unrecorded outcomes"`.
- Added sensitivity variants **R8** (including 951 suspected study-group outbreaks: 5/15 worst case, 13/15 at $\delta=5$), **R9** (excluding 79 healthcare/LTCF outbreaks: 4/15 and 12/15, 0 changes), and **R10** (within-cell equal outbreak weighting: 7/15 worst case, 14/15 at $\delta=5$, 0 reversals), plus **Supp. Table S2** profiling all excluded groups.
- Contextualized $\delta = 0$ bounds against Scallan et al. (2011, 2025) and Dewey-Mattia et al. (2018) in MS §4.

### Major Comment 7: Literature Positioning & Rank Confidence Sets
- Repositioned manuscript as an applied synthesis citing Manski & Molinari (2021), Mogstad et al. (2024), Klein et al. (2020), and Goldstein & Spiegelhalter (1996).
- Computed and reported **simultaneous 95% rank confidence sets** (worst case: STEC `[1,5]`, NoV GII `[2,6]`, others `[1,6]`; at $\delta=5$ pp: STEC `[1,1]`, *Salmonella* `[2,2]`, *Campylobacter* `[3,3]`, lower tier `[4,6]`).

### Major Comment 8: Process Transparency, RECORD Checklist, Declarations & Cross-Platform Reproducibility
- Removed unsubstantiated `"pre-specified"` wording; documented all 16 sensitivity checks transparently; appended full **RECORD Checklist** (Supp. Section S4) and complete author/funding/COI/AI-disclosure declarations.
- Fixed cross-platform `reproduce.py` assertion failures by introducing `NUM_TOL = 1e-5` for simulation/root-finding sections in `compare_results.py` while keeping `1e-9` for deterministic leaves and normalizing CRLF/LF in `check_tables.py`.

---

## Minor Comments (1–20) Summary

1. Fixed STEC − *Campylobacter* worst-case lower difference to **5.5** pp (MS §3.2).
2. Updated exact vs.\ bisection $\delta^{*}$ check to `"to machine precision"` ($< 2 \times 10^{-14}$ pp) (MS §2.2 & Supp. §S2).
3. Included **#4 *Campylobacter*, Restaurant, 2009–2013** (`1.24`), added real-data ranking **Supp. Table S11**, and corrected citation (MS §3.4).
4. Reported full range (`19.2 pp to ∞`) and distinguished confidence tipping points ($\le 1.5$ pp) from sample $\delta^{*}$ (`0.19–4.2` pp) (Abstract & MS §4.2).
5. Harmonized $M_i = 0$ everywhere for fully recorded outbreaks (MS §2.1, §2.4 & Supp. §S2).
6. Distinguished `complete-field consistent records` (`3,976`) from `fully recorded outbreaks` (`2,414`) (MS §2.1 & §3.1).
7. Standardized $\delta$ and difference units consistently to percentage points (`pp` / points) (MS §2.2).
8. Reconciled count as **16** sensitivity variants (15 six-group + 1 five-group R7) (MS §2.4 & §3.4).
9. Explained near-100% bootstrap `"undetermined"` reproduction in wide overlapping intervals (MS Table 2 footnote).
10. Clarified that `"Unrecorded status (%)"` includes $C - H$ from the 52 blank-$K$ records (MS Table 1 footnote).
11. Added **variant R9** excluding 79 healthcare/long-term-care outbreaks (Supp. Table S5).
12. Reported `0 / 2,000` bootstrap replicates with missing $K_{\text{rec}}$ strata (Supp. §S2 & Table S6 caption).
13. Stated fixed-$n_{js}$ stratified sampling implication in Eq. 5 (MS §2.3).
14. Corrected White et al. (2022) paraphrase to four-fold variation in *outbreak reporting rates* (MS §4.1).
15. Moved Cole et al. (2019) from follow-up paragraph to bounds background in Introduction (MS §1).
16. Explicitly labeled Simpson et al. (2022) as a conference abstract (MS §1).
17. Classified Wikswo et al. (2022) under `H/K` known-status examples with all-mode/suspected scope noted (MS §1 & §4).
18. Changed `"developed"` to `"applied and integrated"` in Introduction (MS §1).
19. Clarified in **Fig. 2** legend and caption that $\delta = 5$ pp is an illustrative scenario threshold.
20. Overlaid crude **$H/C$** and **$H/K$** points onto **Fig. 3**.

---

# Round 2 Revision Summary (v3.0 → v4.0)

We thank the Round 2 referee for an exceptionally sharp and constructive evaluation—identifying composition confounding in the within-data missingness check, the cost-model dependence of follow-up prioritization, incomplete calibration of the wild cluster bootstrap, the data-directed 15-vs-30 hypothesis family, and the Scallan et al. (2011) norovirus citation error. Every Major (M1–M7) and Minor (1–14) comment is resolved in `v4.0` as follows:

## Major Comments (M1–M7)

### M1: Composition Confounding in Empirical Missingness Calibration (`inference_voi.py`, Supp. Table S4, Fig. 4)
- Decomposed partially vs. fully recorded comparisons by outbreak size (median 16 vs. 5 for STEC; 19 vs. 7 for *Salmonella*), multistate status (60.0% vs. 11.7% STEC; 34.2% vs. 2.4% *Salmonella*), within-cell standardized rates (`Within-cell Δ = +0.19 pp` for STEC; `+9.02 pp` for *Salmonella*), and single-state vs. multistate strata (`Supp. Table S4`).
- Explicitly defined signed `Std obs − Full` (`-3.22 pp` STEC; `+2.86 pp` *Salmonella*) in MS §2.2, §3.2, and Table S4.
- Replaced calibrated MNAR claims with composition-adjusted reporting and converted Main **Fig. 4** strictly into symmetric (`5 × 5 pp`) and asymmetric (`-15/+5 pp`) stress-test boxes.

### M2: Unified Cost-Conditional Follow-Up Framework (`inference_voi.py`, Supp. Table S10, Fig. 5b–c, Fig. S3a)
- Implemented side-by-side 200-trial budget-filling benchmarks under **three cost models**: per-case ($\text{cost}_i = U_i$), per-outbreak ($\text{cost}_i = 1$), and mixed ($\text{cost}_i = 20 + U_i$ case-equivalents).
- Confirmed strategy reversal under per-outbreak costs (`Largest-first` `7.90` pairs vs. `Random` `3.27` and `Per-case w_s/C_js` `7.38` at 100 outbreaks) and unified dominance of cost-normalized width reduction `w_s U_i / (C_js * cost_i)` across all budgets (`6.29` per-case at 1,000 cases; `9.12` per-outbreak at 100 outbreaks; `4.09` mixed at 1,000 units).
- Moved heuristic DPI to Supplement (§S2, Fig. S3b); added masking-cohort representativeness and prospective-only 2009–2018 limitations in §4.1.

### M3: Closed Calibration Loop, Non-Gaussian Bootstrap Max-T, and Uniformity Caveats (`bootstrap.py`, `inference_voi.py`, Table 3, Table S8)
- Added **studentized non-Gaussian bootstrap Max-T** across all 30 hypotheses directly from the 2,000 bootstrap replicates: `c_boot = 2.725` (worst case) and `2.899` ($\delta = 5$); **STEC > Norovirus GII** $p = 0.0430$ (worst case) and all 12 upper-tier pairs $p \le 0.0020$ at $\delta = 5$.
- Closed the 500-replicate **bootstrap-world calibration** loop across all **five** procedures (`HC0`, `HC1`, `non-Gaussian boot-crit`, `53-state`, `53-state WCB`): empirical 30-test FWER drops from 32.8% / 29.8% (HC0/HC1) to 21.4% / 20.6% (State/WCB) at worst case and 8.8% / 8.0% at $\delta = 5$; cited MacKinnon & Webb (2017) on unequal cluster sizes and removed any claim that calibration validates WCB.
- Clarified Stoye (2009) regularity near $\delta \to 0$ and reported `0 / 72` cells within $10^{-6}$ of piecewise kinks at $\delta = 5$ pp.

### M4: Full 30-Hypothesis Two-Sided Family & Exact Rank Confidence Sets (`bootstrap.py`, `inference_voi.py`, Table 3)
- Upgraded **all** FWER inference (`HC1`, `Bootstrap cov`, `53-State`, `WCB`, `Non-Gaussian Boot`, dynamic $c_{\text{sim},30}(\delta)$) to all **30 ordered two-sided hypotheses** (citing Mogstad et al. 2024, Klein et al. 2020, Andrews et al. 2019).
- Stated the conservative partial-identification rank-set rule ($\Delta^L_{ij} > 0$) and computed exact 30-test rank confidence sets directly for **worst case**, **$\delta = 5$ pp**, and **asymmetric `-15/+5 pp`** (`STEC [1,1], SAL [2,2], CAMP [3,3], lower [4,6]`).

### M5: Corrected Scallan et al. (2011) Norovirus Burden Comparison (MS §4)
- Corrected Scallan et al. (2011) norovirus hospitalization proportion to **0.03%** (per estimated community illness), specified distinct pathogen denominators (lab-confirmed vs. total illness vs. outbreak data), and discussed why NORS outbreak-reported norovirus hospitalization (1.6%) is ~50-fold higher than community burden estimates.

### M6: Accurate-$C_i$ Assumption & Single-State Stratification (`analyze.py`, MS §2.2, §4.1, Table S5)
- Explicitly stated exact-$C_i$/$H_i$/$K_i$ assumption and route-dependent ascertainment bias (Heiman et al. 2015); added sensitivity variant **R11 (`3,659` single-state outbreaks only)** confirming `12` cells, `4` worst-case / `12` $\delta = 5$ verdicts, and `0` reversals.

### M7: Positioning Against Simpson et al. (2022) Preprint & Abstract Condensation
- Cited and contrasted the full NORS severity-index preprint (`simpson2022severity`, DOI `10.21203/rs.3.rs-1720769/v1`) in Introduction §1; condensed Abstract below 250 words with minimal acronyms.

## Minor Comments (1–14) Summary
1. Fixed `"12 upper-tier orderings"` (vs. `14/15` total at $\Gamma=3.0$) in MS §4.
2. Clarified 11 low-missingness states track *fully recorded* rates in MS §4.
3. Added `0.1%`-floor OR sensitivity (`floor = 0.001`: `SAL > CPERF` $\Gamma^* = 955.7$; 14/15 at $\Gamma \le 3.0$) and flagged $\Gamma^* = \infty^{\dagger}$ in Table 2 & Table S4.
4. Rephrased to `"approximates the within-cell missing-at-random anchor"` in MS §4.
5. Replaced `"spuriously raises"` with `"would overstate"` in MS §4.
6. Replaced agreement-based Table 2 bootstrap columns with determination frequencies (`Boot det., worst (%)` and `Boot det., δ = 5 (%)`).
7. Noted in MS §3.4 that R1a, R2, R4a, R5, and R6b shift one low-tier $\delta = 0$ direction.
8. Stated 90.4% (2009–2018) effective weight span in Abstract and data-dependent support selection in MS §2.1 & §4.1.
9. Narrowed Ebel et al. (2016) scope strictly to FoodNet lab-confirmed bacteria in §4.1.
10. Added White et al. (2022) full `5–86` per 10M (>40-fold norovirus) reporting range in §4.1.
11. Condensed Abstract to $\le 250$ words with minimal acronyms.
12. Cited `Table S9` in MS §3.4 and aligned `Tables S1–S12` across `README.md` and `supplementary.tex`.
13. Added verified DOIs across `bibliography.bib` and confirmed `Horowitz & Manski (2000)` (`pages = 77–84`) and `CDC (2026)` publication details.
14. Elevated cost-normalized width reduction (`Eq. 5`) as primary follow-up rule and placed heuristic DPI in Supplement.

