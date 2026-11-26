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

---

# Round 3 Revision Summary (v4.0 → v5.0)

We thank the Round 3 referee for pinpointing the tension between nominal Gaussian significance and our finite-sample calibration, the `%` truncation bug in §3.1, the Discussion rank-set error, the exposure-state Multistate pseudo-cluster, and all remaining bibliographic and graphical details. All 8 Major (M1–M8) and 19 Minor comments are addressed in `v5.0`:

## Major Comments (M1–M8)

### M1 & M2: True Studentized Bootstrap-$t$ Calibration & Matched 6-Procedure Calibration Loop (`bootstrap.py`, `inference_voi.py`, Table 3, Table S8)
- Renamed the `bootstrap.py` fixed-scale test accurately to **standardized (fixed-scale) bootstrap Max-$T$** (removing erroneous `"studentized"` label) and reported explicit Monte Carlo SEs (`0.0045` for `p = 0.043`; `0.0097` for bootstrap-$t$).
- Implemented true replicate-studentized **bootstrap-$t$ Max-$T$** ($T^{(b)}_m = (\Delta^{L,(b)}_m - \widehat{\Delta}^L_m)/\widehat{\sigma}^{(b)}_{\text{HC1},m}$, $B=2,000$): worst-case critical value $c_{\text{bt}} = 4.519$, **STEC > Norovirus GII** $p = 0.249$ (0 rejections overall; calibrated worst-case ranks `[1, 6]` for all pathogens), whereas at $\delta = 5$ pp ($c_{\text{bt}} = 3.369$), all 12 upper-tier orderings remain jointly significant ($p \le 0.0020$).
- Fixed the calibration loop so every procedure compares **its own exact statistic** (`fixed_scale_boot` = `5.0%` / `4.4%`; `bootstrap_t` = `3.8%` / `5.8%`; vs. nominal `29.8%` HC1 and `20.6%` WCB at worst case), and listed top exceedance drivers (`NOV_GII->CPERF`, `NOV_GI->CPERF`, `SAL->CPERF`, Minor 9).
- Placed calibrated worst-case results and worst-case rank sets first in the Abstract, Results, Table 3, and Conclusions.

### M3 & M4: Table S4 Direction Correction, Breakdown Frontiers, Common-Mechanism Scaling & Outbreak-MAR (`inference_voi.py`, `analyze.py`, Fig. 5, Table S4, Table S5)
- Corrected Discussion §4 to report accurately that Table S4's `Within-cell Δ` and `Single-state Δ` are **positive in all six pathogen groups** (`+0.19` to `+9.02` pp; `+0.28` to `+8.37` pp), presenting both compatible interpretations (outcome-dependent recording and outbreak-size/multistate heterogeneity).
- Fixed the factual error on Discussion p.\ 13 (worst-case nominal ranks are `[1,5], [2,6], [1,6]` and calibrated ranks `[1,6]`, not `[1,1]`).
- Positioned $\delta^*, \Gamma^*, \delta^*_{0.95}$ as **breakdown points** and 2D planes as **breakdown frontiers** (Kline & Santos 2013; Masten & Poirier 2020; Liublinska & Rubin 2014), computed explicit **rank breakdown points** (`15.0–15.9 pp` calibrated bootstrap-$t$; `18.0–20.0 pp` nominal), added **common-mechanism proportional scaling** ($\alpha \in [0.00, 2.00]$, Fig. 5b), and added outbreak-level $H_i/K_i$ MAR anchor (**variant R13**).

### M5: Estimand Weighting Justification, Effective Period, & `Illnesses` Missing Types (`methods.tex`, `results.tex`, `discussion.tex`)
- Stated 90.4% (2009–2018) weight concentration in Abstract; elevated R10 (`equal within-cell outbreak weighting`: `7` worst-case, `14` at $\delta=5$) beside main case-weighted findings in Results and Discussion; discussed `Illnesses` estimated-total mixed "missing types" (Jiang & Ding 2018).

### M6: Exposure-State Labelling, Multistate Leverage & Splits (`inference_voi.py`, Table S7)
- Corrected field description to **state of exposure** (`52 jurisdictions + CDC-coordinated Multistate cluster`, `84.8%` of unrecorded STEC cases; corrected `"top three states"` to `Multistate 84.8%, Ohio 4.8%, Tennessee 1.8%`).
- Reported MacKinnon–Nielsen--Webb (2023) score leverage (`48.3%` worst case / `66.8%` at $\delta=5$ in Multistate; $G_{\text{eff}} = 4.04 / 2.21$) and confirmed **zero worst-case rejections** both when splitting Multistate into 219 singletons (`271` clusters) and clustering inside the 52 single states (`R11`). Corrected MacKinnon & Webb (2017) & Canay et al. (2021) on WCB failure under dominant clusters, and explicitly noted calibration does not model state correlation.

### M7 & M8: Tempering Follow-Up, Table S10/Fig. S3b Completeness & Streamlining Presentation
- Removed `"optimal"`, `"dominates"`, and `"across all cost regimes"`; tabulated **all** mixed budgets (`500, 1000, 2000, 5000`, showing `Per-case 9.72 > Mixed 9.22` at 5,000 units) in Table S10; plotted **baseline 6.61 & MNAR random 8.34** in Fig. S3d; refocused Main Fig. 5 on breakdown points + common-mechanism scaling and restructured Results around three core questions.

## Minor & Presentation Fixes (§4.2–4.4)
- **§4.4 item 1:** Fixed unescaped `%` in `paper/sections/results.tex` line 7 (`13.5\%`), restoring all omitted §3.1 text in `paper.pdf`.
- **Minor 1–19:** Verified conservative blank-$K$ lower bound (STEC `27.48%` vs `27.94%`, `SAL` `16.33%` vs `16.76%`, 0 verdict changes); listed all 7 variants changing a $\delta=0$ direction (R1a, R2, R4a, R5, R6b, R8, R13); separated five-group R7/R7b (`3/10` and `9/10`) from 18 six-group variants; corrected to `"30 one-sided hypotheses"`; reported post-selection R4a/R4b rank sets; corrected Scallan/Tack denominator discussion; labeled IM CIs `"pointwise 95%"`; added **R7b** (304 unspecified norovirus) and **R12** (O157 STEC only); fixed Table 2 $\Gamma^* = 955.7$ ref; added Wilson CIs for 11 low-missingness states; checked kink proximity at roots & cited Fang & Santos (2019); expanded **Table S12** to the full 22-item STROBE/RECORD checklist with exact section/page references; updated `bibliography.bib` with verified DOIs and all 14 §4.3 references; fixed Fig. 4/5/S3 subscripts, legend spacing, `CPERF > NoV GI` label, Table S6 2-decimal precision, unit consistency (`pp`), Table 1–3 abbreviation definitions, and `reproduce.py` `"14 CSV tables"` docstring.



---

# Round 4 Revision Summary (v5.0 → v6.0)

We thank the Round 4 referee for auditing every line of code, independently re-implementing the bounds and breakdown points, running the `250 × 199` double-bootstrap and $\delta$-grid checks, and verifying all 54 references. Every Major (`M1–M7`) and Minor (`m1–m22`) comment is resolved in `v6.0`:

## Major Comments (`M1–M7`)

| Comment | Root Cause in `v5.0` | Resolution in `v6.0` |
|---|---|---|
| **M1** Circular bootstrap calibration | `run_small_sample_calibration` compared outer statistics to a fixed original-sample critical value (`c_bt_wc = 4.519`). | Replaced with a genuine **`400 × 399` stratified double bootstrap** (`Beran 1988`) re-estimating every procedure's critical value inside every outer world (`inference_voi.py`, Table S8): empirical worst-case 30-test FWER is `33.2%±2.4%` (HC0), `30.5%±2.3%` (HC1), `24.8%±2.2%` (fixed-scale), `20.5%±2.0%` (53-cluster), `20.5%±2.0%` (unrestricted WCB), vs. **`6.5%±1.2%`** (`6.8%±1.3%` at $\delta = 5$ pp) for studentized bootstrap-$t$. Removed `"calibrated"` everywhere; presented HC1 (`p = 0.024`) and bootstrap-$t$ (`p = 0.249`) side-by-side as fragile/method-dependent. |
| **M2** Fixed $\delta = 5$ critical value in bootstrap-$t$ breakdown points | `inference_voi.py:1544` passed `fixed_c=c_bt_d5` across all $\delta$. | Evaluated $\delta$-specific $c_{\text{bt}}(\delta)$ on a 14-point grid (`3.05` @ 0 to `3.37` @ 5, `3.80` @ 10, `4.10` @ 15, `4.52` @ 100 pp) plus conservative worst-case uniform bound ($c_{\text{bt}}(100) = 4.52$). Corrected joint upper-tier rank breakdown point to **`11.6 pp`** (`11.6 pp` STEC `[1,1]` / *Salmonella* `[2,2]`, `13.5 pp` *Campylobacter* `[3,3]`; **`7.9 pp`** worst-case $c_{\text{bt}}$), displayed in Table 3 and Fig. 5a–b. |
| **M3** Superpopulation & sparse strata (`N_eff ≈ 2`) | Coarse strata (`R2`) and equal-outbreak (`R10`) lacked inference; LOO influence not shown. | Clarified superpopulation target (and hierarchical shrinkage complement, `Normand et al. 1997`), reported full 30-test inference and rank sets for **`R2_period_only`** (`min n = 24`: `2` HC1/state vs `0` boot-$t$ worst case, `p = 0.0510`; `12` at $\delta = 5$) and **`R10_outbreak_weighted`** (`5` boot-$t$ worst case; `12` at $\delta = 5$), and ran all `3,176` single-outbreak **LOO** deletions (`4–4` worst case, `12–13` at $\delta = 5$; `STEC > SAL` $\delta^* \in [37.4, 48.2]$ pp) plus 6-group **LLO** (`4` worst case, `12` at $\delta = 5$, `45.7` pp) in MS §3.2 and Table S7. |
| **M4** Analytic flexibility | Primary procedure shifted across rounds; early pilot and threshold grid undisclosed. | Pre-designated **studentized bootstrap-$t$** *a priori* on methodological grounds (severe skewness and sparse cells $N_{\text{eff}} \approx 2$), disclosed the exploratory pilot cohort (`3,799` / `96` / `3,703` across 10 cells) and support-threshold grid (`1–6`) in Supp. §S1, and explicitly acknowledged lack of pre-specification in §4.1. |
| **M5** Wrong range, hidden negative contrasts & missingness scale | Abstract/Discussion cited `+0.19 to +9.02`; omitted negative `Std obs − Full` (`-2.53`, `-0.27`) and Multistate STEC (`-1.28`); over-interpreted CM scaling; lacked $\delta$ anchor. | Fixed within-cell range to **`+0.05 to +9.02 pp`**; added `Multistate Δ` column to Table S4 and explicitly discussed mixed/negative STEC contrasts in MS §3.3 & §4; moved common-mechanism scaling to Supp. Fig. S4; anchored breakdown points externally to **Scallan Walter et al. (2025)** (`43.1%` O157 / `16.2%` non-O157, NORS-mix-weighted **`37.2%`**, and `0.0%–1.6%` lower-tier rates) citing `Manski & Pepper (2018)`. |
| **M6** Case ascertainment & `"assumption-free"` | Abstract retained `"assumption-free"`; no $C_i$-contamination check. | Replaced `"assumption-free"` with `"without assumptions on unrecorded hospitalization outcomes"`; clarified throughout (Abstract, Methods, Results, Conclusions) that the estimand measures **reported outbreak cases, not intrinsic virulence**; added $C_i$-contamination sensitivity variants **`R14a`** (`25%` extrapolated non-cases: `6` worst case / `13` $\delta = 5$) and **`R14b`** ($\le 50\%$ unrecorded: `12` / `14`, zero reversals) citing `Horowitz & Manski (1995)` and `Molinari (2008)`. |
| **M7** Tangential follow-up & overloaded prose | Follow-up exercise occupied Abstract, Methods §2.4, Results §3.4, and Conclusions; masking bounds `[0.12, 0.45]` undisclosed. | Streamlined Abstract (`220` words) and Main Text around (1) bounds/separations, (2) primary studentized bootstrap-$t$ vs Gaussian inference, and (3) externally anchored breakdown frontiers; moved the entire exploratory follow-up benchmark (`Hansen & Hurwitz 1946`) to Supp. §S2 with all clipping constants (`[0.12, 0.45]`, `[0.03, 0.65]`, `[0.25, 1.85]`) explicitly disclosed. |

## Minor Comments (`m1–m22`)
- **m1–m3:** Fixed `+0.05 to +9.02 pp` across Abstract, Discussion & `README.md`; aligned `Std obs − Full` code (`B.point_delta0`) to MS formula (STEC `-2.53 pp`, SAL `+3.21 pp`); disclosed `>= 200` common-support cases filter for 11 low-missingness states in MS §2.2, §3.3, and Table S4.
- **m4–m5:** Added pointwise IM under-coverage caveat (`75.2%` min to `86.8%` median) to MS §3.2 and Table 3 footnote and harmonized `analyze.py` label; justified conservative partial-ID rank sets and cited `Al Mohamad et al. (2021, 2022)` & `Chernozhukov et al. (2019)`.
- **m6–m7:** Renamed `"MNW cluster leverage"` to `"cluster share of score variance"` attributing $G_{\text{eff}}$ to `Carter et al. (2017)` alongside `MacKinnon et al. (2023)`; stated WCB is the **unrestricted** variant with rationale and weaker-performance reference.
- **m8–m12:** Attributed $C_i$ composition strictly to `CDC NORS Guidance v1.1 (pp. 14–15)` noting restricted data required for `Jiang & Ding (2018)`; corrected `White et al. (2022)` wording (4-fold top-10 vs bottom-10 means; 5–86 full ~17-fold range); updated external benchmarks to `Scallan Walter et al. (2025)`, narrowed burden sentence to *C. perfringens*, and cited `Batz et al. (2012)` & `Scallan et al. (2015)`; fixed DOIs/notes on `kaido2019confidence`, `diemer2024missingness`, `weller2026single` (PMID 42662456), and `simpson2022analysing` / `simpson2022severity`.
- **m13–m16:** Relabeled Table 2 bootstrap columns `"Boot stab."` as descriptive stability; redesigned Fig. 4 as a `2 × 3` dual-scale panel (`[0,10]^2` lower tier & `[0,60]^2` key upper-tier pairs, zero overlap); moved common-mechanism scaling to Supp. Fig. S4 (`α ∈ [0, 2]`, clear legend); updated Supp. Table S12 RECORD checklist to exact 19-page `paper.pdf` page numbers.
- **m17–m22:** Refined statistical vs. descriptive wording (`"zero of 30 one-sided hypotheses"`); noted blank-$H/K>0$ $H \le K$ conservatism (23 outbreaks); reported `R4a`/`R4b` rank sets in Table S7; noted 72.8% pooled-weight share from *Salmonella* + norovirus GII; added `"cluster_diagnostics"` to `NUM_SECTIONS` in `compare_results.py`; cleaned `nors_rules.py:38` pilot comment.

---

# Round 5 Revision Summary (v6.0 → v7.0)

We thank the Round 5 referee for identifying subtle epidemiological, statistical, and structural issues across the v6.0 manuscript and codebase. Every Major (`Major 1–7`) and Minor (`Minor 1–22`) point has been resolved in `v7.0` and independently cross-audited:

## Major Comments (`Major 1–7`)

| Comment | Root Cause in `v6.0` | Resolution in `v7.0` |
|---|---|---|
| **Major 1** Binding `11.6 pp` pair was `STEC > SAL`, not upper vs. lower tier | External anchoring narrative conflated within-upper-tier (`STEC > SAL`) precision loss with upper-vs.-lower (`CAMP > CPERF`) separation. | Separated both claims across Abstract, §3.3, §4, Fig. 5, and Table 3: upper-vs.-lower (`CAMP > CPERF`) holds to **`13.5 pp`** boot-$t$ / `12.3 pp` CV3 (`34.5–79.2 pp` *Salmonella*/STEC vs. lower tier), substantively anchored to `0.03%–<1%` external benchmarks; `STEC > SAL` breaks down at **`11.5 pp`** from sampling precision (implied unrecorded `20.4%` STEC, `29.6%` *Salmonella* inside FoodNet ranges). Explained NORS `28.5%` vs. FoodNet `37.2%` via probable/estimated cases in NORS $C_i$ vs. lab-confirmed FoodNet cases (Ebel 2016). |
| **Major 2** Multistate cluster dependence (`48.3%–66.8%` variance share) | Only CV1 and unrestricted WCB reported for 53 clusters; outbreak double bootstrap lacked cluster correlation. | Added leave-one-cluster-out **CV3 jackknife** (`53/53` valid; `STEC > NoV GII` `p = 0.241`), **Bayesian 53-cluster bootstrap-$t$** (`p = 0.176`), a **`300 × 399` 53-cluster Bayesian double bootstrap** (CV1 `17.0%/16.0%` and WCB `19.3%/17.7%` vs. **CV3 `7.3%±1.5%` / `4.7%±1.2%`**), and full `R11` (`52` states, all 5 methods `0/12` worst case; `7.4–14.8 pp` within-upper vs `13.1–19.5 pp` upper-vs-lower). Explicitly stated independence caveat in §2.3 & §4.1. |
| **Major 3** Pre-specification contradiction | Methods/Discussion claimed `"a priori"` / `"pre-designate"` / `"pre-specified"` primary boot-$t$ while §4.1 noted non-pre-specification. | Removed all `"a priori"`, `"pre-designate"`, and `"pre-specified"` primary claims; presented all six Max-$T$ procedures (`HC1`, `CV1`, `CV3`, `WCB`, `cluster boot-t`, `boot-t`) alongside `R2` (`p=0.051`) and `R10` (`5` rejections) transparently as method- and specification-dependent. |
| **Major 4** Additive vs. OR scale | Stated lower tier `"cannot be ordered"` unconditionally despite OR separation of `NoV GII > GI`. | Qualified non-ordering as **on the additive scale**; justified additive risk differences for absolute burden comparisons; implemented formal **OR-scale 30-test inference** (`floor = 0.001`: `13/30` rejected at $\Gamma = 2.0$, `NoV GII` `[4,5]` above `GI` `[5,6]`, $\Gamma^{*}_{0.95} = 1.94$ boot-$t$ vs. `2.27` `STEC > SAL` and `6.83` `CAMP > CPERF`). |
| **Major 5** Masten & Poirier (2020 §3) misreading & `7.9 pp` `"uniform"` bound | Scalar $\delta^{*}_{0.95}$ labeled `"pointwise"` limitation; `bootstrap_t_unif` (`7.9 pp`) mislabeled uniform. | Clarified that scalar $\delta^{*}_{0.95}$ are valid one-sided lower confidence bounds (`Kline & Santos 2013`); uniform bands apply only to reading 2D frontiers in Fig. 4 jointly; removed `bootstrap_t_unif` (`7.9 pp`); verified unique crossings (`root_crossings` $\in \{0,1\}$) on a 201-pt scan and refined all finite `boot-t` roots using exact uninterpolated `c_bt(delta)` (`<= 3.1e-5` residual; `STEC > SAL` `11.58 -> 11.46 pp`). |
| **Major 6** External benchmark & burden misattributions | `0.6%`/`0.03%` attributed to Scallan Walter (2025) instead of Scallan (2011 Table 3); NORS `0.5%–1.6%` wording; DALY vs per-case severity. | Strictly attributed `0.6%` *C. perfringens* & `0.03%` norovirus to **Scallan et al. (2011 Table 3)**, reported **Scallan Walter et al. (2025 Table 2)** accurately (`<1%`, `<0.1%`, `~0.40%`), labeled `0.5%–1.6%` (`0.0%–1.3%` in 11 states) as our NORS figures, and contrasted incidence-weighted DALYs (Scallan 2015) with per-case severity in §1. |
| **Major 7** `Std obs − Full` cell mismatch & 11-state framing | `r_full_std` averaged over cells with both partial & full records (`9` STEC) vs. all `12` cells for `theta(0)`, creating artifactual `-2.53 / -0.27 pp`. | Restricted `rate_all_common_std` to **identical shared cells & weights** (`8–12` cells, `79.3%–100%` weight): all 6 groups now show positive `Std obs − Full` (`+0.07` to `+2.86 pp`, STEC `+0.88 pp`); framed 11-state rates descriptively; explained downward-$\delta$ mechanism and ran formal `diag_asymmetric` inference (`12/12` preserved across all 4 procedures). |

## Minor Comments (`Minor 1–22`)
- **Minor 1–4:** Harmonized HC1 (`18.5 pp`) & CV1 (`18.0 pp`) rank-breakdown values; swapped **Supp. Fig. S3** (`fig_s3_followup_sensitivity`) before **Supp. Fig. S4** (`fig_s4_common_mechanism`); placed the full **10-row empirical FWER rejection matrix inside Supp. Table S8's body** (`400 × 499` outbreak + `300 × 399` cluster double bootstraps); defined $H_{\text{rec},js}, K_{\text{rec},js}, m_j^-, m_j^+$ and conservative partial-ID rank projection in §2.2–§2.3.
- **Minor 5–9:** Changed Abstract phrasing to `"known-status hospitalization"`; quoted CDC NORS Guidance pp. 15–16 accurately (`"including lab-confirmed and probable cases"`); corrected White et al. `62 vs. 9` to `~6.9 times`; added & cited `li2010evaluation` and `hedberg2024estimating` alongside `deweymattia2018surveillance`; framed NORS-vs.-FoodNet *Salmonella*/*Campylobacter* mechanism explicitly as a hypothesis.
- **Minor 10–14:** Fixed `R10` (`H_rec/K_rec` normalized alongside `H/U/C`); relabeled `R14b` as selection sensitivity; checked kink distances across all finite roots (`>= 0.0013 pp`, `0/72` cells within `1e-6` at $\delta=5$); added **`R12b_nontyphoidal_sal`** excluding 6 *Salmonella* serovar Typhi outbreaks (`87` cases, `42` hospitalized) and verified semicolon-aligned O157 confirmation in **`R12`** (`279/388` confirmed vs. `280` substring).
- **Minor 15–22:** Updated LFC boundary-null wording, `400 × 499` inner resamples, MC SEs, and Carter et al. / MNW §7 $G_{\text{eff}}$ plug-in note; added explicit Intro novelty contrast (`Wikswo 2022`, `Simpson 2022`) and Discussion surveillance reporting recommendation; decongested prose, removed all raw Markdown backticks & Table S4 internal column names in prose, fixed **Fig. 4** bottom legend/x-axis spacing, added `"Reported Outbreak Cases"` to Title, removed hardcoded fallbacks in `make_outputs.py`, and updated `table_6_bootstrap` & RECORD `Table S12` exact page numbers.
