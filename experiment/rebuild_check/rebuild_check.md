# Rebuild check: rebuilt analysis vs preserved results

- Preserved: `results.json` (sha256 `372463400091...`)
- Rebuilt: `results_rebuilt.json` (sha256 `75fc826541dc...`)
- Rule: integers, booleans, strings and nulls identical; floats within an absolute tolerance of 1e-09; run metadata (timestamp, spec hash, wall clock, version, hardware, cohort file hash) not counted.
- Everything except the bootstrap: 18,135 result leaves; bit-identical 18,135; within tolerance 0; not matched 0; largest float deviation 0.
- Including the bootstrap block: 18,654 leaves; not matched 0 (all in the bootstrap block, which uses a different random-stream order; see the Monte Carlo comparison below).

## Bootstrap (Monte Carlo comparison)

- Replicates: 2000 in both runs. Exact leaf match is not expected unless the random stream is consumed in the same order; the tolerance is statistical: every verdict frequency within 3 Monte Carlo standard errors of the difference of two independent runs, every bound percentile within 0.25 bootstrap SD.
- Verdict frequencies: 161 compared; largest absolute difference 0.0000 (z = 0.00); beyond tolerance: 0.
- Bound percentiles: 48 compared; largest difference 0.000 percentage points (0.000 bootstrap SD); beyond tolerance: 0.
- delta* percentiles: infinite/finite status identical: True; largest finite difference 0.00 points (reported, not tested).
- Supported orderings reproduced: preserved 91.6-100.0%, rebuilt 91.6-100.0%. Undetermined orderings staying undetermined: preserved 75.2-100.0%, rebuilt 75.2-100.0%. Modal number of orderings supported without assumption: preserved 4, rebuilt 4; at delta = 5: preserved 12, rebuilt 12.
- Within Monte Carlo tolerance: True.
- Supplementary distribution-free check (added after the comparison above; not the pre-stated criterion): 48 of 48 preserved bound percentiles lie within the rebuilt replicates' order-statistic interval (rank +/- 3 x sqrt(2) x binomial SD).

| Ordering | Frequency | Preserved | Rebuilt | Abs. diff. | z |
|---|---|---:|---:|---:|---:|
| SAL|STEC | share_worst_main_verdict | 0.9940 | 0.9940 | 0.0000 | 0.00 |
| SAL|STEC | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|STEC | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|CAMP | share_worst_main_verdict | 0.9580 | 0.9580 | 0.0000 | 0.00 |
| SAL|CAMP | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|CAMP | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|CPERF | share_worst_main_verdict | 0.7520 | 0.7520 | 0.0000 | 0.00 |
| SAL|CPERF | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|CPERF | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|NOV_GI | share_worst_main_verdict | 0.8545 | 0.8545 | 0.0000 | 0.00 |
| SAL|NOV_GI | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|NOV_GI | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|NOV_GII | share_worst_main_verdict | 0.8575 | 0.8575 | 0.0000 | 0.00 |
| SAL|NOV_GII | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| SAL|NOV_GII | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| STEC|CAMP | share_worst_main_verdict | 0.9705 | 0.9705 | 0.0000 | 0.00 |
| STEC|CAMP | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| STEC|CAMP | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| STEC|CPERF | share_worst_main_verdict | 0.9160 | 0.9160 | 0.0000 | 0.00 |
| STEC|CPERF | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| STEC|CPERF | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| STEC|NOV_GI | share_worst_main_verdict | 0.9360 | 0.9360 | 0.0000 | 0.00 |
| STEC|NOV_GI | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| STEC|NOV_GI | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| STEC|NOV_GII | share_worst_main_verdict | 0.9995 | 0.9995 | 0.0000 | 0.00 |
| STEC|NOV_GII | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| STEC|NOV_GII | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CAMP|CPERF | share_worst_main_verdict | 0.9815 | 0.9815 | 0.0000 | 0.00 |
| CAMP|CPERF | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CAMP|CPERF | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CAMP|NOV_GI | share_worst_main_verdict | 0.9985 | 0.9985 | 0.0000 | 0.00 |
| CAMP|NOV_GI | share_delta5_main_verdict | 0.9840 | 0.9840 | 0.0000 | 0.00 |
| CAMP|NOV_GI | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CAMP|NOV_GII | share_worst_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CAMP|NOV_GII | share_delta5_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CAMP|NOV_GII | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CPERF|NOV_GI | share_worst_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CPERF|NOV_GI | share_delta5_main_verdict | 0.9995 | 0.9995 | 0.0000 | 0.00 |
| CPERF|NOV_GI | share_same_delta0_direction | 0.6020 | 0.6020 | 0.0000 | 0.00 |
| CPERF|NOV_GII | share_worst_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| CPERF|NOV_GII | share_delta5_main_verdict | 0.6695 | 0.6695 | 0.0000 | 0.00 |
| CPERF|NOV_GII | share_same_delta0_direction | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| NOV_GI|NOV_GII | share_worst_main_verdict | 1.0000 | 1.0000 | 0.0000 | 0.00 |
| NOV_GI|NOV_GII | share_delta5_main_verdict | 0.6480 | 0.6480 | 0.0000 | 0.00 |
| NOV_GI|NOV_GII | share_same_delta0_direction | 0.9560 | 0.9560 | 0.0000 | 0.00 |

| Section | Leaves | Exact | Within tol. | Not matched | Max abs. float dev. | Status |
|---|---:|---:|---:|---:|---:|---|
| task | 1 | 1 | 0 | 0 | 0 | exact |
| task_slug | 1 | 1 | 0 | 0 | 0 | exact |
| dataset | 1 | 1 | 0 | 0 | 0 | exact |
| metrics | 5 | 5 | 0 | 0 | 0 | exact |
| metric_definitions | 4 | 4 | 0 | 0 | 0 | exact |
| seeds | 1 | 1 | 0 | 0 | 0 | exact |
| statistics | 2 | 2 | 0 | 0 | 0 | exact |
| simulated | 1 | 1 | 0 | 0 | 0 | exact |
| provenance | 6 | 6 | 0 | 0 | 0 | exact |
| summary | 90 | 90 | 0 | 0 | 0 | exact |
| groups | 6 | 6 | 0 | 0 | 0 | exact |
| delta_grid_pp | 5 | 5 | 0 | 0 | 0 | exact |
| main | 3504 | 3504 | 0 | 0 | 0 | exact |
| delta_fine | 3216 | 3216 | 0 | 0 | 0 | exact |
| robustness.R1a_crude_all | 512 | 512 | 0 | 0 | 0 | exact |
| robustness.R1b_crude_support | 512 | 512 | 0 | 0 | 0 | exact |
| robustness.R2_period_only | 516 | 516 | 0 | 0 | 0 | exact |
| robustness.R3a_single_etiology | 533 | 533 | 0 | 0 | 0 | exact |
| robustness.R3b_complete_only | 533 | 533 | 0 | 0 | 0 | exact |
| robustness.R3c_pilot_rule | 531 | 531 | 0 | 0 | 0 | exact |
| robustness.R4a_threshold_2 | 539 | 539 | 0 | 0 | 0 | exact |
| robustness.R4b_threshold_5 | 521 | 521 | 0 | 0 | 0 | exact |
| robustness.R5_named_settings | 530 | 530 | 0 | 0 | 0 | exact |
| robustness.R6a_w_group_balanced | 533 | 533 | 0 | 0 | 0 | exact |
| robustness.R6b_w_pooled_outbreaks | 533 | 533 | 0 | 0 | 0 | exact |
| robustness.R6c_w_uniform | 533 | 533 | 0 | 0 | 0 | exact |
| robustness.R7_norovirus_combined | 300 | 300 | 0 | 0 | 0 | exact |
| robustness.R8_include_suspected | 541 | 541 | 0 | 0 | 0 | exact |
| robustness.R9_excl_healthcare | 533 | 533 | 0 | 0 | 0 | exact |
| robustness.R10_outbreak_weighted | 534 | 534 | 0 | 0 | 0 | exact |
| retention | 165 | 165 | 0 | 0 | 0 | exact |
| width_decomposition | 221 | 221 | 0 | 0 | 0 | exact |
| bootstrap | 519 | 519 | 0 | 0 | 0 | exact |
| audit | 212 | 212 | 0 | 0 | 0 | exact |
| pilot | 125 | 125 | 0 | 0 | 0 | exact |
| stacked_inference | 779 | 779 | 0 | 0 | 0 | exact |
| cluster_diagnostics | 563 | 563 | 0 | 0 | 0 | exact |
| small_sample_calibration | 63 | 63 | 0 | 0 | 0 | exact |
| empirical_calibration | 278 | 278 | 0 | 0 | 0 | exact |
| followup_masking | 650 | 650 | 0 | 0 | 0 | exact |
| notes | 1 | 1 | 0 | 0 | 0 | exact |
| estimates_note | 1 | 1 | 0 | 0 | 0 | exact |

## Run metadata (expected to differ, not counted)

- `provenance.timestamp`: preserved "2026-11-26T01:21:15Z", rebuilt "2026-11-26T01:29:06Z"
- `provenance.wall_clock_seconds`: preserved 50.8, rebuilt 36.3
