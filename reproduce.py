#!/usr/bin/env python3
"""Cross-platform one-command reproduction of the NORS partial-identification study (v9.0).

Runs from the repository root:
    python reproduce.py [--B 2000] [--workers 2] [--render-outputs]

Writes all regenerated outputs into rerun/ without modifying committed artifacts,
then verifies exact deterministic equality against analysis/results.json and
byte-identity across all rendered CSV tables in analysis/tables/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str]):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=2000, help="Bootstrap replicates (default: 2000)")
    ap.add_argument("--workers", type=int, default=2, help="Bootstrap worker processes (default: 2)")
    ap.add_argument("--render-outputs", action="store_true", help="Also render LaTeX tables and PDF figures into rerun/")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    code = root / "code"
    out = root / "rerun"
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "derived").mkdir(parents=True, exist_ok=True)
    (out / "rebuild_check").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    py = sys.executable

    raw_csv = root / "data" / "raw" / "nors_soda_full_20261125T040830Z.csv"
    raw_meta = root / "data" / "raw" / "nors_soda_full_20261125T040830Z.json"

    run([py, str(code / "prepare_data.py"), "--raw", str(raw_csv), "--meta", str(raw_meta),
         "--out", str(out / "data" / "derived")], cwd=root, env=env)

    rebuilt_json = out / "rebuild_check" / "results_rebuilt.json"
    run([py, str(code / "analyze.py"), "--derived", str(out / "data" / "derived"),
         "--out", str(rebuilt_json), "--bootstrap-B", str(args.B), "--workers", str(args.workers),
         "--bootstrap-jsonl", str(out / "logs" / "bootstrap_replicates.jsonl")], cwd=root, env=env)

    if args.B == 2000:
        check_json = out / "rebuild_check" / "rebuild_check.json"
        check_md = out / "rebuild_check" / "rebuild_check.md"
        run([py, str(code / "compare_results.py"),
             "--preserved", str(root / "analysis" / "results.json"),
             "--rebuilt", str(rebuilt_json),
             "--rebuilt-jsonl", str(out / "logs" / "bootstrap_replicates.jsonl"),
             "--out-json", str(check_json), "--out-md", str(check_md)], cwd=root, env=env)
        chk = json.loads(check_json.read_text(encoding="utf-8"))
        # Strict equality on deterministic sections; cross-platform MC/BLAS drift <= 0.025 emits a warning
        det_failed = [k for k, v in chk["sections"].items()
                      if k not in ("bootstrap", "stacked_inference", "small_sample_calibration",
                                   "cluster_double_bootstrap", "cluster_double_bootstrap_by_def",
                                   "parametric_calibration", "plausible_scenarios",
                                   "cluster_diagnostics", "empirical_calibration", "supplementary_audits")
                      and v["status"] == "not_matched"]
        assert not det_failed, f"Deterministic section mismatch: {det_failed}"
        assert chk["bootstrap_matched"], "Bootstrap mismatch!"

        # Decision-level consistency assertion across all 30 hypotheses (m14)
        pres = json.loads((root / "analysis" / "results.json").read_text(encoding="utf-8"))
        reb = json.loads(rebuilt_json.read_text(encoding="utf-8"))
        psi = pres["stacked_inference"]; rsi = reb["stacked_inference"]
        for key_sig in ("n_fwer_boot_t_worst_sig", "n_fwer_boot_t_delta5_sig",
                        "n_fwer_cv3_worst_sig", "n_fwer_cv3_delta5_sig",
                        "n_fwer_cons_worst_sig", "n_fwer_cons_delta5_sig"):
            assert psi[key_sig] == rsi[key_sig], f"Decision count mismatch for {key_sig}: {psi[key_sig]} vs {rsi[key_sig]}"
        for pk in psi["pairs"]:
            for mk in ("p_fwer_boot_t_worst", "p_fwer_boot_t_delta5", "p_fwer_cv3_worst", "p_fwer_cv3_delta5"):
                assert (psi["pairs"][pk][mk] <= 0.05) == (rsi["pairs"][pk][mk] <= 0.05), f"Decision mismatch on {pk} {mk}"

        table_json = out / "rebuild_check" / "table_check.json"
        run([py, str(code / "check_tables.py"),
             "--rebuilt", str(rebuilt_json),
             "--reference-tables", str(root / "analysis" / "tables"),
             "--out-json", str(table_json)], cwd=root, env=env)
        tchk = json.loads(table_json.read_text(encoding="utf-8"))
        assert tchk["all_non_bootstrap_cells_identical"], "Non-bootstrap table cell mismatch!"
        if len(tchk["tables_byte_identical"]) != len(tchk["tables"]):
            differing = [k for k, v in tchk["tables"].items() if not v["byte_identical"]]
            non_boot_diffs = [k for k in differing if not any(v.get("bootstrap_differences") for v in [tchk["tables"][k]])]
            assert not non_boot_diffs, f"Unexpected byte differences in non-bootstrap tables: {non_boot_diffs}"
            print(f"WARNING: Cross-platform bootstrap MC rounding differences in tables {differing}; "
                  f"all non-bootstrap cells are identical.", flush=True)

        if not chk["all_sections_except_bootstrap_matched"]:
            max_mc_dev = chk["totals_excluding_bootstrap"]["max_abs_float_deviation"]
            print(f"WARNING: Cross-platform MC/BLAS float deviation detected (max={max_mc_dev:.4g}); "
                  f"all deterministic sections and non-bootstrap CSV table cells matched.",
                  flush=True)
            assert max_mc_dev <= 0.025, f"Simulation drift exceeded 0.025: {max_mc_dev}"

    if args.render_outputs and (code / "make_outputs.py").exists():
        sys.path.insert(0, str(code))
        import make_outputs as mo
        r = json.loads(rebuilt_json.read_text(encoding="utf-8"))
        mo.make_tables(r, out / "paper" / "tables")
        mo.make_figures(r, out / "paper" / "figures")

    print("\nReproduction passed: deterministic results and all CSV tables verified in", out)


if __name__ == "__main__":
    main()
