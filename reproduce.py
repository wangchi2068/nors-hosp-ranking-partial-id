#!/usr/bin/env python3
"""Cross-platform one-command reproduction of the NORS partial-identification study.

Runs from the repository root:
    python reproduce.py [--B 2000] [--workers 2] [--render-outputs]

Writes all regenerated outputs into experiment/rerun/ without modifying committed artifacts,
then verifies bit-for-bit equality against experiment/analysis/results.json and all 10 CSV tables.
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
    exp = root / "experiment"
    code = exp / "code"
    out = exp / "rerun"
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "derived").mkdir(parents=True, exist_ok=True)
    (out / "rebuild_check").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    py = sys.executable

    raw_csv = exp / "data" / "raw" / "nors_soda_full_20261125T040830Z.csv"
    raw_meta = exp / "data" / "raw" / "nors_soda_full_20261125T040830Z.json"

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
             "--preserved", str(exp / "analysis" / "results.json"),
             "--rebuilt", str(rebuilt_json),
             "--rebuilt-jsonl", str(out / "logs" / "bootstrap_replicates.jsonl"),
             "--out-json", str(check_json), "--out-md", str(check_md)], cwd=root, env=env)
        chk = json.loads(check_json.read_text(encoding="utf-8"))
        assert chk["all_sections_except_bootstrap_matched"], "Non-bootstrap mismatch!"
        assert chk["bootstrap_matched"], "Bootstrap mismatch!"

        table_json = out / "rebuild_check" / "table_check.json"
        run([py, str(code / "check_tables.py"),
             "--rebuilt", str(rebuilt_json),
             "--reference-tables", str(exp / "data" / "analysis_tables"),
             "--out-json", str(table_json)], cwd=root, env=env)
        tchk = json.loads(table_json.read_text(encoding="utf-8"))
        assert len(tchk["tables_byte_identical"]) == len(tchk["tables"]), "Table byte-identity mismatch!"

    if args.render_outputs:
        sys.path.insert(0, str(code))
        import make_outputs as mo
        r = json.loads(rebuilt_json.read_text(encoding="utf-8"))
        mo.make_tables(r, out / "paper" / "tables")
        mo.make_figures(r, out / "paper" / "figures")

    print("\nReproduction passed: all results and all 10 tables verified bit-for-bit in", out)


if __name__ == "__main__":
    main()
