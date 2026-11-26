"""Render the analysis tables from a rebuilt results JSON with the delivered make_outputs.py and
compare them cell by cell with the delivered tables (rebuilt 2026-11-25).

Usage:
  python check_tables.py --rebuilt rebuild_check/results_rebuilt.json \
      --reference-tables data/analysis_tables --out-json rebuild_check/table_check.json

Columns whose header contains "Bootstrap" (and all value columns of the bootstrap table) are reported separately (the bootstrap is not
bit-reproducible; see rebuild_check.md). Writes nothing outside a temporary directory except --out-json.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuilt", required=True, type=Path)
    ap.add_argument("--reference-tables", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(HERE))
    import make_outputs  # delivered program: tables/figures from results.json only
    r = json.loads(args.rebuilt.read_text(encoding="utf-8"))
    report = OrderedDict(rebuilt=args.rebuilt.name, reference="data/analysis_tables (delivered)", tables=OrderedDict())
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp) / "tables"
        make_outputs.make_tables(r, tdir)
        for ref in sorted(args.reference_tables.glob("table_*.csv")):
            new = tdir / ref.name
            a = pd.read_csv(ref, dtype=str, keep_default_na=False)
            b = pd.read_csv(new, dtype=str, keep_default_na=False)
            entry = OrderedDict(
                byte_identical=ref.read_bytes().replace(b"\r\n", b"\n") == new.read_bytes().replace(b"\r\n", b"\n"),
                same_shape=a.shape == b.shape, same_header=list(a.columns) == list(b.columns))
            if entry["same_shape"] and entry["same_header"]:
                if "bootstrap" in ref.name:  # the bootstrap table: every column except the row label
                    boot_cols = list(a.columns[1:])
                else:
                    boot_cols = [c for c in a.columns if "bootstrap" in c.lower()]
                other = [c for c in a.columns if c not in boot_cols]
                entry["non_bootstrap_cells"] = int(a[other].size)
                entry["non_bootstrap_cells_differing"] = int((a[other] != b[other]).values.sum())
                entry["bootstrap_columns"] = boot_cols
                entry["bootstrap_cells"] = int(a[boot_cols].size)
                entry["bootstrap_cells_differing"] = int((a[boot_cols] != b[boot_cols]).values.sum())
                diffs = []
                for c in boot_cols:
                    for i in range(len(a)):
                        if a.at[i, c] != b.at[i, c]:
                            diffs.append(OrderedDict(row=a.iat[i, 0], column=c, delivered=a.at[i, c], rebuilt=b.at[i, c]))
                entry["bootstrap_differences"] = diffs
            report["tables"][ref.name] = entry
    t = report["tables"].values()
    report["all_non_bootstrap_cells_identical"] = all(e.get("non_bootstrap_cells_differing", 1) == 0 for e in t)
    report["tables_byte_identical"] = [k for k, e in report["tables"].items() if e["byte_identical"]]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("byte-identical:", report["tables_byte_identical"], "| all non-bootstrap cells identical:",
          report["all_non_bootstrap_cells_identical"])


if __name__ == "__main__":
    main()
