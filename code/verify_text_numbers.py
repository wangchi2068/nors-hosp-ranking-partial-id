"""Verify that key quoted numbers in the LaTeX manuscript match analysis/results.json (m14)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    r = json.loads((ROOT / "analysis" / "results.json").read_text(encoding="utf-8"))
    sec_dir = ROOT / "paper" / "sections"
    corpus = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(sec_dir.glob("*.tex"))
    )
    si = r["stacked_inference"]
    m = r["main"]
    sc = r["small_sample_calibration"]
    cc53 = r["cluster_double_bootstrap"]
    pc = r["parametric_calibration"]
    sa = r["supplementary_audits"]
    uc = sa["unrecorded_concentration"]

    checks: list[tuple[str, bool, str]] = [
        ("Raw NORS count 66,713", "66,713" in corpus, "audit count"),
        ("Eligible count 4,133", "4,133" in corpus, "eligibility_flow"),
        ("Retained common-support 3,176", "3,176" in corpus, "retention"),
        ("Retained cases 72,132", "72,132" in corpus, "retention"),
        (
            "Sample worst-case separated 4/15",
            f"{m['n_determined']['worst_case']} of 15" in corpus
            or f"{m['n_determined']['worst_case']}/15" in corpus,
            str(m["n_determined"]["worst_case"]),
        ),
        (
            "Dual-criterion worst-case 0/30",
            f"{si['n_fwer_dual_worst_sig']} of 30" in corpus
            or f"{si['n_fwer_dual_worst_sig']}/30" in corpus,
            str(si["n_fwer_dual_worst_sig"]),
        ),
        (
            "Dual-criterion delta=5 stepdown 12/30",
            f"{si['n_fwer_dual_delta5_sig']} of 30" in corpus
            or f"{si['n_fwer_dual_delta5_sig']}/30" in corpus,
            str(si["n_fwer_dual_delta5_sig"]),
        ),
        (
            "Dual-criterion delta=0 14/30",
            f"{si['n_fwer_dual_delta0_sig']} of 30" in corpus
            or f"{si['n_fwer_dual_delta0_sig']}/30" in corpus,
            str(si["n_fwer_dual_delta0_sig"]),
        ),
        (
            "Top-10% unrecorded share 88.6%",
            f"{uc['top10_pct_outbreaks_share_of_u_pct']:.1f}%"
            in corpus.replace("\\%", "%"),
            f"{uc['top10_pct_outbreaks_share_of_u_pct']:.1f}%",
        ),
        (
            "Outbreak dual FWER 3.0%",
            f"{100 * sc['fwer_worst']['dual_criterion']:.1f}%"
            in corpus.replace("\\%", "%"),
            f"{100 * sc['fwer_worst']['dual_criterion']:.1f}%",
        ),
        (
            "LogNormal dual FWER 4.7%",
            f"{100 * pc['fwer_worst']['dual_criterion']:.1f}%"
            in corpus.replace("\\%", "%"),
            f"{100 * pc['fwer_worst']['dual_criterion']:.1f}%",
        ),
        (
            "53-cluster dual FWER 13.0%",
            f"{100 * cc53['fwer_worst']['dual_criterion']:.1f}%"
            in corpus.replace("\\%", "%"),
            f"{100 * cc53['fwer_worst']['dual_criterion']:.1f}%",
        ),
        (
            "Zero markdown ** remaining in .tex",
            "**" not in corpus,
            "markdown bold check",
        ),
    ]

    failed = [c for c in checks if not c[1]]
    for label, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {label} ({detail})")
    if failed:
        raise SystemExit(f"verify_text_numbers failed {len(failed)} checks")
    print(f"All {len(checks)} narrative numeric checks passed.")


if __name__ == "__main__":
    main()
