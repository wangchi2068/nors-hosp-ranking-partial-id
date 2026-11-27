"""Build the audited cohort, mapping tables, strata counts and audit summaries from the raw
NORS snapshot.

Usage:
  python prepare_data.py --raw data/raw/nors_soda_full_20261125T040830Z.csv \
      --meta data/raw/nors_soda_full_20261125T040830Z.json --out data/derived

Deterministic (no randomness). Outputs (in --out):
  cohort_outbreaks.csv         one row per eligible outbreak (main rule) + pilot-rule rows
  suspected_cohort_outbreaks.csv suspected-etiology study-group outbreaks for sensitivity R8
  excluded_outbreaks_summary.json descriptive profile of excluded suspected and ambiguous outbreaks
  record_audit.csv             one row per audited outbreak (eligible + ambiguous etiology)
  etiology_mapping.csv         raw etiology string -> study group (Food 2009-2023)
  setting_mapping.csv          raw setting token -> harmonized setting group
  strata_counts.csv            group x setting x period sums (main cohort) + support flag, weight
  support_cells.csv            common-support cells (>= 3 outbreaks per group) and weights
  eligibility_flow.json        selection counts
  record_audit_summary.json    audit counts by category x group, variants, pilot reproduction
  pilot_reproduction.csv       pilot counts reproduced
  data_manifest.json           sha256 of the raw snapshot and of every output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

import nors_rules as R
import strata as S

THRESHOLD_MAIN = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(raw: Path) -> pd.DataFrame:
    return pd.read_csv(raw, dtype=str, keep_default_na=False)


def build_cohort(df: pd.DataFrame):
    """Return (food, cohort, audit_rows, flow) for Food 2009-2023."""
    years = pd.to_numeric(df["year"], errors="coerce")
    food = df[(df["primary_mode"] == "Food") & years.between(2009, 2023)].copy()
    food["year"] = food["year"].astype(int)
    cats, rows, audit_rows = [], [], []
    for _, r in food.iterrows():
        cat, grp = R.etiology_category(r["etiology"], r["etiology_status"])
        cats.append(cat)
        main_g = R.is_eligible_main(r["etiology"], r["etiology_status"])
        pilot_g = R.pilot_group(r["etiology"], r["etiology_status"])
        ambiguous = R.is_ambiguous_etiology(r["etiology"], r["etiology_status"])
        if not (main_g or pilot_g or ambiguous):
            continue
        a = R.audit_record(r["illnesses"], r["hospitalizations"], r["info_on_hospitalizations"])
        setting_group = R.harmonize_setting(r["setting"])
        period = R.period_of(int(r["year"]))
        n_entries = len(R.split_tokens(r["etiology"]))
        h_blank = str(r["hospitalizations"]).strip() == ""
        k_blank = str(r["info_on_hospitalizations"]).strip() == ""
        base = OrderedDict(
            outbreak_id=r[":id"], year=int(r["year"]), month=r["month"], state=r["state"],
            etiology=r["etiology"], etiology_status=r["etiology_status"],
            serotype_or_genotype=r.get("serotype_or_genotype", ""),
            n_etiology_entries=n_entries, flow_category=cat,
        )
        if ambiguous:
            audit_rows.append(OrderedDict(
                base, group="none", audit_category="ambiguous_etiology",
                audit_detail="confirmed etiology not attributable to one study group",
                handling=R.AUDIT_HANDLING["ambiguous_etiology"], kept_main=False,
                C=a["C"], H_raw=r["hospitalizations"], K_raw=r["info_on_hospitalizations"]))
            continue
        rec = OrderedDict(
            base,
            group=main_g or pilot_g,
            pilot_group=pilot_g or "",
            in_main=bool(main_g),
            single_etiology=bool(main_g) and n_entries == 1,
            in_pilot_eligible=bool(pilot_g),
            in_pilot_rule=bool(pilot_g) and not (h_blank ^ k_blank),
            setting_raw=r["setting"], setting_group=setting_group, period=period,
            cell=f"{setting_group} | {period}",
            is_healthcare=R.is_healthcare_setting(r["setting"]),
            C_raw=r["illnesses"], H_raw=r["hospitalizations"], K_raw=r["info_on_hospitalizations"],
            audit_category=a["category"], audit_detail=a["detail"],
            handling=R.AUDIT_HANDLING[a["category"]], kept=a["kept"],
            C=a["C"], H=a["H"], K_eff=a["K_eff"], U=a["U"], H_rec=a["H_rec"], K_rec=a["K_rec"],
            recorded=a["recorded"],
        )
        rec["in_complete_only"] = rec["in_main"] and a["category"] == "complete_consistent"
        rows.append(rec)
        if main_g:
            audit_rows.append(OrderedDict(
                base, group=main_g, audit_category=a["category"], audit_detail=a["detail"],
                handling=R.AUDIT_HANDLING[a["category"]], kept_main=a["kept"],
                C=a["C"], H_raw=r["hospitalizations"], K_raw=r["info_on_hospitalizations"]))
    food["flow_category"] = cats
    cohort = pd.DataFrame(rows)
    return food, cohort, pd.DataFrame(audit_rows)


def etiology_mapping(food: pd.DataFrame) -> pd.DataFrame:
    cnt = Counter()
    for e in food["etiology"]:
        for t in R.split_tokens(e):
            if t:
                cnt[t] += 1
    rows = []
    for t, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        g = R.map_token(t)
        if g:
            note = "study group (genus/pathotype rule)"
        elif t in R.NOROVIRUS_UNSPECIFIED:
            note = "excluded: norovirus without GI/GII genogroup"
        else:
            note = "not a study group"
        rows.append(dict(etiology_string=t, n_entries_food_2009_2023=n, group=g or "",
                         pilot_exact_string=t in R.PILOT_STRINGS, note=note))
    rows.append(dict(etiology_string="<multi-entry>", n_entries_food_2009_2023="", group="",
                     pilot_exact_string=False,
                     note="eligible if every entry maps to ONE study group and >= 1 entry is Confirmed; "
                          "entries from >1 group or with a non-study agent -> excluded"))
    return pd.DataFrame(rows)


def setting_mapping(food: pd.DataFrame) -> pd.DataFrame:
    cnt = Counter()
    for s in food["setting"]:
        for t in R.split_tokens(s):
            if t:
                cnt[t] += 1
    rows = []
    for t, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        g = R.SETTING_MAP.get(t, "Other")
        rows.append(dict(setting_token=t, n_food_2009_2023=n, setting_group=g or "(ignored: Unknown)"))
    rows.append(dict(setting_token="<blank>", n_food_2009_2023=int((food["setting"].str.strip() == "").sum()),
                     setting_group="Unknown (if no other token)"))
    rows.append(dict(setting_token="<multi-token>", n_food_2009_2023="",
                     setting_group="drop Unknown tokens; one group -> it; >=2 groups -> Multiple; none -> Unknown"))
    return pd.DataFrame(rows)


def main_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    return cohort[cohort["in_main"] & cohort["kept"]].copy()


def eligibility_flow(raw_rows: int, food: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    main = cohort[cohort["in_main"]]
    cat_counts = food["flow_category"].value_counts()
    order = ["single_target_confirmed", "single_nontarget", "blank", "single_target_suspected_or_blank",
             "norovirus_unspecified_confirmed", "multi_cross_group_or_nontarget",
             "multi_same_group_confirmed", "multi_same_group_unconfirmed"]
    # ties are listed in order of first appearance by year, as in the original summary
    nv = food[food["flow_category"] == "norovirus_unspecified_confirmed"].sort_values("year", kind="mergesort")
    nov = Counter(nv["etiology"].str.strip())
    by = lambda d: {g: int((d["group"] == g).sum()) for g in R.GROUPS}
    kept = main[main["kept"]]
    return OrderedDict(
        snapshot_rows_all_modes_all_years=int(raw_rows),
        food_2009_2023=int(len(food)),
        etiology_category_counts_food_2009_2023={k: int(cat_counts.get(k, 0)) for k in order},
        eligible_main_rule=int(len(main)),
        eligible_by_group_main_rule=by(main),
        eligible_pilot_exact_strings=int(cohort["in_pilot_eligible"].sum()),
        norovirus_unspecified_genogroup_single_confirmed=int(sum(nov.values())),
        norovirus_unspecified_by_string={k: int(v) for k, v in nov.most_common()},
        eligible_single_etiology_by_group=by(main[main["single_etiology"]]),
        eligible_multi_same_group_by_group=by(main[~main["single_etiology"]]),
        retained_main_after_audit=int(len(kept)),
        excluded_main_by_audit=int(len(main) - len(kept)),
    )


def audit_summary(audit: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    cats = sorted(audit["audit_category"].unique())
    categories = OrderedDict()
    for c in cats:
        sub = audit[audit["audit_category"] == c]
        categories[c] = OrderedDict(n_outbreaks=int(len(sub)), cases=float(sub["C"].sum()),
                                    handling=sorted(sub["handling"].unique().tolist()))
    by_cat = OrderedDict()
    for c in cats:
        sub = audit[audit["audit_category"] == c]
        by_cat[c] = OrderedDict((g, int((sub["group"] == g).sum())) for g in sorted(R.GROUPS + ["none"]))
    main = cohort[cohort["in_main"] & cohort["kept"]]
    single = main[main["single_etiology"]]
    complete = main[main["in_complete_only"]]
    pilot = cohort[cohort["in_pilot_rule"] & cohort["kept"]]
    by = lambda d: OrderedDict((g, int((d["group"] == g).sum())) for g in R.GROUPS)
    variants = OrderedDict(
        main=OrderedDict(description="main eligibility (one confirmed study group, genus level, single or same-group "
                                     "multi-entry); keep every record with a valid bound (weaker constraint where a "
                                     "field is missing)", n_outbreaks=int(len(main)), by_group=by(main)),
        single_etiology=OrderedDict(description="single-etiology Confirmed outbreaks only (the step's original "
                                                "default); main audit rule", n_outbreaks=int(len(single)),
                                    by_group=by(single)),
        complete_only=OrderedDict(description="main eligibility; keep only complete_consistent records (any missing "
                                              "H or K excluded)", n_outbreaks=int(len(complete)),
                                  by_group=by(complete)),
        pilot_rule=OrderedDict(description="pilot exact species strings, single etiology, Confirmed; exclude records "
                                           "with exactly one of H/K missing, keep both-missing as all-unknown "
                                           "(reproduces the pilot's 3,799 / 96 / 3,703)", n_outbreaks=int(len(pilot)),
                               by_group=by(pilot)),
    )
    eligible_detail = OrderedDict()
    elig = audit[audit["audit_category"] != "ambiguous_etiology"]
    for c in sorted(elig["audit_category"].unique()):
        sub = elig[elig["audit_category"] == c]
        eligible_detail[c] = OrderedDict((k, int(v)) for k, v in sorted(sub["audit_detail"].value_counts().items()))
    pe = cohort[cohort["in_pilot_eligible"]]
    n_ex = int((pe["H_raw"].str.strip().eq("") ^ pe["K_raw"].str.strip().eq("")).sum())
    pilot_rep = [OrderedDict(eligibility_rule="pilot exact species strings, single etiology",
                             exclusion_rule="exclude exactly-one-of-H/K missing", eligible=int(len(pe)),
                             excluded=n_ex, analysed=int(len(pe) - n_ex),
                             matches_pilot=bool(len(pe) == 3799 and n_ex == 96 and len(pe) - n_ex == 3703))]
    return OrderedDict(
        categories=categories, by_category_x_group=by_cat, variants=variants, eligible_detail=eligible_detail,
        unknown_status_cases_share_main=float(main["U"].sum() / main["C"].sum()),
        pilot_reproduction=pilot_rep,
    )


def strata_counts(main: pd.DataFrame):
    t = S.aggregate(main, R.GROUPS, "setting_period")
    support = S.common_support(t, THRESHOLD_MAIN)
    ts = S.restrict(t, support)
    w = S.weights(ts, "pooled_cases")
    wmap = dict(zip(support, w))
    rows = []
    for i, g in enumerate(R.GROUPS):
        for j, c in enumerate(t.cells):
            s, p = c.split(" | ")
            rows.append(OrderedDict(group=g, setting_group=s, period=p, cell=c, outbreaks=int(t.n[i, j]),
                                    C=t.C[i, j], H=t.H[i, j], U=t.U[i, j], H_rec=t.H_rec[i, j],
                                    K_rec=t.K_rec[i, j], in_common_support=c in wmap,
                                    weight=wmap.get(c, 0.0)))
    cells = pd.DataFrame([OrderedDict(cell=c, weight=wmap[c],
                                      min_outbreaks_any_group=int(t.n[:, t.cells.index(c)].min()),
                                      pooled_cases=float(t.C[:, t.cells.index(c)].sum())) for c in support])
    return pd.DataFrame(rows), cells


def _summarize_excluded_group(rows: list[dict]) -> OrderedDict:
    """Summarize case distribution, top settings, recorded hosp rate, and unrecorded share."""
    n_total = len(rows)
    kept = [r for r in rows if r["kept"]]
    cases = np.array([float(r["C"]) for r in kept], dtype=float)
    set_counts = Counter(r["setting_group"] for r in rows)
    top_set = [
        OrderedDict(
            setting=s,
            outbreaks=int(cnt),
            share_pct=(100.0 * cnt / n_total) if n_total else 0.0,
        )
        for s, cnt in sorted(set_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    ]
    c_sum = float(sum(r["C"] for r in kept))
    u_sum = float(sum(r["U"] for r in kept))
    h_rec = float(sum(r["H_rec"] for r in kept))
    k_rec = float(sum(r["K_rec"] for r in kept))
    return OrderedDict(
        n_outbreaks=int(n_total),
        n_kept_valid_counts=int(len(kept)),
        total_cases=c_sum,
        median_cases=float(np.median(cases)) if len(cases) else None,
        iqr_cases=[float(np.percentile(cases, 25)), float(np.percentile(cases, 75))] if len(cases) else None,
        top_settings=top_set,
        recorded_hosp_rate=(h_rec / k_rec) if k_rec > 0 else None,
        unrecorded_status_share=(u_sum / c_sum) if c_sum > 0 else None,
    )


def build_suspected_and_excluded(food: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, OrderedDict]:
    """Build suspected cohort (R8), confirmed unspecified-norovirus cohort (R7b), and excluded summaries."""
    suspected_rows = []
    unspec_nov_rows = []
    buckets = OrderedDict([
        ("suspected_study_group", []),
        ("norovirus_unspecified_confirmed", []),
        ("multi_cross_group_or_nontarget", []),
        ("multi_cross_with_confirmed_target", []),
    ])
    for _, r in food.iterrows():
        cat, grp = R.etiology_category(r["etiology"], r["etiology_status"])
        if cat not in ("single_target_suspected_or_blank", "multi_same_group_unconfirmed",
                       "norovirus_unspecified_confirmed", "multi_cross_group_or_nontarget"):
            continue
        a = R.audit_record(r["illnesses"], r["hospitalizations"], r["info_on_hospitalizations"])
        setting_group = R.harmonize_setting(r["setting"])
        period = R.period_of(int(r["year"]))
        n_entries = len(R.split_tokens(r["etiology"]))
        is_hc = R.is_healthcare_setting(r["setting"])
        summary_rec = dict(
            setting_group=setting_group,
            kept=bool(a["kept"]),
            C=a["C"] or 0.0,
            U=a["U"] or 0.0,
            H_rec=a["H_rec"] or 0.0,
            K_rec=a["K_rec"] or 0.0,
        )
        row_dict = OrderedDict(
            outbreak_id=r[":id"], year=int(r["year"]), month=r["month"], state=r["state"],
            etiology=r["etiology"], etiology_status=r["etiology_status"],
            serotype_or_genotype=r.get("serotype_or_genotype", ""),
            n_etiology_entries=n_entries, flow_category=cat,
            group=(grp or "NOV_UNSPEC"), pilot_group="", in_main=False,
            single_etiology=(n_entries == 1),
            in_pilot_eligible=False, in_pilot_rule=False,
            setting_raw=r["setting"], setting_group=setting_group, period=period,
            cell=f"{setting_group} | {period}", is_healthcare=is_hc,
            C_raw=r["illnesses"], H_raw=r["hospitalizations"], K_raw=r["info_on_hospitalizations"],
            audit_category=a["category"], audit_detail=a["detail"],
            handling=R.AUDIT_HANDLING[a["category"]], kept=a["kept"],
            C=a["C"], H=a["H"], K_eff=a["K_eff"], U=a["U"], H_rec=a["H_rec"], K_rec=a["K_rec"],
            recorded=a["recorded"],
            in_complete_only=(a["category"] == "complete_consistent"),
        )
        if cat in ("single_target_suspected_or_blank", "multi_same_group_unconfirmed"):
            buckets["suspected_study_group"].append(summary_rec)
            suspected_rows.append(row_dict)
        elif cat == "norovirus_unspecified_confirmed":
            buckets["norovirus_unspecified_confirmed"].append(summary_rec)
            unspec_nov_rows.append(row_dict)
        else:
            buckets["multi_cross_group_or_nontarget"].append(summary_rec)
            if R.is_ambiguous_etiology(r["etiology"], r["etiology_status"]):
                buckets["multi_cross_with_confirmed_target"].append(summary_rec)

    excl_summary = OrderedDict((k, _summarize_excluded_group(v)) for k, v in buckets.items())
    return pd.DataFrame(suspected_rows), pd.DataFrame(unspec_nov_rows), excl_summary


def write_json(obj, path: Path, indent: int = 1):
    # LF line endings everywhere for cross-platform hash consistency (m9)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--meta", type=Path, help="download metadata JSON with the expected sha256")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    raw_sha = sha256(args.raw)
    if args.meta:
        expected = json.load(open(args.meta))["sha256"]
        if raw_sha != expected:
            sys.exit(f"sha256 mismatch: {raw_sha} != {expected}")
        print(f"raw sha256 OK {raw_sha}")
    df = load_raw(args.raw)
    food, cohort, audit = build_cohort(df)
    suspected_cohort, unspec_nov_cohort, excluded_summary = build_suspected_and_excluded(food)
    main_df = main_cohort(cohort)

    cohort.to_csv(args.out / "cohort_outbreaks.csv", index=False, lineterminator="\n")
    suspected_cohort.to_csv(args.out / "suspected_cohort_outbreaks.csv", index=False, lineterminator="\n")
    unspec_nov_cohort.to_csv(args.out / "unspecified_norovirus_outbreaks.csv", index=False, lineterminator="\n")
    audit.to_csv(args.out / "record_audit.csv", index=False, lineterminator="\n")
    etiology_mapping(food).to_csv(args.out / "etiology_mapping.csv", index=False, lineterminator="\n")
    setting_mapping(food).to_csv(args.out / "setting_mapping.csv", index=False, lineterminator="\n")
    sc, cells = strata_counts(main_df)
    sc.to_csv(args.out / "strata_counts.csv", index=False, lineterminator="\n")
    cells.to_csv(args.out / "support_cells.csv", index=False, lineterminator="\n")
    flow = eligibility_flow(len(df), food, cohort)
    summ = audit_summary(audit, cohort)
    write_json(flow, args.out / "eligibility_flow.json")
    write_json(summ, args.out / "record_audit_summary.json")
    write_json(excluded_summary, args.out / "excluded_outbreaks_summary.json")
    pd.DataFrame(summ["pilot_reproduction"]).to_csv(args.out / "pilot_reproduction.csv", index=False,
                                                    lineterminator="\n")
    files = sorted(p for p in args.out.iterdir() if p.is_file() and p.name != "data_manifest.json")
    manifest = OrderedDict(raw_snapshot=str(args.raw.name), raw_sha256=raw_sha, raw_rows=int(len(df)),
                           outputs={p.name: sha256(p) for p in files})
    write_json(manifest, args.out / "data_manifest.json")
    print(f"food 2009-2023: {len(food)}; eligible main: {flow['eligible_main_rule']}; "
          f"support cells: {len(cells)}; pilot: {summ['pilot_reproduction'][0]}")


if __name__ == "__main__":
    main()
