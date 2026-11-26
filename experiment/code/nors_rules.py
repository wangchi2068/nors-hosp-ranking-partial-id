"""Locked rules for the NORS hospitalization-ranking analysis.

Summary of cohort eligibility, record audit, and stratum harmonization rules:

1. Group mapping (genus/pathotype level): SAL = any "Salmonella*"; STEC = "Escherichia coli,
   Shiga toxin-producing"; CAMP = any "Campylobacter*"; CPERF = "Clostridium perfringens";
   NOV_GI / NOV_GII = "Norovirus Genogroup I" / "Norovirus Genogroup II". Other norovirus
   strings (Norovirus, Norovirus unknown/other, GIV, GIX) are not GI/GII and are excluded.
2. Main eligibility: every etiology entry maps to ONE study group and >= 1 entry is Confirmed.
3. Record audit (0 <= H <= K <= C): K blank -> K_eff = H; H blank & K = 0 -> H = 0 (all C unknown);
   H blank & K > 0 -> all C unknown; both blank -> all C unknown; H > K -> K_eff = H;
   H > C or K > C -> exclude; C blank or <= 0 -> exclude. The delta anchor p_obs uses only
   records with both H and K recorded (H_rec / K_rec).
4. Setting harmonization into Restaurant, Catering/banquet, Private home, Institution, Other;
   multi-setting entries: drop Unknown tokens, one group -> it, >= 2 groups -> Multiple,
   none -> Unknown. Periods 2009-2013, 2014-2018, 2019-2023 (7 x 3 = 21 cells).
"""
from __future__ import annotations

import math

GROUPS = ["SAL", "STEC", "CAMP", "CPERF", "NOV_GI", "NOV_GII"]
GROUP_LABELS = {
    "SAL": "Salmonella",
    "STEC": "STEC",
    "CAMP": "Campylobacter",
    "CPERF": "C. perfringens",
    "NOV_GI": "Norovirus GI",
    "NOV_GII": "Norovirus GII",
}
COMBINED_NOV_LABEL = "Norovirus GI+GII"

NOROVIRUS_UNSPECIFIED = {
    "Norovirus", "Norovirus unknown", "Norovirus other",
    "Norovirus Genogroup IV", "Norovirus Genogroup IX",
}

# Exact species strings used by the user's pilot (reproduces 3,799 / 96 / 3,703).
PILOT_STRINGS = {
    "Salmonella enterica": "SAL",
    "Escherichia coli, Shiga toxin-producing": "STEC",
    "Campylobacter jejuni": "CAMP",
    "Clostridium perfringens": "CPERF",
    "Norovirus Genogroup I": "NOV_GI",
    "Norovirus Genogroup II": "NOV_GII",
}

SETTING_GROUPS = ["Restaurant", "Catering/banquet", "Private home", "Institution", "Other",
                  "Multiple", "Unknown"]
PERIODS = ["2009-2013", "2014-2018", "2019-2023"]

# Raw NORS setting token -> harmonized setting group (None = unknown token, dropped).
# Tokens not listed fall into "Other".
SETTING_MAP = {
    "Restaurant: Sit-down dining": "Restaurant",
    "Restaurant: Fast-food (drive up service or pay at counter)": "Restaurant",
    "Restaurant: Other": "Restaurant",
    "Restaurant: Buffet": "Restaurant",
    "Food truck/street vendor": "Restaurant",
    "Caterer": "Catering/banquet",
    "Banquet Facility (food prepared and served on-site)": "Catering/banquet",
    "Event space": "Catering/banquet",
    "Private home/residence": "Private home",
    "Residence - Single-family home": "Private home",
    "Residence - Multi-unit housing": "Private home",
    "School/college/university": "Institution",
    "Long-term care/nursing home/assisted living facility": "Institution",
    "Correctional/detention facility": "Institution",
    "Hospital": "Institution",
    "Other healthcare facility": "Institution",
    "Child daycare/preschool": "Institution",
    "Camp": "Institution",
    "Office/indoor workplace": "Institution",
    "Soup kitchen/meal center": "Institution",
    "Grocery store/bakery/deli/convenience store": "Other",
    "Festival/fair": "Other",
    "Fairground": "Other",
    "Religious facility": "Other",
    "Farm/dairy/agricultural setting": "Other",
    "Hotel/motel": "Other",
    "Ship/boat": "Other",
    "Park/outdoor area": "Other",
    "Campground": "Other",
    "Commissary/rental kitchen": "Other",
    "Other": "Other",
    "Unknown": None,
}

HEALTHCARE_TOKENS = {
    "Hospital",
    "Long-term care/nursing home/assisted living facility",
    "Other healthcare facility",
}


def is_healthcare_setting(setting: str) -> bool:
    """Return True if any raw NORS setting token denotes a hospital or long-term care facility."""
    return any(t in HEALTHCARE_TOKENS for t in split_tokens(setting))


def split_tokens(value: str) -> list[str]:
    if value is None:
        return []
    return [t.strip() for t in str(value).split(";")]


def map_token(token: str) -> str | None:
    """Genus/pathotype-level study group of one etiology string (None = not a study group)."""
    t = token.strip()
    if t.startswith("Salmonella"):
        return "SAL"
    if t == "Escherichia coli, Shiga toxin-producing":
        return "STEC"
    if t.startswith("Campylobacter"):
        return "CAMP"
    if t == "Clostridium perfringens":
        return "CPERF"
    if t == "Norovirus Genogroup I":
        return "NOV_GI"
    if t == "Norovirus Genogroup II":
        return "NOV_GII"
    return None


def etiology_category(etiology: str, status: str) -> tuple[str, str | None]:
    """Return (flow category, study group or None) for one outbreak.

    Categories (eligibility_flow.json): blank, single_target_confirmed,
    single_target_suspected_or_blank, norovirus_unspecified_confirmed, single_nontarget,
    multi_same_group_confirmed, multi_same_group_unconfirmed, multi_cross_group_or_nontarget.
    """
    if etiology is None or str(etiology).strip() == "":
        return "blank", None
    toks = split_tokens(etiology)
    sts = split_tokens(status)
    groups = [map_token(t) for t in toks]
    if len(toks) == 1:
        if groups[0]:
            cat = "single_target_confirmed" if sts[0] == "Confirmed" else "single_target_suspected_or_blank"
            return cat, groups[0]
        if toks[0] in NOROVIRUS_UNSPECIFIED and sts[0] == "Confirmed":
            return "norovirus_unspecified_confirmed", None
        return "single_nontarget", None
    if all(groups) and len(set(groups)) == 1:
        if "Confirmed" in sts:
            return "multi_same_group_confirmed", groups[0]
        return "multi_same_group_unconfirmed", groups[0]
    return "multi_cross_group_or_nontarget", None


def is_eligible_main(etiology: str, status: str) -> str | None:
    """Main rule: all entries map to one study group and >= 1 entry is Confirmed."""
    cat, g = etiology_category(etiology, status)
    if cat in ("single_target_confirmed", "multi_same_group_confirmed"):
        return g
    return None


def is_ambiguous_etiology(etiology: str, status: str) -> bool:
    """Confirmed-but-not-attributable: Norovirus without GI/GII (single, Confirmed) or a
    cross-group / non-target multi-entry outbreak with >= 1 Confirmed study-group entry."""
    cat, _ = etiology_category(etiology, status)
    if cat == "norovirus_unspecified_confirmed":
        return True
    if cat == "multi_cross_group_or_nontarget":
        toks, sts = split_tokens(etiology), split_tokens(status)
        return any(map_token(t) and s == "Confirmed" for t, s in zip(toks, sts))
    return False


def pilot_group(etiology: str, status: str) -> str | None:
    """Pilot rule: single etiology, Confirmed, exact pilot species string."""
    toks, sts = split_tokens(etiology), split_tokens(status)
    if len(toks) == 1 and toks[0] in PILOT_STRINGS and sts[0] == "Confirmed":
        return PILOT_STRINGS[toks[0]]
    return None


def harmonize_setting(setting: str) -> str:
    groups = set()
    for t in split_tokens(setting):
        if not t:
            continue
        g = SETTING_MAP.get(t, "Other")
        if g:
            groups.add(g)
    if not groups:
        return "Unknown"
    if len(groups) == 1:
        return groups.pop()
    return "Multiple"


def period_of(year: int) -> str | None:
    if 2009 <= year <= 2013:
        return "2009-2013"
    if 2014 <= year <= 2018:
        return "2014-2018"
    if 2019 <= year <= 2023:
        return "2019-2023"
    return None


def _num(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    v = float(s)
    return None if math.isnan(v) else v


AUDIT_HANDLING = {
    "complete_consistent": "kept as recorded",
    "single_missing_K": "kept: K_eff=H, C-H unknown (bounds H/C..1)",
    "single_missing_H_K0": "kept: H=0 implied by H<=K; all C unknown",
    "single_missing_H_Kpos": "kept: H in [0,K] not identified; all C treated as unknown (bounds 0..1)",
    "both_missing": "kept: all C unknown (bounds 0..1)",
    "contradiction_H_gt_K": "kept: K_eff=H (H>K)",
    "contradiction_exceeds_C": "excluded: H>C or K>C (no valid bound)",
    "no_case_count": "excluded: C blank or <= 0",
    "ambiguous_etiology": "excluded: outcome not attributable to one study group (counted)",
}

AUDIT_DETAIL = {
    "single_missing_K": "info_on_hospitalizations missing",
    "single_missing_H_K0": "hospitalizations missing, K=0",
    "single_missing_H_Kpos": "hospitalizations missing, K>0",
    "both_missing": "hospitalizations and info_on_hospitalizations missing",
}


def audit_record(C_raw, H_raw, K_raw) -> dict:
    """Classify one eligible outbreak and return the audited counts.

    Returns dict(category, detail, kept, C, H, K_eff, U, H_rec, K_rec, recorded) where U is the
    number of cases with unrecorded hospitalization status and (H_rec, K_rec) contribute to the
    recorded rate p_obs only when both H and K were recorded.
    """
    C, H, K = _num(C_raw), _num(H_raw), _num(K_raw)
    out = dict(C=C, H=None, K_eff=None, U=None, H_rec=0.0, K_rec=0.0, recorded=False)
    if C is None or C <= 0:
        out.update(category="no_case_count", detail="C blank or <= 0", kept=False)
        return out
    if (H is not None and H > C) or (K is not None and K > C):
        out.update(category="contradiction_exceeds_C", detail="H>C or K>C", kept=False)
        return out
    if H is None and K is None:
        out.update(category="both_missing", H=0.0, K_eff=0.0, U=C, kept=True)
    elif K is None:
        out.update(category="single_missing_K", H=H, K_eff=H, U=C - H, kept=True)
    elif H is None:
        cat = "single_missing_H_K0" if K == 0 else "single_missing_H_Kpos"
        out.update(category=cat, H=0.0, K_eff=0.0, U=C, kept=True)
    elif H > K:
        out.update(category="contradiction_H_gt_K", H=H, K_eff=H, U=C - H, kept=True)
    else:
        out.update(category="complete_consistent", H=H, K_eff=K, U=C - K, kept=True,
                   H_rec=H, K_rec=K, recorded=True)
    if out["category"] == "complete_consistent":
        if K == 0:
            out["detail"] = "no hospitalization info (K=0)"
        elif K == C:
            out["detail"] = "all statuses known"
        else:
            out["detail"] = "partial info"
    else:
        out["detail"] = AUDIT_DETAIL.get(out["category"], out["category"])
    return out
