"""
One place for the pass / review / fail cost model.

Used by scripts/review_band.py (the report's numbers) and app.py (the
dashboard), so both compute the same thing from the same code. Every business
figure comes from src/config.py; nothing here hard-codes one.

Outcome cost per unit:
    pass    0 if intact, COST_FN_DOP if damaged
    review  COST_REVIEW_DOP either way (the person catches it)
    fail    COST_FAIL_CONFIRM_DOP if damaged; + false_claim if intact
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.scoring import decide  # noqa: E402

UNITS = 1000
OUTCOMES = ("pass", "review", "fail")
SCORES_CSV = ROOT / "outputs" / "test_scores.csv"


def load_scores(path: Path = SCORES_CSV) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [{"filename": r["filename"], "score": float(r["score"]), "label": int(r["label"]),
                 "main_class": r["main_class"], "dup": int(r["dup_in_trainval"])}
                for r in csv.DictReader(f)]


def unit_costs(false_claim: float) -> dict:
    """Cost of each outcome for an intact and for a damaged unit."""
    return {
        "pass": {"intact": 0.0, "damaged": float(config.COST_FN_DOP)},
        "review": {"intact": float(config.COST_REVIEW_DOP), "damaged": float(config.COST_REVIEW_DOP)},
        "fail": {"intact": float(config.COST_FAIL_CONFIRM_DOP + false_claim),
                 "damaged": float(config.COST_FAIL_CONFIRM_DOP)},
    }


def shares(rows: list[dict], t_low: float, t_high: float) -> dict:
    """Share (and count) of positives and of negatives sent to each outcome."""
    counts = {"pos": dict.fromkeys(OUTCOMES, 0), "neg": dict.fromkeys(OUTCOMES, 0)}
    for r in rows:
        counts["pos" if r["label"] else "neg"][decide(r["score"], t_low, t_high)] += 1
    out = {}
    for grp, c in counts.items():
        n = sum(c.values())
        for o in OUTCOMES:
            out[f"{grp}_{o}"] = c[o] / n if n else 0.0
            out[f"{grp}_{o}_n"] = c[o]
    return out


def cost_per_1000(sh: dict, base_rate: float, false_claim: float) -> dict:
    pos = UNITS * base_rate
    neg = UNITS * (1 - base_rate)
    uc = unit_costs(false_claim)
    exp = {
        "fn": pos * sh["pos_pass"],
        "review": pos * sh["pos_review"] + neg * sh["neg_review"],
        "fail_damaged": pos * sh["pos_fail"],
        "fail_intact": neg * sh["neg_fail"],
        "pass_intact": neg * sh["neg_pass"],
    }
    exp["fail"] = exp["fail_damaged"] + exp["fail_intact"]
    cost = (exp["fn"] * uc["pass"]["damaged"]
            + pos * sh["pos_review"] * uc["review"]["damaged"]
            + neg * sh["neg_review"] * uc["review"]["intact"]
            + exp["fail_damaged"] * uc["fail"]["damaged"]
            + exp["fail_intact"] * uc["fail"]["intact"])
    return {"exp_fn_per_1000": exp["fn"], "exp_review_per_1000": exp["review"],
            "exp_fail_per_1000": exp["fail"], "exp_false_claims_per_1000": exp["fail_intact"],
            "exp_pass_intact_per_1000": exp["pass_intact"],
            "cost_dop_per_1000": cost, "cost_eur_per_1000": cost / config.DOP_PER_EUR}


def outcome_table(sh: dict, base_rate: float) -> list[dict]:
    """The 2x3 table: intact / damaged x pass / review / fail, test counts and per 1,000."""
    pos = UNITS * base_rate
    neg = UNITS * (1 - base_rate)
    rows = []
    for grp, label, scale in (("neg", "intact", neg), ("pos", "damaged", pos)):
        row = {"truth": label}
        for o in OUTCOMES:
            row[f"{o}_test_n"] = sh[f"{grp}_{o}_n"]
            row[f"{o}_per_1000"] = round(scale * sh[f"{grp}_{o}"], 3)
        rows.append(row)
    return rows


def review_load(sh: dict, base_rate: float) -> list[dict]:
    out = []
    for name, n_units in config.CONTAINER_SIZES.items():
        pos, neg = n_units * base_rate, n_units * (1 - base_rate)
        review = pos * sh["pos_review"] + neg * sh["neg_review"]
        fail = pos * sh["pos_fail"] + neg * sh["neg_fail"]
        hours = (review * config.MIN_PER_REVIEW + fail * config.MIN_PER_FAIL_CONFIRM) / 60
        days = hours / (config.STAFF * config.HOURS_PER_DAY)
        out.append({
            "container": name, "units": n_units,
            "units_to_review": round(review, 2), "units_failed": round(fail, 2),
            "units_passed": round(n_units - review - fail, 2),
            "expected_missed_damaged": round(pos * sh["pos_pass"], 3),
            "person_hours": round(hours, 2), "days_to_clear": round(days, 3),
            "staff": config.STAFF, "hours_per_day": config.HOURS_PER_DAY,
            "claim_window_days": config.CLAIM_WINDOW_DAYS,
            "fits_claim_window": int(days <= config.CLAIM_WINDOW_DAYS),
        })
    return out


def option_rows(rows: list[dict], base_rate: float, false_claim: float, single_t: float,
                pairs: dict[str, tuple[float, float]], container_units: int) -> list[dict]:
    """do nothing, person checks everything, model alone, model + band for each named pair."""
    pos = UNITS * base_rate

    def row(option, lo, hi, c, note):
        return {"option": option, "t_low": lo, "t_high": hi,
                "exp_fn_per_1000": round(c["exp_fn_per_1000"], 4),
                "exp_review_per_1000": round(c["exp_review_per_1000"], 4),
                "exp_fail_per_1000": round(c["exp_fail_per_1000"], 4),
                "exp_false_claims_per_1000": round(c.get("exp_false_claims_per_1000", 0.0), 4),
                "cost_dop_per_1000": round(c["cost_dop_per_1000"], 2),
                "cost_eur_per_1000": round(c["cost_eur_per_1000"], 2),
                f"cost_dop_per_container_{container_units}": round(c["cost_dop_per_1000"] * container_units / UNITS, 2),
                f"cost_eur_per_container_{container_units}": round(c["cost_eur_per_1000"] * container_units / UNITS, 2),
                "note": note}

    nothing = {"exp_fn_per_1000": pos, "exp_review_per_1000": 0.0, "exp_fail_per_1000": 0.0,
               "cost_dop_per_1000": pos * config.COST_FN_DOP}
    nothing["cost_eur_per_1000"] = nothing["cost_dop_per_1000"] / config.DOP_PER_EUR
    everything = {"exp_fn_per_1000": 0.0, "exp_review_per_1000": float(UNITS), "exp_fail_per_1000": 0.0,
                  "cost_dop_per_1000": UNITS * config.COST_REVIEW_DOP}
    everything["cost_eur_per_1000"] = everything["cost_dop_per_1000"] / config.DOP_PER_EUR
    out = [
        row("do_nothing", 1.0, 1.0, nothing, "every unit released"),
        row("person_checks_everything", 0.0, 0.0, everything,
            f"every unit reviewed at {config.COST_REVIEW_DOP} DOP; catches everything"),
        row("model_alone", single_t, single_t, cost_per_1000(shares(rows, single_t, single_t), base_rate, false_claim),
            "t_low = t_high at sweep.py's single-threshold optimum; no review band; "
            f"fail costs {config.COST_FAIL_CONFIRM_DOP} DOP, plus {false_claim:g} DOP false claim if intact"),
    ]
    for name, (lo, hi) in pairs.items():
        out.append(row(f"model_plus_review_band_{name}", lo, hi,
                       cost_per_1000(shares(rows, lo, hi), base_rate, false_claim),
                       f"pass / review / fail at the {name} pair"))
    return out


def image_level_rates(rows: list[dict], threshold: float) -> dict:
    """TPR and FPR when every image with score >= threshold is flagged, FPR by class."""
    tp = fn = fp = tn = 0
    fp_c = {c: 0 for c in config.NEGATIVE_CLASSES}
    n_c = {c: 0 for c in config.NEGATIVE_CLASSES}
    for r in rows:
        flagged = r["score"] >= threshold
        if r["label"]:
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
            if r["main_class"] in n_c:
                n_c[r["main_class"]] += 1
                fp_c[r["main_class"]] += flagged
    return {"threshold": threshold, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": tp / (tp + fn) if tp + fn else None, "fpr": fp / (fp + tn) if fp + tn else None,
            "fpr_by_class": {c: (fp_c[c] / n_c[c] if n_c[c] else None) for c in n_c},
            "fp_by_class": fp_c, "negatives_by_class": n_c}
