"""
Re-run the costing with your own assumptions, without retraining or re-predicting.

    uv run python recost.py                       # uses the values in costing.py
    uv run python recost.py --fn-defect 2 --fp 0.1 --defect-rate 0.015 --reviewers 3

Reads artifacts/test_units.csv (one reject score per object on the test belt, written by
train.py) and rewrites the cost parts of artifacts/metrics.json, artifacts/cost_sweep.csv and
artifacts/cost_vs_threshold.png, so the dashboard picks up the new operating point.
Without --save it only prints. To make your values the defaults, edit COSTS, HUMAN,
BASE_RATES and LINE at the top of costing.py instead and run with --save.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import costing

ART = Path(__file__).resolve().parent / "artifacts"


def wilson(k: int, n: int, z: float = 1.96) -> list:
    """95% Wilson interval for k successes out of n (same as train.py)."""
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [c - h, c + h]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fn-defect", type=float, help="EUR per missed defective potato")
    p.add_argument("--fn-stone", type=float, help="EUR per missed stone")
    p.add_argument("--fp", type=float, help="EUR per good potato rejected")
    p.add_argument("--review", type=float, help="EUR per human review")
    p.add_argument("--defect-rate", type=float, help="share of objects that are defective, e.g. 0.01")
    p.add_argument("--stone-rate", type=float, help="share of objects that are stones, e.g. 0.002")
    p.add_argument("--human-miss", type=float, help="share of real defects/stones a reviewer misses")
    p.add_argument("--human-false-reject", type=float, help="share of good potatoes a reviewer bins")
    p.add_argument("--objects-per-hour", type=float)
    p.add_argument("--hours-per-day", type=float)
    p.add_argument("--seconds-per-review", type=float)
    p.add_argument("--utilisation", type=float)
    p.add_argument("--reviewers", type=float, help="people available for the review belt")
    p.add_argument("--save", action="store_true", help="write the results into artifacts/")
    a = p.parse_args()

    for key, val in [("fn_defect", a.fn_defect), ("fn_stone", a.fn_stone), ("fp", a.fp), ("review", a.review)]:
        if val is not None:
            costing.COSTS[key] = val
    for key, val in [("defect", a.defect_rate), ("stone", a.stone_rate)]:
        if val is not None:
            costing.BASE_RATES[key] = val
    for key, val in [("miss", a.human_miss), ("false_reject", a.human_false_reject)]:
        if val is not None:
            costing.HUMAN[key] = val
    for key, val in [("objects_per_hour", a.objects_per_hour), ("hours_per_day", a.hours_per_day),
                     ("seconds_per_review", a.seconds_per_review), ("utilisation", a.utilisation),
                     ("reviewers", a.reviewers)]:
        if val is not None:
            costing.LINE[key] = val

    units = pd.read_csv(ART / "test_units.csv")
    sweep = costing.sweep_single(units)
    best = sweep.loc[sweep.cost_per_1000.idxmin()]
    cap = costing.max_review_share()
    band = costing.best_band(units, max_review_share=cap)
    free = costing.best_band(units)
    alts = costing.alternatives(units, float(best.threshold), band)
    staff = costing.staffing(band["review_share"])
    per_day = costing.LINE["objects_per_hour"] * costing.LINE["hours_per_day"] / 1000

    print(f"inputs: costs {costing.COSTS} | base rates {costing.BASE_RATES} | human {costing.HUMAN}")
    print(f"        line {costing.LINE}")
    print(f"\nmodel alone : t = {best.threshold:g}  ->  EUR {best.cost_per_1000:.2f} per 1,000 "
          f"(defect recall {best.recall_defect:.1%}, stone recall {best.recall_stone:.1%}, "
          f"good rejected {best.false_reject_rate:.2%})")
    print(f"review band : pass < {band['t_low']:g}, review to {band['t_high']:g}, reject >= {band['t_high']:g}"
          f"  ->  EUR {band['cost_per_1000']:.2f} per 1,000, {band['review_share']:.2%} reviewed "
          f"(cap {cap:.2%}), {staff['people_on_line_at_once']:.2f} people at once, "
          f"{staff['reviews_per_day']:,.0f} reviews/day")
    print(f"uncapped    : {free['t_low']:g} to {free['t_high']:g}  ->  EUR {free['cost_per_1000']:.2f}, "
          f"{free['review_share']:.2%} reviewed, "
          f"{costing.staffing(free['review_share'])['people_on_line_at_once']:.2f} people")
    print("\npolicy                      EUR/1,000    EUR/day")
    for name, key in [("do nothing", "do_nothing"), ("manual picking table", "person_checks_everything"),
                      ("model alone", "model_alone"), ("model + review band", "model_plus_review")]:
        print(f"  {name:26s} {alts[key]:9.2f} {alts[key] * per_day:10,.0f}")

    if a.save:
        m = json.loads((ART / "metrics.json").read_text())
        real = units[~units.phantom]
        m.update(costs=costing.COSTS, human=costing.HUMAN, base_rates=costing.BASE_RATES, line=costing.LINE,
                 band_detail=band, band_unconstrained=free, max_review_share=cap,
                 staffing=staff, staffing_unconstrained=costing.staffing(free["review_share"]),
                 alternatives=alts, sensitivity=costing.sensitivity(units))
        op = m["operating_point"]
        op.update(threshold=float(best.threshold), cost_per_1000=float(best.cost_per_1000),
                  t_low=band["t_low"], t_high=band["t_high"], band_cost_per_1000=band["cost_per_1000"],
                  review_share=band["review_share"])
        t = op["threshold"]
        for grp in ["defect", "stone"]:
            s = real.loc[real.truth == grp, "score"]
            k, n = int((s >= t).sum()), len(s)
            op[f"recall_{grp}_test"] = k / n
            op[f"recall_{grp}_ci95"] = wilson(k, n)
        n_good = int((real.truth == "good").sum())
        op["false_rejects_per_good_test"] = float((units[units.truth == "good"].score >= t).sum() / n_good)
        (ART / "metrics.json").write_text(json.dumps(m, indent=2))
        sweep.to_csv(ART / "cost_sweep.csv", index=False)
        costing.plot_sweep(sweep, t, ART / "cost_vs_threshold.png")
        print("\nsaved to artifacts/ (note: the dashboard's cost boxes start from costing.py's values)")


if __name__ == "__main__":
    main()
