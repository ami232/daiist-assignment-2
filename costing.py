"""
Cost model shared by train.py (to pick the operating point) and app.py (to show it live).

Unit of decision: one object on the belt (a potato, a stone or a cut piece).
Positive = should be removed from the line, for one of two reasons with different costs:
  defect - a potato with a rot/blemish spot (dataset class 'bad', plus 'cut')
  stone  - a stone travelling with the potatoes
Every figure below is an assumption argued in REPORT.md; change it here and both the
pipeline and the dashboard follow.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Scenario: Dutch packhouse, 3 kg retail bags of table potatoes, camera after destoning/washing.
# Sources for every figure are in REPORT.md ("What each mistake costs").
LABOUR_EUR_PER_H = 47.90  # Eurostat 2025, Netherlands, whole economy, employers with 10+ staff
BAG_CREDIT = 3.85         # Albert Heijn regular price, 3 kg vastkokende aardappelen (refund to the shopper)
GBP_EUR = 1.1665          # HMRC monthly rate, October 2026
COMPLAINT_CHARGE = 22.50 * GBP_EUR  # mid-point of the GBP 0-45 retailers charge suppliers per complaint (GCA, 2015)
SAFETY_CHARGE = 45.00 * GBP_EUR     # top of that range: all retailers charge for product-safety failings
P_COMPLAIN = 0.05         # about 1 in 20 dissatisfied consumers complains (TARP 1986, via Chebat & Davidow 2005)
COSTS = {
    # a missed defect reaches a bag; 1 in 20 such bags comes back as a complaint: refund + retailer charge
    "fn_defect": P_COMPLAIN * (BAG_CREDIT + COMPLAINT_CHARGE),                           # = 1.505
    # a missed stone: same complaint rate, safety-level charge, plus 45 min of QA investigation
    "fn_stone": P_COMPLAIN * (BAG_CREDIT + SAFETY_CHARGE + 0.75 * LABOUR_EUR_PER_H),     # = 4.614
    "fp": 0.200 * (0.34 - 0.04),   # 200 g x (EUR 0.34/kg LNCN wk 37 2026 midpoint - EUR 0.04/kg salvage) = 0.06
    "tp": 0.0,                     # a rejected bad item was never saleable
    "tn": 0.0,
    "review": LABOUR_EUR_PER_H * 3.0 / (3600 * 0.80),  # 3 s per check, paid idle time allocated = 0.0499
}
# A person inspecting potatoes: catches 85% of defects (See 2015, 82 inspectors) and bins 10% of
# good ones (low end of the 10-20% SiftAI/KPM report for human potato inspectors).
HUMAN = {"miss": 0.15, "false_reject": 0.10}
BASE_RATES = {"defect": 0.010, "stone": 0.0005}  # 1.0% defective (test belt: 31/2,953); 1 stone per 2,000
LINE = {"objects_per_hour": 30_000,               # ~6 t/h camera lane at ~200 g per potato
        "hours_per_day": 8,                       # one shift
        "seconds_per_review": 3.0,
        "utilisation": 0.80,
        "reviewers": 1,                           # at most one person on the review station
        "pickers": 4}                             # today's manual picking table (R S Cockerill: 4-8 per line)

# fine steps at the bottom, where the scores of defective potatoes actually sit
THRESHOLDS = np.unique(np.round(np.concatenate([[0.001, 0.002, 0.003, 0.005, 0.0075],
                                                 np.arange(0.01, 1.00, 0.01)]), 4))


def _rates(units: pd.DataFrame, t_low, t_high):
    """Share of each truth group that is auto-rejected (score >= t_high) and sent to review
    (t_low <= score < t_high). Phantom units (reject predictions on nothing) count as
    negatives but are normalised by the number of real good objects."""
    arr = _arrays(units)
    out = {}
    for grp in ["defect", "stone", "good"]:
        s, n = arr[grp]
        out[grp] = ((s >= t_high).sum() / n, ((s >= t_low) & (s < t_high)).sum() / n)
    return out


_CACHE = {}


def _arrays(units: pd.DataFrame) -> dict:
    key = id(units)
    if key not in _CACHE or _CACHE[key][0] is not units:
        real = units[~units.phantom]
        arr = {g: (real.loc[real.truth == g, "score"].to_numpy(), max(int((real.truth == g).sum()), 1))
               for g in ["defect", "stone"]}
        # phantoms are negatives, normalised by the number of real good objects
        arr["good"] = (units.loc[units.truth == "good", "score"].to_numpy(),
                       max(int((real.truth == "good").sum()), 1))
        _CACHE.clear()
        _CACHE[key] = (units, arr)
    return _CACHE[key][1]


def expected_cost(units, t_low, t_high, costs=COSTS, base=BASE_RATES, human=HUMAN):
    """Expected EUR per 1,000 objects for a pass / review / reject band.
    t_low == t_high gives the model-alone policy with no review."""
    r = _rates(units, t_low, t_high)
    pi = {"defect": base["defect"], "stone": base["stone"]}
    pi["good"] = 1 - pi["defect"] - pi["stone"]
    fn = {}
    for grp in ["defect", "stone"]:
        auto, rev = r[grp]
        fn[grp] = (1 - auto - rev) + rev * human["miss"]
    fp = r["good"][0] + r["good"][1] * human["false_reject"]
    review_share = sum(pi[g] * r[g][1] for g in pi)
    cost = (pi["defect"] * fn["defect"] * costs["fn_defect"]
            + pi["stone"] * fn["stone"] * costs["fn_stone"]
            + pi["good"] * fp * costs["fp"]
            + (pi["defect"] * (1 - fn["defect"]) + pi["stone"] * (1 - fn["stone"])) * costs["tp"]
            + review_share * costs["review"])
    return {"cost_per_1000": 1000 * cost, "review_share": review_share,
            "recall_defect": 1 - fn["defect"], "recall_stone": 1 - fn["stone"],
            "false_reject_rate": fp,
            "cost_fn_defect": 1000 * pi["defect"] * fn["defect"] * costs["fn_defect"],
            "cost_fn_stone": 1000 * pi["stone"] * fn["stone"] * costs["fn_stone"],
            "cost_fp": 1000 * pi["good"] * fp * costs["fp"],
            "cost_review": 1000 * review_share * costs["review"]}


def sweep_single(units, **kw) -> pd.DataFrame:
    rows = [{"threshold": t, **expected_cost(units, t, t, **kw)} for t in THRESHOLDS]
    return pd.DataFrame(rows)


def best_band(units, max_review_share=None, **kw) -> dict:
    best = None
    for i, lo in enumerate(THRESHOLDS):
        for hi in THRESHOLDS[i:]:
            c = expected_cost(units, lo, hi, **kw)
            if max_review_share is not None and c["review_share"] > max_review_share:
                continue
            if best is None or c["cost_per_1000"] < best["cost_per_1000"] - 1e-12:
                best = {"t_low": float(lo), "t_high": float(hi), **c}
    return best


def alternatives(units, threshold, band, **kw) -> dict:
    costs = kw.get("costs", COSTS); base = kw.get("base", BASE_RATES); human = kw.get("human", HUMAN)
    pi_g = 1 - base["defect"] - base["stone"]
    nothing = 1000 * (base["defect"] * costs["fn_defect"] + base["stone"] * costs["fn_stone"])
    everyone = 1000 * (base["defect"] * human["miss"] * costs["fn_defect"]
                       + base["stone"] * human["miss"] * costs["fn_stone"]
                       + pi_g * human["false_reject"] * costs["fp"] + costs["review"])
    picking = manual_picking(costs=costs, base=base, human=human)
    return {"do_nothing": nothing,
            # "a person checks everything" as it is done today: pickers scanning the whole belt
            "person_checks_everything": picking["cost_per_1000"],
            "person_checks_everything_detail": picking,
            # the same done object by object at 3 s each, for scale: needs ~31 people at once
            "person_checks_each_object_3s": everyone,
            "person_checks_each_object_3s_staff": staffing(1.0),
            "model_alone": expected_cost(units, threshold, threshold, **kw)["cost_per_1000"],
            "model_plus_review": band["cost_per_1000"]}


def manual_picking(costs=COSTS, base=BASE_RATES, human=HUMAN, line=LINE) -> dict:
    """Today's practice: a picking table where people pull defects and stones off the belt.
    Same human error rates as a reviewer; labour is the pickers' paid hours spread over
    every object, since they watch the whole belt."""
    pi_g = 1 - base["defect"] - base["stone"]
    errors = (base["defect"] * human["miss"] * costs["fn_defect"]
              + base["stone"] * human["miss"] * costs["fn_stone"]
              + pi_g * human["false_reject"] * costs["fp"])
    labour = line["pickers"] * LABOUR_EUR_PER_H / line["objects_per_hour"]
    return {"cost_per_1000": 1000 * (errors + labour), "errors_per_1000": 1000 * errors,
            "labour_per_1000": 1000 * labour,
            "cost_fn_defect": 1000 * base["defect"] * human["miss"] * costs["fn_defect"],
            "cost_fn_stone": 1000 * base["stone"] * human["miss"] * costs["fn_stone"],
            "cost_fp": 1000 * pi_g * human["false_reject"] * costs["fp"]}


def sensitivity(units) -> list:
    """Re-pick the model-alone threshold and the review band when one assumption moves."""
    cases = [("as assumed", {}, {})]
    for key, label in [("fn_defect", "missed defect"), ("fn_stone", "missed stone"), ("fp", "good potato thrown out")]:
        for f in (0.25, 4.0):
            cases.append((f"{label} cost x{f:g}", {key: COSTS[key] * f}, {}))
    for grp in ["defect", "stone"]:
        for f in (0.25, 4.0):
            v = BASE_RATES[grp] * f
            cases.append((f"{grp} base rate x{f:g} ({100 * v:.3g}%)", {}, {grp: v}))
    rows = []
    for label, dc, db in cases:
        costs = dict(COSTS, **dc); base = dict(BASE_RATES, **db)
        sw = sweep_single(units, costs=costs, base=base)
        b = sw.loc[sw.cost_per_1000.idxmin()]
        band = best_band(units, max_review_share=max_review_share(), costs=costs, base=base)
        rows.append({"case": label, "best_threshold": float(b.threshold),
                     "model_alone_cost": float(b.cost_per_1000),
                     "t_low": band["t_low"], "t_high": band["t_high"],
                     "band_cost": float(band["cost_per_1000"]), "review_share": float(band["review_share"])})
    return rows


def max_review_share(line=LINE) -> float:
    """Largest share of objects the available reviewers can inspect."""
    per_hour = line["reviewers"] * 3600 * line["utilisation"] / line["seconds_per_review"]
    return per_hour / line["objects_per_hour"]


def staffing(review_share: float, line=LINE) -> dict:
    per_hour = review_share * line["objects_per_hour"]
    people = per_hour * line["seconds_per_review"] / (3600 * line["utilisation"])
    return {"reviews_per_hour": per_hour, "reviews_per_day": per_hour * line["hours_per_day"],
            "review_hours_per_day": per_hour * line["hours_per_day"] * line["seconds_per_review"] / 3600,
            "people_on_line_at_once": people}


def unit_confusion(units, t_low, t_high) -> pd.DataFrame:
    """Counts on the test belt: truth x (pass, review, reject)."""
    rows = {}
    for grp, sub in [("good", units[units.truth == "good"]),
                     ("defect", units[units.truth == "defect"]),
                     ("stone", units[units.truth == "stone"])]:
        s = sub.score.to_numpy()
        rows[grp] = {"pass": int((s < t_low).sum()),
                     "review": int(((s >= t_low) & (s < t_high)).sum()),
                     "reject": int((s >= t_high).sum())}
    return pd.DataFrame(rows).T[["pass", "review", "reject"]]


def plot_sweep(sweep: pd.DataFrame, chosen: float, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep.threshold, sweep.cost_per_1000, lw=2.2, label="total")
    ax.plot(sweep.threshold, sweep.cost_fn_defect, lw=1, ls="--", label="missed defects")
    ax.plot(sweep.threshold, sweep.cost_fn_stone, lw=1, ls="--", label="missed stones")
    ax.plot(sweep.threshold, sweep.cost_fp, lw=1, ls="--", label="good potatoes thrown out")
    c = float(sweep.loc[(sweep.threshold - chosen).abs().idxmin(), "cost_per_1000"])
    ax.axvline(chosen, color="k", lw=0.8)
    ax.scatter([chosen], [c], color="k", zorder=5)
    ax.annotate(f"t = {chosen:.2f}\nEUR {c:.2f} / 1,000", (chosen, c), xytext=(10, 25),
                textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("reject threshold (model confidence, log scale)")
    ax.set_ylabel("expected EUR per 1,000 objects")
    ax.set_title("Cost against threshold, model deciding alone")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
