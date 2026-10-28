"""
Single-threshold cost sweep on the test split, rescaled to production prevalence.

Reads outputs/test_scores.csv (image scores and labels from src.evaluate) and
every business figure from src/config.py. For each threshold t in 0.00..1.00
step 0.01 an image is flagged when score >= t. TPR and FPR on the test split
are rescaled to 1,000 received units at a given base rate:

    positives = 1000 * base_rate          negatives = 1000 * (1 - base_rate)
    FN = positives * (1 - TPR)            FP = negatives * FPR
    cost per 1,000 = FN * COST_FN_DOP + FP * COST_FP_DOP     (DOP, and EUR)

Done for BASE_RATE and every value in BASE_RATE_SWEEP, on all 120 test
images and, as a check, on the 114 without a near-duplicate in train/valid.

Writes to outputs/:
    sweep.csv                       one row per subset x base rate x threshold
    sweep_summary.json              minimum, flat band, break-even threshold
    cost_curve.png                  cost vs threshold at BASE_RATE (band shaded)
                                    and minimum-cost threshold vs base rate
    options_single_threshold.csv    the five single-threshold options at BASE_RATE
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.scoring import decide  # noqa: E402

OUT_DIR = ROOT / "outputs"
SCORES_CSV = OUT_DIR / "test_scores.csv"
UNITS = 1000
THRESHOLDS = [float(round(float(t), 2)) for t in np.arange(0.0, 1.0 + 1e-9, 0.01)]


def load_scores() -> list[dict]:
    rows = []
    with SCORES_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "filename": r["filename"],
                "score": float(r["score"]),
                "label": int(r["label"]),
                "main_class": r["main_class"],
                "dup": int(r["dup_in_trainval"]),
            })
    return rows


def rates(rows: list[dict], t: float) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        flagged = decide(r["score"], t, t) == "fail"
        if r["label"]:
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
    pos, neg = tp + fn, fp + tn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": tp / pos if pos else 0.0, "fpr": fp / neg if neg else 0.0}


def cost_row(subset: str, base_rate: float, t: float, r: dict) -> dict:
    positives = UNITS * base_rate
    negatives = UNITS * (1 - base_rate)
    exp_fn = positives * (1 - r["tpr"])
    exp_fp = negatives * r["fpr"]
    cost = exp_fn * config.COST_FN_DOP + exp_fp * config.COST_FP_DOP
    return {
        "subset": subset, "base_rate": base_rate, "threshold": t,
        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
        "tpr": round(r["tpr"], 6), "fpr": round(r["fpr"], 6),
        "exp_fn_per_1000": round(exp_fn, 4), "exp_fp_per_1000": round(exp_fp, 4),
        "cost_dop_per_1000": round(cost, 2),
        "cost_eur_per_1000": round(cost / config.DOP_PER_EUR, 2),
    }


def base_rates() -> list[float]:
    seen, out = set(), []
    for b in [config.BASE_RATE] + list(config.BASE_RATE_SWEEP):
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def flat_band(curve: list[dict], tolerance: float) -> dict:
    """Thresholds whose cost is within `tolerance` of the minimum."""
    costs = [c["cost_dop_per_1000"] for c in curve]
    cmin = min(costs)
    i_min = costs.index(cmin)                      # lowest threshold at the minimum
    within = [c["threshold"] for c in curve if c["cost_dop_per_1000"] <= cmin + tolerance]
    return {
        "min_cost_dop_per_1000": cmin,
        "min_cost_eur_per_1000": round(cmin / config.DOP_PER_EUR, 2),
        "threshold_at_min": curve[i_min]["threshold"],
        "thresholds_at_min": [c["threshold"] for c in curve if c["cost_dop_per_1000"] == cmin],
        "tolerance_dop": round(tolerance, 2),
        "band_low": min(within), "band_high": max(within), "band_n_thresholds": len(within),
        "tpr_at_min": curve[i_min]["tpr"], "fpr_at_min": curve[i_min]["fpr"],
    }


def options(curve_all: list[dict], band: dict, base_rate: float, positives_in_test: int) -> list[dict]:
    positives = UNITS * base_rate
    negatives = UNITS * (1 - base_rate)
    eur = config.DOP_PER_EUR

    def row(option, threshold, tpr, fpr, exp_fn, exp_fp, exp_review, cost, note):
        return {"option": option, "threshold": threshold, "tpr": tpr, "fpr": fpr,
                "exp_fn_per_1000": round(exp_fn, 4), "exp_fp_per_1000": round(exp_fp, 4),
                "exp_review_per_1000": round(exp_review, 4),
                "cost_dop_per_1000": round(cost, 2), "cost_eur_per_1000": round(cost / eur, 2),
                "note": note}

    do_nothing = positives * config.COST_FN_DOP
    everyone = UNITS * config.COST_REVIEW_DOP
    at_min = next(c for c in curve_all if c["threshold"] == band["threshold_at_min"])
    flag_all = negatives * config.COST_FP_DOP
    return [
        row("do_nothing_all_pass", 1.0, 0.0, 0.0, positives, 0.0, 0.0, do_nothing,
            "every unit released; every damaged unit becomes a missed claim"),
        row("person_checks_everything", None, 1.0, 0.0, 0.0, 0.0, UNITS, everyone,
            f"every unit gets the {config.MIN_PER_REVIEW}-min unpack at {config.COST_REVIEW_DOP} DOP; catches everything"),
        row("model_alone_best_threshold", at_min["threshold"], at_min["tpr"], at_min["fpr"],
            at_min["exp_fn_per_1000"], at_min["exp_fp_per_1000"], 0.0, at_min["cost_dop_per_1000"],
            f"flag if score >= t; flat band {band['band_low']}..{band['band_high']}; "
            f"test TPR/FPR from {positives_in_test} positives"),
        row("coco_zero_shot", None, 0.0, 0.0, positives, 0.0, 0.0, do_nothing,
            "identical to do_nothing: COCO has no package class, every image scores 0.0 and passes"),
        row("flag_everything_all_fail", 0.0, 1.0, 1.0, 0.0, negatives, 0.0, flag_all,
            "every unit flagged; cost is every intact unit's unpack, FN-free"),
    ]


def break_even(curve_all: list[dict], band: dict, everyone_cost: float) -> dict:
    """First threshold above the minimum where the model no longer beats
    'person checks everything', plus the lowest threshold that beats it."""
    beats = [c["threshold"] for c in curve_all if c["cost_dop_per_1000"] < everyone_cost]
    above = [c["threshold"] for c in curve_all
             if c["threshold"] > band["threshold_at_min"] and c["cost_dop_per_1000"] >= everyone_cost]
    return {
        "person_checks_everything_cost_dop_per_1000": everyone_cost,
        "model_beats_it_from": min(beats) if beats else None,
        "model_beats_it_until_exclusive": min(above) if above else None,
        "model_beats_it_at_n_thresholds": len(beats),
    }


def plot(curves: dict, band: dict, mins_by_rate: list[dict], everyone_cost: float, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    main = curves[("all", config.BASE_RATE)]
    chk = curves[("dedup", config.BASE_RATE)]
    t = [c["threshold"] for c in main]
    ax1.plot(t, [c["cost_dop_per_1000"] for c in main], color="#4C78A8", lw=2, label="all 120 test images")
    ax1.plot(t, [c["cost_dop_per_1000"] for c in chk], color="#4C78A8", lw=1, ls="--", alpha=0.6,
             label="114 without train/valid twin (check)")
    ax1.axvspan(band["band_low"], band["band_high"], color="#F58518", alpha=0.18,
                label=f"within one test positive of min ({band['band_low']}..{band['band_high']})")
    ax1.plot([band["threshold_at_min"]], [band["min_cost_dop_per_1000"]], "o", color="#E45756", ms=9,
             label=f"min {band['min_cost_dop_per_1000']:.0f} DOP at t={band['threshold_at_min']}")
    ax1.axhline(everyone_cost, color="grey", ls=":", lw=1, label=f"person checks everything ({everyone_cost:.0f})")
    ax1.set_xlabel("threshold t (flag if image score >= t)")
    ax1.set_ylabel(f"expected cost per {UNITS} units (DOP)")
    ax1.set_title(f"Cost vs threshold at base rate {config.BASE_RATE:.1%}")
    ax1.set_ylim(0, max(everyone_cost * 1.25, band["min_cost_dop_per_1000"] * 3))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper center")

    br = [m["base_rate"] for m in mins_by_rate]
    ax2.plot(br, [m["threshold_at_min"] for m in mins_by_rate], "o-", color="#E45756", label="minimum-cost threshold")
    ax2.fill_between(br, [m["band_low"] for m in mins_by_rate], [m["band_high"] for m in mins_by_rate],
                     color="#F58518", alpha=0.18, label="flat band (within one test positive)")
    ax2.axvline(config.BASE_RATE, color="grey", ls=":", lw=1, label=f"assumed base rate {config.BASE_RATE:.1%}")
    ax2.set_xscale("log")
    ax2.set_xticks(br, [f"{b:.1%}" for b in br])
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("base rate (share of received units that are compromised)")
    ax2.set_ylabel("threshold")
    ax2.set_title("Minimum-cost threshold vs base rate")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    rows = load_scores()
    subsets = {"all": rows, "dedup": [r for r in rows if not r["dup"]]}
    positives_in_test = sum(r["label"] for r in rows)

    curves: dict[tuple[str, float], list[dict]] = {}
    table: list[dict] = []
    for subset, data in subsets.items():
        per_t = {t: rates(data, t) for t in THRESHOLDS}
        for b in base_rates():
            curve = [cost_row(subset, b, t, per_t[t]) for t in THRESHOLDS]
            curves[(subset, b)] = curve
            table.extend(curve)

    with (OUT_DIR / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)

    tolerance = UNITS * config.BASE_RATE * (1 / positives_in_test) * config.COST_FN_DOP
    band = flat_band(curves[("all", config.BASE_RATE)], tolerance)
    band_dedup = flat_band(curves[("dedup", config.BASE_RATE)],
                           UNITS * config.BASE_RATE * (1 / sum(r["label"] for r in subsets["dedup"])) * config.COST_FN_DOP)
    mins_by_rate = []
    for b in sorted(base_rates()):
        tol_b = UNITS * b * (1 / positives_in_test) * config.COST_FN_DOP
        mins_by_rate.append({"base_rate": b, **flat_band(curves[("all", b)], tol_b)})

    everyone_cost = UNITS * config.COST_REVIEW_DOP
    opts = options(curves[("all", config.BASE_RATE)], band, config.BASE_RATE, positives_in_test)
    with (OUT_DIR / "options_single_threshold.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(opts[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(opts)

    summary = {
        "source": "outputs/test_scores.csv",
        "units": UNITS,
        "base_rate": config.BASE_RATE,
        "base_rates": sorted(base_rates()),
        "cost_fn_dop": config.COST_FN_DOP, "cost_fp_dop": config.COST_FP_DOP,
        "cost_review_dop": config.COST_REVIEW_DOP, "dop_per_eur": config.DOP_PER_EUR,
        "test_positives": positives_in_test, "test_negatives": len(rows) - positives_in_test,
        "band_tolerance_rule": "cost <= min + 1000 * base_rate * (1 / positives_in_test) * COST_FN_DOP",
        "all": band,
        "dedup_check": band_dedup,
        "min_by_base_rate": mins_by_rate,
        "break_even_vs_person_checks_everything": break_even(curves[("all", config.BASE_RATE)], band, everyone_cost),
        "options_at_base_rate": opts,
    }
    (OUT_DIR / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(curves, band, mins_by_rate, everyone_cost, OUT_DIR / "cost_curve.png")

    print(f"min cost {band['min_cost_dop_per_1000']} DOP/1000 at t={band['threshold_at_min']} "
          f"(ties {band['thresholds_at_min']}); band {band['band_low']}..{band['band_high']} "
          f"(tolerance {band['tolerance_dop']} DOP)")
    print("dedup check:", band_dedup["threshold_at_min"], band_dedup["band_low"], band_dedup["band_high"])
    print("break-even:", summary["break_even_vs_person_checks_everything"])
    print("wrote sweep.csv, sweep_summary.json, cost_curve.png, options_single_threshold.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
