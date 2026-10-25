"""Cost sweep and human-review load.
Reads costs.json (your figures) and outputs/predictions_test.json.
Run: uv run python costing.py   ->  outputs/costing_results.json, outputs/cost_curve.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs")
C = json.loads(Path("costs.json").read_text())
THR = np.append(np.round(np.arange(0.01, 1.0, 0.01), 2), 1.01)  # 1.01 = never fires


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def frame_scores(images):
    """s_pos: top confidence in the frame. s_neg: top confidence among boxes matching no label
    (proxy for an empty frame's score, since the test set has no empty frames)."""
    s_pos, s_neg = [], []
    for v in images.values():
        gts = [(int(c), (x-w/2, y-h/2, x+w/2, y+h/2)) for c, x, y, w, h in v["gt"]]
        confs, unmatched = [0.0], [0.0]
        for c, conf, *box in v["pred"]:
            confs.append(conf)
            if not any(gc == int(c) and iou(box, gb) >= 0.5 for gc, gb in gts):
                unmatched.append(conf)
        s_pos.append(max(confs))
        s_neg.append(max(unmatched))
    return np.array(s_pos), np.array(s_neg)


def params():
    return {
        "c_rev": C["minutes_per_review"] / 60 * C["operator_hourly_eur"],
        "c_fn": C["fn_cost_per_incendio"] / C["frames_per_incendio"],
        "frames_day": C["cameras"] * 24 * 60 / C["frame_interval_min"],
        "capacity": C["operators"] * C["effective_hours_per_operator"] * 60 / C["minutes_per_review"],
    }


def three_action(Fp, Fn, b, P):
    """Cost per frame for every (review threshold i, alert threshold j) with i <= j."""
    pi, pj = Fp[:, None], Fp[None, :]
    ni, nj = Fn[:, None], Fn[None, :]
    pos_pass, pos_rev, pos_alert = 1 - pi, pi - pj, pj + 0 * pi
    neg_rev, neg_alert = ni - nj, nj + 0 * ni
    pos = b * (pos_pass * P["c_fn"] + pos_rev * (P["c_rev"] + C["review_delay_fraction"] * P["c_fn"])
               + pos_alert * P["c_rev"])
    neg = (1 - b) * (neg_rev * P["c_rev"] + neg_alert * (P["c_rev"] + C["alert_false_extra_eur"]))
    share = b * pos_rev + (1 - b) * neg_rev
    cost = np.where(np.triu(np.ones((len(Fp), len(Fp)), bool)), pos + neg, np.inf)
    return cost, share


def best(cost, share, P, constrained):
    c = np.where(share * P["frames_day"] <= P["capacity"], cost, np.inf) if constrained else cost
    i, j = np.unravel_index(np.argmin(c), c.shape)
    return int(i), int(j)


def main():
    data = json.loads((OUT / "predictions_test.json").read_text())
    s_pos, s_neg = frame_scores(data["images"])
    Fp = np.array([(s_pos >= t).mean() for t in THR])
    Fn = np.array([(s_neg >= t).mean() for t in THR])
    P, b = params(), C["base_rate"]

    cost, share = three_action(Fp, Fn, b, P)
    i_u, j_u = best(cost, share, P, False)
    i, j = best(cost, share, P, True)

    alone = b * (1 - Fp) * P["c_fn"] + (1 - b) * Fn * C["false_dispatch_eur"]
    k = int(np.argmin(alone))
    policies = {
        "do_nothing": b * P["c_fn"],
        "human_checks_everything": P["c_rev"],
        "model_decides_alone": float(alone[k]),
        "three_action_unconstrained": float(cost[i_u, j_u]),
        "three_action_chosen": float(cost[i, j]),
    }

    def load(ii, jj):
        r = float(share[ii, jj] * P["frames_day"])
        return {"review_threshold": float(THR[ii]), "alert_threshold": float(THR[jj]),
                "reviews_per_day": r, "share_to_review": float(share[ii, jj]),
                "operators_needed": r * C["minutes_per_review"] / 60 / C["effective_hours_per_operator"],
                "capacity_per_day": P["capacity"]}

    print(f"Derived: review {P['c_rev']:.2f} EUR, FN per frame {P['c_fn']:.2f} EUR, "
          f"{P['frames_day']:.0f} frames/day, capacity {P['capacity']:.0f} reviews/day, base rate {b}")
    print("\nPolicy cost per 1,000 frames / per day (EUR):")
    for name, v in policies.items():
        print(f"  {name:28s} {v*1000:10.2f}  {v*P['frames_day']:10.2f}")
    print("\nUnconstrained optimum:", load(i_u, j_u))
    print("Chosen (within capacity):", load(i, j))
    print(f"model-alone best threshold: {THR[k]}")

    sens = []
    print("\nSensitivity to base rate (chosen policy within capacity):")
    for bb in [1e-5, 1e-4, 1e-3, 1e-2]:
        cc, sh = three_action(Fp, Fn, bb, P)
        ii, jj = best(cc, sh, P, True)
        row = {"base_rate": bb, "three_action": float(cc[ii, jj]) * 1000, "do_nothing": bb * P["c_fn"] * 1000,
               "review_all": P["c_rev"] * 1000, "review_thr": float(THR[ii]), "alert_thr": float(THR[jj]),
               "reviews_per_day": float(sh[ii, jj] * P["frames_day"])}
        sens.append(row)
        print(f"  b={bb:g}: three-action {row['three_action']:.2f} | nothing {row['do_nothing']:.2f} | "
              f"review all {row['review_all']:.2f} per 1000 frames; thr {row['review_thr']}/{row['alert_thr']}, "
              f"{row['reviews_per_day']:.0f} reviews/day")

    plt.figure(figsize=(8, 4.5))
    plt.plot(THR, np.diag(cost) * 1000, label="model + operator, single threshold")
    plt.plot(THR, alone * 1000, label="model decides alone")
    plt.axhline(policies["do_nothing"] * 1000, ls="--", c="gray", label="do nothing")
    plt.axhline(policies["human_checks_everything"] * 1000, ls="--", c="k", label="human checks everything")
    plt.axvline(THR[i], c="g", ls=":", label=f"chosen review thr {THR[i]:.2f}")
    plt.axvline(THR[j], c="r", ls=":", label=f"chosen alert thr {THR[j]:.2f}")
    plt.yscale("log"); plt.xlabel("confidence threshold"); plt.ylabel("expected cost per 1,000 frames (EUR)")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "cost_curve.png", dpi=150)

    (OUT / "costing_results.json").write_text(json.dumps({
        "thresholds": THR.tolist(), "F_pos": Fp.tolist(), "F_neg": Fn.tolist(), "params": P, "costs": C,
        "policies_per_frame": policies, "chosen": load(i, j), "unconstrained": load(i_u, j_u),
        "model_alone_threshold": float(THR[k]), "sensitivity": sens}, indent=2))
    print("\nSaved outputs/costing_results.json and outputs/cost_curve.png")


if __name__ == "__main__":
    main()