"""Photo-level decision simulation, separate from object detection metrics.

No fitted model, no training, and no file access here. All human-performance
and prevalence inputs are explicit scenario assumptions. Costs include time
for true alerts and false alerts, plus the missed-damage loss.
"""
from __future__ import annotations

import numpy as np


PHOTO_WORKFLOW = "photo_triage_incidents_v1"


def capacity_limit(s):
    """None means workload is costed/reported without a fixed daily cap."""
    return s.get("available_staff_minutes_per_day")


def within_capacity(minutes, s):
    limit = capacity_limit(s)
    return None if limit is None else minutes <= limit + 1e-9


def photo_workflow(s):
    return s.get("workflow", "physical_confirmation_v1") == PHOTO_WORKFLOW


def validate_scenario(s):
    if s.get("workflow", "physical_confirmation_v1") not in ("physical_confirmation_v1", PHOTO_WORKFLOW):
        raise ValueError("Unknown workflow.")
    for key in ("damage_prevalence", "photo_reviewer_sensitivity", "photo_reviewer_specificity"):
        if not 0 <= s[key] <= 1:
            raise ValueError(f"{key} must lie between zero and one.")
    for key in ("cases_per_day", "staff_euros_per_hour", "physical_check_minutes",
                "damage_report_extra_minutes", "photo_review_minutes",
                "missed_damage_euros"):
        if not np.isfinite(s[key]) or s[key] < 0:
            raise ValueError(f"Invalid {key}.")
    limit = capacity_limit(s)
    if limit is not None and (not np.isfinite(limit) or limit < 0):
        raise ValueError("Invalid available_staff_minutes_per_day.")
    if photo_workflow(s):
        for key in ("incident_handling_minutes", "false_incident_extra_euros"):
            if not np.isfinite(s[key]) or s[key] < 0:
                raise ValueError(f"Invalid {key}.")


def inputs(scores, truth):
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    if scores.ndim != 1 or scores.shape != truth.shape or not len(scores):
        raise ValueError("One score and known binary truth required per photo.")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and in [0,1].")
    if not truth.any() or truth.all():
        raise ValueError("Both labelled conditions are needed to estimate conditional rates.")
    return scores, truth


def thresholds(scores):
    """Every attainable score partition in [0,1], without splitting score ties."""
    unique = np.unique(scores)
    return np.unique(np.r_[0., (unique[:-1] + unique[1:]) / 2, 1.])


def costs(direct_pos, direct_neg, review_pos, review_neg, s):
    """Inputs are routing fractions CONDITIONAL on each true condition.

    In the photo workflow, a direct alert or human-positive review creates a
    provisional incident. Every incident consumes administrative handling time,
    including false incidents. Handling is not a second condition review.
    Historical scenarios retain ideal physical confirmation and its old costs.
    """
    p = s["damage_prevalence"]
    tp = p * (direct_pos + s["photo_reviewer_sensitivity"] * review_pos)
    fp = (1-p) * (direct_neg + (1-s["photo_reviewer_specificity"]) * review_neg)
    fn = p - tp
    tn = 1-p-fp
    photo = p * review_pos + (1-p) * review_neg
    incidents = tp + fp
    if photo_workflow(s):
        physical = np.zeros_like(incidents)
        minutes = photo*s["photo_review_minutes"] + incidents*s["incident_handling_minutes"]
        false_loss = fp*s["false_incident_extra_euros"]
    else:
        physical = incidents
        incidents = tp  # Only confirmed damage creates a report in the old flow.
        minutes = (photo * s["photo_review_minutes"]
                   + physical * s["physical_check_minutes"]
                   + tp * s["damage_report_extra_minutes"])
        false_loss = np.zeros_like(incidents)
    labour = minutes * s["staff_euros_per_hour"] / 60
    loss = fn * s["missed_damage_euros"]
    return {"tp_rate": tp, "fp_rate": fp, "fn_rate": fn, "tn_rate": tn,
            "photo_review_rate": photo, "physical_check_rate": physical,
            "incident_rate": incidents, "false_incident_loss_per_case": false_loss,
            "minutes_per_case": minutes, "labour_per_case": labour,
            "missed_damage_loss_per_case": loss, "cost_per_case": labour+loss+false_loss}


def evaluate(scores, truth, lower, upper, s):
    validate_scenario(s)
    scores, truth = inputs(scores, truth)
    if not 0 <= lower <= upper <= 1:
        raise ValueError("Require 0 <= lower <= upper <= 1.")
    # Boundary convention is identical in dashboard upload inference.
    route = np.where(scores >= upper, 2, np.where(scores >= lower, 1, 0))
    counts = [[int(np.sum((route == action) & (truth == condition)))
               for condition in (True, False)] for action in range(3)]
    positive, negative = int(truth.sum()), int((~truth).sum())
    c = costs(counts[2][0]/positive, counts[2][1]/negative,
              counts[1][0]/positive, counts[1][1]/negative, s)
    c = {k: float(v) for k, v in c.items()}
    c.update({"lower": float(lower), "upper": float(upper),
              "workflow": s.get("workflow", "physical_confirmation_v1"),
              "routing_counts": counts, "routing_rows": ["pass", "photo_review", "provisional_incident" if photo_workflow(s) else "physical_check"],
              "routing_columns": ["damage", "undamaged"],
              "known_photos": len(scores), "positive_photos": positive, "negative_photos": negative,
              "conditional_damage_miss_rate": c["fn_rate"]/s["damage_prevalence"] if s["damage_prevalence"] else None,
              "cost_per_1000": c["cost_per_case"]*1000,
              "cost_per_day": c["cost_per_case"]*s["cases_per_day"],
              "minutes_per_day": c["minutes_per_case"]*s["cases_per_day"],
              "photo_reviews_per_day": c["photo_review_rate"]*s["cases_per_day"],
              "physical_checks_per_day": c["physical_check_rate"]*s["cases_per_day"],
              "incidents_per_day": c["incident_rate"]*s["cases_per_day"],
              "false_incidents_per_day": c["fp_rate"]*s["cases_per_day"] if photo_workflow(s) else 0.,
              "missed_damage_cases_per_day": c["fn_rate"]*s["cases_per_day"]})
    c["capacity_feasible"] = within_capacity(c["minutes_per_day"], s)
    c["expected_confusion_per_1000"] = [[c["tp_rate"]*1000, c["fn_rate"]*1000],
                                                 [c["fp_rate"]*1000, c["tn_rate"]*1000]]
    return c


def sweep(scores, truth, s):
    """Exhaustive two-threshold sweep of the observed validation partitions.

    A finite capacity optionally constrains expected cumulative workload.
    With no cap, all pairs are eligible and the minimum includes staff cost.
    The numerical pair is a candidate, not an automatic final policy.
    A single-threshold policy is the diagonal: no photo review band.
    """
    validate_scenario(s)
    scores, truth = inputs(scores, truth)
    grid = thresholds(scores)
    pos, neg = np.sort(scores[truth]), np.sort(scores[~truth])
    tail_p = 1 - np.searchsorted(pos, grid, side="left") / len(pos)
    tail_n = 1 - np.searchsorted(neg, grid, side="left") / len(neg)
    dp, dn = tail_p[None, :], tail_n[None, :]
    rp, rn = tail_p[:, None]-dp, tail_n[:, None]-dn
    all_costs = costs(dp, dn, rp, rn, s)
    valid = grid[:, None] <= grid[None, :]
    limit = capacity_limit(s)
    feasible = valid if limit is None else valid & (all_costs["minutes_per_case"]*s["cases_per_day"] <= limit+1e-9)

    def best(mask):
        rows, cols = np.where(mask)
        if not len(rows):
            return None
        # Cost first, then workload, then index for reproducible ties.
        order = np.lexsort((cols, rows, all_costs["minutes_per_case"][rows, cols],
                            all_costs["cost_per_case"][rows, cols]))
        i, j = rows[order[0]], cols[order[0]]
        return evaluate(scores, truth, grid[i], grid[j], s)

    diagonal = np.eye(len(grid), dtype=bool)
    single_curve = [{"threshold": float(t),
                     "cost_per_1000": float(all_costs["cost_per_case"][i, i]*1000),
                     "minutes_per_day": float(all_costs["minutes_per_case"][i, i]*s["cases_per_day"])}
                    for i, t in enumerate(grid)]
    return {"candidate_feasible": best(feasible), "candidate_unconstrained": best(valid),
            "single_threshold_feasible": best(diagonal & feasible),
            "single_threshold_unconstrained": best(diagonal),
            "workload_constraint_applied": limit is not None,
            "partitions": len(grid), "pairs_examined": int(valid.sum()),
            "pairs_feasible": int(feasible.sum()), "single_threshold_curve": single_curve}


def alternatives(s):
    """Scenario-only comparators, independent of the validation class mix."""
    validate_scenario(s)
    output = {}
    for name, args in {"do_nothing": (0., 0., 0., 0.),
                       "physical_check_everything": (1., 1., 0., 0.),
                       "photo_review_everything": (0., 0., 1., 1.)}.items():
        # Physical-all is an explicitly separate, ideal manual comparator.
        comparator = {**s, "workflow": "physical_confirmation_v1"} if name == "physical_check_everything" else s
        c = {k: float(v) for k, v in costs(*args, comparator).items()}
        c.update({"cost_per_1000": c["cost_per_case"]*1000,
                  "cost_per_day": c["cost_per_case"]*s["cases_per_day"],
                  "minutes_per_day": c["minutes_per_case"]*s["cases_per_day"]})
        c["capacity_feasible"] = within_capacity(c["minutes_per_day"], s)
        output[name] = c
    return output


def binary_counts(scores, truth, threshold):
    scores, truth = inputs(scores, truth)
    pred = scores >= threshold
    tp, fp = int(np.sum(pred & truth)), int(np.sum(pred & ~truth))
    fn, tn = int(np.sum(~pred & truth)), int(np.sum(~pred & ~truth))
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn),
            "false_negative_rate": fn/(tp+fn), "false_positive_rate": fp/(fp+tn)}
