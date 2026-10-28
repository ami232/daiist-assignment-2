import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COSTS_FILE = ROOT / "costs.json"


def load_costs(path: Path = COSTS_FILE) -> dict:
    costs = json.loads(path.read_text())
    return costs


def decision_rates(scores: np.ndarray, is_positive: np.ndarray, low: float, high: float) -> dict:
    high = max(low, high)
    fail = scores >= high
    passed = scores < low
    review = ~fail & ~passed
    rates = {}
    for label, mask in (("pos", is_positive), ("neg", ~is_positive)):
        n = max(mask.sum(), 1)
        rates[label] = {
            "pass": passed[mask].sum() / n,
            "review": review[mask].sum() / n,
            "fail": fail[mask].sum() / n,
        }
    return rates


def expected_cost(rates: dict, costs: dict) -> dict:
    pi = costs["base_rate"]
    pos, neg = rates["pos"], rates["neg"]
    model_cost = (
        pi * (pos["fail"] * costs["cost_tp"] + pos["pass"] * costs["cost_fn"] + pos["review"] * costs["cost_tp"])
        + (1 - pi) * (neg["fail"] * costs["cost_fp"] + neg["pass"] * costs["cost_tn"] + neg["review"] * costs["cost_tn"])
    )
    review_share = pi * pos["review"] + (1 - pi) * neg["review"]
    review_cost = review_share * costs["cost_review"]
    return {
        "cost_per_case": model_cost + review_cost,
        "review_share": review_share,
        "review_cost_per_case": review_cost,
    }


def operating_summary(scores: np.ndarray, is_positive: np.ndarray, low: float, high: float, costs: dict) -> dict:
    rates = decision_rates(scores, is_positive, low, high)
    result = expected_cost(rates, costs)
    reviews_per_day = result["review_share"] * costs["cases_per_day"]
    review_hours = reviews_per_day * costs["review_seconds"] / 3600
    result.update(
        rates=rates,
        cost_per_1000=result["cost_per_case"] * 1000,
        reviews_per_day=reviews_per_day,
        reviewers_needed=review_hours / costs["reviewer_hours_per_day"],
    )
    return result


def sweep_single_threshold(scores: np.ndarray, is_positive: np.ndarray, costs: dict, steps: int = 101) -> pd.DataFrame:
    rows = []
    for t in np.linspace(0, 1, steps):
        rates = decision_rates(scores, is_positive, t, t)
        rows.append({
            "threshold": t,
            "recall": rates["pos"]["fail"],
            "false_positive_rate": rates["neg"]["fail"],
            "cost_per_1000": expected_cost(rates, costs)["cost_per_case"] * 1000,
        })
    return pd.DataFrame(rows)


def best_band(scores: np.ndarray, is_positive: np.ndarray, costs: dict, steps: int = 51) -> dict:
    grid = np.linspace(0, 1, steps)
    best = None
    for low in grid:
        for high in grid[grid >= low]:
            summary = operating_summary(scores, is_positive, low, high, costs)
            if best is None or summary["cost_per_case"] < best["cost_per_case"] - 1e-12:
                best = {"low": float(low), "high": float(high), **summary}
    return best


def reference_policies(scores: np.ndarray, is_positive: np.ndarray, threshold: float, costs: dict) -> pd.DataFrame:
    policies = {
        "Do nothing (always continue)": (1.01, 1.01),
        "Always decelerate": (0.0, 0.0),
        "Warn the driver on every frame": (0.0, 1.01),
        "Model alone (continue or decelerate)": (threshold, threshold),
    }
    rows = []
    for name, (low, high) in policies.items():
        s = operating_summary(scores, is_positive, low, high, costs)
        rows.append({
            "policy": name,
            "cost_per_1000": round(s["cost_per_1000"], 2),
            "review_share": round(s["review_share"], 4),
            "reviewers_needed": round(s["reviewers_needed"], 2),
        })
    return pd.DataFrame(rows)
