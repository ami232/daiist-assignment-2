import numpy as np

FP_EUR = 10.0
FN_EUR = 200.0
REVIEW_EUR = 1.0
BASE_RATE = 0.10
IMAGES_PER_DAY = 500
REVIEW_SECONDS = 30
REVIEW_CAP = 0.25


def triage(scores, low, high):
    if low > high:
        raise ValueError("Lower threshold exceeds upper threshold")
    scores = np.asarray(scores, dtype=float)
    return np.where(scores < low, 0, np.where(scores >= high, 2, 1))


def outcome(scores, truth, low, high, base_rate=BASE_RATE):
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)

    if len(scores) != len(truth) or len(scores) == 0:
        raise ValueError("Invalid scores or labels")

    positives = int(truth.sum())
    negatives = len(truth) - positives

    if not positives or not negatives:
        raise ValueError("Both classes must be represented")

    actions = triage(scores, low, high)

    result = {
        "TP": int(((actions == 2) & truth).sum()),
        "FP": int(((actions == 2) & ~truth).sum()),
        "TN": int(((actions == 0) & ~truth).sum()),
        "FN": int(((actions == 0) & truth).sum()),
        "review_positive": int(((actions == 1) & truth).sum()),
        "review_negative": int(((actions == 1) & ~truth).sum()),
    }

    wp = 1000 * base_rate / positives
    wn = 1000 * (1 - base_rate) / negatives

    reviews = (
        result["review_positive"] * wp
        + result["review_negative"] * wn
    )

    cost = (
        result["FP"] * wn * FP_EUR
        + result["FN"] * wp * FN_EUR
        + reviews * REVIEW_EUR
    )

    share = reviews / 1000
    daily = share * IMAGES_PER_DAY

    result.update({
        "eur_per_1000": round(float(cost), 4),
        "review_share": round(float(share), 6),
        "review_images_per_day": round(float(daily), 2),
        "review_minutes_per_day": round(float(daily * REVIEW_SECONDS / 60), 2),
        "review_staff_fte": round(float(daily * REVIEW_SECONDS / 28800), 4),
        "base_rate": float(base_rate),
    })

    return result


def sweep(scores, truth, base_rate=BASE_RATE):
    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        result = outcome(scores, truth, t, t, base_rate)
        rows.append({
            "threshold": round(float(t), 4),
            "eur_per_1000": result["eur_per_1000"],
        })

    best = min(rows, key=lambda x: x["eur_per_1000"])
    return rows, best


def choose_triage(scores, truth, base_rate=BASE_RATE, max_review=REVIEW_CAP):
    grid = np.linspace(0.01, 0.99, 50)
    candidates = []

    for i, low in enumerate(grid):
        for high in grid[i:]:
            result = outcome(scores, truth, low, high, base_rate)

            if result["review_share"] <= max_review:
                candidates.append((
                    result["eur_per_1000"],
                    result["review_share"],
                    float(low),
                    float(high),
                    result,
                ))

    if not candidates:
        raise RuntimeError("No feasible thresholds")

    _, _, low, high, result = min(candidates)

    return {
        "lower": low,
        "upper": high,
        "review_cap": max_review,
        "validation": result,
    }


def sensitivity(scores, truth):
    return {
        str(rate): sweep(scores, truth, rate)[1]
        for rate in [0.05, 0.10, 0.20]
    }
