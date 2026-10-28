import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"

ASSUMPTIONS = {
    "base_rate": 0.02,
    "cost_fp": 1.0,
    "cost_fn": 20.0,
    "cost_tp": 0.5,
    "cost_tn": 0.0,
}


def evaluate_binary(frame, threshold, assumptions=None):
    costs = ASSUMPTIONS if assumptions is None else assumptions

    positive = frame["positive"].to_numpy() == 1
    hold = frame["score"].to_numpy() >= threshold

    tp = int((positive & hold).sum())
    fn = int((positive & ~hold).sum())
    fp = int((~positive & hold).sum())
    tn = int((~positive & ~hold).sum())

    if tp + fn == 0 or fp + tn == 0:
        raise ValueError("Both actual classes must be present.")

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    prevalence = costs["base_rate"]

    expected_cost = (
        prevalence * (
            tpr * costs["cost_tp"]
            + (1 - tpr) * costs["cost_fn"]
        )
        + (1 - prevalence) * (
            fpr * costs["cost_fp"]
            + (1 - fpr) * costs["cost_tn"]
        )
    )

    return {
        "threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "recall": tpr,
        "false_positive_rate": fpr,
        "cost_per_1000": 1000 * expected_cost,
    }


def main():
    scores = pd.read_csv(OUT / "image_scores.csv")
    validation = scores[scores["split"] == "valid"]
    test = scores[scores["split"] == "test"]

    sweep = pd.DataFrame([
        evaluate_binary(validation, threshold)
        for threshold in np.linspace(0, 1, 1001)
    ])

    # For equal minimum costs, choose the lowest threshold.
    minimum = sweep["cost_per_1000"].min()
    candidates = sweep[
        np.isclose(sweep["cost_per_1000"], minimum, rtol=0, atol=1e-9)
    ]
    chosen = float(candidates.iloc[0]["threshold"])

    validation_result = evaluate_binary(validation, chosen)
    test_result = evaluate_binary(test, chosen)

    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)

    output = {
        "assumptions": ASSUMPTIONS,
        "selection_split": "valid",
        "selection_rule": "minimum cost; lowest threshold in a tie",
        "chosen_threshold": chosen,
        "validation": validation_result,
        "test": test_result,
    }
    (OUT / "chosen_threshold.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["threshold"], sweep["cost_per_1000"])
    ax.axvline(
        chosen, color="red", linestyle="--",
        label=f"Chosen threshold: {chosen:.3f}",
    )
    ax.set(
        xlabel="Confidence threshold",
        ylabel="Expected cost per 1,000 cases (€)",
        title="Validation cost sweep — assumed 2% positive rate",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cost_vs_threshold.png", dpi=160)
    plt.close(fig)

    print("\nVALIDATION: SELECTED THRESHOLD")
    print(json.dumps(validation_result, indent=2))
    print("\nTEST: SAME FIXED THRESHOLD")
    print(json.dumps(test_result, indent=2))
    print(
        "\nMinimum-cost validation grid points span: "
        f"{candidates['threshold'].min():.3f} to "
        f"{candidates['threshold'].max():.3f}"
    )
    print("\nConfusion counts describe the dataset.")
    print("Costs are reweighted to the assumed 2% operational base rate.")


    from shutil import copy2
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "weights" / "best.pt"))
    diagnostic = model.val(
        data=str(ROOT / "datasets" / "missing-pills-v1" / "evaluation.yaml"),
        split="test",
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        conf=chosen,
        plots=True,
        project=str(ROOT / "runs"),
        name="selected_threshold_detection",
    )

    copy2(
        Path(diagnostic.save_dir) / "confusion_matrix.png",
        OUT / "detection_confusion_matrix.png",
    )
    print("Detection confusion matrix saved in analysis/.")




if __name__ == "__main__":
    main()