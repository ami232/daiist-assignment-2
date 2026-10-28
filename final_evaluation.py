"""Evaluate the already-frozen policy on test; never optimise on test."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from photo_predictions import ROOT, INFERENCE, digest, read, records_from_model, write
from costing import alternatives, binary_counts, evaluate, photo_workflow


def main():
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT / "data/.matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT / "data/.ultralytics"))
    import numpy as np
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset
    from ultralytics.utils.metrics import ConfusionMatrix
    from train import choose_device

    policy_file = ROOT / "operating_policy.json"
    policy = read(policy_file)
    assert policy["status"] == "frozen_before_test" and policy["selection_split"] == "valid"
    assert policy["test_must_not_select_or_tune_policy"]
    assert digest(ROOT / "scenario.json") == policy["scenario_sha256"]
    assert digest(ROOT / policy["weights"]) == policy["weights_sha256"]
    assert digest(ROOT / "results/validation_analysis/photo_predictions.json") == policy["validation_predictions_sha256"]
    assert digest(ROOT / "results/validation_analysis/analysis.json") == policy["validation_analysis_sha256"]
    policy_hash = digest(policy_file)
    out = ROOT / "results/final_evaluation" / policy_hash[:12]
    out.mkdir(parents=True,exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.exists():
        result=read(summary_path)
        assert result["policy_sha256"] == policy_hash
        write(ROOT / "results/final_evaluation.json", {"summary":str(summary_path.relative_to(ROOT)),"policy_sha256":policy_hash})
        print("Using completed final evaluation for the identical frozen policy.",flush=True)
        return result
    # Copy the frozen inputs into the output BEFORE opening test images/labels.
    write(out / "policy.json",policy)
    scenario = read(ROOT / "scenario.json")
    write(out / "scenario.json",scenario)
    config = read(ROOT / "dataset_config.json")
    dataset = ROOT / config["data_dir"]
    files = sorted(p for p in (dataset / "test/images").iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"})
    assert len(files) == config["measured_export_images"]["test"]
    device = choose_device(["cuda","mps","cpu"])
    runtime = out / "test_runtime.yaml"
    runtime.write_text(yaml.safe_dump({"path":str(dataset),"train":"train/images","val":"valid/images",
                                       "test":"test/images","names":config["classes"],"nc":len(config["classes"])}))
    weights=ROOT / policy["weights"]
    print(f"FINAL TEST: frozen {policy['model_key']}; {len(files)} images; device={device}",flush=True)
    detection_path = out / "detection_metrics.json"
    if detection_path.exists():
        detection_metrics = read(detection_path)
        assert detection_metrics["weights_sha256"] == policy["weights_sha256"]
    else:
        model=YOLO(str(weights))
        metrics = model.val(data=str(runtime),split="test",device=device,imgsz=640,batch=1,rect=False,
                            conf=.001,iou=.7,max_det=300,workers=0,plots=True,
                            project=str(out),name="detection",exist_ok=True,verbose=False)
        detection_metrics={"weights_sha256":policy["weights_sha256"],"metrics":metrics.results_dict,
                           "per_class":metrics.summary(decimals=6)}
        write(detection_path,detection_metrics)
    detections = records_from_model(weights,files,out / "photo_predictions.json",config["classes"],device,split="test")
    pretrained = ROOT / policy.get("pretrained_weights", "data/pretrained/yolo11n.pt")
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    pretrained = Path(attempt_download_asset(str(pretrained)))
    if policy.get("pretrained_weights_sha256"):
        assert digest(pretrained) == policy["pretrained_weights_sha256"], "COCO baseline differs from the training initialization."
    zero = records_from_model(pretrained,files,out / "coco_zero_shot_predictions.json",
                              config["classes"],device,coco=True,split="test")
    assert zero["zero_shot"]["mapped_target_prediction_count"] == 0
    known=[r for r in detections["images"] if r["condition_ground_truth_known"]]
    truth=[r["has_damage_label"] for r in known]
    scores=[r["max_damage_confidence"] for r in known]
    operational = evaluate(scores,truth,policy["lower"],policy["upper"],scenario)
    model_only = None
    if "model_only_threshold" in policy:
        assert policy["model_only_selection_split"] == "valid"
        threshold = policy["model_only_threshold"]
        model_only = evaluate(scores,truth,threshold,threshold,scenario)
    # A separate box-level matrix at the upper alert cutoff, using original labels.
    matrix=ConfusionMatrix(names={i:n for i,n in enumerate(config["classes"])})
    for record in detections["images"]:
        pred,gt=record["predictions"],record["truth"]
        matrix.process_batch({"cls":torch.tensor([b["class_id"] for b in pred]),
                              "conf":torch.tensor([b["confidence"] for b in pred]),
                              "bboxes":torch.tensor([b["xyxy"] for b in pred]).reshape(-1,4)},
                             {"cls":torch.tensor([b["class_id"] for b in gt]),
                              "bboxes":torch.tensor([b["xyxy"] for b in gt]).reshape(-1,4)},
                             conf=policy["upper"],iou_thres=.45)
    baseline = {"mAP50":0.,"mAP50_95":0.,"per_class":[
        {"Class":name,"AP50":0.,"AP50_95":0.,"recall":0.,"precision":None} for name in config["classes"]],
        "precision_note":"Undefined without positive predictions.",
        "photo_confusion_no_alerts":binary_counts([0.]*len(known),truth,.4),
        "scenario":alternatives(scenario)["do_nothing"]}
    failures=[]
    for record in known:
        score=record["max_damage_confidence"]
        if (record["has_damage_label"] and score < policy["lower"]) or (not record["has_damage_label"] and score >= policy["upper"]):
            failures.append({"image":record["image"],"damage_truth":record["has_damage_label"],"score":score,
                             "type":"damage_passed_directly" if record["has_damage_label"] else ("unnecessary_automatic_incident" if photo_workflow(scenario) else "unnecessary_direct_physical_check")})
    result={"evaluation_split":"test","test_evaluated":True,"completed_utc":datetime.now(timezone.utc).isoformat(),
            "evaluation_context":policy.get("evaluation_context","first_holdout_evaluation"),
            "policy_sha256":policy_hash,"policy":policy,"inference":INFERENCE,
            "detection_images":len(files),"known_condition_images":len(known),
            "unknown_condition_images":[r["image"] for r in detections["images"] if not r["condition_ground_truth_known"]],
            "metrics":detection_metrics["metrics"],"per_class":detection_metrics["per_class"],
            "detection_precision_recall_note":"Per-box metrics at test evaluator's F1 point; this summary does not alter the frozen operating thresholds.",
            "box_confusion":{"values":matrix.matrix.tolist(),"confidence":policy["upper"],"matching_iou":.45,
                             "rows":"predicted class; final row=background/miss","columns":"true class; final column=unmatched prediction",
                             "classes":config["classes"]+["background"]},
            "photo_condition_at_upper":binary_counts(scores,truth,policy["upper"]),
            "frozen_policy_scenario":operational,"alternatives":alternatives(scenario),
            "model_only_scenario":model_only,
            "baselines":{"coco_zero_shot":{**baseline,**zero["zero_shot"],"inference_ran":True,
                                             "weights":pretrained.relative_to(ROOT).as_posix(),"weights_sha256":zero["weights_sha256"]},
                         "predict_nothing":{**baseline,"definition":"No boxes or damage alerts.","inference_ran":False}},
            "failures":failures,
            "limitations":["Human-review and monetary outcomes are simulated; photo routing and scores are observed.",
                           "Costs use the same 5% scenario prevalence, not the test positive fraction.",
                           "No test threshold sweep or model selection. Do not tune this policy using these results.",
                           "Human condition review is assumed, not measured; one photo is only a proxy for one return.",
                           "A secondary evaluation of a previously seen test set is not a new independent holdout."]}
    assert digest(policy_file)==policy_hash and digest(weights)==policy["weights_sha256"]
    write(summary_path,result)
    write(ROOT / "results/final_evaluation.json",{"summary":str(summary_path.relative_to(ROOT)),"policy_sha256":policy_hash})
    print("FINAL TEST COMPLETE: "+str(summary_path.relative_to(ROOT)),flush=True)
    print(json.dumps({"metrics":result["metrics"],"routing":operational["routing_counts"],
                      "expected_cost_per_day":operational["cost_per_day"],"expected_minutes_per_day":operational["minutes_per_day"],
                      "expected_damage_miss_rate":operational["conditional_damage_miss_rate"]}),flush=True)
    return result


if __name__ == "__main__":
    main()
