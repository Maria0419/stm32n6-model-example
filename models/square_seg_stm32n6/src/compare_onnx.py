import argparse
import json
from pathlib import Path

from config_utils import load_pipeline_config, resolve_path
from dataset import SquareSegDataset
from onnx_metrics import create_session, evaluate_session


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "compare.yaml"


def main():
    parser = argparse.ArgumentParser(description="Compare float and INT8 ONNX segmentation accuracy")
    parser.parse_args()

    config_path, train_path, train_cfg, cfg = load_pipeline_config(CONFIG_PATH, "compare")
    float_model, int8_model = (resolve_path(config_path, cfg[key]) for key in ("float_model", "int8_model"))

    dataset = SquareSegDataset(
        resolve_path(train_path, train_cfg["test_dir"]),
        train_cfg["image_height"],
        train_cfg["image_width"],
        augment=False,
    )
    threshold = float(cfg.get("threshold", train_cfg.get("threshold", 0.5)))
    bce_weight = float(train_cfg.get("bce_loss_weight", 1.0))
    pos_weight = float(train_cfg.get("pos_weight", 1.0))
    float_metrics = evaluate_session(
        create_session(float_model),
        dataset,
        threshold,
        bce_weight=bce_weight,
        pos_weight=pos_weight,
    )
    int8_metrics = evaluate_session(
        create_session(int8_model),
        dataset,
        threshold,
        bce_weight=bce_weight,
        pos_weight=pos_weight,
    )
    degradation = float_metrics["iou"] - int8_metrics["iou"]
    min_iou = float(cfg.get("minimum_int8_iou", 0.95))
    max_degradation = float(cfg.get("maximum_iou_degradation", 0.02))
    accepted = int8_metrics["iou"] >= min_iou and degradation <= max_degradation
    report = {
        "config": str(CONFIG_PATH),
        "dataset": str(dataset.root),
        "threshold": threshold,
        "loss": {"bce_weight": bce_weight, "pos_weight": pos_weight},
        "float": {"model": str(float_model), **float_metrics},
        "int8": {"model": str(int8_model), **int8_metrics},
        "iou_degradation": degradation,
        "acceptance": {
            "minimum_int8_iou": min_iou,
            "maximum_iou_degradation": max_degradation,
            "passed": accepted,
        },
    }
    report_path = resolve_path(config_path, cfg["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
