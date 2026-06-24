import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

from config_utils import load_pipeline_config, resolve_path
from dataset import SquareSegDataset


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "quantize.yaml"


class DatasetCalibrationReader(CalibrationDataReader):
    def __init__(self, dataset, input_name, indices):
        self.dataset = dataset
        self.input_name = input_name
        self.indices = iter(indices)

    def get_next(self):
        try:
            image, _ = self.dataset[next(self.indices)]
        except StopIteration:
            return None
        return {self.input_name: image.numpy()[None, ...].astype(np.float32, copy=False)}


def get_model_opset(path):
    model = onnx.load(path)
    if model.opset_import:
        return model.opset_import[0].version
    return None


def main():
    parser = argparse.ArgumentParser(description="Statically quantize the ONNX model for STM32N6")
    args = parser.parse_args()

    config_path, train_path, train_cfg, cfg = load_pipeline_config(CONFIG_PATH, "quantize")
    source = resolve_path(config_path, cfg["input"])
    output = resolve_path(config_path, cfg["output"])

    source_opset = get_model_opset(source)
    dataset = SquareSegDataset(
        resolve_path(train_path, train_cfg["train_dir"]),
        train_cfg["image_height"],
        train_cfg["image_width"],
        augment=False,
    )

    requested_count = cfg.get("calibration_samples", 256)
    count = min(requested_count, len(dataset))
    seed = cfg.get("seed", train_cfg.get("seed", 42))
    indices = random.Random(seed).sample(range(len(dataset)), count)
    input_name = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(source),
        str(output),
        DatasetCalibrationReader(dataset, input_name, indices),
        quant_format=getattr(QuantFormat, cfg.get("format", "QDQ")),
        activation_type=getattr(QuantType, cfg.get("activation_type", "QInt8")),
        weight_type=getattr(QuantType, cfg.get("weight_type", "QInt8")),
        calibrate_method=getattr(CalibrationMethod, cfg.get("calibration_method", "MinMax")),
        per_channel=cfg.get("per_channel", True),
    )

    model = onnx.load(output)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    image, _ = dataset[indices[0]]
    result = session.run(None, {session.get_inputs()[0].name: image.numpy()[None, ...]})
    operators = {}
    for node in model.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1

    report = {
        "config": str(CONFIG_PATH),
        "model": str(output),
        "source_model": str(source),
        "source_opset": source_opset,
        "source_opset_ok": source_opset is None or source_opset >= 13,
        "quantized_opset": model.opset_import[0].version,
        "calibration": {
            "dataset": str(dataset.root),
            "requested_samples": requested_count,
            "samples": count,
            "seed": seed,
            "preprocessing": {"color": "grayscale", "size": [dataset.size[1], dataset.size[0]], "scale": "divide_by_255"},
            "sample_indices": indices,
        },
        "quantization": {
            key: cfg.get(key)
            for key in ("format", "activation_type", "weight_type", "calibration_method", "per_channel")
        },
        "operators": operators,
        "input": {"name": input_name, "dtype": "float32"},
        "output_shape": list(result[0].shape),
        "onnx_checker": "passed",
        "onnxruntime_inference": "passed",
    }
    report_path = resolve_path(config_path, cfg["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
