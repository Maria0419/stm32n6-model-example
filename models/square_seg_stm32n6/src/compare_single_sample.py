import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from config_utils import load_pipeline_config, resolve_path
from uart_inference import (
    decode_prediction,
    load_config as load_uart_config,
    load_mask,
    resolve_paths as resolve_uart_paths,
    send_image_and_get_result,
    wait_for_handshake,
)


def sigmoid(x):
    positive = x >= 0.0
    negative = ~positive
    result = np.empty_like(x, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    result[negative] = exp_x / (1.0 + exp_x)
    return result


def compute_metrics(prediction, target):
    intersection = float(np.logical_and(prediction, target).sum())
    prediction_sum = float(prediction.sum())
    target_sum = float(target.sum())
    return {
        "dice": (2.0 * intersection + 1.0) / (prediction_sum + target_sum + 1.0),
        "iou": (intersection + 1.0) / (prediction_sum + target_sum - intersection + 1.0),
        "foreground_pixels": int(prediction_sum),
        "target_pixels": int(target_sum),
        "intersection_pixels": int(intersection),
    }


def load_image_for_session(image_path, size):
    with Image.open(image_path) as source:
        image = source.convert("L")
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array[None, None, :, :]


def load_mask_for_size(mask_path, size):
    with Image.open(mask_path) as source:
        mask = source.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 127


def collect_samples(root, limit):
    samples = []
    for image_path in sorted(Path(root).rglob("*_image.tif")):
        mask_path = image_path.with_name(image_path.name.replace("_image.tif", "_label.tif"))
        if mask_path.is_file():
            samples.append((image_path, mask_path))
        if len(samples) >= limit:
            break
    return samples


def run_onnx(session_path, image_path, mask_path, threshold):
    session = ort.InferenceSession(str(session_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    height = input_info.shape[2]
    width = input_info.shape[3]
    size = (width, height)

    image = load_image_for_session(image_path, size)
    logits = session.run([output_name], {input_info.name: image})[0][0, 0]
    if threshold == 0.5:
        prediction = logits > 0.0
    else:
        prediction = sigmoid(logits) > threshold

    target = load_mask_for_size(mask_path, size)
    return {
        "prediction": prediction,
        "target": target,
        "metrics": compute_metrics(prediction, target),
        "size": [width, height],
    }


def compare_predictions(a, b):
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return {"same_shape": False}

    equal = a == b
    different_pixels = int((~equal).sum())
    total_pixels = int(equal.size)
    return {
        "same_shape": True,
        "different_pixels": different_pixels,
        "equal_pixels": total_pixels - different_pixels,
        "equal_ratio": (total_pixels - different_pixels) / total_pixels,
    }


def save_mask(path, prediction):
    Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(path)


def summarize_runs(entries):
    if not entries:
        return None

    return {
        "count": len(entries),
        "iou_mean": sum(entry["metrics"]["iou"] for entry in entries) / len(entries),
        "dice_mean": sum(entry["metrics"]["dice"] for entry in entries) / len(entries),
        "inference_time_us_mean": sum(entry["inference_time_us"] for entry in entries) / len(entries),
        "total_time_ms_mean": sum(entry["total_time_ms"] for entry in entries) / len(entries),
    }


def run_uart_sample(connection, metadata, image_path, mask_path):
    started_at = time.perf_counter()
    inference_us, raw_output = send_image_and_get_result(connection, image_path, metadata)
    prediction = decode_prediction(raw_output, metadata)
    if prediction is None:
        return None

    target = load_mask(mask_path, metadata)
    total_time_ms = (time.perf_counter() - started_at) * 1000.0
    return {
        "image": str(image_path),
        "mask": str(mask_path),
        "prediction": prediction,
        "metrics": compute_metrics(prediction, target),
        "inference_time_us": inference_us,
        "total_time_ms": total_time_ms,
    }


def run_uart_benchmark(uart_config_path, single_sample, multi_samples, single_runs):
    import serial

    config_path, config = load_uart_config(uart_config_path)
    paths = resolve_uart_paths(config_path, config)
    serial_config = config["serial"]
    handshake_wait_seconds = serial_config.get("handshake_wait_seconds", 60)

    with serial.Serial(
        serial_config["port"],
        baudrate=serial_config["baudrate"],
        timeout=serial_config["timeout_seconds"],
        write_timeout=serial_config["timeout_seconds"],
    ) as connection:
        metadata = wait_for_handshake(connection, serial, handshake_wait_seconds)
        if not metadata:
            return None

        single_entries = []
        for _ in range(single_runs):
            entry = run_uart_sample(connection, metadata, single_sample[0], single_sample[1])
            if entry is not None:
                single_entries.append(entry)

        multi_entries = []
        for image_path, mask_path in multi_samples:
            entry = run_uart_sample(connection, metadata, image_path, mask_path)
            if entry is not None:
                multi_entries.append(entry)

    single_summary = summarize_runs(single_entries)
    multi_summary = summarize_runs(multi_entries)
    first_single = single_entries[0] if single_entries else None

    return {
        "tensor": metadata,
        "single_image": single_summary,
        "multi_image": multi_summary,
        "single_prediction": None if first_single is None else first_single["prediction"],
        "single_metrics": None if first_single is None else first_single["metrics"],
        "single_inference_time_us": None if first_single is None else first_single["inference_time_us"],
        "single_total_time_ms": None if first_single is None else first_single["total_time_ms"],
        "multi_samples": [
            {
                "image": entry["image"],
                "mask": entry["mask"],
                "iou": entry["metrics"]["iou"],
                "dice": entry["metrics"]["dice"],
                "inference_time_us": entry["inference_time_us"],
                "total_time_ms": entry["total_time_ms"],
            }
            for entry in multi_entries
        ],
        "paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare one sample across float ONNX, INT8 ONNX and UART board inference")
    parser.add_argument("--compare-config", default="models/square_seg_stm32n6/configs/compare.yaml")
    parser.add_argument("--uart-config", default="models/square_seg_stm32n6/configs/uart_inference.yaml")
    parser.add_argument("--output-dir", default="models/square_seg_stm32n6/artifacts/single_sample_compare")
    parser.add_argument("--single-runs", type=int, default=10)
    parser.add_argument("--multi-count", type=int, default=8)
    args = parser.parse_args()

    compare_config_path, train_path, train_cfg, compare_cfg = load_pipeline_config(args.compare_config, "compare")
    uart_config_path = Path(args.uart_config).expanduser().resolve()
    _, uart_config = load_uart_config(uart_config_path)

    float_model = resolve_path(compare_config_path, compare_cfg["float_model"])
    int8_model = resolve_path(compare_config_path, compare_cfg["int8_model"])
    threshold = compare_cfg.get("threshold", train_cfg.get("threshold", 0.5))

    sample = uart_config["sample"]
    image_path = resolve_path(uart_config_path, sample["image"])
    mask_path = resolve_path(uart_config_path, sample["mask"])

    test_dir = resolve_path(train_path, train_cfg["test_dir"])
    multi_samples = collect_samples(test_dir, args.multi_count)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    float_result = run_onnx(float_model, image_path, mask_path, threshold)
    int8_result = run_onnx(int8_model, image_path, mask_path, threshold)
    uart_result = run_uart_benchmark(
        uart_config_path,
        (image_path, mask_path),
        multi_samples,
        args.single_runs,
    )

    save_mask(output_dir / "float_prediction.png", float_result["prediction"])
    save_mask(output_dir / "int8_prediction.png", int8_result["prediction"])
    if uart_result is not None and uart_result["single_prediction"] is not None:
        save_mask(output_dir / "uart_prediction.png", uart_result["single_prediction"])

    report = {
        "image": str(image_path),
        "mask": str(mask_path),
        "threshold": threshold,
        "float": {
            "model": str(float_model),
            "size": float_result["size"],
            **float_result["metrics"],
        },
        "int8": {
            "model": str(int8_model),
            "size": int8_result["size"],
            **int8_result["metrics"],
        },
        "uart": None,
        "prediction_match": {
            "float_vs_int8": compare_predictions(float_result["prediction"], int8_result["prediction"]),
            "float_vs_uart": compare_predictions(float_result["prediction"], None if uart_result is None else uart_result["single_prediction"]),
            "int8_vs_uart": compare_predictions(int8_result["prediction"], None if uart_result is None else uart_result["single_prediction"]),
        },
        "outputs": {
            "float_prediction": str(output_dir / "float_prediction.png"),
            "int8_prediction": str(output_dir / "int8_prediction.png"),
            "uart_prediction": str(output_dir / "uart_prediction.png"),
            "report": str(output_dir / "report.json"),
        },
    }

    if uart_result is not None:
        report["uart"] = {
            "single_image_runs": args.single_runs,
            "single_image_inference_time_us_mean": None if uart_result["single_image"] is None else uart_result["single_image"]["inference_time_us_mean"],
            "single_image_total_time_ms_mean": None if uart_result["single_image"] is None else uart_result["single_image"]["total_time_ms_mean"],
            "single_image_iou": None if uart_result["single_metrics"] is None else uart_result["single_metrics"]["iou"],
            "single_image_dice": None if uart_result["single_metrics"] is None else uart_result["single_metrics"]["dice"],
            "eight_image_count": len(uart_result["multi_samples"]),
            "eight_image_iou_mean": None if uart_result["multi_image"] is None else uart_result["multi_image"]["iou_mean"],
            "eight_image_dice_mean": None if uart_result["multi_image"] is None else uart_result["multi_image"]["dice_mean"],
            "eight_image_inference_time_us_mean": None if uart_result["multi_image"] is None else uart_result["multi_image"]["inference_time_us_mean"],
            "eight_image_total_time_ms_mean": None if uart_result["multi_image"] is None else uart_result["multi_image"]["total_time_ms_mean"],
            "tensor": uart_result["tensor"],
            "samples": uart_result["multi_samples"],
        }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
