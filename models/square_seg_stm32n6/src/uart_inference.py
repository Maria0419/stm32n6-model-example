import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "uart_inference.yaml"
MAGIC = b"SSEG"
VERSION = 1
HEADER = struct.Struct("<4sBBHII")
HELLO_METADATA = struct.Struct("<HHIIfhfh")
RESULT_TIMING = struct.Struct("<Q")

MSG_HELLO = 0x01
MSG_IMAGE = 0x02
MSG_HELLO_ACK = 0x81
MSG_RESULT = 0x82
MSG_ERROR = 0xFF


def resolve_path(config_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config_path, config


def resolve_paths(config_path, config):
    sample = config["sample"]
    output = config["output"]
    benchmark = config.get("benchmark", {})
    output_dir = resolve_path(config_path, output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "image": resolve_path(config_path, sample["image"]),
        "mask": resolve_path(config_path, sample["mask"]),
        "dataset_dir": resolve_path(config_path, benchmark.get("dataset_dir", sample["image"])).parent,
        "mask_output": output_dir / output["mask_filename"],
        "report_output": output_dir / output["report_filename"],
    }


def pack_frame(message, payload=b"", status=0):
    checksum = zlib.crc32(payload) if payload else 0
    return HEADER.pack(MAGIC, VERSION, message, status, len(payload), checksum) + payload


def read_exact(connection, length):
    data = bytearray()
    while len(data) < length:
        chunk = connection.read(length - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def read_frame(connection, expected_message=None):
    header = read_exact(connection, HEADER.size)
    if len(header) != HEADER.size:
        return None

    magic, version, message, status, length, checksum = HEADER.unpack(header)
    payload = read_exact(connection, length)
    payload_checksum = zlib.crc32(payload) if payload else 0

    if len(payload) != length:
        return None
    if magic != MAGIC or version != VERSION:
        return None
    if payload_checksum != checksum:
        return None
    if message == MSG_ERROR or status:
        return None
    if expected_message is not None and message != expected_message:
        return None
    return payload


def parse_metadata(payload):
    if not payload or len(payload) != HELLO_METADATA.size:
        return None

    values = HELLO_METADATA.unpack(payload)
    keys = (
        "width",
        "height",
        "input_size",
        "output_size",
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
    )
    return dict(zip(keys, values))


def wait_for_handshake(connection, serial_module, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    original_timeout = connection.timeout
    connection.timeout = 15.0

    try:
        while time.monotonic() < deadline:
            try:
                connection.reset_input_buffer()
                connection.write(pack_frame(MSG_HELLO))
                connection.flush()
                metadata = parse_metadata(read_frame(connection, MSG_HELLO_ACK))
                if metadata:
                    return metadata
            except serial_module.SerialException:
                pass
            time.sleep(0.5)
    finally:
        connection.timeout = original_timeout

    return None


def quantize_image(image_path, metadata):
    with Image.open(image_path) as source:
        image = source.convert("L")
        size = (metadata["width"], metadata["height"])
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        normalized = np.asarray(image, dtype=np.float32) / 255.0

    quantized = np.rint(normalized / metadata["input_scale"] + metadata["input_zero_point"])
    return np.clip(quantized, -128, 127).astype(np.int8)


def load_mask(mask_path, metadata):
    with Image.open(mask_path) as source:
        mask = source.convert("L")
        size = (metadata["width"], metadata["height"])
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 127


def send_image_and_get_result(connection, image_path, metadata):
    quantized = quantize_image(image_path, metadata)
    connection.write(pack_frame(MSG_IMAGE, quantized.tobytes()))
    connection.flush()

    payload = read_frame(connection, MSG_RESULT)
    if not payload or len(payload) < RESULT_TIMING.size:
        return None, None

    inference_us = RESULT_TIMING.unpack(payload[: RESULT_TIMING.size])[0]
    raw_output = payload[RESULT_TIMING.size :]
    return inference_us, raw_output


def decode_prediction(raw_output, metadata):
    if raw_output is None:
        return None
    size = metadata["height"] * metadata["width"]
    output = np.frombuffer(raw_output[:size], dtype=np.int8)
    if output.size != size:
        return None
    output = output.reshape(metadata["height"], metadata["width"])
    return output.astype(np.int16) > metadata["output_zero_point"]


def compute_metrics(prediction, target):
    intersection = float(np.logical_and(prediction, target).sum())
    prediction_sum = float(prediction.sum())
    target_sum = float(target.sum())
    return {
        "dice": (2.0 * intersection + 1.0) / (prediction_sum + target_sum + 1.0),
        "iou": (intersection + 1.0) / (prediction_sum + target_sum - intersection + 1.0),
    }


def collect_samples(dataset_dir, limit):
    samples = []
    for image_path in sorted(Path(dataset_dir).rglob("*_image.tif")):
        mask_path = image_path.with_name(image_path.name.replace("_image.tif", "_label.tif"))
        if mask_path.is_file():
            samples.append((image_path, mask_path))
        if len(samples) >= limit:
            break
    return samples


def run_sample(connection, metadata, image_path, mask_path):
    started_at = time.perf_counter()
    inference_us, raw_output = send_image_and_get_result(connection, image_path, metadata)
    prediction = decode_prediction(raw_output, metadata)
    if prediction is None:
        return None

    target = load_mask(mask_path, metadata)
    metrics = compute_metrics(prediction, target)
    total_time_ms = (time.perf_counter() - started_at) * 1000.0
    return {
        "image": str(image_path),
        "mask": str(mask_path),
        "prediction": prediction,
        "metrics": metrics,
        "inference_time_us": inference_us,
        "inference_time_ms": None if inference_us is None else inference_us / 1000.0,
        "total_time_ms": total_time_ms,
    }


def summarize_entries(entries):
    if not entries:
        return None
    return {
        "count": len(entries),
        "iou_mean": sum(entry["metrics"]["iou"] for entry in entries) / len(entries),
        "dice_mean": sum(entry["metrics"]["dice"] for entry in entries) / len(entries),
        "inference_time_us_mean": sum(entry["inference_time_us"] for entry in entries) / len(entries),
        "inference_time_ms_mean": sum(entry["inference_time_ms"] for entry in entries) / len(entries),
        "total_time_ms_mean": sum(entry["total_time_ms"] for entry in entries) / len(entries),
    }


def build_report(paths, metadata, single_entry, single_summary, multi_entries, multi_summary):
    if single_entry is None:
        return None

    report = {
        "image": single_entry["image"],
        "ground_truth_mask": single_entry["mask"],
        "prediction_mask": str(paths["mask_output"]),
        "inference_time_us": single_entry["inference_time_us"],
        "inference_time_ms": single_entry["inference_time_ms"],
        "total_time_ms": single_entry["total_time_ms"],
        **single_entry["metrics"],
        "protocol": {"magic": MAGIC.decode("ascii"), "version": VERSION},
        "tensor": {
            "width": metadata["width"],
            "height": metadata["height"],
            "input_size_bytes": metadata["input_size"],
            "output_size_bytes": metadata["output_size"],
            "input_scale": metadata["input_scale"],
            "input_zero_point": metadata["input_zero_point"],
            "output_scale": metadata["output_scale"],
            "output_zero_point": metadata["output_zero_point"],
        },
        "single_image": {
            "image": single_entry["image"],
            "ground_truth_mask": single_entry["mask"],
            "prediction_mask": str(paths["mask_output"]),
            "runs": None if single_summary is None else single_summary["count"],
            "iou": single_entry["metrics"]["iou"],
            "dice": single_entry["metrics"]["dice"],
            "inference_time_us": single_entry["inference_time_us"],
            "inference_time_ms": single_entry["inference_time_ms"],
            "total_time_ms": single_entry["total_time_ms"],
            "inference_time_us_mean": None if single_summary is None else single_summary["inference_time_us_mean"],
            "inference_time_ms_mean": None if single_summary is None else single_summary["inference_time_ms_mean"],
            "total_time_ms_mean": None if single_summary is None else single_summary["total_time_ms_mean"],
        },
        "multi_image": {
            "count": 0 if multi_summary is None else multi_summary["count"],
            "iou_mean": None if multi_summary is None else multi_summary["iou_mean"],
            "dice_mean": None if multi_summary is None else multi_summary["dice_mean"],
            "inference_time_us_mean": None if multi_summary is None else multi_summary["inference_time_us_mean"],
            "inference_time_ms_mean": None if multi_summary is None else multi_summary["inference_time_ms_mean"],
            "total_time_ms_mean": None if multi_summary is None else multi_summary["total_time_ms_mean"],
            "samples": [
                {
                    "image": entry["image"],
                    "ground_truth_mask": entry["mask"],
                    "iou": entry["metrics"]["iou"],
                    "dice": entry["metrics"]["dice"],
                    "inference_time_us": entry["inference_time_us"],
                    "inference_time_ms": entry["inference_time_ms"],
                    "total_time_ms": entry["total_time_ms"],
                }
                for entry in multi_entries
            ],
        },
    }
    return report


def run(config_path=CONFIG_PATH):
    import serial

    config_path, config = load_config(config_path)
    paths = resolve_paths(config_path, config)
    serial_config = config["serial"]
    benchmark = config.get("benchmark", {})
    handshake_wait_seconds = serial_config.get("handshake_wait_seconds", 60)
    single_runs = benchmark.get("single_runs", 10)
    multi_image_count = benchmark.get("multi_image_count", 8)
    multi_samples = collect_samples(paths["dataset_dir"], multi_image_count)

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
            entry = run_sample(connection, metadata, paths["image"], paths["mask"])
            if entry is not None:
                single_entries.append(entry)

        multi_entries = []
        for image_path, mask_path in multi_samples:
            entry = run_sample(connection, metadata, image_path, mask_path)
            if entry is not None:
                multi_entries.append(entry)

    if not single_entries:
        return None

    single_entry = single_entries[0]
    single_summary = summarize_entries(single_entries)
    multi_summary = summarize_entries(multi_entries)

    Image.fromarray(single_entry["prediction"].astype(np.uint8) * 255, mode="L").save(paths["mask_output"])
    report = build_report(paths, metadata, single_entry, single_summary, multi_entries, multi_summary)
    paths["report_output"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    run()


if __name__ == "__main__":
    main()
