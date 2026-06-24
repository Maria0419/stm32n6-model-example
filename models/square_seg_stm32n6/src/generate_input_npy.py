import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from config_utils import load_train_config, resolve_path


def load_image_tensor(image_path, size):
    with Image.open(image_path) as source:
        image = source.convert("L")
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array[None, :, :]


def load_mask_tensor(mask_path, size):
    with Image.open(mask_path) as source:
        mask = source.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
    return array[None, :, :]


def resolve_samples(input_dir):
    samples = []
    for image_path in sorted(input_dir.rglob("*_image.tif")):
        mask_path = image_path.with_name(image_path.name.replace("_image.tif", "_label.tif"))
        if mask_path.is_file():
            samples.append((image_path, mask_path))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate NPY input/output batches from dataset images for ONNX validation")
    parser.add_argument("--config", default="models/square_seg_stm32n6/configs/train.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--input-output", type=Path, default=None)
    parser.add_argument("--output-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    config_path, train_cfg = load_train_config(args.config)
    default_dir = train_cfg["train_dir"] if args.split == "train" else train_cfg["test_dir"]
    input_dir = resolve_path(config_path, args.input_dir or default_dir)
    artifacts_dir = resolve_path(config_path, train_cfg["artifacts_dir"])
    input_output_path = resolve_path(config_path, args.input_output) if args.input_output else artifacts_dir / f"{args.split}_inputs_{args.limit}.npy"
    output_output_path = resolve_path(config_path, args.output_output) if args.output_output else artifacts_dir / f"{args.split}_outputs_{args.limit}.npy"
    input_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_output_path.parent.mkdir(parents=True, exist_ok=True)

    size = (train_cfg["image_width"], train_cfg["image_height"])
    samples = resolve_samples(input_dir)
    if not samples:
        return

    offset = min(max(args.offset, 0), len(samples) - 1)
    selected = samples[offset : offset + args.limit]
    if not selected:
        return

    input_batch = np.stack([load_image_tensor(image_path, size) for image_path, _ in selected], axis=0).astype(np.float32)
    output_batch = np.stack([load_mask_tensor(mask_path, size) for _, mask_path in selected], axis=0).astype(np.float32)
    np.save(input_output_path, input_batch)
    np.save(output_output_path, output_batch)

    list_path = input_output_path.with_suffix(".txt")
    lines = [f"{image_path}	{mask_path}" for image_path, mask_path in selected]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
