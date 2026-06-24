import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config_utils import load_export_config, resolve_path
from model import build_model


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "export.yaml"


def main():
    parser = argparse.ArgumentParser(description="Export the square_seg model to ONNX")
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--skip-ort-check", action="store_true")
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args()

    config_path, train_path, train_cfg, export_cfg = load_export_config(CONFIG_PATH)
    artifacts_dir = resolve_path(train_path, train_cfg["artifacts_dir"])
    model_path = artifacts_dir / "model.pt"

    default_output = export_cfg.get("output", artifacts_dir / "model.onnx")
    output_path = resolve_path(config_path, args.output or default_output)
    opset = args.opset or export_cfg.get("opset", 13)
    tolerance = args.tolerance if args.tolerance is not None else export_cfg.get("tolerance", 1e-4)
    skip_ort_check = export_cfg.get("skip_ort_check", False) or args.skip_ort_check
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_state = torch.load(model_path, map_location="cpu", weights_only=True)
    checkpoint_cfg = model_state.get("config", train_cfg)
    model = build_model(checkpoint_cfg["model"])
    model.load_state_dict(model_state["model_state"])
    model.eval()

    dummy = torch.zeros(
        1,
        1,
        checkpoint_cfg["image_height"],
        checkpoint_cfg["image_width"],
        dtype=torch.float32,
    )
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
        external_data=False,
    )

    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    ops = sorted({node.op_type for node in onnx_model.graph.node})

    report = {
        "config": str(CONFIG_PATH),
        "onnx": str(output_path),
        "ops": ops,
        "opset": opset,
    }
    if not skip_ort_check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_out = session.run(None, {"input": dummy.numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        report["max_abs_diff_torch_vs_ort"] = max_abs_diff
        report["ort_check_passed"] = max_abs_diff <= tolerance
        report["ort_tolerance"] = tolerance

    (artifacts_dir / "onnx_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
