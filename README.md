# STM32N6 ONNX Model Example

End-to-end host and firmware workflow for binary square segmentation on STM32N6.
The repository covers ONNX export, INT8 quantization, a committed STM32Cube.AI
project, signed flashing to external XSPI, and UART-based host/board
inference on 128x128 grayscale images.

## Repository layout

- `models/square_seg_stm32n6/configs/train.yaml`: dataset paths, model shape, and training hyperparameters.
- `models/square_seg_stm32n6/configs/export.yaml`: float ONNX export settings.
- `models/square_seg_stm32n6/configs/quantize.yaml`: INT8 QDQ quantization settings.
- `models/square_seg_stm32n6/configs/compare.yaml`: optional float-vs-INT8 accuracy gate.
- `models/square_seg_stm32n6/configs/uart_inference.yaml`: serial port, sample image, reference mask, and output paths for board inference.
- `models/square_seg_stm32n6/src/`: host-side Python scripts.
- `models/square_seg_stm32n6/artifacts/`: generated ONNX, INT8, report, and UART output files.
- `stm32-model-example-int8/`: STM32 project generated in STM32Cube.AI from the INT8 model, then customized for the final UART workflow.

## Python environment

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default presets expect the dataset under `../data/data`.

## Host-side model preparation

The clean host flow is:

```bash
python models/square_seg_stm32n6/src/export_onnx.py

python models/square_seg_stm32n6/src/quantize_onnx.py

```

Important deployment constraints preserved by this flow:

- The float ONNX model is exported with `opset 13`, which is required by the INT8 QDQ flow in this project.
- Quantization uses `256` deterministic calibration samples with grayscale `128x128` preprocessing and divide-by-`255` normalization.

Useful generated reports:

- `models/square_seg_stm32n6/artifacts/onnx_report.json`
- `models/square_seg_stm32n6/artifacts/quantization_report.json`

The optional float-vs-INT8 comparison step is still available:

```bash
python models/square_seg_stm32n6/src/compare_onnx.py
```

## STM32Cube.AI project

The committed firmware project `stm32-model-example-int8/` was generated in
STM32Cube.AI by importing the INT8 model and generating the STM32 project
there, then customized manually afterward.

Important generated directories:

- `stm32-model-example-int8/.ai/`: Cube.AI workspace, generated network output, and reports.
- `stm32-model-example-int8/Appli/AI/App/`: integrated AI runtime sources used by the final application.
- `stm32-model-example-int8/FSBL/`: first-stage bootloader sources.
- `stm32-model-example-int8/STM32CubeIDE/`: CubeIDE workspace with separate `FSBL` and `Appli` projects.

The final board behavior is implemented mainly in
`stm32-model-example-int8/Appli/Core/Src/main.c`, which adds the UART
handshake, metadata exchange, image transfer, board-side timing return, and
prediction-mask return on top of the generated AI runtime.

## Build and deploy

Open `stm32-model-example-int8/STM32CubeIDE/` in STM32CubeIDE and build both:

- `FSBL`
- `Appli`

The final deployment command is:

```bash
bash stm32-model-example-int8/tools/deploy_signed.sh
```

This script:

- Finds the latest `FSBL.bin` and `Appli.bin` outputs.
- Adds trusted headers with `STM32_SigningTool_CLI`.
- Uses `MX66UW1G45G_STM32N6570-DK.stldr` as the external loader.
- Flashes trusted `FSBL` to `0x70000000`.
- Flashes trusted `Appli` to `0x70100000`.
- Resets the target.

Board state required while programming:

- DEV boot mode
- BOOT1 high

If `stm32-model-example-int8/out/deploy` is not writable, the script falls back
to a temporary work directory under `/tmp`.

## UART inference

The board application receives one quantized `128x128` image over virtual
`USART1`, runs inference, and returns the raw INT8 output mask plus the NPU
inference time. The protocol uses:

- magic `SSEG`
- protocol version `1`
- a 16-byte header
- `HELLO`, `HELLO_ACK`, `IMAGE`, `RESULT`, and `ERROR` messages
- CRC32-protected payloads

Edit the serial port, sample image, reference mask, and output paths in
`models/square_seg_stm32n6/configs/uart_inference.yaml`, then run:

```bash
python models/square_seg_stm32n6/src/uart_inference.py
```

The script waits for the board handshake, reads tensor metadata, quantizes the
input image, sends it to the board, receives the output mask and inference time,
computes Dice and IoU against the reference mask, and saves:

- `models/square_seg_stm32n6/artifacts/uart_inference/prediction.png`
- `models/square_seg_stm32n6/artifacts/uart_inference/report.json`

The JSON report includes inference time, Dice, IoU, protocol information, and
tensor metadata.
