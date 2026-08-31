"""Command-line entry point for STM32N6 post-training quantization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .quantization import (
    DEFAULT_CALIBRATION_SAMPLES,
    QuantizationError,
    quantize_for_stm32n6,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Quantize a static float ONNX model to STM32N6 INT8 QDQ ONNX."
    )
    result.add_argument("--model", required=True, type=Path, help="Float ONNX input.")
    result.add_argument("--output", required=True, type=Path, help="QDQ ONNX output.")
    result.add_argument(
        "--preprocess",
        required=True,
        type=Path,
        help="Python hook defining preprocess(image), as used by the submission.",
    )
    result.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help=f"Training examples used for calibration (default: {DEFAULT_CALIBRATION_SAMPLES}).",
    )
    result.add_argument("--report", type=Path, help="Optional JSON report destination.")
    result.add_argument(
        "--per-tensor",
        action="store_true",
        help="Use per-tensor weights instead of the STM32N6 per-channel default.",
    )
    result.add_argument(
        "--no-moving-average",
        action="store_true",
        help="Disable MinMax calibration moving average.",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    try:
        report = quantize_for_stm32n6(
            arguments.model,
            arguments.output,
            arguments.preprocess,
            calibration_samples=arguments.samples,
            per_channel=not arguments.per_tensor,
            moving_average=not arguments.no_moving_average,
            report_path=arguments.report,
        )
    except QuantizationError as error:
        print(json.dumps({"type": "quantization_error", "error": str(error)}))
        raise SystemExit(2) from error
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
