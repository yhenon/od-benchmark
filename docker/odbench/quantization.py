"""Post-training QDQ quantization for the STM32N6 hardware target.

This module deliberately exposes a small, reproducible subset of ONNX Runtime's
quantizer.  Calibration examples come from the benchmark's public training
split and are transformed with the same ``preprocess(image)`` hook used by the
submission evaluator.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .datasets import load_dataset


TARGET = "stm32n6"
TARGET_OPSET = 17
DEFAULT_CALIBRATION_SAMPLES = 256
MAX_CALIBRATION_SAMPLES = 4096


class QuantizationError(RuntimeError):
    """Raised when an input cannot be safely quantized for the target."""


def _load_hook(path: Path) -> ModuleType:
    if not path.is_file():
        raise QuantizationError(f"preprocess hook is missing: {path}")
    spec = importlib.util.spec_from_file_location("odbench_calibration_preprocess", path)
    if spec is None or spec.loader is None:
        raise QuantizationError(f"cannot import preprocess hook: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise QuantizationError(f"failed to import preprocess hook: {error}") from error
    if not callable(getattr(module, "preprocess", None)):
        raise QuantizationError("preprocess hook must define preprocess(image)")
    return module


def _dataset_image(item: Any) -> Any:
    if isinstance(item, (tuple, list)):
        if not item:
            raise QuantizationError("calibration dataset returned an empty item")
        return item[0]
    return item


def _element_type_to_dtype(element_type: int) -> np.dtype[Any]:
    # Import lazily so the dataset API remains usable in minimal environments.
    import onnx

    mapping = {
        onnx.TensorProto.FLOAT: np.dtype(np.float32),
        onnx.TensorProto.DOUBLE: np.dtype(np.float64),
        onnx.TensorProto.FLOAT16: np.dtype(np.float16),
        onnx.TensorProto.INT8: np.dtype(np.int8),
        onnx.TensorProto.UINT8: np.dtype(np.uint8),
        onnx.TensorProto.INT16: np.dtype(np.int16),
        onnx.TensorProto.UINT16: np.dtype(np.uint16),
        onnx.TensorProto.INT32: np.dtype(np.int32),
        onnx.TensorProto.UINT32: np.dtype(np.uint32),
        onnx.TensorProto.INT64: np.dtype(np.int64),
        onnx.TensorProto.UINT64: np.dtype(np.uint64),
        onnx.TensorProto.BOOL: np.dtype(np.bool_),
    }
    try:
        return mapping[element_type]
    except KeyError as error:
        raise QuantizationError(
            f"unsupported calibration input element type: {element_type}"
        ) from error


@dataclass(frozen=True)
class ModelInput:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]


def _model_inputs(model: Any) -> tuple[ModelInput, ...]:
    initializer_names = {value.name for value in model.graph.initializer}
    inputs: list[ModelInput] = []
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            raise QuantizationError(f"model input {value.name!r} has no shape")
        dimensions: list[int] = []
        for dimension in tensor.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                raise QuantizationError(
                    f"model input {value.name!r} must have a fully static shape; "
                    "export with batch size 1 and no dynamic axes"
                )
            dimensions.append(int(dimension.dim_value))
        if not dimensions or dimensions[0] != 1:
            raise QuantizationError(
                f"model input {value.name!r} must use a static batch size of 1"
            )
        inputs.append(
            ModelInput(
                name=value.name,
                shape=tuple(dimensions),
                dtype=_element_type_to_dtype(tensor.elem_type),
            )
        )
    if not inputs:
        raise QuantizationError("ONNX model has no runtime inputs")
    return tuple(inputs)


def _normalize_feed(value: Any, inputs: Sequence[ModelInput]) -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        raw = dict(value)
    elif len(inputs) == 1:
        raw = {inputs[0].name: value}
    else:
        raise QuantizationError(
            "preprocess must return an input-name mapping for a multi-input model"
        )
    expected_names = {item.name for item in inputs}
    actual_names = set(raw)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise QuantizationError(
            f"preprocess input names do not match ONNX model; missing={missing}, extra={extra}"
        )
    result: dict[str, np.ndarray] = {}
    for model_input in inputs:
        array = np.asarray(raw[model_input.name])
        if tuple(array.shape) != model_input.shape:
            raise QuantizationError(
                f"preprocess produced {model_input.name!r} shape {tuple(array.shape)}; "
                f"expected {model_input.shape}"
            )
        if array.dtype != model_input.dtype:
            if not np.can_cast(array.dtype, model_input.dtype, casting="same_kind"):
                raise QuantizationError(
                    f"preprocess produced {model_input.name!r} dtype {array.dtype}; "
                    f"expected {model_input.dtype}"
                )
            array = array.astype(model_input.dtype, copy=False)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise QuantizationError(
                f"preprocess produced non-finite values for {model_input.name!r}"
            )
        result[model_input.name] = np.ascontiguousarray(array)
    return result


class DatasetCalibrationReader:
    """ONNX Runtime calibration reader backed by deterministic training examples."""

    def __init__(
        self,
        dataset: Sequence[Any],
        preprocess: Any,
        inputs: Sequence[ModelInput],
        samples: int,
    ) -> None:
        if samples < 1:
            raise QuantizationError("calibration sample count must be positive")
        if len(dataset) < 1:
            raise QuantizationError("calibration dataset is empty")
        count = min(samples, len(dataset))
        # Cover the whole public training split while remaining deterministic.
        self.indices = tuple(
            int(value) for value in np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
        )
        self.dataset = dataset
        self.preprocess = preprocess
        self.inputs = tuple(inputs)
        self._iterator: Iterator[int] = iter(self.indices)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            index = next(self._iterator)
        except StopIteration:
            return None
        try:
            image = _dataset_image(self.dataset[index])
            transformed = self.preprocess(image)
            return _normalize_feed(transformed, self.inputs)
        except QuantizationError:
            raise
        except Exception as error:
            raise QuantizationError(
                f"preprocess failed for calibration sample {index}: {error}"
            ) from error

    def rewind(self) -> None:
        self._iterator = iter(self.indices)


def _opset(model: Any) -> int:
    versions = [item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}]
    if len(versions) != 1:
        raise QuantizationError("ONNX model must declare exactly one ai.onnx opset")
    return int(versions[0])


def _graph_summary(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(path, load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {
        "nodes": len(model.graph.node),
        "opset": _opset(model),
        "quantize_linear_nodes": counts.get("QuantizeLinear", 0),
        "dequantize_linear_nodes": counts.get("DequantizeLinear", 0),
        "operators": dict(sorted(counts.items())),
    }


def _compare_models(
    float_model: Path,
    quantized_model: Path,
    feed: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    float_session = ort.InferenceSession(
        str(float_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    quantized_session = ort.InferenceSession(
        str(quantized_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    float_outputs = float_session.run(None, dict(feed))
    quantized_outputs = quantized_session.run(None, dict(feed))
    if len(float_outputs) != len(quantized_outputs):
        raise QuantizationError("quantized model changed the number of outputs")
    comparisons: list[dict[str, Any]] = []
    for index, (reference, candidate) in enumerate(zip(float_outputs, quantized_outputs)):
        reference_array = np.asarray(reference)
        candidate_array = np.asarray(candidate)
        if reference_array.shape != candidate_array.shape:
            raise QuantizationError(f"quantized model changed output {index} shape")
        if not np.isfinite(candidate_array).all():
            raise QuantizationError(f"quantized model produced non-finite output {index}")
        difference = np.abs(reference_array.astype(np.float64) - candidate_array.astype(np.float64))
        comparisons.append(
            {
                "index": index,
                "shape": list(candidate_array.shape),
                "mean_absolute_error": float(difference.mean()) if difference.size else 0.0,
                "max_absolute_error": float(difference.max()) if difference.size else 0.0,
            }
        )
    return {"passed": True, "outputs": comparisons}


def quantize_for_stm32n6(
    model: str | os.PathLike[str],
    output: str | os.PathLike[str],
    preprocess: str | os.PathLike[str],
    *,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    per_channel: bool = True,
    moving_average: bool = True,
    target_opset: int = TARGET_OPSET,
    report_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Produce a static INT8 QDQ ONNX model calibrated on public training data."""

    if isinstance(calibration_samples, bool) or not isinstance(calibration_samples, int):
        raise QuantizationError("calibration_samples must be an integer")
    if not 1 <= calibration_samples <= MAX_CALIBRATION_SAMPLES:
        raise QuantizationError(
            f"calibration_samples must be between 1 and {MAX_CALIBRATION_SAMPLES}"
        )
    if target_opset != TARGET_OPSET:
        raise QuantizationError(f"{TARGET} quantization requires opset {TARGET_OPSET}")

    source = Path(model).resolve()
    destination = Path(output).resolve()
    hook_path = Path(preprocess).resolve()
    if not source.is_file() or source.suffix.lower() != ".onnx":
        raise QuantizationError(f"float ONNX model is missing: {source}")
    if destination == source:
        raise QuantizationError("quantized output must not overwrite the float model")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        import onnx
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as error:
        raise QuantizationError(
            "STM32N6 quantization requires onnx and onnxruntime.quantization"
        ) from error

    try:
        float_model = onnx.load(source, load_external_data=True)
        onnx.checker.check_model(float_model)
    except Exception as error:
        raise QuantizationError(f"invalid float ONNX model: {error}") from error
    original_opset = _opset(float_model)

    with tempfile.TemporaryDirectory(prefix="odbench-quantize-") as temporary:
        converted_input = Path(temporary) / "float-opset17.onnx"
        quantizer_input = Path(temporary) / "float-opset17-preprocessed.onnx"
        try:
            if original_opset == target_opset:
                onnx.save_model(float_model, converted_input, save_as_external_data=False)
            else:
                converted = onnx.version_converter.convert_version(float_model, target_opset)
                onnx.checker.check_model(converted)
                onnx.save_model(converted, converted_input, save_as_external_data=False)
        except Exception as error:
            raise QuantizationError(
                f"cannot convert ONNX opset {original_opset} to target opset {target_opset}: {error}"
            ) from error

        preprocessing_mode = "symbolic_shape_and_optimization"
        try:
            quant_pre_process(
                input_model=converted_input,
                output_model_path=quantizer_input,
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
                save_as_external_data=False,
            )
        except Exception:
            # Symbolic shape inference does not support every otherwise valid
            # graph. Retain ordinary ONNX shape inference as a safe fallback.
            preprocessing_mode = "onnx_shape_only"
            try:
                quant_pre_process(
                    input_model=converted_input,
                    output_model_path=quantizer_input,
                    skip_optimization=True,
                    skip_onnx_shape=False,
                    skip_symbolic_shape=True,
                    save_as_external_data=False,
                )
            except Exception as error:
                raise QuantizationError(
                    f"ONNX Runtime quantization preprocessing failed: {error}"
                ) from error

        converted_model = onnx.load(quantizer_input, load_external_data=False)
        inputs = _model_inputs(converted_model)
        hook = _load_hook(hook_path)
        dataset = load_dataset(split="train")
        reader = DatasetCalibrationReader(
            dataset,
            hook.preprocess,
            inputs,
            calibration_samples,
        )
        try:
            quantize_static(
                model_input=str(quantizer_input),
                model_output=str(destination),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=per_channel,
                reduce_range=False,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={
                    "ActivationSymmetric": False,
                    "WeightSymmetric": True,
                    "CalibMovingAverage": moving_average,
                    "CalibMovingAverageConstant": 0.01,
                },
            )
        except Exception as error:
            destination.unlink(missing_ok=True)
            raise QuantizationError(f"ONNX Runtime quantization failed: {error}") from error

        try:
            quantized = onnx.load(destination, load_external_data=False)
            onnx.checker.check_model(quantized)
            reader.rewind()
            smoke_feed = reader.get_next()
            if smoke_feed is None:
                raise QuantizationError("calibration reader produced no samples")
            smoke_test = _compare_models(quantizer_input, destination, smoke_feed)
        except QuantizationError:
            destination.unlink(missing_ok=True)
            raise
        except Exception as error:
            destination.unlink(missing_ok=True)
            raise QuantizationError(f"quantized ONNX validation failed: {error}") from error

    graph = _graph_summary(destination)
    if graph["quantize_linear_nodes"] == 0 or graph["dequantize_linear_nodes"] == 0:
        destination.unlink(missing_ok=True)
        raise QuantizationError("quantizer produced a model without Q/DQ nodes")
    report = {
        "schema_version": 1,
        "type": "stm32n6_quantization",
        "target": TARGET,
        "source_model": str(source),
        "output_model": str(destination),
        "output_bytes": destination.stat().st_size,
        "original_opset": original_opset,
        "target_opset": target_opset,
        "calibration": {
            "dataset_split": "train",
            "requested_samples": calibration_samples,
            "used_samples": len(reader.indices),
            "selection": "deterministic_evenly_spaced",
            "preprocess": str(hook_path),
        },
        "settings": {
            "format": "QDQ",
            "activations": "int8",
            "weights": "int8",
            "per_channel": per_channel,
            "calibration_method": "MinMax",
            "calibration_moving_average": moving_average,
            "float_input_output": True,
            "preprocessing": preprocessing_mode,
        },
        "graph": graph,
        "smoke_test": smoke_test,
    }
    report_destination = (
        Path(report_path).resolve()
        if report_path is not None
        else destination.with_suffix(".quantization.json")
    )
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(report_destination)
    report_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
