"""Verified, offline pretrained models bundled with the benchmark image."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/opt/odbench/pretrained")


def _root() -> Path:
    return Path(os.environ.get("ODBENCH_PRETRAINED_ROOT", DEFAULT_ROOT))


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    path = _root() / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"pretrained model manifest is unavailable: {path}") from error
    if value.get("schema_version") != 1 or not isinstance(value.get("models"), list):
        raise RuntimeError("pretrained model manifest has an unsupported schema")
    return value


def list_pretrained() -> list[dict[str, Any]]:
    """Return the public metadata for all locally available model initializers."""
    omitted = {"url", "filename", "sha256", "bytes"}
    return [
        {key: value for key, value in model.items() if key not in omitted}
        for model in _manifest()["models"]
    ]


def _model_metadata(model_id: str) -> dict[str, Any]:
    for model in _manifest()["models"]:
        if model.get("id") == model_id:
            return model
    available = ", ".join(model["id"] for model in _manifest()["models"])
    raise ValueError(f"unknown pretrained model {model_id!r}; available: {available}")


@lru_cache(maxsize=None)
def _verified_weight_path(model_id: str) -> Path:
    metadata = _model_metadata(model_id)
    path = _root() / metadata["filename"]
    try:
        size = path.stat().st_size
    except OSError as error:
        raise RuntimeError(f"pretrained weights are unavailable: {path}") from error
    if size != metadata["bytes"]:
        raise RuntimeError(
            f"pretrained weights have the wrong size: {path} ({size} != {metadata['bytes']})"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != metadata["sha256"]:
        raise RuntimeError(f"pretrained weights failed SHA-256 verification: {path}")
    return path


def _state_dict(model_id: str) -> dict[str, Any]:
    import torch

    return torch.load(_verified_weight_path(model_id), map_location="cpu", weights_only=True)


def load_backbone(model_id: str):
    """Load an ImageNet initializer as a four-scale feature extractor.

    The returned module emits a tuple whose channels and spatial reductions are
    available as ``feature_channels`` and ``feature_reductions`` attributes.
    It accepts arbitrary practical image sizes; odd dimensions are rounded by
    the underlying stride-2 convolutions.
    """
    import torch
    from torchvision import models

    metadata = _model_metadata(model_id)
    if metadata["kind"] != "backbone":
        raise ValueError(f"{model_id!r} is a {metadata['kind']}, not a backbone")

    architecture = metadata["architecture"]
    if architecture == "mobilenet_v2":
        base = models.mobilenet_v2(weights=None)
    elif architecture == "mobilenet_v3_small":
        base = models.mobilenet_v3_small(weights=None)
    elif architecture == "shufflenet_v2_x0_5":
        base = models.shufflenet_v2_x0_5(weights=None)
    else:
        raise RuntimeError(f"unsupported pretrained backbone architecture: {architecture}")
    base.load_state_dict(_state_dict(model_id), strict=True)

    if architecture.startswith("mobilenet_"):
        out_indices = frozenset(metadata["feature_indices"])

        class MobileNetFeatures(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = base.features
                self.feature_channels = tuple(metadata["feature_channels"])
                self.feature_reductions = tuple(metadata["feature_reductions"])

            def forward(self, inputs):
                outputs = []
                value = inputs
                for index, layer in enumerate(self.features):
                    value = layer(value)
                    if index in out_indices:
                        outputs.append(value)
                return tuple(outputs)

        return MobileNetFeatures()

    class ShuffleNetFeatures(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = base.conv1
            self.maxpool = base.maxpool
            self.stage2 = base.stage2
            self.stage3 = base.stage3
            self.stage4 = base.stage4
            self.feature_channels = tuple(metadata["feature_channels"])
            self.feature_reductions = tuple(metadata["feature_reductions"])

        def forward(self, inputs):
            reduction4 = self.maxpool(self.conv1(inputs))
            reduction8 = self.stage2(reduction4)
            reduction16 = self.stage3(reduction8)
            reduction32 = self.stage4(reduction16)
            return reduction4, reduction8, reduction16, reduction32

    return ShuffleNetFeatures()


def load_detector(model_id: str, *, num_classes: int | None = None):
    """Load the bundled SSDLite detector, optionally replacing its COCO class head.

    ``num_classes=None`` preserves the 91-class COCO head. Passing another
    positive class count initializes only incompatible classification tensors
    from scratch while retaining all compatible COCO-trained parameters.
    """
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    metadata = _model_metadata(model_id)
    if metadata["kind"] != "detector":
        raise ValueError(f"{model_id!r} is a {metadata['kind']}, not a detector")
    if num_classes is not None and (isinstance(num_classes, bool) or num_classes < 2):
        raise ValueError("num_classes must include background and be at least 2")

    classes = 91 if num_classes is None else num_classes
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=classes,
    )
    state = _state_dict(model_id)
    if classes == 91:
        model.load_state_dict(state, strict=True)
        return model

    target = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target and value.shape == target[key].shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    if not incompatible.missing_keys:
        raise RuntimeError("expected a replacement detector head, but no tensors differed")
    return model


def load_pretrained(model_id: str, *, num_classes: int | None = None):
    """Load a registry entry using its appropriate backbone/detector loader."""
    kind = _model_metadata(model_id)["kind"]
    if kind == "backbone":
        if num_classes is not None:
            raise ValueError("num_classes is only valid for detector initializers")
        return load_backbone(model_id)
    if kind == "detector":
        return load_detector(model_id, num_classes=num_classes)
    raise RuntimeError(f"unsupported pretrained model kind: {kind}")
