"""Runtime hook for the training-only WIDER FACE image."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class WiderFaceDetectionTrain(Dataset):
    """WIDER FACE train images with torchvision-style detection targets."""

    classes = ("face",)
    class_ids = (1,)

    def __init__(
        self,
        root: Path,
        transforms: Callable[[Image.Image, dict[str, torch.Tensor]], Any] | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        target_transform: Callable[[dict[str, torch.Tensor]], Any] | None = None,
    ) -> None:
        self.root = root
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        self.samples = json.loads((root / "annotations.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.samples)

    def _target(self, index: int) -> dict[str, torch.Tensor]:
        boxes: list[list[float]] = []
        areas: list[float] = []
        ignored_boxes: list[list[float]] = []
        for values in self.samples[index]["boxes"]:
            x, y, width, height = values[:4]
            invalid = values[7]
            if width <= 0 or height <= 0:
                continue
            box = [float(x), float(y), float(x + width), float(y + height)]
            if invalid:
                ignored_boxes.append(box)
            else:
                boxes.append(box)
                areas.append(float(width * height))
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "ignored_boxes": torch.tensor(ignored_boxes, dtype=torch.float32).reshape(-1, 4),
        }

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        with Image.open(self.root / "images" / self.samples[index]["file"]) as opened:
            image: Any = opened.convert("RGB")
        target: Any = self._target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return image, target


def load(root: Path, split: str, **kwargs: Any) -> WiderFaceDetectionTrain:
    if split != "train":
        raise ValueError("the WIDER FACE agent image contains only the training split")
    return WiderFaceDetectionTrain(root=root, **kwargs)
