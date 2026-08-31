"""Runtime hook for the training-only VisDrone2019-DET image."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class VisDroneDetectionTrain(Dataset):
    """VisDrone train images with torchvision-style detection targets."""

    classes = (
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    )
    class_ids = tuple(range(1, 11))

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
        self.images = sorted((root / "images").glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.images)

    def _target(self, index: int) -> dict[str, torch.Tensor]:
        path = self.root / "annotations" / f"{self.images[index].stem}.txt"
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        ignored_boxes: list[list[float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = [int(value) for value in line.rstrip(",").split(",")]
            x, y, width, height, score, category = values[:6]
            if width <= 0 or height <= 0:
                continue
            box = [float(x), float(y), float(x + width), float(y + height)]
            if category == 0:
                ignored_boxes.append(box)
            elif score == 1 and 1 <= category <= 10:
                boxes.append(box)
                labels.append(category)
                areas.append(float(width * height))

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
            "ignored_boxes": torch.tensor(ignored_boxes, dtype=torch.float32).reshape(-1, 4),
        }

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        with Image.open(self.images[index]) as opened:
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


def load(root: Path, split: str, **kwargs: Any) -> VisDroneDetectionTrain:
    if split != "train":
        raise ValueError("the VisDrone agent image contains only the training split")
    return VisDroneDetectionTrain(root=root, **kwargs)
