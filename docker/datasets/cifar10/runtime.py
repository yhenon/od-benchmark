"""Runtime hook for the training-only CIFAR-10 image."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10Train(Dataset):
    """The 50,000 public CIFAR-10 training examples."""

    classes = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    _files = tuple(f"data_batch_{index}" for index in range(1, 6))

    def __init__(
        self,
        root: Path,
        transform: Callable[[Image.Image], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        images: list[np.ndarray] = []
        targets: list[int] = []
        for filename in self._files:
            with (self.root / filename).open("rb") as stream:
                batch = pickle.load(stream, encoding="latin1")
            images.append(batch["data"])
            targets.extend(batch["labels"])

        self.data = np.concatenate(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image: Any = Image.fromarray(self.data[index])
        target: Any = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def load(root: Path, split: str, **kwargs: Any) -> CIFAR10Train:
    if split != "train":
        raise ValueError("the CIFAR-10 image contains only the training split")
    return CIFAR10Train(root=root, **kwargs)
