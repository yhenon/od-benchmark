"""Trusted reader for the unlabeled CIFAR-10 evaluation images."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image


def iter_examples(root: Path) -> Iterator[tuple[str, Image.Image]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    images = np.load(root / "images.npy", mmap_mode="r", allow_pickle=False)
    prefix = manifest["id_prefix"]
    for index in range(len(images)):
        yield f"{prefix}{index:05d}", Image.fromarray(np.array(images[index], copy=True))
