"""Trusted reader for unlabeled WIDER FACE validation images."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from PIL import Image


def iter_examples(root: Path) -> Iterator[tuple[str, Image.Image]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for example in manifest["examples"]:
        with Image.open(root / example["file"]) as opened:
            yield example["id"], opened.convert("RGB")
