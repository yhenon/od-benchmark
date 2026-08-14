"""Generic access to the single dataset bundled into an agent image."""

from __future__ import annotations

import importlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_ROOT = Path(os.environ.get("ODBENCH_DATA_ROOT", "/opt/odbench/data"))


def dataset_id() -> str:
    identifier = os.environ.get("ODBENCH_DATASET")
    if not identifier:
        raise RuntimeError("this is the generic base image; no dataset is installed")
    return identifier


def dataset_root() -> Path:
    return DATA_ROOT / dataset_id()


@lru_cache(maxsize=1)
def dataset_manifest() -> dict[str, Any]:
    """Return the installed hook's machine-readable manifest."""

    path = dataset_root() / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid dataset manifest: {path}") from error

    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported dataset manifest schema")
    if manifest.get("id") != dataset_id():
        raise RuntimeError("dataset manifest id does not match the image")
    if not isinstance(manifest.get("splits"), dict) or not manifest["splits"]:
        raise RuntimeError("dataset manifest must declare at least one split")
    if manifest.get("default_split") not in manifest["splits"]:
        raise RuntimeError("dataset manifest has an invalid default split")
    return manifest


def load_dataset(split: str | None = None, **kwargs: Any) -> Any:
    """Construct an installed dataset split using its trusted runtime hook."""

    manifest = dataset_manifest()
    selected_split = split or manifest["default_split"]
    if selected_split not in manifest["splits"]:
        available = ", ".join(sorted(manifest["splits"]))
        raise ValueError(f"split {selected_split!r} is unavailable; choose from: {available}")

    runtime = importlib.import_module("odbench_dataset")
    return runtime.load(root=dataset_root(), split=selected_split, **kwargs)
