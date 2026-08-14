"""Validate that an evaluator image contains examples but no label file."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path


def main() -> None:
    identifier = os.environ["ODBENCH_EVAL_DATASET"]
    root = Path(os.environ["ODBENCH_EVAL_DATA_ROOT"]) / identifier
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("id") != identifier:
        raise RuntimeError("invalid evaluation dataset manifest")
    allowed_files = {"manifest.json", *manifest.get("files", [])}
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if actual_files != allowed_files:
        raise RuntimeError("evaluation dataset contains undeclared files")
    if any("label" in path.lower() for path in actual_files):
        raise RuntimeError("evaluation image must not contain a label file")

    runtime = importlib.import_module("eval_dataset")
    count = sum(1 for _ in runtime.iter_examples(root))
    if count != manifest.get("num_examples"):
        raise RuntimeError("evaluation example count does not match its manifest")
    print(f"validated unlabeled dataset={identifier} examples={count}")


if __name__ == "__main__":
    main()
