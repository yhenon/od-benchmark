"""Validate the common dataset hook contract while building an image."""

from __future__ import annotations

from .datasets import dataset_manifest, load_dataset


def main() -> None:
    manifest = dataset_manifest()
    split = manifest["default_split"]
    dataset = load_dataset(split)
    expected_examples = manifest["splits"][split].get("num_examples")
    if expected_examples is not None and len(dataset) != expected_examples:
        raise RuntimeError(
            f"dataset length mismatch: expected {expected_examples}, got {len(dataset)}"
        )
    if len(dataset) == 0:
        raise RuntimeError("default dataset split is empty")
    dataset[0]
    print(f"validated dataset={manifest['id']} split={split} examples={len(dataset)}")


if __name__ == "__main__":
    main()
