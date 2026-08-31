"""Prepare CIFAR-10 test images and export labels into a separate artifact."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np


ARCHIVE_URL = "https://huggingface.co/datasets/xingslong/cifar-10-batches-py/resolve/main/cifar-10-python.tar.gz"
CANONICAL_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
TEST_BATCH_MD5 = "40351d587109b95175f43aff81a1287e"
TEST_MEMBER = PurePosixPath("cifar-10-batches-py/test_batch")
CLASSES = (
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


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(dataset_destination: Path, label_destination: Path) -> None:
    dataset_destination.mkdir(parents=True, exist_ok=False)
    label_destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive = Path(temporary_directory) / "cifar-10-python.tar.gz"
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
        if md5(archive) != ARCHIVE_MD5:
            raise RuntimeError("CIFAR-10 archive checksum mismatch")
        batch_path = Path(temporary_directory) / "test_batch"
        with tarfile.open(archive, mode="r:gz") as bundle:
            member = bundle.getmember(str(TEST_MEMBER))
            source = bundle.extractfile(member)
            if source is None or not member.isfile():
                raise RuntimeError("CIFAR-10 test batch is missing")
            with source, batch_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        if md5(batch_path) != TEST_BATCH_MD5:
            raise RuntimeError("CIFAR-10 test batch checksum mismatch")
        with batch_path.open("rb") as stream:
            batch = pickle.load(stream, encoding="latin1")

    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = [int(label) for label in batch["labels"]]
    np.save(dataset_destination / "images.npy", images, allow_pickle=False)

    manifest = {
        "schema_version": 1,
        "id": "cifar10",
        "display_name": "CIFAR-10",
        "task": "image_classification",
        "split": "test",
        "num_examples": len(images),
        "num_classes": len(CLASSES),
        "classes": list(CLASSES),
        "id_prefix": "cifar10-test-",
        "files": ["images.npy"],
        "source": {"url": CANONICAL_URL, "archive_md5": ARCHIVE_MD5},
    }
    (dataset_destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    label_export = {
        "schema_version": 1,
        "dataset": "cifar10",
        "split": "test",
        "num_examples": len(labels),
        "labels": [
            {"id": f"cifar10-test-{index:05d}", "class": label}
            for index, label in enumerate(labels)
        ],
    }
    (label_destination / "labels.json").write_text(
        json.dumps(label_export, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "usage: prepare.py DATASET_DESTINATION LABEL_DESTINATION [SOURCE]"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]))
