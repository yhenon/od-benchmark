"""Prepare unlabeled WIDER FACE validation images and a private label artifact."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from PIL import Image


DATASET_ID = "widerface"
EXPECTED_IMAGES = 3_226
MAX_DETECTIONS = 2_000


def _load_parser():
    path = Path(__file__).resolve().parents[3] / "datasets" / "widerface" / "prepare.py"
    if not path.is_file():
        # In the evaluator build context only this file is copied, so keep parsing local.
        return None
    spec = importlib.util.spec_from_file_location("widerface_train_prepare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_annotations


def parse_annotations(path: Path) -> list[tuple[str, list[list[int]]]]:
    parser = _load_parser()
    if parser is not None:
        return parser(path)

    # The Docker build copies this hook alone. Parse the already validated public format here.
    lines = path.read_text(encoding="utf-8").splitlines()
    result: list[tuple[str, list[list[int]]]] = []
    index = 0
    while index < len(lines):
        image_name = lines[index].strip()
        index += 1
        count = int(lines[index])
        index += 1
        if count < 0 or not image_name.endswith(".jpg") or image_name.startswith("/"):
            raise RuntimeError(f"invalid WIDER FACE annotation header: {image_name!r}")
        if count == 0 and index < len(lines) and not lines[index].strip().endswith(".jpg"):
            if [int(value) for value in lines[index].split()] != [0] * 10:
                raise RuntimeError(f"invalid zero-count sentinel for {image_name}")
            index += 1
        boxes = []
        for _ in range(count):
            values = [int(value) for value in lines[index].split()]
            index += 1
            if len(values) != 10 or min(values[:4]) < 0:
                raise RuntimeError(f"invalid box for {image_name}")
            boxes.append(values)
        result.append((image_name, boxes))
    return result


def main(dataset_destination: Path, label_destination: Path, source: Path) -> None:
    images = source / "WIDER_val" / "images"
    annotations = source / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    if not images.is_dir() or not annotations.is_file():
        raise RuntimeError(
            "WIDER FACE source must contain WIDER_val/images/ and "
            "wider_face_split/wider_face_val_bbx_gt.txt"
        )
    samples = parse_annotations(annotations)
    if len(samples) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_IMAGES} WIDER FACE val images, found {len(samples)}"
        )
    declared = {name for name, _ in samples}
    actual = {
        path.relative_to(images).as_posix()
        for path in images.rglob("*")
        if path.is_file() and path.suffix.lower() == ".jpg"
    }
    if declared != actual:
        raise RuntimeError("WIDER FACE validation images and annotations do not match")

    dataset_destination.mkdir(parents=True, exist_ok=False)
    label_destination.mkdir(parents=True, exist_ok=False)
    image_destination = dataset_destination / "images"
    image_destination.mkdir()
    examples: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    declared_files: list[str] = []
    for index, (image_name, boxes) in enumerate(samples):
        identifier = f"widerface-val-{index:05d}"
        relative_image = f"images/{image_name}"
        image_path = images / image_name
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        output = image_destination / image_name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, output)
        objects = [
            {
                "bbox": values[:4],
                "class": 1,
                "ignore": bool(values[7]),
            }
            for values in boxes
            if values[2] > 0 and values[3] > 0
        ]
        examples.append({"id": identifier, "file": relative_image})
        labels.append(
            {
                "id": identifier,
                "width": width,
                "height": height,
                "objects": objects,
                "ignored_regions": [],
            }
        )
        declared_files.append(relative_image)

    manifest = {
        "schema_version": 1,
        "id": DATASET_ID,
        "display_name": "WIDER FACE",
        "task": "object_detection",
        "split": "val",
        "num_examples": len(examples),
        "num_classes": 1,
        "classes": ["face"],
        "class_ids": [1],
        "box_format": "xyxy",
        "max_detections": MAX_DETECTIONS,
        "examples": examples,
        "files": declared_files,
        "source": {"name": "WIDER FACE val"},
    }
    (dataset_destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    label_export = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "task": "object_detection",
        "split": "val",
        "num_examples": len(labels),
        "num_classes": 1,
        "class_ids": [1],
        "iou_thresholds": [0.5],
        "max_detections": MAX_DETECTIONS,
        "recall_limits": [],
        "reported_metrics": ["AP"],
        "labels": labels,
    }
    (label_destination / "labels.json").write_text(
        json.dumps(label_export, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: prepare.py DATASET_DESTINATION LABEL_DESTINATION SOURCE"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
