"""Prepare unlabeled VisDrone validation images and a private label artifact."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image


DATASET_ID = "visdrone"
EXPECTED_IMAGES = 548
CLASSES = (
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


def _parse_annotations(path: Path) -> tuple[list[dict[str, object]], list[list[int]]]:
    objects: list[dict[str, object]] = []
    ignored_regions: list[list[int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.rstrip(",").split(",")
        if len(fields) != 8:
            raise RuntimeError(f"{path.name}:{line_number}: expected 8 fields")
        try:
            x, y, width, height, score, category, truncation, occlusion = (
                int(value) for value in fields
            )
        except ValueError as error:
            raise RuntimeError(f"{path.name}:{line_number}: non-integer annotation") from error
        if x < 0 or y < 0 or width < 0 or height < 0:
            raise RuntimeError(f"{path.name}:{line_number}: invalid bounding box")
        if score not in {0, 1} or not 0 <= category <= 11:
            raise RuntimeError(f"{path.name}:{line_number}: invalid score or category")
        if truncation not in {0, 1} or occlusion not in {0, 1, 2}:
            raise RuntimeError(f"{path.name}:{line_number}: invalid visibility attributes")
        if width == 0 or height == 0:
            continue
        bbox = [x, y, width, height]
        if category == 0:
            ignored_regions.append(bbox)
        elif 1 <= category <= 10:
            objects.append(
                {
                    "bbox": bbox,
                    "class": category,
                    "ignore": score == 0,
                }
            )
    return objects, ignored_regions


def main(dataset_destination: Path, label_destination: Path, source: Path) -> None:
    images = source / "images"
    annotations = source / "annotations"
    if not images.is_dir() or not annotations.is_dir():
        raise RuntimeError("VisDrone source must contain images/ and annotations/")
    image_files = sorted(images.glob("*.jpg"))
    annotation_files = {path.stem: path for path in annotations.glob("*.txt")}
    if len(image_files) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_IMAGES} VisDrone val images, found {len(image_files)}"
        )
    image_stems = {path.stem for path in image_files}
    if image_stems != annotation_files.keys():
        raise RuntimeError("VisDrone validation images and annotations do not match")

    dataset_destination.mkdir(parents=True, exist_ok=False)
    label_destination.mkdir(parents=True, exist_ok=False)
    image_destination = dataset_destination / "images"
    image_destination.mkdir()
    examples: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    declared_files: list[str] = []
    for index, image_path in enumerate(image_files):
        identifier = f"visdrone-val-{index:05d}"
        relative_image = f"images/{image_path.name}"
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        objects, ignored_regions = _parse_annotations(annotation_files[image_path.stem])
        shutil.copyfile(image_path, image_destination / image_path.name)
        examples.append({"id": identifier, "file": relative_image})
        labels.append(
            {
                "id": identifier,
                "width": width,
                "height": height,
                "objects": objects,
                "ignored_regions": ignored_regions,
            }
        )
        declared_files.append(relative_image)

    manifest = {
        "schema_version": 1,
        "id": DATASET_ID,
        "display_name": "VisDrone2019-DET",
        "task": "object_detection",
        "split": "val",
        "num_examples": len(examples),
        "num_classes": len(CLASSES),
        "classes": list(CLASSES),
        "class_ids": list(range(1, len(CLASSES) + 1)),
        "box_format": "xyxy",
        "max_detections": 500,
        "examples": examples,
        "files": declared_files,
        "source": {"name": "VisDrone2019-DET-val"},
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
        "num_classes": len(CLASSES),
        "class_ids": list(range(1, len(CLASSES) + 1)),
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
