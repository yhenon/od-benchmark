"""Build the agent-visible VisDrone2019-DET training split from local files."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


DATASET_ID = "visdrone"
EXPECTED_IMAGES = 6_471
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


def _paired_files(source: Path) -> list[tuple[Path, Path]]:
    images = source / "images"
    annotations = source / "annotations"
    if not images.is_dir() or not annotations.is_dir():
        raise RuntimeError("VisDrone source must contain images/ and annotations/")
    image_files = sorted(images.glob("*.jpg"))
    annotation_files = {path.stem: path for path in annotations.glob("*.txt")}
    if not image_files:
        raise RuntimeError("VisDrone source contains no .jpg images")
    image_stems = {path.stem for path in image_files}
    if image_stems != annotation_files.keys():
        missing = sorted(image_stems - annotation_files.keys())[:3]
        extra = sorted(annotation_files.keys() - image_stems)[:3]
        raise RuntimeError(
            f"VisDrone images/annotations do not match (missing={missing}, extra={extra})"
        )
    return [(image, annotation_files[image.stem]) for image in image_files]


def _validate_annotation(path: Path) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.rstrip(",").split(",")
        if len(fields) != 8:
            raise RuntimeError(f"{path.name}:{line_number}: expected 8 fields")
        try:
            values = [int(value) for value in fields]
        except ValueError as error:
            raise RuntimeError(f"{path.name}:{line_number}: non-integer annotation") from error
        x, y, width, height, score, category, truncation, occlusion = values
        if x < 0 or y < 0 or width < 0 or height < 0:
            raise RuntimeError(f"{path.name}:{line_number}: invalid bounding box")
        if score not in {0, 1} or not 0 <= category <= 11:
            raise RuntimeError(f"{path.name}:{line_number}: invalid score or category")
        if truncation not in {0, 1} or occlusion not in {0, 1, 2}:
            raise RuntimeError(f"{path.name}:{line_number}: invalid visibility attributes")


def main(destination: Path, source: Path) -> None:
    pairs = _paired_files(source)
    if len(pairs) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_IMAGES} VisDrone train images, found {len(pairs)}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    image_destination = destination / "images"
    annotation_destination = destination / "annotations"
    image_destination.mkdir()
    annotation_destination.mkdir()
    for image, annotation in pairs:
        _validate_annotation(annotation)
        shutil.copyfile(image, image_destination / image.name)
        shutil.copyfile(annotation, annotation_destination / annotation.name)

    manifest = {
        "schema_version": 1,
        "id": DATASET_ID,
        "display_name": "VisDrone2019-DET",
        "task": "object_detection",
        "default_split": "train",
        "splits": {"train": {"num_examples": len(pairs)}},
        "classes": list(CLASSES),
        "class_ids": list(range(1, len(CLASSES) + 1)),
        "box_format": "xyxy",
        "source": {"name": "VisDrone2019-DET-train"},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare.py DESTINATION SOURCE")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
