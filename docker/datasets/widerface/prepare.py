"""Build the agent-visible WIDER FACE training split from local files."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path, PurePosixPath


DATASET_ID = "widerface"
EXPECTED_IMAGES = 12_880


def parse_annotations(path: Path) -> list[tuple[str, list[list[int]]]]:
    """Parse the WIDER FACE text format, including its zero-count sentinel rows."""

    lines = path.read_text(encoding="utf-8").splitlines()
    samples: list[tuple[str, list[list[int]]]] = []
    index = 0
    while index < len(lines):
        image_name = lines[index].strip()
        index += 1
        relative = PurePosixPath(image_name)
        if (
            not image_name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() != ".jpg"
        ):
            raise RuntimeError(f"invalid WIDER FACE image path: {image_name!r}")
        if index >= len(lines):
            raise RuntimeError(f"missing annotation count for {image_name}")
        try:
            count = int(lines[index])
        except ValueError as error:
            raise RuntimeError(f"invalid annotation count for {image_name}") from error
        index += 1
        if count < 0:
            raise RuntimeError(f"negative annotation count for {image_name}")

        # Some official text releases put one all-zero placeholder after a zero count.
        if count == 0 and index < len(lines) and not lines[index].strip().endswith(".jpg"):
            try:
                sentinel = [int(value) for value in lines[index].split()]
            except ValueError as error:
                raise RuntimeError(f"invalid zero-count sentinel for {image_name}") from error
            if sentinel != [0] * 10:
                raise RuntimeError(f"invalid zero-count sentinel for {image_name}")
            index += 1

        boxes: list[list[int]] = []
        for box_index in range(count):
            if index >= len(lines):
                raise RuntimeError(f"missing box {box_index + 1} for {image_name}")
            try:
                values = [int(value) for value in lines[index].split()]
            except ValueError as error:
                raise RuntimeError(f"non-integer box for {image_name}") from error
            index += 1
            if len(values) != 10:
                raise RuntimeError(f"box for {image_name} must contain 10 fields")
            x, y, width, height, blur, expression, illumination, invalid, occlusion, pose = values
            if x < 0 or y < 0 or width < 0 or height < 0:
                raise RuntimeError(f"invalid box geometry for {image_name}")
            if blur not in {0, 1, 2} or expression not in {0, 1}:
                raise RuntimeError(f"invalid blur/expression attributes for {image_name}")
            if illumination not in {0, 1} or invalid not in {0, 1}:
                raise RuntimeError(f"invalid illumination/invalid attributes for {image_name}")
            if occlusion not in {0, 1, 2} or pose not in {0, 1}:
                raise RuntimeError(f"invalid occlusion/pose attributes for {image_name}")
            boxes.append(values)
        samples.append((image_name, boxes))

    names = [name for name, _ in samples]
    if len(set(names)) != len(names):
        raise RuntimeError("WIDER FACE annotations contain duplicate image paths")
    return samples


def main(destination: Path, source: Path) -> None:
    images = source / "WIDER_train" / "images"
    annotations = source / "wider_face_split" / "wider_face_train_bbx_gt.txt"
    if not images.is_dir() or not annotations.is_file():
        raise RuntimeError(
            "WIDER FACE source must contain WIDER_train/images/ and "
            "wider_face_split/wider_face_train_bbx_gt.txt"
        )
    samples = parse_annotations(annotations)
    if len(samples) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_IMAGES} WIDER FACE train images, found {len(samples)}"
        )
    declared = {name for name, _ in samples}
    actual = {
        path.relative_to(images).as_posix()
        for path in images.rglob("*")
        if path.is_file() and path.suffix.lower() == ".jpg"
    }
    if declared != actual:
        missing = sorted(declared - actual)[:3]
        extra = sorted(actual - declared)[:3]
        raise RuntimeError(
            f"WIDER FACE images/annotations do not match (missing={missing}, extra={extra})"
        )

    destination.mkdir(parents=True, exist_ok=False)
    image_destination = destination / "images"
    image_destination.mkdir()
    for image_name, _ in samples:
        output = image_destination / image_name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(images / image_name, output)
    (destination / "annotations.json").write_text(
        json.dumps(
            [{"file": name, "boxes": boxes} for name, boxes in samples],
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "id": DATASET_ID,
        "display_name": "WIDER FACE",
        "task": "object_detection",
        "default_split": "train",
        "splits": {"train": {"num_examples": len(samples)}},
        "classes": ["face"],
        "class_ids": [1],
        "box_format": "xyxy",
        "source": {"name": "WIDER FACE train"},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare.py DESTINATION SOURCE")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
