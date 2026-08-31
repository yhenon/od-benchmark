#!/usr/bin/env python3
"""Trusted outer scorer for classification and object-detection tasks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from typing import Any


IOU_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(10))
MAX_DETECTIONS = (1, 10, 100, 500)


def read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _common_validation(labels: dict[str, Any], predictions: dict[str, Any]) -> int:
    if labels.get("schema_version") != 1:
        raise ValueError("unsupported label schema")
    if predictions.get("schema_version") != 1:
        raise ValueError("unsupported prediction schema")
    for field in ("dataset", "split", "num_examples"):
        if labels.get(field) != predictions.get(field):
            raise ValueError("labels and predictions do not describe the same evaluation set")
    expected = labels.get("num_examples")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise ValueError("invalid evaluation example count")
    return expected


def _unique_items(items: Any, expected: int, value_name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list) or len(items) != expected:
        raise ValueError(f"{value_name} must contain the expected number of items")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"invalid {value_name} item")
        if item["id"] in result:
            raise ValueError(f"duplicate id in {value_name}")
        result[item["id"]] = item
    return result


def score_classification(
    labels_document: dict[str, Any], predictions_document: dict[str, Any]
) -> dict[str, float]:
    expected = _common_validation(labels_document, predictions_document)
    label_items = _unique_items(labels_document.get("labels"), expected, "labels")
    prediction_items = _unique_items(
        predictions_document.get("predictions"), expected, "predictions"
    )
    if label_items.keys() != prediction_items.keys():
        raise ValueError("labels and predictions must contain the exact expected example ids")
    correct = 0
    for identifier, label in label_items.items():
        expected_class = label.get("class")
        predicted_class = prediction_items[identifier].get("class")
        if isinstance(expected_class, bool) or not isinstance(expected_class, int):
            raise ValueError("invalid classification label")
        if isinstance(predicted_class, bool) or not isinstance(predicted_class, int):
            raise ValueError("invalid classification prediction")
        correct += predicted_class == expected_class
    return {"top1_accuracy": correct / expected}


def _bbox(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be an [x, y, width, height] list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} coordinates must be numbers")
    x, y, width, height = (float(item) for item in value)
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        raise ValueError(f"{name} coordinates must be finite")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"{name} is invalid")
    return x, y, width, height


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def _covered_area(
    box: tuple[float, float, float, float],
    regions: Iterable[tuple[float, float, float, float]],
) -> float:
    """Area of box covered by the union of axis-aligned ignored regions."""

    clipped: list[tuple[float, float, float, float]] = []
    box_right = box[0] + box[2]
    box_bottom = box[1] + box[3]
    for region in regions:
        left = max(box[0], region[0])
        top = max(box[1], region[1])
        right = min(box_right, region[0] + region[2])
        bottom = min(box_bottom, region[1] + region[3])
        if left < right and top < bottom:
            clipped.append((left, top, right, bottom))
    x_values = sorted({value for rectangle in clipped for value in (rectangle[0], rectangle[2])})
    area = 0.0
    for left, right in zip(x_values, x_values[1:]):
        intervals = sorted(
            (top, bottom)
            for rect_left, top, rect_right, bottom in clipped
            if rect_left < right and rect_right > left
        )
        covered_height = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_height += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_height += end - start
        area += (right - left) * covered_height
    return area


def _in_ignored_region(
    box: tuple[float, float, float, float],
    ignored_regions: list[tuple[float, float, float, float]],
) -> bool:
    return _covered_area(box, ignored_regions) / (box[2] * box[3]) >= 0.5


def _overlap(
    detection: tuple[float, float, float, float],
    ground_truth: tuple[float, float, float, float],
    ignored: bool,
) -> float:
    intersection = _intersection_area(detection, ground_truth)
    if ignored:
        union = detection[2] * detection[3]
    else:
        union = detection[2] * detection[3] + ground_truth[2] * ground_truth[3] - intersection
    return intersection / union if union > 0 else 0.0


def _match_image(
    ground_truths: list[tuple[tuple[float, float, float, float], bool]],
    detections: list[tuple[tuple[float, float, float, float], float]],
    threshold: float,
) -> list[tuple[float, int]]:
    ordered_gt = sorted(ground_truths, key=lambda item: item[1])
    matched = [False] * len(ordered_gt)
    results: list[tuple[float, int]] = []
    for detection, score in sorted(detections, key=lambda item: item[1], reverse=True):
        best_overlap = threshold
        best_index: int | None = None
        best_match = 0
        for index, (ground_truth, ignored) in enumerate(ordered_gt):
            if matched[index] and not ignored:
                continue
            if best_match and ignored:
                break
            overlap = _overlap(detection, ground_truth, ignored)
            if overlap < best_overlap:
                continue
            best_overlap = overlap
            best_index = index
            best_match = -1 if ignored else 1
        if best_index is not None and best_match == 1:
            matched[best_index] = True
        results.append((score, best_match))
    return results


def _voc_ap(recall: list[float], precision: list[float]) -> float:
    mrec = [0.0, *recall, 1.0]
    mpre = [0.0, *precision, 0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    return sum(
        (mrec[index] - mrec[index - 1]) * mpre[index]
        for index in range(1, len(mrec))
        if mrec[index] != mrec[index - 1]
    )


def score_detection(
    labels_document: dict[str, Any], predictions_document: dict[str, Any]
) -> dict[str, float]:
    expected = _common_validation(labels_document, predictions_document)
    if predictions_document.get("type") != "object_detection_predictions":
        raise ValueError("prediction type does not match object detection labels")
    class_ids = labels_document.get("class_ids")
    if (
        not isinstance(class_ids, list)
        or not class_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in class_ids)
        or len(set(class_ids)) != len(class_ids)
    ):
        raise ValueError("invalid detection class ids")
    class_id_set = set(class_ids)
    threshold_values = labels_document.get("iou_thresholds", list(IOU_THRESHOLDS))
    if (
        not isinstance(threshold_values, list)
        or not threshold_values
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 < value <= 1
            for value in threshold_values
        )
    ):
        raise ValueError("invalid detection IoU thresholds")
    thresholds = tuple(float(value) for value in threshold_values)
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("detection IoU thresholds must be unique")
    maximum_detections = labels_document.get("max_detections", 500)
    if (
        isinstance(maximum_detections, bool)
        or not isinstance(maximum_detections, int)
        or maximum_detections <= 0
    ):
        raise ValueError("invalid maximum detection count")
    recall_limits = labels_document.get("recall_limits", list(MAX_DETECTIONS))
    if (
        not isinstance(recall_limits, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value <= maximum_detections
            for value in recall_limits
        )
        or len(set(recall_limits)) != len(recall_limits)
    ):
        raise ValueError("invalid detection recall limits")
    reported_metrics = labels_document.get(
        "reported_metrics",
        ["AP", "AP50", "AP75", *(f"AR{value}" for value in recall_limits)],
    )
    allowed_metrics = {"AP", "AP50", "AP75", *(f"AR{value}" for value in recall_limits)}
    if (
        not isinstance(reported_metrics, list)
        or not reported_metrics
        or any(value not in allowed_metrics for value in reported_metrics)
        or len(set(reported_metrics)) != len(reported_metrics)
        or "AP" not in reported_metrics
    ):
        raise ValueError("invalid reported detection metrics")
    label_items = _unique_items(labels_document.get("labels"), expected, "labels")
    prediction_items = _unique_items(
        predictions_document.get("predictions"), expected, "predictions"
    )
    if label_items.keys() != prediction_items.keys():
        raise ValueError("labels and predictions must contain the exact expected example ids")

    prepared: dict[str, dict[str, Any]] = {}
    available_classes: set[int] = set()
    for identifier, label_item in label_items.items():
        ignored_values = label_item.get("ignored_regions")
        object_values = label_item.get("objects")
        if not isinstance(ignored_values, list) or not isinstance(object_values, list):
            raise ValueError("invalid detection labels")
        ignored_regions = [_bbox(value, "ignored region") for value in ignored_values]
        objects: list[tuple[tuple[float, float, float, float], int, bool]] = []
        for item in object_values:
            if not isinstance(item, dict):
                raise ValueError("invalid detection object")
            category = item.get("class")
            ignored = item.get("ignore")
            if category not in class_id_set or not isinstance(ignored, bool):
                raise ValueError("invalid detection object class or ignore flag")
            box = _bbox(item.get("bbox"), "ground-truth box")
            if not _in_ignored_region(box, ignored_regions):
                objects.append((box, category, ignored))
                if not ignored:
                    available_classes.add(category)

        detections_value = prediction_items[identifier].get("detections")
        if (
            not isinstance(detections_value, list)
            or len(detections_value) > maximum_detections
        ):
            raise ValueError(
                f"each image must contain at most {maximum_detections} detections"
            )
        detections: list[tuple[tuple[float, float, float, float], float, int]] = []
        for item in detections_value:
            if not isinstance(item, dict):
                raise ValueError("invalid detection")
            category = item.get("class")
            score = item.get("score")
            if category not in class_id_set:
                raise ValueError("invalid detection class")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0 <= score <= 1
            ):
                raise ValueError("invalid detection score")
            box = _bbox(item.get("bbox"), "detection box")
            if not _in_ignored_region(box, ignored_regions):
                detections.append((box, float(score), category))
        detections.sort(key=lambda item: item[1], reverse=True)
        prepared[identifier] = {"objects": objects, "detections": detections}

    if not available_classes:
        raise ValueError("detection labels contain no evaluable classes")
    ap_values: dict[tuple[int, int], float] = {}
    ar_values: dict[tuple[int, int, int], float] = {}
    for category in sorted(available_classes):
        for threshold_index, threshold in enumerate(thresholds):
            for maximum in [*recall_limits, maximum_detections]:
                matches: list[tuple[float, int]] = []
                number_of_ground_truths = 0
                for item in prepared.values():
                    ground_truths = [
                        (box, ignored)
                        for box, object_category, ignored in item["objects"]
                        if object_category == category
                    ]
                    # The official toolkit caps all detections per image before
                    # selecting the current category.
                    detections = [
                        (box, score)
                        for box, score, detection_category in item["detections"][:maximum]
                        if detection_category == category
                    ]
                    number_of_ground_truths += sum(
                        not ignored for _, ignored in ground_truths
                    )
                    matches.extend(_match_image(ground_truths, detections, threshold))
                matches.sort(key=lambda item: item[0], reverse=True)
                true_positives = 0
                false_positives = 0
                recall: list[float] = []
                precision: list[float] = []
                for _, match in matches:
                    true_positives += match == 1
                    false_positives += match == 0
                    recall.append(true_positives / max(1, number_of_ground_truths))
                    precision.append(true_positives / max(1, true_positives + false_positives))
                ar_values[(category, threshold_index, maximum)] = max(recall, default=0.0)
                if maximum == maximum_detections:
                    ap_values[(category, threshold_index)] = _voc_ap(recall, precision)

    categories = sorted(available_classes)
    ap = sum(
        ap_values[(category, index)]
        for category in categories
        for index in range(len(thresholds))
    )
    ap /= len(categories) * len(thresholds)
    all_metrics = {"AP": ap}
    for name, target_threshold in (("AP50", 0.5), ("AP75", 0.75)):
        if name not in reported_metrics:
            continue
        try:
            threshold_index = thresholds.index(target_threshold)
        except ValueError as error:
            raise ValueError(f"{name} requires IoU threshold {target_threshold}") from error
        all_metrics[name] = sum(
            ap_values[(category, threshold_index)] for category in categories
        ) / len(categories)
    for maximum in recall_limits:
        value = sum(
            ar_values[(category, index, maximum)]
            for category in categories
            for index in range(len(thresholds))
        ) / (len(categories) * len(thresholds))
        all_metrics[f"AR{maximum}"] = value
    return {name: all_metrics[name] for name in reported_metrics}


# Kept as a compatibility name for callers that imported the original scorer.
score_visdrone = score_detection


def score_documents(
    labels_document: dict[str, Any], predictions_document: dict[str, Any]
) -> dict[str, Any]:
    if labels_document.get("task") == "object_detection":
        metrics = score_detection(labels_document, predictions_document)
    else:
        metrics = score_classification(labels_document, predictions_document)
    return {
        "schema_version": 1,
        "dataset": labels_document.get("dataset"),
        "split": labels_document.get("split"),
        "num_examples": labels_document.get("num_examples"),
        "submission_sha256": predictions_document.get("submission_sha256"),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels")
    parser.add_argument("predictions", nargs="?", default="-")
    arguments = parser.parse_args()
    try:
        result = score_documents(read_json(arguments.labels), read_json(arguments.predictions))
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
