from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "odbench_eval_score", REPO_ROOT / "docker" / "eval" / "score.py"
)
assert SPEC is not None and SPEC.loader is not None
SCORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORE)


def labels(objects, ignored_regions=None):
    return {
        "schema_version": 1,
        "dataset": "visdrone",
        "task": "object_detection",
        "split": "val",
        "num_examples": 1,
        "num_classes": 10,
        "class_ids": list(range(1, 11)),
        "labels": [
            {
                "id": "image-1",
                "width": 100,
                "height": 100,
                "objects": objects,
                "ignored_regions": ignored_regions or [],
            }
        ],
    }


def predictions(detections):
    return {
        "schema_version": 1,
        "type": "object_detection_predictions",
        "dataset": "visdrone",
        "split": "val",
        "num_examples": 1,
        "submission_sha256": "abc",
        "predictions": [{"id": "image-1", "detections": detections}],
    }


class DetectionScoringTests(unittest.TestCase):
    def test_perfect_detection_scores_one_at_every_threshold(self):
        document = SCORE.score_documents(
            labels([{"bbox": [10, 20, 30, 40], "class": 1, "ignore": False}]),
            predictions(
                [{"bbox": [10, 20, 30, 40], "score": 0.9, "class": 1}]
            ),
        )
        self.assertEqual(
            document["metrics"],
            {
                "AP": 1.0,
                "AP50": 1.0,
                "AP75": 1.0,
                "AR1": 1.0,
                "AR10": 1.0,
                "AR100": 1.0,
                "AR500": 1.0,
            },
        )

    def test_detection_inside_ignored_region_does_not_become_false_positive(self):
        document = SCORE.score_documents(
            labels(
                [{"bbox": [60, 60, 20, 20], "class": 1, "ignore": False}],
                ignored_regions=[[0, 0, 50, 50]],
            ),
            predictions(
                [
                    {"bbox": [5, 5, 10, 10], "score": 0.99, "class": 1},
                    {"bbox": [60, 60, 20, 20], "score": 0.9, "class": 1},
                ]
            ),
        )
        self.assertEqual(document["metrics"]["AP"], 1.0)

    def test_missing_detection_scores_zero(self):
        document = SCORE.score_documents(
            labels([{"bbox": [10, 20, 30, 40], "class": 1, "ignore": False}]),
            predictions([]),
        )
        self.assertEqual(document["metrics"]["AP"], 0.0)
        self.assertEqual(document["metrics"]["AR500"], 0.0)


if __name__ == "__main__":
    unittest.main()
