#!/usr/bin/env python3
"""Trusted outer scorer: join provider predictions with private labels."""

import argparse
import json
import sys


def read_json(path):
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels")
    parser.add_argument("predictions", nargs="?", default="-")
    arguments = parser.parse_args()
    labels_document = read_json(arguments.labels)
    predictions_document = read_json(arguments.predictions)

    if labels_document.get("schema_version") != 1:
        raise SystemExit("unsupported label schema")
    if predictions_document.get("schema_version") != 1:
        raise SystemExit("unsupported prediction schema")
    for field in ("dataset", "split", "num_examples"):
        if labels_document.get(field) != predictions_document.get(field):
            raise SystemExit("labels and predictions do not describe the same evaluation set")

    expected = labels_document["num_examples"]
    label_items = labels_document["labels"]
    prediction_items = predictions_document["predictions"]
    labels = {item["id"]: item["class"] for item in label_items}
    predictions = {item["id"]: item["class"] for item in prediction_items}
    if (
        len(label_items) != expected
        or len(prediction_items) != expected
        or len(labels) != expected
        or len(predictions) != expected
        or labels.keys() != predictions.keys()
    ):
        raise SystemExit("labels and predictions must contain the exact expected example ids")

    correct = sum(predictions[identifier] == label for identifier, label in labels.items())
    result = {
        "schema_version": 1,
        "dataset": labels_document["dataset"],
        "split": labels_document["split"],
        "num_examples": expected,
        "submission_sha256": predictions_document["submission_sha256"],
        "metrics": {"top1_accuracy": correct / expected},
    }
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
