from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from odbench_outer.tools import ToolRuntimeError, validate_submission_artifact


def make_submission(root: Path, model_bytes: bytes, artifact_path: str = "model.onnx") -> None:
    root.mkdir()
    (root / "model.onnx").write_bytes(model_bytes)
    (root / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": {"format": "onnx", "path": artifact_path},
                "preprocess": "preprocess.py",
                "postprocess": "postprocess.py",
            }
        ),
        encoding="utf-8",
    )


class SubmissionLimitTests(unittest.TestCase):
    def test_accepts_artifact_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "submission"
            make_submission(submission, b"12345678")
            self.assertEqual(validate_submission_artifact(submission, 8), 8)

    def test_rejects_artifact_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "submission"
            make_submission(submission, b"123456789")
            with self.assertRaisesRegex(ToolRuntimeError, "9 bytes; limit is 8 bytes"):
                validate_submission_artifact(submission, 8)

    def test_rejects_escaping_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "submission"
            make_submission(submission, b"model", "../model.onnx")
            with self.assertRaisesRegex(ToolRuntimeError, "escapes"):
                validate_submission_artifact(submission, 8)


if __name__ == "__main__":
    unittest.main()
