"""Trusted coordinator for unlabeled evaluation-set inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import signal
import stat
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from .protocol import MAX_RESPONSE_BYTES, receive_frame, send_frame
from .worker import child_main


HOOK_SIZE_LIMIT = 1024 * 1024
MODEL_SIZE_LIMIT = 2 * 1024 * 1024 * 1024


def _contained_file(root: Path, relative_path: Any, size_limit: int) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("submission paths must be non-empty strings")
    path = (root / relative_path).resolve()
    if path.parent != root and root not in path.parents:
        raise ValueError("submission path escapes its directory")
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"submission input is not a regular file: {relative_path}")
    if file_stat.st_size > size_limit:
        raise ValueError(f"submission input is too large: {relative_path}")
    return path


def _stage_submission(source_directory: Path) -> tuple[Path, dict[str, Any]]:
    root = source_directory.resolve()
    manifest_path = _contained_file(root, "submission.json", 64 * 1024)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported submission manifest schema")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("format") != "onnx":
        raise ValueError("the artifact format must be onnx")

    sources = {
        "model.onnx": _contained_file(root, artifact.get("path"), MODEL_SIZE_LIMIT),
        "preprocess.py": _contained_file(root, manifest.get("preprocess"), HOOK_SIZE_LIMIT),
        "postprocess.py": _contained_file(root, manifest.get("postprocess"), HOOK_SIZE_LIMIT),
    }

    staged = Path(tempfile.mkdtemp(prefix="submission-", dir="/tmp"))
    for destination_name, source in sources.items():
        destination = staged / destination_name
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        destination.chmod(0o444)
    staged.chmod(0o555)
    return staged, manifest


def _submission_digest(staged: Path) -> str:
    digest = hashlib.sha256()
    for name in ("model.onnx", "preprocess.py", "postprocess.py"):
        digest.update(name.encode("utf-8") + b"\0")
        with (staged / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _terminate_worker(process_id: int) -> None:
    try:
        waited_process, _ = os.waitpid(process_id, os.WNOHANG)
        if waited_process == process_id:
            return
    except ChildProcessError:
        return
    try:
        os.killpg(process_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.waitpid(process_id, 0)
    except ChildProcessError:
        pass


def _spawn_worker(submission: Path) -> tuple[int, int, int]:
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    process_id = os.fork()
    if process_id == 0:
        os.close(request_write)
        os.close(response_read)
        child_main(request_read, response_write, submission)
    os.close(request_read)
    os.close(response_write)
    return process_id, request_write, response_read


def _load_eval_manifest() -> tuple[Path, dict[str, Any]]:
    identifier = os.environ["ODBENCH_EVAL_DATASET"]
    root = Path(os.environ["ODBENCH_EVAL_DATA_ROOT"]) / identifier
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("id") != identifier or manifest.get("schema_version") != 1:
        raise RuntimeError("invalid installed evaluation dataset manifest")
    return root, manifest


def _encode_image(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def run_provider(submission_directory: Path, timeout: float) -> dict[str, Any]:
    staged_submission, _ = _stage_submission(submission_directory)
    submission_sha256 = _submission_digest(staged_submission)
    process_id, request_fd, response_fd = _spawn_worker(staged_submission)
    try:
        ready = receive_frame(response_fd, maximum_size=MAX_RESPONSE_BYTES, timeout=60.0)
        if ready != b"R":
            raise RuntimeError("submission worker failed to initialize")

        # Import and load the evaluation data only after the worker has forked and
        # dropped privileges, so no images or labels can exist in its memory.
        import eval_dataset

        dataset_root, dataset_manifest = _load_eval_manifest()
        predictions: list[dict[str, Any]] = []
        number_of_classes = int(dataset_manifest["num_classes"])
        for example_id, image in eval_dataset.iter_examples(dataset_root):
            send_frame(request_fd, _encode_image(image))
            response = receive_frame(
                response_fd, maximum_size=MAX_RESPONSE_BYTES, timeout=timeout
            )
            if len(response) != 9 or response[:1] != b"P":
                phases = {
                    b"E1": "image decoding or preprocessing",
                    b"E2": "ONNX inference",
                    b"E3": "postprocessing",
                }
                phase = phases.get(response, "worker execution")
                raise RuntimeError(
                    f"submission failed during {phase} for example {example_id}"
                )
            prediction = struct.unpack("!q", response[1:])[0]
            if not 0 <= prediction < number_of_classes:
                raise RuntimeError("postprocess returned an out-of-range class index")
            predictions.append({"id": example_id, "class": prediction})

        send_frame(request_fd, b"")
        os.close(request_fd)
        request_fd = -1
        _, status = os.waitpid(process_id, 0)
        process_id = -1
        if status != 0:
            raise RuntimeError("submission worker exited abnormally")

        return {
            "schema_version": 1,
            "type": "classification_predictions",
            "dataset": dataset_manifest["id"],
            "split": dataset_manifest["split"],
            "num_examples": len(predictions),
            "submission_sha256": submission_sha256,
            "predictions": predictions,
        }
    finally:
        if request_fd >= 0:
            os.close(request_fd)
        os.close(response_fd)
        if process_id > 0:
            _terminate_worker(process_id)
        shutil.rmtree(staged_submission, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--example-timeout",
        type=float,
        default=float(os.environ.get("ODBENCH_EXAMPLE_TIMEOUT", "10")),
    )
    arguments = parser.parse_args()
    try:
        result = run_provider(arguments.submission, arguments.example_timeout)
    except Exception as error:
        print(f"evaluation inference failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
