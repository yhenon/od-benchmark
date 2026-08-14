"""Lower-privilege process that executes submitted hooks and ONNX inference."""

from __future__ import annotations

import importlib.util
import io
import os
import struct
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnxruntime
from PIL import Image

from .protocol import MAX_IMAGE_BYTES, receive_frame, send_frame


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load hook: {path.name}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _normalize_inputs(session: onnxruntime.InferenceSession, value: Any) -> dict[str, np.ndarray]:
    if isinstance(value, np.ndarray):
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise TypeError("preprocess returned an array but the model has multiple inputs")
        return {inputs[0].name: np.ascontiguousarray(value)}
    if isinstance(value, dict) and all(
        isinstance(name, str) and isinstance(array, np.ndarray)
        for name, array in value.items()
    ):
        return {name: np.ascontiguousarray(array) for name, array in value.items()}
    raise TypeError("preprocess must return a numpy array or a string-to-array mapping")


def run_worker(request_fd: int, response_fd: int, submission: Path) -> None:
    preprocess_module = _load_module("submission_preprocess", submission / "preprocess.py")
    postprocess_module = _load_module("submission_postprocess", submission / "postprocess.py")
    preprocess = getattr(preprocess_module, "preprocess", None)
    postprocess = getattr(postprocess_module, "postprocess", None)
    if not callable(preprocess) or not callable(postprocess):
        raise TypeError("hooks must define callable preprocess and postprocess functions")

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(
        str(submission / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [output.name for output in session.get_outputs()]
    send_frame(response_fd, b"R")

    while True:
        encoded_image = receive_frame(request_fd, maximum_size=MAX_IMAGE_BYTES)
        if not encoded_image:
            return
        try:
            with Image.open(io.BytesIO(encoded_image)) as opened_image:
                image = opened_image.convert("RGB")
            model_inputs = _normalize_inputs(session, preprocess(image))
        except Exception:
            send_frame(response_fd, b"E1")
            return
        try:
            values = session.run(output_names, model_inputs)
            outputs = dict(zip(output_names, values, strict=True))
        except Exception:
            send_frame(response_fd, b"E2")
            return
        try:
            prediction = postprocess(outputs)
            if isinstance(prediction, bool) or not isinstance(prediction, (int, np.integer)):
                raise TypeError("postprocess must return an integer class index")
            send_frame(response_fd, b"P" + struct.pack("!q", int(prediction)))
        except Exception:
            send_frame(response_fd, b"E3")
            return


def child_main(request_fd: int, response_fd: int, submission: Path) -> None:
    try:
        os.setsid()
        null_fd = os.open("/dev/null", os.O_RDWR)
        for standard_fd in (0, 1, 2):
            os.dup2(null_fd, standard_fd)
        if null_fd > 2:
            os.close(null_fd)

        worker_uid = int(os.environ["ODBENCH_WORKER_UID"])
        worker_gid = int(os.environ["ODBENCH_WORKER_GID"])
        os.setgroups([])
        os.setgid(worker_gid)
        os.setuid(worker_uid)
        os.chdir(submission)
        os.environ.clear()
        os.environ.update(
            {
                "HOME": str(submission),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        run_worker(request_fd, response_fd, submission)
    except Exception:
        try:
            send_frame(response_fd, b"E")
        except Exception:
            pass
    finally:
        os._exit(0)
