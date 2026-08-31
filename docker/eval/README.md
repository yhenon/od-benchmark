# Evaluation inference provider

The evaluator is split across two trust domains:

1. `od-benchmark-evaluator:<dataset>-dev` contains evaluation images and opaque
   example IDs, but no per-example labels.
2. The trusted outer service owns labels, receives provider predictions, joins
   them by ID, computes aggregate metrics, and returns only those metrics to the
   agent.

For development, `docker/evaluator export-labels DESTINATION` exports labels
from a separate Docker build target. The final evaluator image never copies
that target. Production labels should come from private benchmark storage.

## Submission contract

A submission directory contains `submission.json`, one ONNX artifact, and two
single-file Python hooks. Manifest schema version 1 is:

```json
{
  "schema_version": 1,
  "artifact": {"format": "onnx", "path": "model.onnx"},
  "preprocess": "preprocess.py",
  "postprocess": "postprocess.py"
}
```

`preprocess(image)` receives one RGB `PIL.Image`. It returns either a NumPy
array for a single-input model or a mapping from ONNX input names to NumPy
arrays. `postprocess(outputs)` receives a mapping from ONNX output names to
NumPy arrays. For classification it returns one integer class index. For object
detection it returns a mapping with NumPy `boxes`, `scores`, and `classes`
arrays; boxes are `[N, 4]` original-image `xyxy` coordinates and at most 500
detections are accepted. A hook declared as `postprocess(outputs, image_size)`
also receives the original `(width, height)`, which is useful for reversing
resize or letterbox geometry. Inference is single-example, CPU-only, and uses
ONNX Runtime.

The ONNX artifact must be self-contained and no larger than 16 MiB
(16,777,216 bytes). Hooks remain limited to 1 MiB each.

The provider returns a prediction document to the trusted outer loop. It
contains opaque example IDs, bounded predictions, and a SHA-256 digest over the
artifact and hooks. `docker/eval/score.py` joins this document with the private
label document and emits the dataset's aggregate metrics. VisDrone uses the
official DET IoU thresholds (0.50:0.05:0.95), ignored-region treatment, and a
500-detection cap.

## Security boundary

The root coordinator loads evaluation images only after forking the worker.
The worker then runs as UID/GID 10002 and cannot read the root-only dataset
directory. It has no network, GPU, writable root filesystem, or access to the
outer label file. Hook output and errors are reduced to a fixed binary protocol;
arbitrary stdout/stderr are discarded. Per-example timeouts and container
memory, CPU, process, and file-size limits are enforced.

The coordinator retains only `SETUID`, `SETGID`, and `KILL` capabilities so it
can create and terminate the lower-privilege worker. The worker loses those
capabilities when it changes UID. Stronger isolation against native-library or
kernel exploits would require a microVM or a second container boundary.
