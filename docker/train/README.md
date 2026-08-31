# Metered training service

The training service is a CPU prototype of the future GPU-backed `train` tool.
It deliberately uses a small filesystem mailbox instead of requiring the
training process to host a server.

## Trust and filesystem layout

At launch, the outer controller copies the agent workspace into a fresh job
directory. The snapshot rejects symlinks and is mounted read-only at
`/job/input`. The training container gets exactly four job mounts:

| Container path | Access | Purpose |
| --- | --- | --- |
| `/job/input` | read-only | Frozen training source and hooks |
| `/job/output` | read-write | ONNX artifacts and checkpoints |
| `/job/events` | read-write | Epoch-complete messages |
| `/job/decisions` | read-only | Continue/stop replies |

It has no network or other host mounts, a read-only image filesystem, no Linux
capabilities, and CPU, memory, process, and swap limits. The CIFAR-10 training
set comes from the selected dataset image. Evaluation data and labels are not
present.

For a production tool call, use `--agent-container NAME`; the controller copies
`/workspace` out of that sandbox before launching training. `--workspace PATH`
is the local development equivalent.

## Epoch boundary API

Training code exports an ONNX model and optional checkpoint, then calls:

The SDK is also installed in the agent workbench so training files can be
import-checked there. `epoch_end()` itself requires the job environment created
by `train_start` and fails immediately with a clear message outside it.

```python
from odbench_train import epoch_end

decision = epoch_end(
    epoch=epoch,
    artifact=f"model-{epoch}.onnx",
    checkpoint=f"checkpoint-{epoch}.pt",
    preprocess="preprocess.py",
    postprocess="postprocess.py",
    metrics={"train_loss": train_loss, "float_dev_accuracy": float_dev_accuracy},
)
if decision.stop:
    return
```

Output paths are relative to `/job/output`; hook paths are relative to the
immutable `/job/input` snapshot. The current artifact contract is one
self-contained ONNX file. PyTorch exports must use `opset_version=17`,
`dynamo=False`, and `external_data=False`; the dynamo exporter can emit opset
18 despite an opset-17 request, which is not a valid quantizer input.
The file must not exceed 16 MiB (16,777,216 bytes); the controller checks this
before it stages or evaluates an epoch artifact.

Create hook files in the agent's `/workspace` before `train_start`; they then
appear in the job under `/job/input`. Generating hooks only under `/job/output`
does not satisfy `epoch_end`. If quantization runs inside the training job, pass
the input hook explicitly (for example, `--preprocess /job/input/preprocess.py`)
while writing the quantized model under `/job/output`.

`epoch_end` atomically publishes an event and then waits for a decision file.
No socket, callback server, or shared database is needed.

The trusted controller sees the event and immediately pauses the whole
container. It copies the fixed artifact and hooks into a private staging area,
runs the existing evaluator, and returns a `train_epoch_complete` notification.
The caller can then issue `continue` or `stop`. Evaluation time and the time
spent waiting for the agent are excluded from the active training meter.

The notification keeps metric provenance explicit. `train_metrics` contains
arbitrary diagnostics supplied by the training script and normally describes an
agent-defined public split and model stage. `evaluation.metrics` is the trusted
hidden-set result for the staged ONNX submission. These sections are not
directly comparable; in particular, their difference is not a quantization-loss
measurement. Training scripts should use precise names such as
`float_dev_accuracy` and `quantized_dev_accuracy` rather than `val_acc`.

Every epoch notification therefore represents a live paused job, even when
the training script would exit immediately after `epoch_end` returns. It must
be continued once to exit naturally or explicitly stopped.

When the outer `start` command receives `--resume-checkpoint`, it copies that
single trusted file into the otherwise immutable input snapshot at
`/job/input/.odbench_resume/checkpoint.pt` and sets
`ODBENCH_RESUME_CHECKPOINT` to that path. Training entrypoints can use the
environment variable as their resume-file default.

Nonzero terminal results include the last bounded portions of container stdout
and stderr when those logs remain available, so entrypoint failures can be
diagnosed without host or Docker access from the agent workspace.

## Reference commands

```sh
docker/trainer build
docker/evaluator export-labels /trusted/cifar10-labels

start="$(docker/trainer start \
  --workspace examples/cifar10-train \
  --entrypoint train.py \
  --budget-seconds 300)"

docker/trainer await JOB_ID --labels /trusted/cifar10-labels/labels.json
docker/trainer continue JOB_ID epoch-000000
docker/trainer await JOB_ID --labels /trusted/cifar10-labels/labels.json
docker/trainer stop JOB_ID
```

`start` returns the job ID. `await` is the blocking portion of the outer-loop
tool backend: it returns on an epoch result, terminal error, normal completion,
or exhausted budget. A real agent integration maps these commands to
`train.start`, a wake-up notification, `train.continue`, and `train.stop`; the
agent never receives Docker or label-store access.

Job metadata, immutable snapshots, staged submissions, aggregate results, and
checkpoints are kept under the owning `runs/<run-id>/training-jobs/` directory
for auditability. Run
`verification/train` for a complete two-epoch pause/evaluate/continue/stop test.

## Persistent SSH GPU worker

The local-network prototype keeps the agent sandbox and trusted evaluator on
the benchmark machine while executing only the agent-authored training process
on a persistent Docker GPU host. The remote training container retains the same
read-only input, four-directory mailbox, no-network, capability, memory, PID,
and wall-time boundaries as the local controller.

Training profiles also set an explicit `/dev/shm` ceiling for multiprocessing
PyTorch DataLoaders. Shared-memory use remains charged to the container's
overall memory limit.

Configure the trusted SSH destination and sync this repository without deleting
unrelated remote files:

```sh
export ODBENCH_TRAIN_HOST=odbench@192.168.1.106
export ODBENCH_REMOTE_REPO=/home/odbench/od-benchmark
export ODBENCH_REMOTE_JOBS_ROOT=/var/lib/odbench/jobs

docker/remote-trainer sync
docker/remote-trainer build
docker/remote-trainer verify-gpu
```

Select the dataset with `ODBENCH_DATASET`. Datasets sourced from local files
also need a one-time, separate training-data sync. For VisDrone the helper
defaults to the public `data/VisDrone2019-DET-train` directory:

```sh
export ODBENCH_DATASET=visdrone
docker/remote-trainer sync
docker/remote-trainer sync-data
docker/remote-trainer build
docker/remote-trainer verify-gpu
docker/prepare visdrone --skip-trainer-build
```

Override the local source with `ODBENCH_TRAIN_DATA_SOURCE`, the persistent
remote dataset root with `ODBENCH_REMOTE_DATA_ROOT`, or the exact remote source
with `ODBENCH_REMOTE_TRAIN_DATA_SOURCE`. Repository sync excludes all `data/`
and `.odbench/` content, while `sync-data` copies only the selected public
training source. Evaluation data and private labels remain local.

`docker/trainer` automatically selects its SSH transport for `start`, `await`,
`continue`, `stop`, and `status` whenever `ODBENCH_TRAIN_HOST` is set. No change
to the outer-loop tool protocol is required. `start` safely snapshots the local
agent container, streams that snapshot and an optional resume checkpoint over
SSH, and launches the configured trainer image remotely.

At an epoch boundary the remote controller pauses the GPU container and stages
the submitted ONNX model, hooks, and checkpoint. The local SSH client downloads
that fixed stage, invokes the existing trusted local evaluator with the private
labels, records only the aggregate evaluation in remote job state, and returns
the usual `train_epoch_complete` result. Private labels, evaluator credentials,
the OpenRouter key, and hardware credentials are never copied to the worker.

VisDrone already selects the checked-in
`hardware/training/lan-rtx3070-8gb.json` profile. Other tasks can select it from
their task definition before preparing the task. The remote host must already
provide Docker, the NVIDIA Container Toolkit, SSH key authentication for the
trusted worker account, and the Docker Buildx plugin. The current prototype
provides one transport but not yet a cross-run GPU scheduler; do not start
overlapping outer-loop runs against the same single-GPU worker.

## CUDA migration

The mailbox and controller contracts do not depend on CPU execution. A CUDA
image can replace `BASE_IMAGE` and the launcher can add a narrowly selected GPU
device. Docker pause keeps that allocation reserved while the agent decides;
if decision latency becomes expensive, the same checkpoint field supports a
later exit-and-resume implementation without changing training scripts.
