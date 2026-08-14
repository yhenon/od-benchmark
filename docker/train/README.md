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

```python
from odbench_train import epoch_end

decision = epoch_end(
    epoch=epoch,
    artifact=f"model-{epoch}.onnx",
    checkpoint=f"checkpoint-{epoch}.pt",
    preprocess="preprocess.py",
    postprocess="postprocess.py",
    metrics={"train_loss": train_loss},
)
if decision.stop:
    return
```

Output paths are relative to `/job/output`; hook paths are relative to the
immutable `/job/input` snapshot. The current artifact contract is one
self-contained ONNX file, so PyTorch exports should use `external_data=False`.
`epoch_end` atomically publishes an event and then waits for a decision file.
No socket, callback server, or shared database is needed.

The trusted controller sees the event and immediately pauses the whole
container. It copies the fixed artifact and hooks into a private staging area,
runs the existing evaluator, and returns a `train_epoch_complete` notification.
The caller can then issue `continue` or `stop`. Evaluation time and the time
spent waiting for the agent are excluded from the active training meter.

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
checkpoints are kept under `.odbench/jobs/` for auditability. Run
`docker/train-verify` for a complete two-epoch pause/evaluate/continue/stop test.

## CUDA migration

The mailbox and controller contracts do not depend on CPU execution. A CUDA
image can replace `BASE_IMAGE` and the launcher can add a narrowly selected GPU
device. Docker pause keeps that allocation reserved while the agent decides;
if decision latency becomes expensive, the same checkpoint field supports a
later exit-and-resume implementation without changing training scripts.
