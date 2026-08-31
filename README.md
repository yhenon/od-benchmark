# OD Benchmark

This repository is an early implementation of the benchmark described in
`project.md`. Its agent sandbox is an air-gapped, CPU-only environment for
ordinary LLM workspace commands. Training, evaluation, profiling, and
submission remain separate trusted services.

## Image layout

The root `Dockerfile` has two useful layers:

- `agent-base` is dataset-agnostic. It contains CPU builds of PyTorch and
  torchvision, ONNX and ONNX Runtime, ONNX Script, NumPy/SciPy/scikit-learn,
  OpenCV, Pillow, pycocotools, safetensors, common build tools, and a small
  verified registry of offline torchvision initializers.
- `dataset-runtime` adds exactly one dataset hook. The current default is
  CIFAR-10, producing `od-benchmark-agent:cifar10-dev`.

Build the generic base, or the current CIFAR-10 task image:

```sh
docker/sandbox build-base
docker/sandbox build
verification/sandbox
```

The verification checks that the base contains no dataset, then tests the
CIFAR-10 hook, isolation settings, and a PyTorch-to-ONNX inference round trip.

The CIFAR-10 hook contains all 50,000 training examples and deliberately
excludes the public test batch. Inside the image, dataset-independent code uses:

```python
from odbench import dataset_manifest, load_dataset

print(dataset_manifest())
training_data = load_dataset()
```

Pretrained weights are downloaded only while building the trusted image, then
checked against the exact byte counts and SHA-256 hashes in
`docker/pretrained/manifest.json`. The sandbox and trainer inherit the same
read-only files, while the separate evaluator image does not contain them:

```python
from odbench import list_pretrained, load_backbone, load_detector

print(list_pretrained())
features = load_backbone(
    "torchvision/mobilenet_v3_small_imagenet1k_v1"
)
detector = load_detector(
    "torchvision/ssdlite320_mobilenet_v3_large_coco_v1",
    num_classes=11,
)
```

The initial catalog contains MobileNetV3 Small, MobileNetV2, and ShuffleNetV2
x0.5 ImageNet backbones plus a COCO-trained SSDLite320 MobileNetV3 Large
detector. `load_backbone` returns four feature maps at reductions 4, 8, 16, and
32. Catalog metadata includes parameter counts, reference GFLOPS at the named
input size, and normalization. The API is deliberately narrower than
torchvision/timm model discovery so benchmark code cannot accidentally attempt
a network download. The VisDrone task prompt documents the bundled detector's
training API and the recommended raw-head ONNX deployment split.

Start a persistent, disposable workspace and execute commands in it:

```sh
docker/sandbox start
docker/sandbox exec python -c 'from odbench import load_dataset; print(len(load_dataset()))'
docker/sandbox shell
docker/sandbox stop
```

The image also contains a benchmark-owned STM32N6 quantization recipe. Given a
static opset-17 float model and the same preprocessing hook used for evaluation:

```sh
python -m odbench.quantize \
  --model model-float.onnx \
  --output submission/model.onnx \
  --preprocess submission/preprocess.py \
  --samples 256
```

It calibrates an INT8 QDQ ONNX model on deterministic samples from the public
training split and emits a validation report beside the output model.

The running container has no network, GPU, host mounts, or Linux capabilities.
Its image filesystem and bundled dataset are read-only. Only `/workspace` and
`/tmp` are writable in memory; stopping the container deletes them. It runs as
UID/GID 10001 with CPU, memory, process, and swap limits.

## Adding a dataset

Each dataset is isolated under `docker/datasets/<dataset-id>/` and implements
the preparation/runtime contract in `docker/datasets/README.md`. A runnable
benchmark also needs `tasks/<task-id>.json`, which selects its prompts, images,
limits, model-call policy, evaluation objective, a training-compute profile
under `hardware/training/`, and a physical deployment requirement under
`hardware/targets/`. See `hardware/README.md` for the distinction. For example:

```sh
ODBENCH_DATASET=another_dataset docker/sandbox build
```

This produces `od-benchmark-agent:another_dataset-dev` without changing the
generic Dockerfile or runtime API. Each task image should contain exactly one
agent-visible training dataset; held-out data belongs only in the evaluation
service.

Dataset preparation does not write a host-side `out/` directory. Build hooks
run in disposable Docker build stages: the permitted training split becomes a
layer in the agent/trainer images, the unlabeled held-out split becomes a layer
in the evaluator image, and private labels are exported only to the prepared
task store under `.odbench/prepared-tasks/`. For ad hoc hook debugging, use a
path under the gitignored `tmp/` directory and delete it when finished.

Local-source datasets are passed as isolated BuildKit contexts and excluded
from version control. Place VisDrone under `data/VisDrone2019-DET-train` and
`data/VisDrone2019-DET-val`, then build the complete train/validation image pair
and prepared task with:

```sh
docker/prepare visdrone --skip-trainer-build
```

Override those locations when needed:

```sh
docker/prepare visdrone \
  --skip-trainer-build \
  --train-data /data/VisDrone2019-DET-train \
  --eval-data /data/VisDrone2019-DET-val
```

The agent and trainer images contain only the 6,471 training images and their
annotations. The evaluator contains only the 548 validation images; its label
export is kept in trusted outer-loop storage.

For WIDER FACE, extract the official archives as `data/WIDER_train`,
`data/WIDER_val`, and `data/wider_face_split`, then prepare the task with:

```sh
docker/prepare widerface --skip-trainer-build
```

The WIDER FACE evaluator reports a single face-class AP at IoU 0.50 across all
valid validation annotations; it does not create easy/medium/hard subsets.

`docker/sandbox` is the reference launcher and therefore part of the security
boundary. Do not add bind mounts, the Docker socket, host networking, devices,
or elevated privileges when integrating it with the future command executor.

## Evaluation provider

The separate `Dockerfile.eval` builds a CPU-only inference provider containing
evaluation images but no labels. Build it with:

```sh
docker/evaluator build
```

A trusted operator can export CIFAR-10 labels into separate outer-loop storage
and evaluate the example submission as follows:

```sh
docker/evaluator export-labels /trusted/path/cifar10-labels
docker/evaluator evaluate /path/to/submission /trusted/path/cifar10-labels/labels.json
```

The evaluator runs the submitted ONNX model and hooks over unlabeled images and
returns predictions only to the trusted outer scorer. The scorer joins opaque
example IDs with its private labels and emits the task's aggregate metric:
top-1 accuracy for CIFAR-10, official-style AP/AR for VisDrone, or AP@0.50 for
WIDER FACE. The agent-facing `evaluate` tool exposes only that final aggregate document.
Self-contained ONNX artifacts are capped at 16 MiB across training, evaluation,
and final submission.
See `docker/eval/README.md` for the submission and security contracts.
Run `verification/eval` for a complete generated-model smoke test.

## Metered training

`Dockerfile.train` adds a small epoch-boundary SDK to the matching agent dataset
image. The trusted `docker/trainer` controller snapshots the agent workspace,
starts an isolated training container, meters active wall time, pauses it at
each atomic filesystem event, evaluates the submitted ONNX artifact, and waits
for a continue/stop decision.

```sh
docker/trainer build
verification/train
```

The training container contains CIFAR-10 train data but no evaluation images or
labels. Its only host access is a dedicated four-directory job mailbox; the
workspace snapshot and decision directory are read-only. See
`docker/train/README.md` for the SDK and outer-loop contracts.

Training can also run on a persistent trusted GPU host over SSH while evaluation
and private labels remain local. Set `ODBENCH_TRAIN_HOST`, sync/build with
`docker/remote-trainer`, and select the matching CUDA hardware profile. The
outer-loop `train_start`, `train_continue`, and `train_stop` interface is
unchanged.

VisDrone is configured to use the checked-in RTX 3070 worker profile. Provision
its public training split and CUDA image once (and again only when the dataset
or image changes):

```sh
export ODBENCH_TRAIN_HOST=odbench@192.168.1.106
export ODBENCH_DATASET=visdrone
docker/remote-trainer sync
docker/remote-trainer sync-data
docker/remote-trainer build
docker/remote-trainer verify-gpu
docker/prepare visdrone --skip-trainer-build
```

The default VisDrone source is `data/VisDrone2019-DET-train`; override it with
`ODBENCH_TRAIN_DATA_SOURCE`. `sync-data` never sends the validation split or the
prepared task's private labels. Keep `ODBENCH_TRAIN_HOST` set when launching the
outer loop; `train_start` will then use SSH automatically:

```sh
uv run python main.py --model YOUR_OPENROUTER_MODEL --task visdrone
```

## OpenRouter agent loop

The first complete outer loop is available through `main.py`. It calls models
only through OpenRouter and exposes bounded Bash, unified-diff patching,
training lifecycle, evaluation, physical-target verification, and one-shot
submission tools. The agent sandbox remains offline; only the trusted host
process holds the OpenRouter key, private evaluation-label path, ST tools, and
USB access to the NUCLEO-N657X0-Q. Final submission is accepted only after the
ONNX artifact runs within the configured hardware timing limit.

```sh
docker/rebuild cifar10
export OPENROUTER_API_KEY='...'
uv run python main.py \
  --model YOUR_OPENROUTER_MODEL \
  --task cifar10
```

`docker/rebuild DATASET` rebuilds the configured agent, trainer, and evaluator
images, refreshes the prepared task, verifies all three image tags, and checks
that the trainer can import `odbench.quantize` and `odbench_train`. Docker's
content-addressed cache remains enabled, so unchanged dependency layers are
reused safely. Use `docker/prepare DATASET` directly when more control over task
preparation is needed.

See `docs/outer-loop.md` for setup, limits, persistence, and security details.
Run `verification/agent` for the complete offline tool and lifecycle test.

Generate submitted-solution accuracy scatter plots by token usage and recorded
cost, plus an accepted-results-by-model summary, for each task after running
benchmarks:

```sh
uv run python -m odbench_outer.plot
```

The command reads `runs/<run-id>/events.jsonl` and writes plots to `runs/plots/`.
Provider colors and embedded icons are loaded from `icons/`. Each plot uses one
point per completed run and reads the accepted submission's final metric from
`submission-result.json`; intermediate candidates are excluded because they
may not pass hardware verification. Regenerating also removes legacy
best-candidate plots from `runs/plots/`. The model summary uses accuracy as its
primary bar and inference throughput as a secondary bar normalized to the
fastest accepted model, so longer means better for both bars.

Every durable outcome of an agent run lives under its own gitignored
`runs/<run-id>/` directory: event and summary logs, evaluated candidates,
training-job records, hardware reports, the accepted submission, and the final
submission result. `.odbench/` is reserved for prepared tasks and other local
runner state that is not itself a run outcome.

## Live run dashboard

Start the local dashboard alongside a benchmark run:

```sh
uv run python -m odbench_outer.dashboard
```

It opens `http://127.0.0.1:8765/` and refreshes the append-only run logs every
two seconds. The view includes trusted score progression, training and
evaluation budgets, token and cost usage, hardware status, and an expandable
timeline of every tool call. Both active and completed runs are available from
the run picker.

Use `--no-open` on a headless host, `--port PORT` to choose another local port,
or `--run-id RUN_ID` to select a run initially. The server binds only to
`127.0.0.1` by default.
