# OpenRouter outer loop

The first agent runner uses OpenRouter's Chat Completions endpoint directly and
has no model-provider SDK dependency. All LLM traffic goes through OpenRouter;
the agent sandbox itself remains offline.

The outer loop sends one system message, the task, and nine function tools:

| Tool | Behavior |
| --- | --- |
| `workspace_exec` | Bounded Bash in the persistent agent workspace |
| `workspace_apply_patch` | Git-style unified diff applied in the workspace |
| `train_start` | Snapshot, start metered training, and wait for an event |
| `train_continue` | Resume and wait for the next training event |
| `train_stop` | Stop a job and release compute |
| `evaluate` | Return aggregate hidden-set statistics |
| `analyze_for_hw` | Compile an ONNX model and report accelerator mapping without touching the board |
| `verify_on_hw` | Generate, flash, and profile a workspace ONNX model on the physical target |
| `submit` | Hardware-gate, evaluate, preserve, and end the run |

Underscores are used in function names because they are portable across
OpenRouter's heterogeneous model providers. The conceptual API remains
`workspace.exec`, `workspace.apply_patch`, and so on.

## Preparing and running the CIFAR-10 prototype

Prepare the complete task once:

```sh
docker/prepare cifar10
```

This builds the dataset-specific agent, trainer, and evaluator images; exports
development labels into trusted outer storage; freezes the task and system
prompts; hashes the private labels and prompts; and writes
`.odbench/prepared-tasks/cifar10/task.json`. Re-running the command validates and reuses
the existing private labels. `--skip-build` is available when the images are
already current.

Before a run, `docker/rebuild DATASET` performs the same preparation and then
verifies the configured agent, trainer, and evaluator tags and smoke-tests the
trainer runtime imports. Docker's content-addressed cache remains enabled, so
unchanged dependency layers are reused safely.

Provide credentials only through the environment. The runner deliberately does
not read repository key files:

```sh
export OPENROUTER_API_KEY='...'
uv run python main.py \
  --model YOUR_OPENROUTER_MODEL \
  --task cifar10
```

Select any OpenRouter model supporting [tool
calling](https://openrouter.ai/docs/guides/features/tool-calling).
The runner accepts only the model and prepared task. Task settings live in
`tasks/<task-id>.json`; compute resources live in the named
`hardware/training/` configuration and the physical deployment requirement
lives in the named `hardware/targets/` configuration. Preparation validates and
freezes both into the prepared manifest, so a run has no command-line limit or
path overrides.

## Physical hardware verification

The `nucleo-n657x0-q` target accepts a self-contained ONNX file and has a strict
5-millisecond mean inference target over ten samples. The agent owns export and
quantization. Dataset images stay inside the agent/training image; its
`odbench.quantize` command performs static INT8 QDQ calibration using the public
training split and the submission preprocessing hook:

```sh
python -m odbench.quantize \
  --model model-float.onnx \
  --output submission/model.onnx \
  --preprocess submission/preprocess.py \
  --samples 256
```

The command pins the STM32N6 recipe to opset 17, MinMax calibration, signed
INT8 Q/DQ activations and weights, per-channel weights, and float model I/O. It
checks static batch-1 input shapes, validates the result, runs a float/quantized
smoke comparison, and records a JSON report next to the output model.
Float models exported from PyTorch must set `opset_version=17`, `dynamo=False`,
and `external_data=False`. The installed PyTorch version's dynamo exporter can
retain opset 18 after a failed downgrade even when opset 17 was requested.

The trusted host performs ST Edge AI Neural-ART generation,
rejects generated external-memory images larger than 4 MiB, programs accepted
external weights at a conservative 200 kHz SWD rate with a 900-second timeout,
builds and loads ST's N6 validation firmware, and runs `checker.py` over the
board's virtual COM port. Raw generation, command, and profiling logs are
retained under the run's `hardware/verification-*` directory, while the tool
result includes structured timing data and bounded report text.

`analyze_for_hw` runs only Neural-ART generation and does not require the board
to be connected. It returns pure hardware, hybrid, and pure software epoch
counts, accelerator epoch percentage, compiler memory totals, and warnings. A
successful compilation with 0% accelerator epochs is therefore visible before
the slower flash step.

An explicit `verify_on_hw` call uses the task's strict
`limits.max_inference_runtime_seconds` threshold. `submit` runs the same flow
before hidden-set evaluation and applies the physical target's submission
tolerance. For example, CIFAR-10 configures 5 milliseconds while VisDrone
configures 50 milliseconds; their final thresholds are 5.25 and 52.5
milliseconds with the current 5% tolerance. Compilation, flashing, protocol,
and timing failures reject the submission without consuming the reserved final
evaluation slot or ending the agent run.

The current host defaults match the standard macOS installations used during
development. They can be overridden without exposing paths to the agent:

```sh
export ODBENCH_STEDGEAI_ROOT=/Applications/ST/STEdgeAI/4.0
export ODBENCH_STEDGEAI_BIN=/Applications/ST/STEdgeAI/4.0/Utilities/macarm/stedgeai
export ODBENCH_CUBEIDE_PATH=/Applications/STM32CubeIDE.app/Contents/Eclipse
export ODBENCH_STM32_PROGRAMMER_CLI=/path/to/STM32_Programmer_CLI
export ODBENCH_NUCLEO_SERIAL_PORT=/dev/cu.usbmodem1102
# Optional when more than one ST-Link is connected:
export ODBENCH_STLINK_SERIAL=004400303335511135383531
```

The physical workflow intentionally remains in the outer loop for now: the
agent container has neither host mounts nor USB devices. A process lock
serializes access so concurrent benchmark runs cannot program the same board at
the same time.

`max_evaluations` is one shared hidden-set quota. Every epoch event, explicit
`evaluate`, and final `submit` evaluation increments the same counter. The last
slot is reserved for `submit`, preventing an adaptive evaluation from making a
valid final submission impossible. Tool results report total, used, remaining,
and adaptive-remaining counts. Failed evaluator attempts consume their reserved
slot; a training wait that finishes without an epoch event releases it.

The runner resolves the dataset, image tags, frozen prompts, model settings,
limits, training hardware, physical hardware target, and private label path
from the prepared manifest. The API key remains an environment credential and
is never part of task configuration.

The CIFAR-10 label export is a development convenience because its public test
labels are available during the image build. A production preparer should
provision the same `private/labels.json` contract from the benchmark's private
asset store; no runner or tool change is required.

The ONNX artifact is capped at 16 MiB (16,777,216 bytes). The training
controller enforces the cap before staging an epoch, the outer loop validates
workspace submissions before spending an evaluation call, and the evaluator
enforces the same cap again inside its container. For float32-only weights this
is an upper bound of roughly four million values before graph metadata and
other tensors; serialized bytes are the actual rule.

OpenRouter requests are non-streaming and set `parallel_tool_calls=false`.
Blank or malformed responses, remote failures, and providers that nevertheless
return parallel tool calls receive a corrective retry up to the task's
`model.max_response_retries` limit. Transport retries and request timeout are
configured alongside it. No parallel tool calls are executed. Assistant
[`reasoning_details`](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
are replayed unchanged when supplied by the selected model. Opaque encrypted
reasoning data is omitted from the durable event log; its metadata and readable
reasoning summaries remain available there.

## State and wake-ups

`train_start` and `train_continue` synchronously wait in the trusted outer
process while the LLM is not running. At each epoch event, the controller
pauses training, evaluates the artifact, publishes the frozen stage back under
`/workspace/.odbench/training/<job>/<event>/`, and sends its aggregate result to
the model in the next OpenRouter request. The hidden labels never enter either
container.

An epoch notification is never terminal: `train_epoch_complete` and
`train_epoch_failed` both leave the process paused inside `epoch_end`. Their
tool results report `job_status: "paused"`, the required next action, and that
a new start is unavailable. The caller must issue `train_continue` or
`train_stop`. Only a terminal training result releases the active-job slot.

For cross-job continuation, `train_start.checkpoint_path` accepts a path
returned by a prior epoch result. The outer runtime resolves that trusted
published checkpoint, stages only that file at
`/job/input/.odbench_resume/checkpoint.pt`, and sets
`ODBENCH_RESUME_CHECKPOINT` for the new process. The original `.odbench` path
is not included in ordinary workspace snapshots. This avoids recursively
snapshotting agent state while making checkpoint continuation first-class.
If a training process exits nonzero, its terminal result includes bounded
`stdout_tail` and `stderr_tail` fields when container logs are available.

Each model response counts as one agent turn, including a response that calls
`train_continue`. The synchronous wait inside the tool does not count as an
additional turn. This keeps turn accounting aligned with model inference,
token, and cost accounting.

Training scripts should publish their first `epoch_end` early—at epoch 1 when
the runtime is not yet known, or within roughly 10–20% of the job budget after
a small fixed-batch benchmark. If the job budget expires before the first
boundary, the run consumes training time without producing a candidate.

Every model response and tool result is appended and fsynced under
`runs/<run-id>/events.jsonl`. A summary is written at termination. The same run
directory owns `candidates/`, `hardware/`, `training-jobs/`, the accepted
`submission/`, and `submission-result.json`, so deleting or archiving a run
cannot leave related outcome data scattered elsewhere.

## Workspace process boundary

Commands run through a fixed in-container supervisor. It starts Bash in a new
process group, captures at most 1 MiB of combined output, enforces a timeout,
kills the process group, and removes background processes created during the
call. If the supervisor itself wedges, the outer process kills the disposable
sandbox rather than risk reusing it.

The workspace-command timeout is separate from the metered training budget.
It defaults to 60 seconds and is clamped to the task's `max_command_seconds`.
Results report both `requested_timeout_seconds` and
`effective_timeout_seconds`, so a clamp is visible in the event log.

Workspace snapshots and artifacts cross the tmpfs boundary through a fixed tar
stream implementation. Both sides reject absolute paths, `..`, links, device
files, duplicate files, and size/count limit violations. No host shell evaluates
model-provided strings.

Run `verification/agent` for offline unit tests plus a real isolated lifecycle:
Bash, patching, timeout cleanup, workspace transfer, training, evaluation,
artifact publication, and final submission. It makes no OpenRouter request.
