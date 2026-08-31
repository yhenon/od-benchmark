# Verification

This directory keeps heavyweight Docker and end-to-end checks out of the
runtime implementation:

- `sandbox` checks the base and dataset sandbox images and their isolation.
- `eval` checks evaluator isolation, artifact limits, and scoring.
- `train` checks the two-epoch training lifecycle.
- `agent` runs unit tests plus the sandbox and outer-tool integrations.
- `pretrained` loads every bundled initializer with networking disabled, checks
  multiscale feature shapes, adapts the detector head, and exports a backbone
  to ONNX.

Run a check from the repository root, for example:

```sh
verification/agent
```

Fast unit tests remain under `tests/` and do not require Docker.
