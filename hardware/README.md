# Hardware configuration

Hardware has two separate roles in a benchmark task:

- `training/` describes the trusted training container's compute quota and
  environment (CPU/GPU allocation, memory, shared memory, process limits, and
  thread settings).
- `targets/` describes the physical deployment target and its default
  acceptance threshold. It does not grant compute resources to training.

Task definitions name one file from each directory through the
`training_hardware` and `target_hardware` fields. `docker/prepare` validates and
copies both configurations into the private prepared-task bundle.

Each task may override the target's default per-sample inference allowance with
`limits.max_inference_runtime_seconds`. The strict `verify_on_hw` limit uses
that value directly; final submission applies the target profile's configured
submission tolerance.
