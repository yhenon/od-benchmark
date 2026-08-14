# Dataset hooks

Each benchmark task builds the generic `agent-base` stage together with exactly
one trusted dataset hook. A hook lives at `docker/datasets/<dataset-id>/` and has
two files:

- `prepare.py DESTINATION` runs with network access only at image build time. It
  writes the permitted data and a `manifest.json` into `DESTINATION`.
- `runtime.py` is copied into the final image as `odbench_dataset`. It exposes
  `load(root: Path, split: str, **kwargs) -> Dataset`.

Manifest schema version 1 requires `id`, `display_name`, `task`,
`default_split`, and a non-empty `splits` object. Each split may declare
`num_examples`; the generic image validator checks it against `len(dataset)`.

Dataset hooks are benchmark-owned code and part of the trusted build boundary.
They must copy only agent-visible training data. Held-out validation/test data
must stay in the separate evaluation service and must never be named in the
manifest or emitted by the preparation hook.

Build a hook with `ODBENCH_DATASET=<dataset-id> docker/sandbox build`. The result
is tagged `od-benchmark-agent:<dataset-id>-dev` by default.
