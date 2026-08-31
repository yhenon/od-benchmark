# Dataset hooks

Each benchmark task builds the generic `agent-base` stage together with exactly
one trusted dataset hook. A hook lives at `docker/datasets/<dataset-id>/` and has
two files:

- `prepare.py DESTINATION SOURCE` runs at image build time. It writes the
  permitted data and a `manifest.json` into `DESTINATION`. Download-backed
  hooks may ignore `SOURCE`; local datasets receive it through a read-only
  BuildKit context selected by `ODBENCH_TRAIN_DATA_SOURCE`.
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

Preparation happens inside the Docker build's disposable `dataset-builder`
stage; there is no repository `out/` contract. To inspect a hook manually, pass
a destination below the gitignored `tmp/` directory, for example
`tmp/datasets/<dataset-id>/train`. These files are scratch data, not benchmark
inputs or run outcomes.
