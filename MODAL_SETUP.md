# Modal training on this checkout

This checkout is nlyu1/assignments. Training runs on Modal, so local CUDA and local data downloads are unnecessary.

## Configured access

- GitHub: `nlyu1/assignments`, with upstream `deep-learning-alchemy/assignments`.
- Modal profile/workspace: `cs312-f26`.
- Assigned environment: `cs312-nlyu` (set in `utils.py` and the local Modal CLI).
- GPU: one H100 per default run. The course permits two concurrent GPUs; pass `max_parallel_runs=2` when launching larger sweeps.
- Shared read-only data: `hard-dl-dclm-v1` in `cs312-shared-data`, mounted at `/root/shared_data`.
- Writable outputs: `volume-dl_alchemy` in `cs312-nlyu`, mounted at `/root/data`; models/checkpoints are in `/root/data/ckpts`.
- W&B: `lyuxingjian-na/assignments`. API key is stored only in Modal secret `dl-alchemy-wandb`; it is injected into training containers. No local W&B login is required to train or use the report exporter.

## Run

```sh
uv sync --frozen
uv run python -m experiments.smoke.modal_smoke_train
```

The default smoke launcher is a full training run (d8, 600,000 sequences, context 1024, AdamW, learning rate 0.003, linear decay). It launches detached and sets `force_run=True`, so each invocation requests another run. Avoid launching duplicates accidentally.

## Monitor and retrieve

```sh
uv run modal app logs --env cs312-nlyu APP_ID
uv run modal volume ls --env cs312-nlyu volume-dl_alchemy /ckpts
uv run modal volume get --env cs312-nlyu volume-dl_alchemy /ckpts/RUN_NAME ./downloaded-model
uv run modal run --env cs312-nlyu scripts/modal_run_report.py --run-id WANDB_RUN_ID --output-dir reports/WANDB_RUN_ID
uv run python -m scripts.modal_usage --env cs312-nlyu
```

The report command runs a small CPU function on Modal to read W&B using the existing secret and writes `run-report.json` locally. Model weights and resume checkpoints remain on the Modal volume; W&B logs metrics separately.

The previous checkout is preserved at `../assignments.backup-20260927-075130`.

## Compatibility fixes verified for this setup

- `train.py` creates a standard UUID for the W&B run ID; W&B 0.30 removed the old `wandb.util.generate_id` helper.
- `setuptools` is an explicit locked dependency because Triton 3.2 needs it at runtime when compiling CUDA kernels.
- To recheck GPU compilation after dependency changes: `uv run modal run --env cs312-nlyu -m scripts.modal_runtime_check`.
