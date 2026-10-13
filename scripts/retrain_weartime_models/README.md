# Wear-time revalidation and retraining

Run these commands from the repository root. Use a Python environment supported
by TensorFlow and install the wear-time dependencies:

```bash
uv sync --extra weartime
```

## Local setup

Create a repository-local `.env` file with paths for this machine:

| Variable | What to provide |
| --- | --- |
| `MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH` | Local SUSTAIN wear-time dataset directory. |
| `MOBGAP_CACHE_DIR_PATH` | Writable disk cache, preferably shared across worktrees. |
| `MOBGAP_VALIDATION_DATA_PATH` | Local checkout of the companion evaluation-results repository, for revalidation. |
| `MOBGAP_N_JOBS` | CPU worker count for retraining and evaluation, appropriate for the available CPU and memory. |

The raw dataset is not included in this repository. When using a worktree, copy
the main checkout's `.env` and adjust its paths. For a code branch, use a matching
branch and worktree in the companion results repository.

Review the selected algorithms and resource settings in the scripts before
starting. Change configuration in the scripts; there are no command-line options.
Full runs can take a long time. CNN runs should use a machine with a GPU.
Set `CUDA_VISIBLE_DEVICES` in the launch environment to select GPUs, or use the
devices assigned by the scheduler. Set it before starting Python. Retraining
runs independent CNN trials across the visible GPUs, capped by the configured
worker count, and refits the selected model on the first GPU. Without visible
GPUs, CNN tuning uses the configured worker count on CPU. Parallel searches
use temporary Optuna storage under the cache directory and remove it afterward,
including on failure. Keep that directory on local disk.

## Revalidation

Generate results, then run their analysis:

```bash
uv run python revalidation/weartime/_02_wtd_result_generation_no_exc.py
uv run python revalidation/weartime/_01_wtd_analysis_no_exc.py
```

To run selected algorithms, temporarily comment out the other entries in the
generation script and align the analysis selection with the available outputs.
Restore those edits afterward. Reuse the existing hyperparameter-search presets.

Results are written to the companion results checkout configured in `.env`.
Check `results_base_path` in the scripts for the destination. Keep the raw result
files together so the analysis can find them. Review the analysis before deciding
whether to retain and publish the companion results.

## Retraining production models

Review the algorithm selection and paths in `train.py`, then run:

```bash
uv run python scripts/retrain_weartime_models/train.py
```

The script writes trained models and their associated configuration and search
artifacts directly into the package's model directories:

| Algorithm | Destination |
| --- | --- |
| XGBoost | `src/mobgap/weartime/xgboost/_models/` |
| CNN and CNN-LSTM | `src/mobgap/weartime/cnn/_models/` |

Existing files with matching names are overwritten. Back up any production
artifacts you need to retain before retraining. Keep the exported files and
filenames together; the packaged model loaders expect them. Retraining does not
produce revalidation results. Run revalidation separately when results are needed.
