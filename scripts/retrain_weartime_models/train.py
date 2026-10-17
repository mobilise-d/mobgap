"""Tune wear-time models with inner CV and refit on all eligible SUSTAIN days."""

import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import optuna
import tensorflow as tf
from joblib import Memory, effective_n_jobs
from sqlalchemy.pool import NullPool

from mobgap import PROJECT_ROOT
from mobgap.data import SustainWearTimeDataset
from mobgap.utils.misc import get_env_var
from mobgap.weartime.cnn import WtdMegaritisCNNTrainer
from mobgap.weartime.optimization import WearTimeOptunaSearch
from mobgap.weartime.xgboost import WtdMegaritisXGBoost, WtdMegaritisXGBoostTrainer

# Edit paths and optimizer configuration here before running.
DATASET_PATH = None  # Defaults to MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH.
CACHE_DIR = None  # Defaults to MOBGAP_CACHE_DIR_PATH or .cache/mobgap.
ARTIFACTS = {
    "WtdMegaritisXGBoost": (
        PROJECT_ROOT / "src/mobgap/weartime/xgboost/_models",
        "xgboost_sustain_weartime",
    ),
    "WtdMegaritisCNN": (
        PROJECT_ROOT / "src/mobgap/weartime/cnn/_models",
        "cnn_sustain_weartime",
    ),
    "WtdMegaritisCNNLSTM": (
        PROJECT_ROOT / "src/mobgap/weartime/cnn/_models",
        "cnn_lstm_sustain_weartime",
    ),
}


def get_study_params(seed: int, *, database: Path, study_name: str) -> dict:
    """Create a fresh persistent study shared by this model's trial workers."""
    return {
        "storage": optuna.storages.RDBStorage(
            f"sqlite:///{database.resolve().as_posix()}",
            engine_kwargs={"poolclass": NullPool},
        ),
        "study_name": study_name,
        "load_if_exists": False,
        "direction": "maximize",
        "sampler": optuna.samplers.TPESampler(seed=seed),
    }


def main() -> None:
    """Export selected hyperparameters and models refitted on all indexed days."""
    cache_dir = Path(CACHE_DIR or get_env_var("MOBGAP_CACHE_DIR_PATH", ".cache/mobgap")).expanduser()
    n_jobs = effective_n_jobs(int(get_env_var("MOBGAP_N_JOBS", 1)))
    dataset = SustainWearTimeDataset(
        Path(DATASET_PATH or get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")).expanduser(),
        additional_sensors_enabled=(),
        memory=Memory(cache_dir, verbose=0),
    )

    # Cap independent CNN trial workers at the scheduler-visible GPU count.
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    cnn_n_jobs = min(n_jobs, gpu_count) if gpu_count else n_jobs
    gpu_ids = tuple(range(cnn_n_jobs)) if gpu_count else None
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Run trials in separate processes; XGBoost extracts recordings serially with feature threads.
    # Rank candidates on participant-grouped human validation folds only.
    # The preset samples human and simulated non-wear days in each inner training set.
    optimizers = {
        "WtdMegaritisXGBoost": WearTimeOptunaSearch(
            WtdMegaritisXGBoostTrainer(
                algo=WtdMegaritisXGBoost(
                    memory=Memory(cache_dir, compress=3, verbose=0),
                ),
                n_jobs=1,
            ),
            **{
                **WtdMegaritisXGBoostTrainer.HyperparameterSearchPresets.sustain_weartime,
                "n_jobs": n_jobs,
            },
        ),
        "WtdMegaritisCNN": WearTimeOptunaSearch(
            WtdMegaritisCNNTrainer(**WtdMegaritisCNNTrainer.PredefinedParameters.empty_cnn),
            **{
                **WtdMegaritisCNNTrainer.HyperparameterSearchPresets.sustain_weartime,
                "n_jobs": cnn_n_jobs,
                "gpu_ids": gpu_ids,
            },
        ),
        "WtdMegaritisCNNLSTM": WearTimeOptunaSearch(
            WtdMegaritisCNNTrainer(**WtdMegaritisCNNTrainer.PredefinedParameters.empty_cnn_lstm),
            **{
                **WtdMegaritisCNNTrainer.HyperparameterSearchPresets.sustain_weartime,
                "n_jobs": cnn_n_jobs,
                "gpu_ids": gpu_ids,
            },
        ),
    }

    # Refit the winner on all indexed human and simulated days, without inner subsampling.
    # Windows with uncertain labels remain excluded. There is no outer evaluation here.
    for name, optimizer in optimizers.items():
        # Parallel trials share a fresh local study; remove it after fitting/export or failure.
        with (
            TemporaryDirectory(prefix="weartime-study-", dir=cache_dir) if optimizer.n_jobs > 1 else nullcontext(None)
        ) as study_dir:
            if study_dir is not None:
                optimizer.set_params(
                    get_study_params=partial(
                        get_study_params,
                        database=Path(study_dir) / "study.sqlite3",
                        study_name=f"{name}-{uuid4().hex}",
                    ),
                )
            optimizer.optimize(dataset)
            trainer = optimizer.optimized_pipeline_
            output_dir, artifact_name = ARTIFACTS[name]
            trainer.save_model(output_dir, prefix=artifact_name)
            with (output_dir / f"{artifact_name}_best_params.json").open("w") as file:
                json.dump(optimizer.best_params_, file, indent=2)
                file.write("\n")


if __name__ == "__main__":
    main()
