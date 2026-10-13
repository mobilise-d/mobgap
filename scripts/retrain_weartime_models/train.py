"""Tune wear-time models with inner CV and refit on all eligible SUSTAIN days."""

import json
from pathlib import Path

from joblib import Memory

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


def main() -> None:
    """Export selected hyperparameters and models refitted on all indexed days."""
    cache_dir = Path(CACHE_DIR or get_env_var("MOBGAP_CACHE_DIR_PATH", ".cache/mobgap")).expanduser()
    n_jobs = int(get_env_var("MOBGAP_N_JOBS", 1))
    dataset = SustainWearTimeDataset(
        Path(DATASET_PATH or get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")).expanduser(),
        additional_sensors_enabled=(),
        memory=Memory(cache_dir, verbose=0),
    )

    # Rank candidates on participant-grouped human validation folds only.
    # The preset samples human and simulated non-wear days in each inner training set.
    optimizers = {
        "WtdMegaritisXGBoost": WearTimeOptunaSearch(
            WtdMegaritisXGBoostTrainer(
                algo=WtdMegaritisXGBoost(
                    memory=Memory(cache_dir, compress=3, verbose=0),
                ).set_params(clf__n_jobs=n_jobs),
                n_jobs=n_jobs,
            ),
            **WtdMegaritisXGBoostTrainer.HyperparameterSearchPresets.sustain_weartime,
        ),
        "WtdMegaritisCNN": WearTimeOptunaSearch(
            WtdMegaritisCNNTrainer(**WtdMegaritisCNNTrainer.PredefinedParameters.empty_cnn),
            **WtdMegaritisCNNTrainer.HyperparameterSearchPresets.sustain_weartime,
        ),
        "WtdMegaritisCNNLSTM": WearTimeOptunaSearch(
            WtdMegaritisCNNTrainer(**WtdMegaritisCNNTrainer.PredefinedParameters.empty_cnn_lstm),
            **WtdMegaritisCNNTrainer.HyperparameterSearchPresets.sustain_weartime,
        ),
    }

    # Refit the winner on all indexed human and simulated days, without inner subsampling.
    # Windows with uncertain labels remain excluded. There is no outer evaluation here.
    for name, optimizer in optimizers.items():
        optimizer.optimize(dataset)
        trainer = optimizer.optimized_pipeline_
        output_dir, artifact_name = ARTIFACTS[name]
        trainer.save_model(output_dir, prefix=artifact_name)
        with (output_dir / f"{artifact_name}_best_params.json").open("w") as file:
            json.dump(optimizer.best_params_, file, indent=2)
            file.write("\n")


if __name__ == "__main__":
    main()
