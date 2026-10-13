"""Tune wear-time models with inner CV and refit on all eligible SUSTAIN days."""

import json
from pathlib import Path

from joblib import Memory

from mobgap.data import SustainWearTimeDataset
from mobgap.utils.misc import get_env_var
from mobgap.weartime.cnn import WtdMegaritisCNNTrainer
from mobgap.weartime.optimization import WearTimeOptunaSearch
from mobgap.weartime.xgboost import WtdMegaritisXGBoost, WtdMegaritisXGBoostTrainer

# Edit paths and optimizer configuration here before running.
DATASET_PATH = None  # Defaults to MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH.
CACHE_DIR = None  # Defaults to MOBGAP_CACHE_DIR_PATH or .cache/mobgap.
OUTPUT_DIR = Path(".cache/weartime_models")
ARTIFACT_PREFIXES = {
    "WtdMegaritisXGBoost": "xgboost_sustain_weartime",
    "WtdMegaritisCNN": "cnn_sustain_weartime",
    "WtdMegaritisCNNLSTM": "cnn_lstm_sustain_weartime",
}


def main() -> None:
    """Export selected hyperparameters and models refitted on all indexed days."""
    cache_dir = Path(CACHE_DIR or get_env_var("MOBGAP_CACHE_DIR_PATH", ".cache/mobgap")).expanduser()
    dataset = SustainWearTimeDataset(
        Path(DATASET_PATH or get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")).expanduser(),
        additional_sensors_enabled=(),
        memory=Memory(cache_dir, verbose=0),
    )

    # Rank candidates on three participant-grouped human validation folds only.
    # Each inner training set samples 40% of its human days and five simulated non-wear days.
    optimizers = {
        "WtdMegaritisXGBoost": WearTimeOptunaSearch(
            WtdMegaritisXGBoostTrainer(
                algo=WtdMegaritisXGBoost(
                    **WtdMegaritisXGBoost.PredefinedParameters.baseline_sustain_weartime,
                    memory=Memory(cache_dir / "xgboost_features", compress=3, verbose=0),
                ),
            ),
            **WtdMegaritisXGBoostTrainer.OptimizationPresets.sustain_weartime,
        ),
        "WtdMegaritisCNN": WearTimeOptunaSearch(
            WtdMegaritisCNNTrainer(**WtdMegaritisCNNTrainer.PredefinedParameters.empty_cnn),
            **WtdMegaritisCNNTrainer.OptimizationPresets.sustain_weartime,
        ),
        "WtdMegaritisCNNLSTM": WearTimeOptunaSearch(
            WtdMegaritisCNNTrainer(**WtdMegaritisCNNTrainer.PredefinedParameters.empty_cnn_lstm),
            **WtdMegaritisCNNTrainer.OptimizationPresets.sustain_weartime,
        ),
    }

    # Refit the winner on all indexed human and simulated days, without inner subsampling.
    # Windows with uncertain labels remain excluded. There is no outer evaluation here.
    for name, optimizer in optimizers.items():
        optimizer.optimize(dataset)
        trainer = optimizer.optimized_pipeline_
        output_dir = OUTPUT_DIR / name
        artifact_name = ARTIFACT_PREFIXES[name]
        trainer.save_model(output_dir, prefix=artifact_name)
        with (output_dir / f"{artifact_name}_best_params.json").open("w") as file:
            json.dump(optimizer.best_params_, file, indent=2)
            file.write("\n")


if __name__ == "__main__":
    main()
