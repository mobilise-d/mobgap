"""Tune wear-time models with inner CV and refit on all eligible SUSTAIN days."""

import json
import pickle
from pathlib import Path

from joblib import Memory
from sklearn.base import clone as clone_untrained_estimator

from mobgap.data import SustainWearTimeDataset
from mobgap.utils.misc import get_env_var
from mobgap.weartime import WtdMegaritisCNN, WtdMegaritisCNNTrainer, WtdMegaritisXGBoost, WtdMegaritisXGBoostTrainer
from mobgap.weartime.optimization import WearTimeOptunaSearch

# Edit paths and optimizer configuration here before running.
DATASET_PATH = None  # Defaults to MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH.
CACHE_DIR = None  # Defaults to MOBGAP_CACHE_DIR_PATH or .cache/mobgap.
OUTPUT_DIR = Path(".cache/weartime_models")


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
            WtdMegaritisCNNTrainer(),
            **WtdMegaritisCNNTrainer.OptimizationPresets.sustain_weartime,
        ),
    }

    # Refit the winner on all indexed human and simulated days, without inner subsampling.
    # Windows with uncertain labels remain excluded. There is no outer evaluation here.
    for name, optimizer in optimizers.items():
        optimizer.optimize(dataset)
        trainer = optimizer.optimized_pipeline_
        detector = trainer.algo
        output_dir = OUTPUT_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = "cnn_sustain_weartime" if isinstance(detector, WtdMegaritisCNN) else "xgboost_sustain_weartime"
        with (output_dir / f"{artifact_name}_best_params.json").open("w") as file:
            json.dump(optimizer.best_params_, file, indent=2)
            file.write("\n")
        model_config = detector.model.model_config if isinstance(detector, WtdMegaritisCNN) else detector.model_config
        with (output_dir / f"{artifact_name}_model_config.json").open("w") as file:
            json.dump(model_config.to_dict(), file, indent=2)
            file.write("\n")
        if isinstance(detector, WtdMegaritisCNN):
            detector.model.model.save(output_dir / f"{artifact_name}_model.keras")
        else:
            with (output_dir / f"{artifact_name}_training_params.json").open("w") as file:
                json.dump(
                    {name: value for name, value in trainer.get_params(deep=False).items() if name != "algo"},
                    file,
                    indent=2,
                )
                file.write("\n")
            with (output_dir / f"{artifact_name}_classifier_template.pkl").open("wb") as file:
                pickle.dump(clone_untrained_estimator(detector.clf), file)
            with (output_dir / f"{artifact_name}_model.pkl").open("wb") as file:
                pickle.dump(detector.clf, file)


if __name__ == "__main__":
    main()
