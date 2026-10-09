"""Tune and refit wear-time models on the complete SUSTAIN dataset."""

import pickle
from pathlib import Path

from joblib import Memory

from mobgap.data import SustainWearTimeDataset
from mobgap.utils.misc import get_env_var
from mobgap.weartime import MegaritisCnnWeartimeModel, WtdMegaritisCNN, WtdMegaritisXGBoost
from mobgap.weartime.optimization import WearTimeOptunaSearch
from mobgap.weartime.pipeline import WtdEmulationPipeline

# Edit paths and optimizer configuration here before running.
DATASET_PATH = None  # Defaults to MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH.
CACHE_DIR = None  # Defaults to MOBGAP_CACHE_DIR_PATH or .cache/mobgap.
OUTPUT_DIR = Path(".cache/weartime_models")


def main() -> None:
    """Tune on inner CV, then export models refitted on the complete dataset."""
    cache_dir = Path(CACHE_DIR or get_env_var("MOBGAP_CACHE_DIR_PATH", ".cache/mobgap")).expanduser()
    dataset = SustainWearTimeDataset(
        DATASET_PATH or get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH"),
        additional_sensors_enabled=(),
        memory=Memory(cache_dir, verbose=0),
    )

    # These presets rank human inner validation folds and add simulated training days.
    optimizers = {
        "WtdMegaritisXGBoost": WearTimeOptunaSearch(
            WtdEmulationPipeline(
                WtdMegaritisXGBoost(
                    **WtdMegaritisXGBoost.PredefinedParameters.untrained_lightweight,
                    memory=Memory(cache_dir / "xgboost_features", compress=3, verbose=0),
                )
            ),
            **WtdMegaritisXGBoost.OptimizationPresets.sustain_weartime,
        ),
        "WtdMegaritisCNN": WearTimeOptunaSearch(
            WtdEmulationPipeline(WtdMegaritisCNN(model=MegaritisCnnWeartimeModel())),
            **WtdMegaritisCNN.OptimizationPresets.sustain_weartime,
        ),
    }

    # Refit each search winner on all days; this script performs no outer evaluation.
    for name, optimizer in optimizers.items():
        optimizer.optimize(dataset)
        detector = optimizer.optimized_pipeline_.algo
        output_dir = OUTPUT_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(detector, WtdMegaritisCNN):
            detector.model._model.save(output_dir / "model.keras")
        else:
            with (output_dir / "model.pkl").open("wb") as file:
                pickle.dump(detector.clf, file)
            with (output_dir / "feature_order.pkl").open("wb") as file:
                pickle.dump(list(detector.feature_names), file)


if __name__ == "__main__":
    main()
