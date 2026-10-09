"""Evaluate the CNN wear-time model with daily participant LOSO."""

from __future__ import annotations

from datetime import datetime
from importlib import import_module
from pathlib import Path

import joblib
from sklearn.model_selection import LeaveOneGroupOut
from tpcp.optimize import Optimize
from tpcp.validate import CombinedSplitter, DatasetSplitter, NoSplit

from mobgap.data import SustainWearTimeDataset, split_by_utc_day
from mobgap.utils.evaluation import EvaluationCV
from mobgap.utils.misc import get_env_var
from mobgap.weartime import MegaritisCnnWeartimeModel, WtdMegaritisCNN
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.pipeline import WtdEmulationPipeline

# Edit configuration here before running the script.
DATASET_PATH = None  # Defaults to MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH.
CACHE_DIR = None  # Defaults to MOBGAP_CACHE_DIR_PATH or .cache/mobgap.
OUTPUT_DIR = Path(".cache/weartime_loso_runs")
RUN_NAME = None  # Defaults to a timestamped directory.
PARTICIPANT_IDS = None  # None selects all human participants.
MAX_PARTICIPANTS = None
SEED = 42
OVERLAP = 0.75
EPOCHS = 60
BATCH_SIZE = 8192
WINDOW_BATCH_SIZE = None  # Defaults to BATCH_SIZE.
SHUFFLE_BUFFER_SIZE = 8192
STANDARDIZE_IN_MODEL = True
FIT_VERBOSE = 1
CV_N_JOBS = 1


def main() -> None:
    """Train and evaluate the configured model."""
    run_name = RUN_NAME or f"loso_daily_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_DIR / run_name

    # Configure the dataset and the human fold selector.
    dataset_path = Path(DATASET_PATH or get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")).expanduser()
    cache_dir = Path(CACHE_DIR or get_env_var("MOBGAP_CACHE_DIR_PATH", ".cache/mobgap")).expanduser()
    base_dataset = SustainWearTimeDataset(
        dataset_path,
        additional_sensors_enabled=(),
        splitter=split_by_utc_day,
        memory=joblib.Memory(cache_dir, verbose=0),
    )

    def select_human_days(days: SustainWearTimeDataset) -> SustainWearTimeDataset:
        human = days.get_subset(recording_type="human_movement")
        if PARTICIPANT_IDS:
            human = human.get_subset(index=human.index[human.index["participant_id"].isin(PARTICIPANT_IDS)])
        if MAX_PARTICIPANTS is not None:
            participant_ids = human.index["participant_id"].drop_duplicates().iloc[:MAX_PARTICIPANTS]
            human = human.get_subset(index=human.index[human.index["participant_id"].isin(participant_ids)])
        return human

    # Hold out one human participant; repeat a fixed 50/50 Part B day split in every outer fold.
    splitter = CombinedSplitter(
        parts=[
            (
                select_human_days,
                DatasetSplitter(base_splitter=LeaveOneGroupOut(), groupby="participant_id"),
            ),
            (
                lambda days: days.get_subset(recording_type="simulated_movements"),
                NoSplit(
                    None,
                    train=lambda days: days.get_subset(
                        index=days.index.sample(frac=1, random_state=SEED).iloc[: len(days.index) // 2]
                    ),
                    test=lambda days: days.get_subset(
                        index=days.index.sample(frac=1, random_state=SEED).iloc[len(days.index) // 2 :]
                    ),
                ),
            ),
        ]
    )

    # Configure TensorFlow and choose safe fold parallelism.
    tf = import_module("tensorflow")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    n_jobs = CV_N_JOBS
    if gpus:
        n_jobs = 1

    # Train and score the CNN through the standard TPCP pipeline.
    pipeline = WtdEmulationPipeline(
        WtdMegaritisCNN(
            model=MegaritisCnnWeartimeModel(
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                fit_verbose=FIT_VERBOSE,
                window_batch_size=WINDOW_BATCH_SIZE or BATCH_SIZE,
                shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
                standardize_in_model=STANDARDIZE_IN_MODEL,
                overlap=OVERLAP,
            )
        )
    )
    evaluation = EvaluationCV(
        dataset=base_dataset,
        scoring=wtd_score,
        cv_iterator=splitter,
        cv_params={
            "n_jobs": n_jobs,
            "return_train_score": False,
            "progress_bar": True,
        },
    )
    optimizer = Optimize(pipeline)

    evaluation.run(optimizer)

    # Export only the held-out metrics.
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.get_aggregated_results_as_df(group="test").to_csv(output_dir / "fold_results.csv")
    evaluation.get_single_results_as_df(group="test").to_csv(output_dir / "daily_results.csv")

    # Fit the final export on all selected human days and both Part B halves after scoring.
    selected_labels = list(
        dict.fromkeys(label for train, test in splitter.split(base_dataset) for label in [*train, *test])
    )
    optimizer.optimize(base_dataset.get_subset(group_labels=selected_labels))
    optimizer.optimized_pipeline_.algo.model._model.save(output_dir / "model.keras")


if __name__ == "__main__":
    main()
