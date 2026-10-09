"""Evaluate the XGBoost wear-time model with daily participant LOSO."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tpcp.validate import CombinedSplitter, DatasetSplitter, NoSplit, SubsetSplitter

from mobgap.data import SustainWearTimeDataset, split_by_utc_day
from mobgap.utils.evaluation import EvaluationCV
from mobgap.utils.misc import get_env_var
from mobgap.weartime import WtdMegaritisXGBoost
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.optimization import WearTimeOptunaSearch
from mobgap.weartime.pipeline import WtdEmulationPipeline

# Edit configuration here before running the script.
DATASET_PATH = None  # Defaults to MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH.
CACHE_DIR = None  # Defaults to MOBGAP_CACHE_DIR_PATH or .cache/mobgap.
OUTPUT_DIR = Path(".cache/weartime_loso_runs")
RUN_NAME = None  # Defaults to a timestamped directory.
PARTICIPANT_IDS = None  # None selects all human participants.
MAX_PARTICIPANTS = None
SIMULATED_NON_WEAR_DAY_COUNT = None  # None keeps all simulated non-wear training days; an integer samples days.
SEED = 42
OVERLAP = 0.75
WINDOW_BATCH_SIZE = 8192
FEATURE_N_JOBS = 1
CV_N_JOBS = 1
N_TRIALS = 20
INNER_FOLDS = 5
SEARCH_TRAIN_FRACTION = 0.4


def main() -> None:
    """Train and evaluate the configured model."""
    run_name = RUN_NAME or f"loso_daily_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # Hold out one human participant; repeat a fixed 50/50 simulated non-wear recording split in every outer fold.
    outer_splitter = CombinedSplitter(
        parts=[
            (
                "human",
                SubsetSplitter(
                    select_human_days, DatasetSplitter(base_splitter=LeaveOneGroupOut(), groupby="participant_id")
                ),
            ),
            (
                "simulated_non_wear",
                SubsetSplitter(
                    lambda days: days.get_subset(recording_type="simulated_movements"),
                    NoSplit(
                        None,
                        train=lambda days: days.get_subset(
                            recording_id=days.index["recording_id"]
                            .drop_duplicates()
                            .sort_values()
                            .sample(frac=1, random_state=SEED)
                            .iloc[: days.index["recording_id"].nunique() // 2]
                            .tolist()
                        ),
                        test=lambda days: days.get_subset(
                            recording_id=days.index["recording_id"]
                            .drop_duplicates()
                            .sort_values()
                            .sample(frac=1, random_state=SEED)
                            .iloc[days.index["recording_id"].nunique() // 2 :]
                            .tolist()
                        ),
                    ),
                ),
            ),
        ]
    )

    # Configure XGBoost with cached recording features.
    pipeline = WtdEmulationPipeline(
        WtdMegaritisXGBoost(
            **WtdMegaritisXGBoost.PredefinedParameters.untrained_lightweight,
            window_batch_size=WINDOW_BATCH_SIZE,
            n_jobs=FEATURE_N_JOBS,
            overlap=OVERLAP,
            memory=joblib.Memory(cache_dir / "xgboost_features", compress=3, verbose=0),
        )
    )

    # Keep the preset's human-only inner validation and train-only simulated non-wear composition.
    optimization_preset = WtdMegaritisXGBoost.OptimizationPresets.sustain_weartime
    optimization_preset["cv"].set_params(parts__human__splitter__base_splitter=INNER_FOLDS)

    evaluation = EvaluationCV(
        dataset=base_dataset,
        scoring=wtd_score,
        cv_iterator=outer_splitter,
        cv_params={
            "n_jobs": CV_N_JOBS,
            "return_train_score": False,
            "progress_bar": True,
        },
    )
    optimizer = WearTimeOptunaSearch(
        pipeline=pipeline,
        **{
            **optimization_preset,
            "n_trials": N_TRIALS,
            "random_seed": SEED,
            # Sample human training days and optional simulated non-wear days independently.
            "train_dataset_transform": lambda inner_train_days: inner_train_days.get_subset(
                index=pd.concat(
                    [
                        inner_train_days.index.query("recording_type == 'human_movement'").sample(
                            n=max(
                                1,
                                round(
                                    (inner_train_days.index["recording_type"] == "human_movement").sum()
                                    * SEARCH_TRAIN_FRACTION
                                ),
                            ),
                            random_state=SEED,
                        ),
                        (
                            inner_train_days.index.query("recording_type == 'simulated_movements'").sample(
                                n=SIMULATED_NON_WEAR_DAY_COUNT, random_state=SEED
                            )
                            if SIMULATED_NON_WEAR_DAY_COUNT is not None
                            else inner_train_days.index.query("recording_type == 'simulated_movements'")
                        ),
                    ]
                )
            ),
        },
    )

    # Finally run the full evaluation
    evaluation.run(optimizer)

    # Export only the held-out metrics.
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.get_aggregated_results_as_df(group="test").to_csv(output_dir / "fold_results.csv")
    evaluation.get_single_results_as_df(group="test").to_csv(output_dir / "daily_results.csv")

    # Fit the final export on all selected human days and both simulated non-wear halves after scoring.
    selected_labels = list(
        dict.fromkeys(label for train, test in outer_splitter.split(base_dataset) for label in [*train, *test])
    )
    optimizer.optimize(base_dataset.get_subset(group_labels=selected_labels))
    detector = optimizer.optimized_pipeline_.algo
    with (output_dir / "model.pkl").open("wb") as file:
        pickle.dump(detector.clf, file)
    with (output_dir / "feature_order.pkl").open("wb") as file:
        pickle.dump(list(detector.feature_names), file)


if __name__ == "__main__":
    main()
