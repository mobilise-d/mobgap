"""Run daily aggregate participant-grouped evaluation for the SUSTAIN wear-time XGBoost model.

This mirrors ``loso_daily_cnn.py``: the dataset is split by recording day, while CV groups by participant so every fold
holds out all days of one human participant. Two part B recordings are added to each training fold.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from loso_daily_cnn import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    _fold_metadata,
    _write_results,
)
from optimizable_optuna_search import OptimizableOptunaSearch
from optuna import Trial
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from tpcp.optimize import Optimize
from tpcp.validate import CombinedSplitter, DatasetSplitter, NoSplit, cross_validate

from mobgap.data import SustainWearTimeDataset
from mobgap.utils.evaluation import EvaluationCV
from mobgap.utils.misc import get_env_var
from mobgap.weartime import WtdMegaritisXGBoost
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.pipeline import WtdEmulationPipeline

LOGGER = logging.getLogger(__name__)


def _study_params(seed: int) -> dict[str, Any]:
    return {"direction": "maximize", "sampler": optuna.samplers.TPESampler(seed=seed)}


def _write_search_results(evaluation: EvaluationCV, output_dir: Path) -> None:
    best_rows = []
    trial_frames = []
    for fold, optimizer in enumerate(evaluation.results_["optimizer"]):
        best_rows.append({"fold": fold, "inner_accuracy": optimizer.best_score_, **optimizer.best_params_})
        trials = optimizer.study_.trials_dataframe()
        trials.insert(0, "fold", fold)
        trial_frames.append(trials)
    pd.DataFrame(best_rows).to_csv(output_dir / "optuna_best_by_fold.csv", index=False)
    pd.concat(trial_frames, ignore_index=True).to_csv(output_dir / "optuna_trials.csv", index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        help="Path to the SUSTAIN Wear-time folder. Defaults to MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH.",
    )
    parser.add_argument(
        "--cache-dir",
        help=f"joblib cache directory. Defaults to MOBGAP_CACHE_DIR_PATH or {DEFAULT_CACHE_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for LOSO result artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--run-name", help="Artifact folder name. Defaults to loso_daily_xgboost_<timestamp>.")
    parser.add_argument(
        "--part-b-recording-id",
        action="append",
        help="Part B recording to add to every training fold. Pass twice; defaults to the first two sorted IDs.",
    )
    parser.add_argument(
        "--participant-id",
        action="append",
        help="Restrict evaluation to one participant. Can be passed multiple times; mainly useful for smoke tests.",
    )
    parser.add_argument(
        "--max-participants",
        type=int,
        help="Restrict to the first N participants in dataset order; mainly useful for smoke tests.",
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=8192,
        help="Number of XGBoost feature windows processed together.",
    )
    parser.add_argument("--overlap", type=float, default=0.75, help="Fractional overlap of XGBoost windows.")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per outer participant fold.")
    parser.add_argument("--inner-folds", type=int, default=5, help="Participant-grouped folds per Optuna trial.")
    parser.add_argument(
        "--search-train-fraction",
        type=float,
        default=0.4,
        help="Fraction of human inner training days sampled for each Optuna trial; part B is excluded from search.",
    )
    parser.add_argument("--search-seed", type=int, default=42, help="Seed for Optuna and inner training-day sampling.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of process workers for datapoint-level XGBoost feature extraction during training.",
    )
    parser.add_argument(
        "--cv-n-jobs",
        type=int,
        default=1,
        help="Number of CV folds to evaluate in parallel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the dataset and LOSO split metadata; do not train or score models.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915 - Keep the LOSO composition visible in one place.
    """Run the XGBoost LOSO evaluation."""
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    run_name = args.run_name or f"loso_daily_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir).expanduser() / run_name

    # Select the human days to evaluate.
    dataset_path = Path(args.dataset_path or get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")).expanduser()
    cache_dir = Path(args.cache_dir or get_env_var("MOBGAP_CACHE_DIR_PATH", str(DEFAULT_CACHE_DIR))).expanduser()
    base_dataset = SustainWearTimeDataset(
        dataset_path,
        additional_sensors_enabled=(),
        warn_thres_for_sampling_rate_deviations_hz=None,
        split_by_day=True,
        memory=joblib.Memory(cache_dir, verbose=0),
    )
    dataset = base_dataset.get_subset(recording_type="human_movement")
    if args.participant_id:
        dataset = dataset.get_subset(index=dataset.index[dataset.index["participant_id"].isin(args.participant_id)])
    if args.max_participants is not None:
        selected_participants = dataset.index["participant_id"].drop_duplicates().iloc[: args.max_participants]
        dataset = dataset.get_subset(index=dataset.index[dataset.index["participant_id"].isin(selected_participants)])
    if len(dataset.index) == 0:
        raise ValueError("The selected SUSTAIN human split-by-day subset is empty.")

    # Add the same two Part B recordings to every outer training fold.
    part_b_index = base_dataset.get_subset(recording_type="simulated_movements").index
    available_ids = sorted(part_b_index["recording_id"].unique())
    selected_ids = available_ids[:2] if args.part_b_recording_id is None else args.part_b_recording_id
    if len(selected_ids) != 2 or len(set(selected_ids)) != 2 or not set(selected_ids).issubset(available_ids):
        raise ValueError("Select exactly two distinct part B recording IDs present in the dataset.")
    training_only_index = part_b_index[part_b_index["recording_id"].isin(selected_ids)].reset_index(drop=True)
    evaluation_index = pd.concat([dataset.index, training_only_index], ignore_index=True)
    evaluation_dataset = base_dataset.get_subset(index=evaluation_index)

    # Hold out one human participant per outer fold; keep Part B out of test folds.
    outer_splitter = CombinedSplitter(
        parts=[
            (
                lambda days: days.get_subset(recording_type="human_movement"),
                DatasetSplitter(base_splitter=LeaveOneGroupOut(), groupby="participant_id"),
            ),
            (
                lambda days: days.get_subset(recording_type="simulated_movements"),
                NoSplit(dataset.index["participant_id"].nunique(), train=lambda days: days),
            ),
        ]
    )
    fold_metadata = _fold_metadata(evaluation_dataset, outer_splitter, training_only_index)

    LOGGER.info("Selected split-by-day human datapoints: %s", len(dataset.index))
    LOGGER.info("Selected participants: %s", dataset.index["participant_id"].nunique())
    LOGGER.info("Participant-grouped CV folds: %s", len(fold_metadata))
    LOGGER.info(
        "Part B recordings added to every training fold: %s", sorted(training_only_index["recording_id"].unique())
    )
    LOGGER.info("XGBoost datapoint feature workers: %s", args.n_jobs)
    LOGGER.info("CV fold workers: %s", args.cv_n_jobs)
    LOGGER.info("Optuna trials per outer fold: %s", args.n_trials)
    LOGGER.info("Output directory: %s", output_dir)

    # Preview the outer fold plan without fitting a model.
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.index.to_csv(output_dir / "dataset_index.csv", index=False)
        training_only_index.to_csv(output_dir / "training_only_index.csv", index=False)
        fold_metadata.to_csv(output_dir / "fold_metadata.csv", index=False)
        LOGGER.info("Dry run complete. Wrote fold metadata and human and part B dataset indices.")
        return

    # Check settings that are only needed for Optuna training.
    if not 0 < args.search_train_fraction <= 1:
        raise ValueError("--search-train-fraction must be in (0, 1].")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive.")
    if args.inner_folds < 2 or args.inner_folds > dataset.index["participant_id"].nunique() - 1:
        raise ValueError("--inner-folds must be between 2 and the number of outer training participants.")

    # Configure XGBoost with cached recording features.
    pipeline = WtdEmulationPipeline(
        WtdMegaritisXGBoost(
            **WtdMegaritisXGBoost.PredefinedParameters.untrained_lightweight,
            window_batch_size=args.window_batch_size,
            n_jobs=args.n_jobs,
            overlap=args.overlap,
            memory=joblib.Memory(cache_dir / "xgboost_features", compress=3, verbose=0),
        )
    )

    # Search human-only inner folds, then refit on the full outer training fold.
    inner_splitter = CombinedSplitter(
        parts=[
            (
                lambda days: days.get_subset(recording_type="human_movement"),
                DatasetSplitter(GroupKFold(n_splits=args.inner_folds), groupby="participant_id"),
            )
        ]
    )

    def objective(trial: Trial, candidate: WtdEmulationPipeline, train_days: SustainWearTimeDataset) -> float:
        params = {
            "algo__clf__n_estimators": trial.suggest_categorical("algo__clf__n_estimators", [50, 100, 200]),
            "algo__clf__max_depth": trial.suggest_categorical("algo__clf__max_depth", [3, 5]),
            "algo__clf__learning_rate": trial.suggest_categorical("algo__clf__learning_rate", [0.05, 0.1]),
            "algo__clf__subsample": trial.suggest_categorical("algo__clf__subsample", [0.7, 0.8, 0.9]),
            "algo__clf__colsample_bytree": trial.suggest_categorical("algo__clf__colsample_bytree", [0.6, 0.8]),
            "algo__clf__min_child_weight": trial.suggest_categorical("algo__clf__min_child_weight", [1, 3]),
        }
        candidate.set_params(**params)
        inner_optimizer = Optimize(
            candidate,
            train_dataset_transform=lambda inner_train_days: inner_train_days.get_subset(
                index=inner_train_days.index.sample(
                    n=max(1, round(len(inner_train_days.index) * args.search_train_fraction)),
                    random_state=args.search_seed,
                )
            ),
        )
        scores = cross_validate(
            inner_optimizer,
            train_days,
            scoring=wtd_score,
            cv=inner_splitter,
            n_jobs=1,
            return_train_score=False,
            progress_bar=False,
        )
        return float(np.mean(scores["test__agg__combined__accuracy"]))

    evaluation = EvaluationCV(
        dataset=evaluation_dataset,
        scoring=wtd_score,
        cv_iterator=outer_splitter,
        cv_params={
            "n_jobs": args.cv_n_jobs,
            "return_train_score": False,
            "return_optimizer": True,
            "progress_bar": True,
        },
    )
    optimizer = OptimizableOptunaSearch(
        pipeline, _study_params, objective, n_trials=args.n_trials, random_seed=args.search_seed
    )

    evaluation.run(optimizer)

    # Write fold metrics, search results, and evaluator timings.
    run_metadata = {
        "run_name": run_name,
        "xgboost_version": version("xgboost"),
        "numpy_version": np.__version__,
        "n_jobs": args.n_jobs,
        "cv_n_jobs": args.cv_n_jobs,
        "return_train_score": False,
        "hyperparameter_search": {
            "method": "Optuna TPE",
            "n_trials_per_outer_fold": args.n_trials,
            "inner_participant_folds": args.inner_folds,
            "inner_training_day_fraction": args.search_train_fraction,
            "seed": args.search_seed,
            "objective": "mean inner held-out participant combined accuracy",
        },
        "hyperparameters": {
            "model_type": "XGBoost",
            "version": "lightweight",
            "window_batch_size": args.window_batch_size,
            "overlap": pipeline.algo.overlap,
            "window_sec": pipeline.algo.window_sec,
        },
    }
    _write_results(
        evaluation=evaluation,
        dataset=dataset,
        training_only_index=training_only_index,
        fold_metadata=fold_metadata,
        output_dir=output_dir,
        run_metadata=run_metadata,
    )
    _write_search_results(evaluation, output_dir)

    LOGGER.info("Results written to %s", output_dir)


if __name__ == "__main__":
    main()
