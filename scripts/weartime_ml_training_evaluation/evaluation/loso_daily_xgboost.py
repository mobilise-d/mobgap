"""Run daily aggregate participant-grouped evaluation for the SUSTAIN wear-time XGBoost model.

This mirrors ``loso_daily_cnn.py``: the dataset is split by recording day, while CV groups by participant so every fold
holds out all days of one human participant. Two part B recordings are added to each training fold.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
from loso_daily_cnn import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    _combined_splitter,
    _fold_metadata,
    _make_base_dataset,
    _make_dataset,
    _path_from_env_or_arg,
    _select_training_only_index,
    _write_results,
)
from optuna import Study, Trial
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from tpcp.optimize import Optimize
from tpcp.optimize.optuna import CustomOptunaOptimize
from tpcp.validate import DatasetSplitter, cross_validate

from mobgap.utils.evaluation import EvaluationCV
from mobgap.weartime import WtdMegaritisXGBoost
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.pipeline import WtdEmulationPipeline

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.trial import FrozenTrial

    from mobgap.data import SustainWearTimeDataset


def _study_params(seed: int) -> dict[str, Any]:
    return {"direction": "maximize", "sampler": optuna.samplers.TPESampler(seed=seed)}


def _sample_inner_training_days(
    train_dataset: SustainWearTimeDataset, *, fraction: float, seed: int
) -> SustainWearTimeDataset:
    human_index = train_dataset.get_subset(recording_type="human_movement").index
    n_days = max(1, round(len(human_index) * fraction))
    sampled_human_index = human_index.sample(n=n_days, random_state=seed)
    training_only_index = train_dataset.get_subset(recording_type="simulated_movements").index
    return train_dataset.get_subset(index=pd.concat([sampled_human_index, training_only_index], ignore_index=True))


class XGBoostOptunaOptimize(CustomOptunaOptimize):
    """Tune within one outer LOSO training fold, then refit on all its training days."""

    def __init__(
        self,
        pipeline: WtdEmulationPipeline,
        *,
        n_trials: int = 20,
        inner_folds: int = 5,
        train_fraction: float = 0.4,
        random_seed: int = 42,
        get_study_params: Callable[[int], dict[str, Any]] = _study_params,
        timeout: float | None = None,
        callbacks: list[Callable[[Study, FrozenTrial], None]] | None = None,
        gc_after_trial: bool = False,
        n_jobs: int = 1,
        eval_str_paras: Sequence[str] = (),
        show_progress_bar: bool = False,
        return_optimized: bool = True,
    ) -> None:
        self.inner_folds = inner_folds
        self.train_fraction = train_fraction
        super().__init__(
            pipeline,
            get_study_params,
            n_trials=n_trials,
            random_seed=random_seed,
            timeout=timeout,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            n_jobs=n_jobs,
            eval_str_paras=eval_str_paras,
            show_progress_bar=show_progress_bar,
            return_optimized=return_optimized,
        )

    def create_objective(self) -> Callable[[Trial, WtdEmulationPipeline, SustainWearTimeDataset], float]:
        """Score each candidate on participant-grouped inner validation folds."""

        def objective(trial: Trial, pipeline: WtdEmulationPipeline, dataset: SustainWearTimeDataset) -> float:
            params = {
                "algo__clf__n_estimators": trial.suggest_categorical("algo__clf__n_estimators", [50, 100, 200]),
                "algo__clf__max_depth": trial.suggest_categorical("algo__clf__max_depth", [3, 5]),
                "algo__clf__learning_rate": trial.suggest_categorical("algo__clf__learning_rate", [0.05, 0.1]),
                "algo__clf__subsample": trial.suggest_categorical("algo__clf__subsample", [0.7, 0.8, 0.9]),
                "algo__clf__colsample_bytree": trial.suggest_categorical("algo__clf__colsample_bytree", [0.6, 0.8]),
                "algo__clf__min_child_weight": trial.suggest_categorical("algo__clf__min_child_weight", [1, 3]),
            }
            pipeline.set_params(**params)
            inner_cv = _combined_splitter(
                DatasetSplitter(GroupKFold(n_splits=self.inner_folds), groupby="participant_id"),
                self.inner_folds,
            )
            inner_optimizer = Optimize(
                pipeline,
                train_dataset_transform=partial(
                    _sample_inner_training_days,
                    fraction=self.train_fraction,
                    seed=self.random_seed,
                ),
            )
            scores = cross_validate(
                inner_optimizer,
                dataset,
                scoring=wtd_score,
                cv=inner_cv,
                n_jobs=1,
                return_train_score=False,
                progress_bar=False,
            )
            return float(np.mean(scores["test__agg__combined__accuracy"]))

        return objective

    def return_optimized_pipeline(
        self, pipeline: WtdEmulationPipeline, dataset: SustainWearTimeDataset, study: Study
    ) -> WtdEmulationPipeline:
        """Refit the winning parameters with every outer training day."""
        pipeline.set_params(**study.best_params)
        return Optimize(pipeline).optimize(dataset).optimized_pipeline_


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


def _xgboost_version() -> str | None:
    try:
        xgboost = import_module("xgboost")
    except ImportError:
        return None
    return xgboost.__version__


def _classifier_params(clf: Any) -> dict[str, Any]:
    get_params = getattr(clf, "get_params", None)
    if get_params is None:
        return {}
    params = {}
    for key, value in get_params().items():
        if not isinstance(value, (str, int, float, bool, type(None))):
            continue
        params[key] = None if isinstance(value, float) and not np.isfinite(value) else value
    return params


def _make_pipeline(args: argparse.Namespace, cache_dir: Path) -> WtdEmulationPipeline:
    return WtdEmulationPipeline(
        WtdMegaritisXGBoost(
            **WtdMegaritisXGBoost.PredefinedParameters.untrained_lightweight,
            window_batch_size=args.window_batch_size,
            n_jobs=args.n_jobs,
            overlap=args.overlap,
            feature_memory=joblib.Memory(cache_dir / "xgboost_features", compress=3, verbose=0),
        )
    )


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
        help="Fraction of human inner training days sampled for each trial; all selected part B days stay included.",
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


def main() -> None:
    """Run the XGBoost LOSO evaluation."""
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    run_name = args.run_name or f"loso_daily_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir).expanduser() / run_name
    base_dataset = _make_base_dataset(args)
    dataset = _make_dataset(args, base_dataset)
    training_only_index = _select_training_only_index(base_dataset, args.part_b_recording_id)
    evaluation_index = pd.concat([dataset.index, training_only_index], ignore_index=True)
    evaluation_dataset = base_dataset.get_subset(index=evaluation_index)
    splitter = _combined_splitter(
        DatasetSplitter(base_splitter=LeaveOneGroupOut(), groupby="participant_id"),
        dataset.index["participant_id"].nunique(),
    )
    fold_metadata = _fold_metadata(evaluation_dataset, splitter, training_only_index)

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

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.index.to_csv(output_dir / "dataset_index.csv", index=False)
        training_only_index.to_csv(output_dir / "training_only_index.csv", index=False)
        fold_metadata.to_csv(output_dir / "fold_metadata.csv", index=False)
        LOGGER.info("Dry run complete. Wrote fold metadata and human and part B dataset indices.")
        return

    if not 0 < args.search_train_fraction <= 1:
        raise ValueError("--search-train-fraction must be in (0, 1].")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive.")
    if args.inner_folds < 2 or args.inner_folds > dataset.index["participant_id"].nunique() - 1:
        raise ValueError("--inner-folds must be between 2 and the number of outer training participants.")

    cache_dir = _path_from_env_or_arg(args.cache_dir, "MOBGAP_CACHE_DIR_PATH", fallback=DEFAULT_CACHE_DIR)
    pipeline = _make_pipeline(args, cache_dir)
    evaluation = EvaluationCV(
        dataset=evaluation_dataset,
        scoring=wtd_score,
        cv_iterator=splitter,
        cv_params={
            "n_jobs": args.cv_n_jobs,
            "return_train_score": False,
            "return_optimizer": True,
            "progress_bar": True,
        },
    )
    optimizer = XGBoostOptunaOptimize(
        pipeline,
        n_trials=args.n_trials,
        inner_folds=args.inner_folds,
        train_fraction=args.search_train_fraction,
        random_seed=args.search_seed,
    )

    total_start = time.time()
    evaluation.run(optimizer)
    total_time = time.time() - total_start

    run_metadata = {
        "run_name": run_name,
        "training_date": datetime.now().isoformat(),
        "total_runtime_seconds": int(total_time),
        "xgboost_version": _xgboost_version(),
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
            "classifier": _classifier_params(pipeline.algo.clf),
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

    LOGGER.info("LOSO XGBoost evaluation complete in %s", timedelta(seconds=int(total_time)))
    LOGGER.info("Results written to %s", output_dir)


if __name__ == "__main__":
    main()
