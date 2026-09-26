"""Run daily aggregate participant-grouped evaluation for the SUSTAIN wear-time CNN.

The evaluation dataset is split by recording day, but the cross-validation groups by participant. This means every
fold holds out all days of a fixed number of human participants and trains on all days from all other human
participants.

The script intentionally uses the standard :class:`~mobgap.weartime.pipeline.WtdEmulationPipeline` and
:data:`~mobgap.weartime.evaluation.wtd_score` scorer so that metrics match the rest of the wear-time evaluation code.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from tpcp.optimize import Optimize
from tpcp.validate import DatasetSplitter

from mobgap.data import SustainWearTimeDataset
from mobgap.utils.evaluation import EvaluationCV
from mobgap.utils.misc import get_env_var
from mobgap.weartime import MegaritisCnnWeartimeModel, WtdMegaritisCNN
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.pipeline import WtdEmulationPipeline

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(".cache") / "weartime_loso_runs"
DEFAULT_CACHE_DIR = Path(".cache") / "mobgap"

if TYPE_CHECKING:
    from collections.abc import Iterator


class FixedSizeTestGroupKFold(BaseCrossValidator):
    """Group CV splitter with a fixed number of complete groups in each test fold."""

    def __init__(self, *, test_groups_per_fold: int) -> None:
        self.test_groups_per_fold = test_groups_per_fold

    def split(self, x: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[list[int], list[int]]]:
        """Yield train/test indices with complete groups assigned to each test fold."""
        del y
        if groups is None:
            raise ValueError("FixedSizeTestGroupKFold requires group labels.")

        group_labels = np.asarray(groups)
        all_indices = np.arange(len(group_labels))
        unique_groups = pd.unique(group_labels)
        n_splits = self.get_n_splits(x, groups=group_labels)
        for fold_index in range(n_splits):
            start = fold_index * self.test_groups_per_fold
            test_groups = unique_groups[start : start + self.test_groups_per_fold]
            test_mask = np.isin(group_labels, test_groups)
            yield all_indices[~test_mask].tolist(), all_indices[test_mask].tolist()

    def get_n_splits(self, x: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the dynamically derived number of folds."""
        del x, y
        if groups is None:
            raise ValueError("FixedSizeTestGroupKFold requires group labels.")

        n_test_groups = int(self.test_groups_per_fold)
        if n_test_groups <= 0:
            raise ValueError("`test_groups_per_fold` must be positive.")

        n_groups = len(pd.unique(groups))
        if n_groups < n_test_groups:
            raise ValueError(
                f"Cannot create folds with {n_test_groups} test groups from only {n_groups} available groups."
            )
        if n_groups % n_test_groups != 0:
            raise ValueError(
                f"The number of groups ({n_groups}) must be divisible by test_groups_per_fold "
                f"({n_test_groups}) so every test fold contains exactly {n_test_groups} groups."
            )
        return n_groups // n_test_groups


def _path_from_env_or_arg(value: str | None, env_var: str, *, fallback: Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser()
    env_value = get_env_var(env_var, default=None) if fallback is not None else get_env_var(env_var)
    if env_value is not None:
        return Path(env_value).expanduser()
    return fallback


def _configure_tensorflow() -> dict[str, Any]:
    tf = import_module("tensorflow")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return {
        "tensorflow_version": tf.__version__,
        "gpu_devices": [str(gpu) for gpu in gpus],
    }


def _resolve_n_jobs(requested_n_jobs: int, tf_metadata: dict[str, Any]) -> int:
    if tf_metadata["gpu_devices"] and requested_n_jobs != 1:
        LOGGER.warning(
            "GPU devices are available, so LOSO folds must run sequentially. Overriding --n-jobs=%s to 1.",
            requested_n_jobs,
        )
        return 1
    return requested_n_jobs


def _make_dataset(args: argparse.Namespace) -> SustainWearTimeDataset:
    dataset_path = _path_from_env_or_arg(args.dataset_path, "MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")
    cache_dir = _path_from_env_or_arg(args.cache_dir, "MOBGAP_CACHE_DIR_PATH", fallback=DEFAULT_CACHE_DIR)
    dataset = SustainWearTimeDataset(
        dataset_path,
        additional_sensors_enabled=(),
        warn_thres_for_sampling_rate_deviations_hz=None,
        split_by_day=True,
        memory=joblib.Memory(cache_dir, verbose=0),
    ).get_subset(recording_type="human_movement")

    if args.participant_id:
        subset_index = dataset.index[dataset.index["participant_id"].isin(args.participant_id)]
        dataset = dataset.get_subset(index=subset_index)
    if args.max_participants is not None:
        selected_participants = dataset.index["participant_id"].drop_duplicates().iloc[: args.max_participants]
        subset_index = dataset.index[dataset.index["participant_id"].isin(selected_participants)]
        dataset = dataset.get_subset(index=subset_index)
    if len(dataset.index) == 0:
        raise ValueError("The selected SUSTAIN human split-by-day subset is empty.")
    return dataset


def _make_participant_group_splitter(test_participants_per_fold: int) -> DatasetSplitter:
    return DatasetSplitter(
        base_splitter=FixedSizeTestGroupKFold(test_groups_per_fold=test_participants_per_fold),
        groupby="participant_id",
    )


def _fold_metadata(
    dataset: SustainWearTimeDataset, splitter: DatasetSplitter, *, test_participants_per_fold: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset)):
        train_index = dataset.index.iloc[train_idx]
        test_index = dataset.index.iloc[test_idx]
        test_participants = sorted(test_index["participant_id"].unique())
        if len(test_participants) != test_participants_per_fold:
            raise RuntimeError(
                f"Fold {fold} expected {test_participants_per_fold} held-out participants, got {test_participants}."
            )
        train_participants = sorted(train_index["participant_id"].unique())
        rows.append(
            {
                "fold": fold,
                "held_out_participant_ids": ",".join(str(participant_id) for participant_id in test_participants),
                "n_train_days": len(train_index),
                "n_test_days": len(test_index),
                "n_train_participants": len(train_participants),
                "n_test_participants": len(test_participants),
                "train_participant_ids": ",".join(str(participant_id) for participant_id in train_participants),
                "test_recording_ids": ",".join(sorted(test_index["recording_id"].unique())),
                "test_recording_days": ",".join(str(day) for day in sorted(test_index["recording_day"].unique())),
            }
        )
    return pd.DataFrame(rows)


def _make_pipeline(args: argparse.Namespace) -> WtdEmulationPipeline:
    window_batch_size = args.window_batch_size or args.batch_size
    low_level_model = MegaritisCnnWeartimeModel(
        batch_size=args.batch_size,
        epochs=args.epochs,
        fit_verbose=args.fit_verbose,
        window_batch_size=window_batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        standardize_in_model=args.standardize_in_model,
        overlap=args.overlap,
    )
    return WtdEmulationPipeline(WtdMegaritisCNN(model=low_level_model))


def _numeric_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric = frame.select_dtypes(include=[np.number])
    return {
        column: {
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std()),
            "min": float(numeric[column].min()),
            "max": float(numeric[column].max()),
        }
        for column in numeric.columns
    }


def _write_results(
    *,
    evaluation: EvaluationCV,
    dataset: SustainWearTimeDataset,
    fold_metadata: pd.DataFrame,
    output_dir: Path,
    run_metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = evaluation.get_aggregated_results_as_df(group="test")
    daily_results = evaluation.get_single_results_as_df(group="test")
    raw_results = evaluation.get_raw_results(group="test")
    train_fold_results = evaluation.get_aggregated_results_as_df(group="train")
    train_daily_results = evaluation.get_single_results_as_df(group="train")
    train_raw_results = evaluation.get_raw_results(group="train")

    fold_results = fold_results.join(fold_metadata.set_index("fold"), how="left")
    train_fold_results = train_fold_results.join(fold_metadata.set_index("fold"), how="left")
    fold_results.to_csv(output_dir / "fold_results.csv")
    daily_results.to_csv(output_dir / "daily_results.csv")
    train_fold_results.to_csv(output_dir / "train_fold_results.csv")
    train_daily_results.to_csv(output_dir / "train_daily_results.csv")
    fold_metadata.to_csv(output_dir / "fold_metadata.csv", index=False)

    for name, value in raw_results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"raw_{name}.csv")
    for name, value in train_raw_results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"raw_train_{name}.csv")

    dataset.index.to_csv(output_dir / "dataset_index.csv", index=False)

    summary = {
        **run_metadata,
        "n_days": len(dataset.index),
        "n_participants": int(dataset.index["participant_id"].nunique()),
        "n_folds": len(fold_metadata),
        "fold_metric_summary": _numeric_summary(fold_results),
        "daily_metric_summary": _numeric_summary(daily_results.reset_index(drop=True)),
        "train_fold_metric_summary": _numeric_summary(train_fold_results),
        "train_daily_metric_summary": _numeric_summary(train_daily_results.reset_index(drop=True)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


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
    parser.add_argument("--run-name", help="Artifact folder name. Defaults to loso_daily_cnn_<timestamp>.")
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
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs per LOSO fold.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Keras training batch size.")
    parser.add_argument(
        "--window-batch-size",
        type=int,
        help="Number of windows materialized from each recording at once. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=8192,
        help="TensorFlow shuffle buffer size in windows. Set 0 to disable shuffling.",
    )
    parser.add_argument("--overlap", type=float, default=0.75, help="Window overlap used by the Keras model.")
    parser.add_argument(
        "--standardize-in-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize windows in the Keras model instead of in the Python window generator.",
    )
    parser.add_argument("--fit-verbose", type=int, default=1, help="Keras fit verbosity.")
    parser.add_argument(
        "--test-participants-per-fold",
        type=int,
        default=1,
        help="Number of complete participants held out in each test fold. The number of selected participants must "
        "be divisible by this value.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of CV folds to evaluate in parallel.")
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
    """Run the LOSO evaluation."""
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    run_name = args.run_name or f"loso_daily_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir).expanduser() / run_name
    dataset = _make_dataset(args)
    splitter = _make_participant_group_splitter(args.test_participants_per_fold)
    fold_metadata = _fold_metadata(
        dataset,
        splitter,
        test_participants_per_fold=args.test_participants_per_fold,
    )

    LOGGER.info("Selected split-by-day human datapoints: %s", len(dataset.index))
    LOGGER.info("Selected participants: %s", dataset.index["participant_id"].nunique())
    LOGGER.info("Test participants per fold: %s", args.test_participants_per_fold)
    LOGGER.info("Participant-grouped CV folds: %s", len(fold_metadata))
    LOGGER.info("Output directory: %s", output_dir)

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.index.to_csv(output_dir / "dataset_index.csv", index=False)
        fold_metadata.to_csv(output_dir / "fold_metadata.csv", index=False)
        LOGGER.info("Dry run complete. Wrote fold metadata and dataset index.")
        return

    tf_metadata = _configure_tensorflow()
    n_jobs = _resolve_n_jobs(args.n_jobs, tf_metadata)
    LOGGER.info("TensorFlow %s", tf_metadata["tensorflow_version"])
    LOGGER.info("GPU devices: %s", tf_metadata["gpu_devices"] or "none")

    pipeline = _make_pipeline(args)
    evaluation = EvaluationCV(
        dataset=dataset,
        scoring=wtd_score,
        cv_iterator=splitter,
        cv_params={
            "n_jobs": n_jobs,
            "return_train_score": True,
            "progress_bar": True,
        },
    )
    optimizer = Optimize(pipeline)

    total_start = time.time()
    evaluation.run(optimizer)
    total_time = time.time() - total_start

    window_model = pipeline.algo.model
    run_metadata = {
        "run_name": run_name,
        "training_date": datetime.now().isoformat(),
        "total_runtime_seconds": int(total_time),
        "tensorflow_version": tf_metadata["tensorflow_version"],
        "gpu_devices": tf_metadata["gpu_devices"],
        "n_jobs": n_jobs,
        "test_participants_per_fold": args.test_participants_per_fold,
        "return_train_score": True,
        "hyperparameters": {
            "model_type": "CNN_1D",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "window_batch_size": args.window_batch_size or args.batch_size,
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "standardize_in_model": args.standardize_in_model,
            "overlap": args.overlap,
            "window_sec": window_model.window_sec,
            "filters": list(window_model.filters),
            "kernel_size": window_model.kernel_size,
            "pool_size": window_model.pool_size,
            "dropout_rate": window_model.dropout_rate,
            "dense_units": window_model.dense_units,
            "learning_rate": window_model.learning_rate,
        },
    }
    _write_results(
        evaluation=evaluation,
        dataset=dataset,
        fold_metadata=fold_metadata,
        output_dir=output_dir,
        run_metadata=run_metadata,
    )

    LOGGER.info("LOSO evaluation complete in %s", timedelta(seconds=int(total_time)))
    LOGGER.info("Results written to %s", output_dir)


if __name__ == "__main__":
    main()
