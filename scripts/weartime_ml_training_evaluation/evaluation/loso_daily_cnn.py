"""Run daily aggregate participant-grouped evaluation for the SUSTAIN wear-time CNN.

The evaluation dataset is split by recording day, but the cross-validation groups by participant. This means every
fold holds out all days of one human participant and trains on all days from all other human
participants, plus the same two part B recordings assigned to training by a combined splitter.

The script intentionally uses the standard :class:`~mobgap.weartime.pipeline.WtdEmulationPipeline` and
:data:`~mobgap.weartime.evaluation.wtd_score` scorer so that metrics match the rest of the wear-time evaluation code.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tpcp.optimize import Optimize
from tpcp.validate import CombinedSplitter, DatasetSplitter, NoSplit

from mobgap.data import SustainWearTimeDataset
from mobgap.utils.evaluation import EvaluationCV
from mobgap.utils.misc import get_env_var
from mobgap.weartime import MegaritisCnnWeartimeModel, WtdMegaritisCNN
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.pipeline import WtdEmulationPipeline

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(".cache") / "weartime_loso_runs"
DEFAULT_CACHE_DIR = Path(".cache") / "mobgap"


def _fold_metadata(
    dataset: SustainWearTimeDataset, splitter: CombinedSplitter, training_only_index: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, (train_labels, test_labels) in enumerate(splitter.split(dataset)):
        train_index = dataset.get_subset(group_labels=train_labels).index
        test_index = dataset.get_subset(group_labels=test_labels).index
        test_participants = sorted(test_index["participant_id"].unique())
        train_participants = sorted(train_index["participant_id"].unique())
        rows.append(
            {
                "fold": fold,
                "held_out_participant_ids": ",".join(str(participant_id) for participant_id in test_participants),
                "n_train_days": len(train_index),
                "n_training_only_days": len(training_only_index),
                "n_test_days": len(test_index),
                "n_train_participants": len(train_participants),
                "n_test_participants": len(test_participants),
                "train_participant_ids": ",".join(str(participant_id) for participant_id in train_participants),
                "training_only_recording_ids": ",".join(sorted(training_only_index["recording_id"].unique())),
                "test_recording_ids": ",".join(sorted(test_index["recording_id"].unique())),
                "test_recording_days": ",".join(str(day) for day in sorted(test_index["recording_day"].unique())),
            }
        )
    return pd.DataFrame(rows)


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
    training_only_index: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    output_dir: Path,
    run_metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = evaluation.get_aggregated_results_as_df(group="test")
    daily_results = evaluation.get_single_results_as_df(group="test")
    raw_results = evaluation.get_raw_results(group="test")
    fold_results = fold_results.join(fold_metadata.set_index("fold"), how="left")
    fold_results.to_csv(output_dir / "fold_results.csv")
    daily_results.to_csv(output_dir / "daily_results.csv")
    fold_metadata.to_csv(output_dir / "fold_metadata.csv", index=False)

    for name, value in raw_results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"raw_{name}.csv")
    dataset.index.to_csv(output_dir / "dataset_index.csv", index=False)
    training_only_index.to_csv(output_dir / "training_only_index.csv", index=False)

    summary = {
        **run_metadata,
        "n_days": len(dataset.index),
        "n_participants": int(dataset.index["participant_id"].nunique()),
        "n_training_only_days": len(training_only_index),
        "training_only_recording_ids": sorted(training_only_index["recording_id"].unique()),
        "n_folds": len(fold_metadata),
        "fold_metric_summary": _numeric_summary(fold_results),
        "daily_metric_summary": _numeric_summary(daily_results.reset_index(drop=True)),
    }
    if run_metadata["return_train_score"]:
        train_fold_results = evaluation.get_aggregated_results_as_df(group="train")
        train_daily_results = evaluation.get_single_results_as_df(group="train")
        train_fold_results = train_fold_results.join(fold_metadata.set_index("fold"), how="left")
        train_fold_results.to_csv(output_dir / "train_fold_results.csv")
        train_daily_results.to_csv(output_dir / "train_daily_results.csv")
        for name, value in evaluation.get_raw_results(group="train").items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(output_dir / f"raw_train_{name}.csv")
        summary["train_fold_metric_summary"] = _numeric_summary(train_fold_results)
        summary["train_daily_metric_summary"] = _numeric_summary(train_daily_results.reset_index(drop=True))
    else:
        for filename in ("train_fold_results.csv", "train_daily_results.csv"):
            (output_dir / filename).unlink(missing_ok=True)
        for path in output_dir.glob("raw_train_*.csv"):
            path.unlink()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "timings.json").write_text(json.dumps(evaluation.perf_, indent=2) + "\n")


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


def main() -> None:  # noqa: PLR0915 - Keep the LOSO composition visible in one place.
    """Run the LOSO evaluation."""
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    run_name = args.run_name or f"loso_daily_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir).expanduser() / run_name
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

    part_b_index = base_dataset.get_subset(recording_type="simulated_movements").index
    available_ids = sorted(part_b_index["recording_id"].unique())
    selected_ids = available_ids[:2] if args.part_b_recording_id is None else args.part_b_recording_id
    if len(selected_ids) != 2 or len(set(selected_ids)) != 2 or not set(selected_ids).issubset(available_ids):
        raise ValueError("Select exactly two distinct part B recording IDs present in the dataset.")
    training_only_index = part_b_index[part_b_index["recording_id"].isin(selected_ids)].reset_index(drop=True)
    evaluation_index = pd.concat([dataset.index, training_only_index], ignore_index=True)
    evaluation_dataset = base_dataset.get_subset(index=evaluation_index)
    splitter = CombinedSplitter(
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
    fold_metadata = _fold_metadata(evaluation_dataset, splitter, training_only_index)

    LOGGER.info("Selected split-by-day human datapoints: %s", len(dataset.index))
    LOGGER.info("Selected participants: %s", dataset.index["participant_id"].nunique())
    LOGGER.info("Participant-grouped CV folds: %s", len(fold_metadata))
    LOGGER.info(
        "Part B recordings added to every training fold: %s", sorted(training_only_index["recording_id"].unique())
    )
    LOGGER.info("Output directory: %s", output_dir)

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.index.to_csv(output_dir / "dataset_index.csv", index=False)
        training_only_index.to_csv(output_dir / "training_only_index.csv", index=False)
        fold_metadata.to_csv(output_dir / "fold_metadata.csv", index=False)
        LOGGER.info("Dry run complete. Wrote fold metadata and human and part B dataset indices.")
        return

    tf = import_module("tensorflow")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf_metadata = {"tensorflow_version": tf.__version__, "gpu_devices": [str(gpu) for gpu in gpus]}
    n_jobs = args.n_jobs
    if gpus and n_jobs != 1:
        LOGGER.warning(
            "GPU devices are available, so LOSO folds must run sequentially. Overriding --n-jobs=%s to 1.", n_jobs
        )
        n_jobs = 1
    LOGGER.info("TensorFlow %s", tf_metadata["tensorflow_version"])
    LOGGER.info("GPU devices: %s", tf_metadata["gpu_devices"] or "none")

    pipeline = WtdEmulationPipeline(
        WtdMegaritisCNN(
            model=MegaritisCnnWeartimeModel(
                batch_size=args.batch_size,
                epochs=args.epochs,
                fit_verbose=args.fit_verbose,
                window_batch_size=args.window_batch_size or args.batch_size,
                shuffle_buffer_size=args.shuffle_buffer_size,
                standardize_in_model=args.standardize_in_model,
                overlap=args.overlap,
            )
        )
    )
    evaluation = EvaluationCV(
        dataset=evaluation_dataset,
        scoring=wtd_score,
        cv_iterator=splitter,
        cv_params={
            "n_jobs": n_jobs,
            "return_train_score": True,
            "progress_bar": True,
        },
    )
    optimizer = Optimize(pipeline)

    evaluation.run(optimizer)

    window_model = pipeline.algo.model
    run_metadata = {
        "run_name": run_name,
        "tensorflow_version": tf_metadata["tensorflow_version"],
        "gpu_devices": tf_metadata["gpu_devices"],
        "n_jobs": n_jobs,
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
        training_only_index=training_only_index,
        fold_metadata=fold_metadata,
        output_dir=output_dir,
        run_metadata=run_metadata,
    )

    LOGGER.info("Results written to %s", output_dir)


if __name__ == "__main__":
    main()
