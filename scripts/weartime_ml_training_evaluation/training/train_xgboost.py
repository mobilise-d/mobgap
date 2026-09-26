"""Train the Megaritis XGBoost wear-time model through the dataset-backed emulation pipeline.

The script trains directly from raw SUSTAIN CWA recordings. It uses
``WtdEmulationPipeline.self_optimize`` so raw data loading stays lazy: the pipeline reads sampling-rate and
``n_samples`` metadata first, then the XGBoost detector loads one dataset datapoint at a time while extracting window
features. By default, raw recordings are split into daily datapoints to enable datapoint-level parallel feature
extraction.

The final sklearn-style classifier still needs the full extracted feature matrix in memory for ``fit``. The detector
preallocates that matrix from the dataset sample counts to avoid retaining per-batch DataFrames.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from mobgap import PROJECT_ROOT
from mobgap.data import SustainWearTimeDataset
from mobgap.weartime import WtdMegaritisXGBoost
from mobgap.weartime.pipeline import WtdEmulationPipeline
from mobgap.weartime.utils.ml_feature_extraction import window_count_from_sample_count

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(".cache") / "weartime_training_runs"
DEFAULT_CACHE_DIR = Path(".cache") / "mobgap"


def _load_dotenv() -> None:
    try:
        load_dotenv = import_module("dotenv").load_dotenv
    except ImportError:
        return
    load_dotenv(Path(PROJECT_ROOT) / ".env")


def _path_from_env_or_arg(value: str | None, env_var: str, *, fallback: Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser()
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value).expanduser()
    if fallback is not None:
        return fallback
    raise ValueError(f"Pass the path explicitly or set `{env_var}`.")


def _dataset_index_for_metadata(dataset: SustainWearTimeDataset) -> list[dict[str, str]]:
    return [{column: str(value) for column, value in row.items()} for row in dataset.index.to_dict(orient="records")]


def _training_window_metadata(dataset: SustainWearTimeDataset, detector: WtdMegaritisXGBoost) -> dict[str, Any]:
    sampling_rates = [float(datapoint.sampling_rate_hz) for datapoint in dataset]
    sampling_rate_hz = sampling_rates[0]
    if not all(np.isclose(sampling_rate, sampling_rate_hz) for sampling_rate in sampling_rates):
        raise ValueError("All selected recordings must use the same sampling rate.")

    recording_sample_counts = tuple(int(datapoint.n_samples) for datapoint in dataset)
    window_samples, step_samples = detector._window_parameters(sampling_rate_hz)
    total_windows = sum(
        window_count_from_sample_count(sample_count, window_samples, step_samples)
        for sample_count in recording_sample_counts
    )
    return {
        "sampling_rate_hz": sampling_rate_hz,
        "recording_sample_counts": recording_sample_counts,
        "window_samples": window_samples,
        "step_samples": step_samples,
        "total_windows": total_windows,
    }


def _untrained_config(version: Literal["full", "lightweight"]) -> dict[str, Any]:
    if version == "full":
        return dict(WtdMegaritisXGBoost.PredefinedParameters.untrained_full)
    return dict(WtdMegaritisXGBoost.PredefinedParameters.untrained_lightweight)


def _classifier_params(clf: Any) -> dict[str, Any]:
    get_params = getattr(clf, "get_params", None)
    if get_params is None:
        return {}
    return {key: value for key, value in get_params().items() if isinstance(value, (str, int, float, bool, type(None)))}


def _xgboost_version() -> str | None:
    try:
        xgboost = import_module("xgboost")
    except ImportError:
        return None
    return xgboost.__version__


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        help="Path to the SUSTAIN Wear-time folder. Defaults to MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for trained model artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        help=f"joblib cache directory. Defaults to MOBGAP_CACHE_DIR_PATH or {DEFAULT_CACHE_DIR}.",
    )
    parser.add_argument("--run-name", help="Artifact stem. Defaults to xgboost_<version>_lowback_<timestamp>.")
    parser.add_argument("--version", default="lightweight", choices=["lightweight", "full"], help="Feature set to use.")
    parser.add_argument(
        "--recording-type",
        default="human_movement",
        choices=["human_movement", "simulated_movements"],
        help="SUSTAIN recording type to train on.",
    )
    parser.add_argument(
        "--split-by-day",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split raw CWA recordings into daily datapoints before training. Enabled by default.",
    )
    parser.add_argument(
        "--participant-id",
        action="append",
        help="Restrict training to one participant. Can be passed multiple times.",
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=8192,
        help="Number of windows feature-extracted before filling the training matrix.",
    )
    parser.add_argument("--overlap", type=float, default=0.75, help="Fractional overlap of XGBoost windows.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of process workers for datapoint-level feature extraction. Use -1 for all workers.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """Train and save the XGBoost wear-time model."""
    _load_dotenv()
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    dataset_path = _path_from_env_or_arg(args.dataset_path, "MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")
    output_dir = Path(args.output_dir).expanduser()
    cache_dir = _path_from_env_or_arg(args.cache_dir, "MOBGAP_CACHE_DIR_PATH", fallback=DEFAULT_CACHE_DIR)
    run_name = args.run_name or f"xgboost_{args.version}_lowback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dataset = SustainWearTimeDataset(
        dataset_path,
        additional_sensors_enabled=(),
        warn_thres_for_sampling_rate_deviations_hz=None,
        split_by_day=args.split_by_day,
        memory=joblib.Memory(cache_dir, verbose=0),
    ).get_subset(recording_type=args.recording_type)
    if args.participant_id:
        subset_index = dataset.index[dataset.index["participant_id"].isin(args.participant_id)]
        dataset = dataset.get_subset(index=subset_index)
    if len(dataset.index) == 0:
        raise ValueError("The selected SUSTAIN subset is empty.")

    detector = WtdMegaritisXGBoost(
        **_untrained_config(args.version),
        window_batch_size=args.window_batch_size,
        n_jobs=args.n_jobs,
        overlap=args.overlap,
        feature_memory=joblib.Memory(cache_dir / "xgboost_features", compress=3, verbose=0),
    )
    training_window_metadata = _training_window_metadata(dataset, detector)

    LOGGER.info("Training datapoints: %s", len(dataset.index))
    LOGGER.info("Raw recordings: %s", dataset.index["recording_id"].nunique())
    LOGGER.info("Participants: %s", dataset.index["participant_id"].nunique())
    LOGGER.info("Feature version: %s", args.version)
    LOGGER.info("Training windows: %s", f"{training_window_metadata['total_windows']:,}")
    LOGGER.info("Output run name: %s", run_name)

    pipeline = WtdEmulationPipeline(detector)

    total_start = time.time()
    pipeline.self_optimize(dataset)
    total_time = time.time() - total_start

    trained_detector = pipeline.algo
    if trained_detector.clf is None or trained_detector.feature_names is None:
        raise RuntimeError("Training finished without a trained XGBoost classifier.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{run_name}_model.pkl"
    feature_order_path = output_dir / f"{run_name}_feature_order.pkl"
    metadata_path = output_dir / f"{run_name}_metadata.json"

    with model_path.open("wb") as file:
        pickle.dump(trained_detector.clf, file)
    with feature_order_path.open("wb") as file:
        pickle.dump(list(trained_detector.feature_names), file)

    metadata = {
        "model_type": "XGBoost",
        "version": args.version,
        "training_date": datetime.now().isoformat(),
        "recording_type": args.recording_type,
        "split_by_day": bool(args.split_by_day),
        "n_datapoints": len(dataset.index),
        "n_recordings": int(dataset.index["recording_id"].nunique()),
        "n_participants": int(dataset.index["participant_id"].nunique()),
        "n_recording_samples": int(sum(training_window_metadata["recording_sample_counts"])),
        "n_windows": int(training_window_metadata["total_windows"]),
        "window_samples": int(training_window_metadata["window_samples"]),
        "step_samples": int(training_window_metadata["step_samples"]),
        "overlap": float(args.overlap),
        "window_batch_size": int(args.window_batch_size),
        "n_jobs": int(args.n_jobs),
        "sampling_rate_hz": float(training_window_metadata["sampling_rate_hz"]),
        "feature_names": list(trained_detector.feature_names),
        "hyperparameters": _classifier_params(trained_detector.clf),
        "training_time_seconds": int(total_time),
        "xgboost_version": _xgboost_version(),
        "numpy_version": np.__version__,
        "dataset_index": _dataset_index_for_metadata(dataset),
        "model_path": str(model_path),
        "feature_order_path": str(feature_order_path),
        "notes": "Trained from raw SUSTAIN CWA recordings via WtdEmulationPipeline. No feature scaling is applied.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    LOGGER.info("Training complete in %s", timedelta(seconds=int(total_time)))
    LOGGER.info("Model saved: %s", model_path)
    LOGGER.info("Feature order saved: %s", feature_order_path)
    LOGGER.info("Metadata saved: %s", metadata_path)


if __name__ == "__main__":
    main()
