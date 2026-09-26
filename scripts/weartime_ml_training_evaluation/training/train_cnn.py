"""Train the Megaritis CNN wear-time model through the dataset-backed emulation pipeline.

The script trains directly from raw SUSTAIN CWA recordings. It uses
``WtdEmulationPipeline.self_optimize`` so data loading stays lazy: the pipeline reads sampling-rate and ``n_samples``
metadata first, then the low-level Keras model loads one recording at a time while TensorFlow consumes the windows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from mobgap import PROJECT_ROOT
from mobgap.data import SustainWearTimeDataset
from mobgap.weartime import MegaritisCnnWeartimeModel, WtdMegaritisCNN
from mobgap.weartime._keras_weartime_model import _steps_per_epoch_from_recording_sample_counts
from mobgap.weartime.pipeline import WtdEmulationPipeline

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


def _configure_tensorflow() -> dict[str, Any]:
    tf = import_module("tensorflow")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return {
        "tensorflow_version": tf.__version__,
        "gpu_devices": [str(gpu) for gpu in gpus],
    }


def _dataset_index_for_metadata(dataset: SustainWearTimeDataset) -> list[dict[str, str]]:
    return [{column: str(value) for column, value in row.items()} for row in dataset.index.to_dict(orient="records")]


def _history_as_jsonable(history: Any) -> dict[str, list[float]]:
    if history is None:
        return {}
    return {key: [float(value) for value in values] for key, values in getattr(history, "history", {}).items()}


def _final_metric(history: dict[str, list[float]], key: str) -> float | None:
    values = history.get(key)
    if not values:
        return None
    return float(values[-1])


def _training_window_metadata(
    dataset: SustainWearTimeDataset,
    model: MegaritisCnnWeartimeModel,
    *,
    batch_size: int,
) -> dict[str, Any]:
    sampling_rates = [float(datapoint.sampling_rate_hz) for datapoint in dataset]
    sampling_rate_hz = sampling_rates[0]
    if not all(np.isclose(sampling_rate, sampling_rate_hz) for sampling_rate in sampling_rates):
        raise ValueError("All selected recordings must use the same sampling rate.")

    recording_sample_counts = tuple(int(datapoint.n_samples) for datapoint in dataset)
    window_samples, step_samples = model._window_parameters(sampling_rate_hz)
    total_windows, steps_per_epoch = _steps_per_epoch_from_recording_sample_counts(
        recording_sample_counts,
        window_samples=window_samples,
        step_samples=step_samples,
        batch_size=batch_size,
    )
    return {
        "sampling_rate_hz": sampling_rate_hz,
        "recording_sample_counts": recording_sample_counts,
        "window_samples": window_samples,
        "step_samples": step_samples,
        "total_windows": total_windows,
        "steps_per_epoch": steps_per_epoch,
    }


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
    parser.add_argument("--run-name", help="Artifact stem. Defaults to cnn_lowback_pipeline_<timestamp>.")
    parser.add_argument(
        "--recording-type",
        default="human_movement",
        choices=["human_movement", "simulated_movements"],
        help="SUSTAIN recording type to train on.",
    )
    parser.add_argument(
        "--participant-id",
        action="append",
        help="Restrict training to one participant. Can be passed multiple times.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Keras training batch size.")
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
    parser.add_argument(
        "--standardize-in-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize windows in the Keras model instead of in the Python window generator.",
    )
    parser.add_argument("--fit-verbose", type=int, default=1, help="Keras fit verbosity.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """Train and save the CNN wear-time model."""
    _load_dotenv()
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    dataset_path = _path_from_env_or_arg(args.dataset_path, "MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")
    output_dir = Path(args.output_dir).expanduser()
    cache_dir = _path_from_env_or_arg(args.cache_dir, "MOBGAP_CACHE_DIR_PATH", fallback=DEFAULT_CACHE_DIR)
    run_name = args.run_name or f"cnn_lowback_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    window_batch_size = args.window_batch_size or args.batch_size

    tf_metadata = _configure_tensorflow()
    LOGGER.info("TensorFlow %s", tf_metadata["tensorflow_version"])
    LOGGER.info("GPU devices: %s", tf_metadata["gpu_devices"] or "none")

    dataset = SustainWearTimeDataset(
        dataset_path,
        additional_sensors_enabled=(),
        warn_thres_for_sampling_rate_deviations_hz=None,
        memory=joblib.Memory(cache_dir, verbose=0),
    ).get_subset(recording_type=args.recording_type)
    if args.participant_id:
        subset_index = dataset.index[dataset.index["participant_id"].isin(args.participant_id)]
        dataset = dataset.get_subset(index=subset_index)
    if len(dataset.index) == 0:
        raise ValueError("The selected SUSTAIN subset is empty.")

    LOGGER.info("Training recordings: %s", len(dataset.index))
    LOGGER.info("Participants: %s", dataset.index["participant_id"].nunique())
    LOGGER.info("Output run name: %s", run_name)

    low_level_model = MegaritisCnnWeartimeModel(
        batch_size=args.batch_size,
        epochs=args.epochs,
        fit_verbose=args.fit_verbose,
        window_batch_size=window_batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        standardize_in_model=args.standardize_in_model,
    )
    training_window_metadata = _training_window_metadata(
        dataset,
        low_level_model,
        batch_size=args.batch_size,
    )
    LOGGER.info("Training windows: %s", f"{training_window_metadata['total_windows']:,}")
    LOGGER.info("Steps per epoch: %s", training_window_metadata["steps_per_epoch"])

    pipeline = WtdEmulationPipeline(WtdMegaritisCNN(model=low_level_model))

    total_start = time.time()
    pipeline.self_optimize(dataset)
    total_time = time.time() - total_start

    trained_model = pipeline.algo.model
    if trained_model is None or trained_model._model is None:
        raise RuntimeError("Training finished without a trained Keras model.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{run_name}.keras"
    metadata_path = output_dir / f"{run_name}.json"
    history_path = output_dir / f"{run_name}_history.json"

    trained_model._model.save(model_path)
    history = _history_as_jsonable(getattr(trained_model._model, "history", None))
    history_path.write_text(json.dumps(history, indent=2) + "\n")

    metadata = {
        "model_type": "CNN_1D",
        "version": "pipeline_training",
        "training_date": datetime.now().isoformat(),
        "recording_type": args.recording_type,
        "n_recordings": len(dataset.index),
        "n_participants": int(dataset.index["participant_id"].nunique()),
        "n_recording_samples": int(sum(training_window_metadata["recording_sample_counts"])),
        "n_windows": int(training_window_metadata["total_windows"]),
        "steps_per_epoch": int(training_window_metadata["steps_per_epoch"]),
        "input_shape": [int(training_window_metadata["window_samples"]), len(trained_model.sensor_cols)],
        "hyperparameters": {
            "num_conv_layers": 3,
            "filters": list(trained_model.filters),
            "kernel_size": trained_model.kernel_size,
            "pool_size": trained_model.pool_size,
            "dropout_rate": trained_model.dropout_rate,
            "dense_units": trained_model.dense_units,
            "learning_rate": trained_model.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "window_sec": trained_model.window_sec,
            "overlap": trained_model.overlap,
            "window_batch_size": window_batch_size,
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "standardize_in_model": args.standardize_in_model,
        },
        "final_train_accuracy": _final_metric(history, "accuracy"),
        "final_train_loss": _final_metric(history, "loss"),
        "training_time_seconds": int(total_time),
        "tensorflow_version": tf_metadata["tensorflow_version"],
        "numpy_version": np.__version__,
        "gpu_devices": tf_metadata["gpu_devices"],
        "dataset_index": _dataset_index_for_metadata(dataset),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "notes": "Trained from raw SUSTAIN CWA recordings via WtdEmulationPipeline with per-window standardization.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    LOGGER.info("Training complete in %s", timedelta(seconds=int(total_time)))
    LOGGER.info("Final accuracy: %s", metadata["final_train_accuracy"])
    LOGGER.info("Final loss: %s", metadata["final_train_loss"])
    LOGGER.info("Model saved: %s", model_path)
    LOGGER.info("Metadata saved: %s", metadata_path)
    LOGGER.info("History saved: %s", history_path)


if __name__ == "__main__":
    main()
