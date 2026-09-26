"""Run daily aggregate participant-grouped evaluation for the SUSTAIN wear-time XGBoost model.

This mirrors ``loso_daily_cnn.py``: the dataset is split by recording day, while CV groups by participant so every fold
holds out all days of one participant.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loso_daily_cnn import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    _fold_metadata,
    _make_dataset,
    _path_from_env_or_arg,
    _write_results,
)
from sklearn.model_selection import LeaveOneGroupOut
from tpcp.optimize import Optimize
from tpcp.validate import DatasetSplitter

from mobgap.utils.evaluation import EvaluationCV
from mobgap.weartime import WtdMegaritisXGBoost
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.pipeline import WtdEmulationPipeline

LOGGER = logging.getLogger(__name__)


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
    dataset = _make_dataset(args)
    splitter = DatasetSplitter(base_splitter=LeaveOneGroupOut(), groupby="participant_id")
    fold_metadata = _fold_metadata(dataset, splitter)

    LOGGER.info("Selected split-by-day human datapoints: %s", len(dataset.index))
    LOGGER.info("Selected participants: %s", dataset.index["participant_id"].nunique())
    LOGGER.info("Participant-grouped CV folds: %s", len(fold_metadata))
    LOGGER.info("XGBoost datapoint feature workers: %s", args.n_jobs)
    LOGGER.info("CV fold workers: %s", args.cv_n_jobs)
    LOGGER.info("Output directory: %s", output_dir)

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.index.to_csv(output_dir / "dataset_index.csv", index=False)
        fold_metadata.to_csv(output_dir / "fold_metadata.csv", index=False)
        LOGGER.info("Dry run complete. Wrote fold metadata and dataset index.")
        return

    cache_dir = _path_from_env_or_arg(args.cache_dir, "MOBGAP_CACHE_DIR_PATH", fallback=DEFAULT_CACHE_DIR)
    pipeline = _make_pipeline(args, cache_dir)
    evaluation = EvaluationCV(
        dataset=dataset,
        scoring=wtd_score,
        cv_iterator=splitter,
        cv_params={
            "n_jobs": args.cv_n_jobs,
            "return_train_score": False,
            "progress_bar": True,
        },
    )
    optimizer = Optimize(pipeline)

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
        fold_metadata=fold_metadata,
        output_dir=output_dir,
        run_metadata=run_metadata,
    )

    LOGGER.info("LOSO XGBoost evaluation complete in %s", timedelta(seconds=int(total_time)))
    LOGGER.info("Results written to %s", output_dir)


if __name__ == "__main__":
    main()
