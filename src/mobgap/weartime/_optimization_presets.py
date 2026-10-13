"""Shared inner training composition for the SUSTAIN wear-time presets.

Participant-grouped folds rank candidates on human-movement days only.
Each inner training subset independently samples human and simulated non-wear
days with a fixed seed. Supply enough simulated days for the configured sample
size in the training pool. These are computational-budget settings for this dataset,
not a published or validated tuning protocol. The outer train/test split is
configured by the caller; final refitting uses the complete provided dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from tpcp.validate import CombinedSplitter, DatasetSplitter, NoSplit, SubsetSplitter

if TYPE_CHECKING:
    from mobgap.data.base import BaseGaitDataset


def _sample_sustain_training_days(
    days: BaseGaitDataset, *, simulated_non_wear_day_count: int | None = 5
) -> BaseGaitDataset:
    human_days = days.index.query("recording_type == 'human_movement'")
    simulated_non_wear_days = days.index.query("recording_type == 'simulated_movements'")
    return days.get_subset(
        index=pd.concat(
            [
                human_days.sample(n=max(1, round(len(human_days) * 0.4)), random_state=42),
                simulated_non_wear_days.sample(n=simulated_non_wear_day_count, random_state=42)
                if simulated_non_wear_day_count is not None
                else simulated_non_wear_days,
            ]
        )
    )


def _sustain_weartime_optimization_defaults() -> dict[str, Any]:
    """Configure inner validation and training subsets for the SUSTAIN search presets.

    ``SustainWearTimeDataset`` contains human-movement days and simulated non-wear days.
    Hyperparameter selection uses human-movement days, which contain real movement data.
    Simulated non-wear days can help the algorithms learn specificity, so they are included
    in training but never used to rank hyperparameter candidates.

    The combined splitter groups human-movement days by participant for cross-validation.
    It also includes the supplied simulated non-wear pool in every fold's training set,
    keeping those days out of the validation sets.

    The training transform then independently subsamples human and simulated non-wear days
    to limit computation during hyperparameter optimization. These sampling settings can
    be adjusted independently of the splitter to run a more complete search.
    """
    from mobgap.weartime.evaluation import wtd_score  # noqa: PLC0415 - Avoid the scorer/pipeline import cycle.
    from mobgap.weartime.optimization import _seeded_study_params  # noqa: PLC0415 - Resolve after trainer imports.

    return {
        "scoring": wtd_score,
        "score_name": "combined__accuracy",
        "cv": CombinedSplitter(
            parts=[
                (
                    "human",
                    SubsetSplitter(
                        lambda days: days.get_subset(recording_type="human_movement"),
                        DatasetSplitter(3, groupby="participant_id"),
                    ),
                ),
                (
                    "simulated_non_wear",
                    SubsetSplitter(
                        lambda days: days.get_subset(recording_type="simulated_movements"),
                        NoSplit(None, train=lambda days: days),
                    ),
                ),
            ]
        ),
        "train_dataset_transform": _sample_sustain_training_days,
        "get_study_params": _seeded_study_params,
        "n_trials": 50,
        "random_seed": 42,
        "timeout": None,
        "callbacks": None,
        "gc_after_trial": False,
        "n_jobs": 1,
        "eval_str_paras": (),
        "show_progress_bar": False,
        "return_optimized": True,
    }
