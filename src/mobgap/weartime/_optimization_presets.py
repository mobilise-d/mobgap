"""Shared inner training composition for the SUSTAIN wear-time presets.

Three participant-grouped folds rank candidates on human-movement days only.
Each inner training subset samples 40% of its human days and five simulated
non-wear days independently with seed 42. Supply at least five simulated days
in the training pool. These are computational-budget settings for this dataset,
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
    from mobgap.weartime.evaluation import wtd_score  # noqa: PLC0415 - Avoid the scorer/pipeline import cycle.

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
    }
