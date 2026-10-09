"""Shared inner training composition for the SUSTAIN wear-time presets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.model_selection import GroupKFold
from tpcp.validate import CombinedSplitter, DatasetSplitter, NoSplit

if TYPE_CHECKING:
    from mobgap.data.base import BaseGaitDataset


def _sample_sustain_training_days(days: BaseGaitDataset, *, part_b_day_count: int | None = 5) -> BaseGaitDataset:
    human_days = days.index.query("recording_type == 'human_movement'")
    part_b_days = days.index.query("recording_type == 'simulated_movements'")
    return days.get_subset(
        index=pd.concat(
            [
                human_days.sample(n=max(1, round(len(human_days) * 0.4)), random_state=42),
                part_b_days.sample(n=part_b_day_count, random_state=42)
                if part_b_day_count is not None
                else part_b_days,
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
                    lambda days: days.get_subset(recording_type="human_movement"),
                    DatasetSplitter(GroupKFold(n_splits=3), groupby="participant_id"),
                ),
                (
                    lambda days: days.get_subset(recording_type="simulated_movements"),
                    NoSplit(None, train=lambda days: days),
                ),
            ]
        ),
        "train_dataset_transform": _sample_sustain_training_days,
    }
