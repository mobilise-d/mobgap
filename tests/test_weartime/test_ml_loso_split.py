"""Check the training-only days in the daily ML evaluation split plan."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from tpcp.validate import DatasetSplitter

from mobgap.data import SustainWearTimeDataset

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def evaluation_scripts(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    """Import the command-line modules from their script directory."""
    pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "weartime_ml_training_evaluation" / "evaluation"
    monkeypatch.syspath_prepend(str(script_dir))
    return importlib.import_module("loso_daily_cnn"), importlib.import_module("loso_daily_xgboost")


def test_outer_and_inner_splits_keep_part_b_only_in_training(evaluation_scripts: tuple[ModuleType, ModuleType]) -> None:
    """Part B stays in train while both splitter levels hold out only human participants."""
    cnn, xgboost = evaluation_scripts

    def part_b_days(index: pd.DataFrame) -> set[tuple[str, str]]:
        return set(
            index.loc[index["recording_type"] == "simulated_movements", ["recording_id", "recording_day"]].itertuples(
                index=False, name=None
            )
        )

    rows = [
        {
            "recording_type": "human_movement",
            "participant_id": participant,
            "recording_id": f"human_{participant}",
            "recording_day": f"2020-01-{day:02d}",
            "file_path": f"human_{participant}.cwa",
        }
        for participant in ("001", "002", "003")
        for day in range(1, 6)
    ]
    rows.extend(
        {
            "recording_type": "simulated_movements",
            "participant_id": participant,
            "recording_id": f"part_b_{participant}",
            "recording_day": f"2020-02-{day:02d}",
            "file_path": f"part_b_{participant}.cwa",
        }
        for participant in ("020", "021")
        for day in (1, 2)
    )
    dataset = SustainWearTimeDataset(Path("unused"), split_by_day=True, subset_index=pd.DataFrame(rows))
    expected_part_b = part_b_days(dataset.index)

    for human_splitter, n_folds in ((LeaveOneGroupOut(), 3), (GroupKFold(n_splits=2), 2)):
        splitter = cnn._combined_splitter(DatasetSplitter(human_splitter, groupby="participant_id"), n_folds)
        folds = list(splitter.split(dataset))
        assert len(folds) == n_folds
        for train_labels, test_labels in folds:
            train = dataset.get_subset(group_labels=train_labels)
            test = dataset.get_subset(group_labels=test_labels)
            assert part_b_days(train.index) == expected_part_b
            assert set(test.index["recording_type"]) == {"human_movement"}
            assert set(train.index["participant_id"]).isdisjoint(set(test.index["participant_id"]))

            sampled = xgboost._sample_inner_training_days(train, fraction=0.4, seed=0)
            human_count = sum(train.index["recording_type"] == "human_movement")
            sampled_human_count = sum(sampled.index["recording_type"] == "human_movement")
            assert sampled_human_count == max(1, round(0.4 * human_count))
            assert part_b_days(sampled.index) == expected_part_b
