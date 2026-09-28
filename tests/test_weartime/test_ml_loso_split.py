"""Check the training-only days in the daily ML evaluation fold plans."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from mobgap.data import SustainWearTimeDataset

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def evaluation_scripts(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    """Import the command-line modules when their optional ML dependencies are installed."""
    pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "weartime_ml_training_evaluation" / "evaluation"
    monkeypatch.syspath_prepend(str(script_dir))
    return importlib.import_module("loso_daily_cnn"), importlib.import_module("loso_daily_xgboost")


def test_loso_dry_runs_keep_part_b_only_in_training(
    evaluation_scripts: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every held-out participant fold keeps both part B recordings in training."""
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

    for script in evaluation_scripts:
        monkeypatch.setattr(script, "_make_base_dataset", lambda _: dataset)
        monkeypatch.setattr(
            sys,
            "argv",
            [script.__name__, "--dry-run", "--output-dir", str(tmp_path), "--run-name", script.__name__],
        )
        script.main()

        output_dir = tmp_path / script.__name__
        training_only = pd.read_csv(output_dir / "training_only_index.csv")
        fold_plan = pd.read_csv(output_dir / "fold_metadata.csv")
        assert len(training_only) == 4
        assert set(training_only["recording_id"]) == {"part_b_020", "part_b_021"}
        assert len(fold_plan) == 3
        assert set(fold_plan["n_train_days"]) == {14}
        assert set(fold_plan["n_training_only_days"]) == {4}
        assert set(fold_plan["n_test_days"]) == {5}
        assert all(recording_id.startswith("human_") for recording_id in fold_plan["test_recording_ids"])
