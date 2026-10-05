"""Check human and Part B assignments in daily ML evaluation folds."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from tpcp.optimize import Optimize
from tpcp.validate import CombinedSplitter

from mobgap.data import SustainWearTimeDataset, split_by_utc_day
from mobgap.utils.evaluation import EvaluationCV

if TYPE_CHECKING:
    from types import ModuleType

    from tpcp.validate import BaseDatasetSplitter


@pytest.fixture
def evaluation_scripts(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    """Import the command-line modules when their optional ML dependencies are installed."""
    pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "weartime_ml_training_evaluation" / "evaluation"
    monkeypatch.syspath_prepend(str(script_dir))
    return importlib.import_module("loso_daily_cnn"), importlib.import_module("loso_daily_xgboost")


def test_loso_outer_training_and_human_only_inner_search(  # noqa: PLR0915 - Cover the complete nested split flow.
    evaluation_scripts: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Outer training includes part B while inner search uses human recordings only."""
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
        for participant in ("020", "021", "022", "023")
        for day in (1, 2)
    )
    dataset = SustainWearTimeDataset(Path("unused"), splitter=split_by_utc_day, subset_index=pd.DataFrame(rows))

    for script in evaluation_scripts:
        monkeypatch.setattr(script, "SustainWearTimeDataset", lambda *_, **__: dataset)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                script.__name__,
                "--dataset-path",
                "unused",
                "--dry-run",
                "--output-dir",
                str(tmp_path),
                "--run-name",
                script.__name__,
            ],
        )
        script.main()

        output_dir = tmp_path / script.__name__
        fold_plan = pd.read_csv(output_dir / "fold_metadata.csv")
        assert len(fold_plan) == 3
        assert set(fold_plan["n_train_days"]) == {14}
        assert set(fold_plan["n_test_days"]) == {5}
        assert all(recording_id.startswith("human_") for recording_id in fold_plan["test_recording_ids"])

        monkeypatch.setattr(
            sys,
            "argv",
            [
                script.__name__,
                "--dataset-path",
                "unused",
                "--dry-run",
                "--output-dir",
                str(tmp_path),
                "--run-name",
                f"{script.__name__}_selected",
                "--participant-id",
                "002",
                "--participant-id",
                "003",
                "--max-participants",
                "2",
                "--part-b-recording-count",
                "1",
            ],
        )
        script.main()
        selected_folds = pd.read_csv(tmp_path / f"{script.__name__}_selected" / "fold_metadata.csv")
        assert set(selected_folds["held_out_participant_ids"].astype(str).str.zfill(3)) == {"002", "003"}
        assert set(selected_folds["n_train_days"]) == {7}

    cnn = evaluation_scripts[0]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "loso_daily_cnn",
            "--dataset-path",
            "unused",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "cnn_part_b_days",
            "--part-b-day-count",
            "3",
        ],
    )
    cnn.main()
    cnn_day_plan = pd.read_csv(tmp_path / "cnn_part_b_days" / "fold_metadata.csv")
    assert set(cnn_day_plan["n_train_days"]) == {13}

    xgboost = evaluation_scripts[1]
    captured_splitters: dict[str, BaseDatasetSplitter] = {}
    captured_datasets: dict[str, SustainWearTimeDataset] = {}
    captured_wrappers: list[object] = []

    def capture_run(evaluation: EvaluationCV, optimizer: xgboost.OptimizableOptunaSearch) -> EvaluationCV:
        captured_splitters["outer"] = evaluation.cv_iterator
        captured_datasets["outer"] = evaluation.dataset
        captured_wrappers.append(optimizer)
        return evaluation

    monkeypatch.setattr(EvaluationCV, "run", capture_run)
    monkeypatch.setattr(xgboost, "_write_results", lambda **_: None)
    monkeypatch.setattr(xgboost, "_write_search_results", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["loso_daily_xgboost", "--dataset-path", "unused", "--inner-folds", "2"])
    xgboost.main()

    evaluation_dataset = captured_datasets["outer"]
    for outer_train_labels, _ in captured_splitters["outer"].split(evaluation_dataset):
        outer_train = evaluation_dataset.get_subset(group_labels=outer_train_labels)
        part_b_train = outer_train.get_subset(recording_type="simulated_movements").index
        assert len(part_b_train) == 4
        assert set(part_b_train["recording_id"]) == {"part_b_021", "part_b_023"}

    captured_inner: dict[str, object] = {}

    def capture_inner_cv(
        optimizable: Optimize, inner_dataset: SustainWearTimeDataset, *, cv: BaseDatasetSplitter, **_: object
    ) -> dict[str, list[float]]:
        captured_inner["optimizer"] = optimizable
        captured_inner["dataset"] = inner_dataset
        captured_inner["splitter"] = cv
        return {"test__agg__combined__accuracy": [0.75]}

    monkeypatch.setattr(xgboost, "cross_validate", capture_inner_cv)
    refit_datasets: list[SustainWearTimeDataset] = []

    def capture_refit(optimizer: Optimize, train_dataset: SustainWearTimeDataset) -> Optimize:
        refit_datasets.append(train_dataset)
        optimizer.optimized_pipeline_ = optimizer.pipeline
        return optimizer

    monkeypatch.setattr(Optimize, "optimize", capture_refit)
    wrapper = captured_wrappers[0]
    assert isinstance(wrapper, xgboost.OptimizableOptunaSearch)
    first_train_labels, _ = next(captured_splitters["outer"].split(evaluation_dataset))
    outer_train = evaluation_dataset.get_subset(group_labels=first_train_labels)
    search = wrapper.clone().set_params(n_trials=1).optimize(outer_train)
    assert search.best_score_ == 0.75
    assert len(refit_datasets) == 1
    assert len(refit_datasets[0].get_subset(recording_type="simulated_movements").index) == 4
    inner_dataset = captured_inner["dataset"]
    assert isinstance(inner_dataset, SustainWearTimeDataset)
    assert len(inner_dataset.get_subset(recording_type="simulated_movements").index) == 4
    assert isinstance(captured_inner["splitter"], CombinedSplitter)
    for inner_train_labels, inner_test_labels in captured_inner["splitter"].split(inner_dataset):
        inner_train = inner_dataset.get_subset(group_labels=inner_train_labels)
        inner_test = inner_dataset.get_subset(group_labels=inner_test_labels)
        assert set(inner_test.index["recording_type"]) == {"human_movement"}
        assert set(inner_train.index["recording_type"]) == {"human_movement"}
    inner_optimizer = captured_inner["optimizer"]
    assert isinstance(inner_optimizer, Optimize)
    sample_human_days = inner_optimizer.train_dataset_transform
    assert sample_human_days is not None
    inner_train_labels, _ = next(captured_inner["splitter"].split(inner_dataset))
    human_training_days = inner_dataset.get_subset(group_labels=inner_train_labels)
    sampled = sample_human_days(human_training_days)
    assert set(sampled.index["recording_type"]) == {"human_movement"}
    assert len(sampled.index) == max(1, round(0.4 * len(human_training_days.index)))

    monkeypatch.setattr(
        sys,
        "argv",
        ["loso_daily_xgboost", "--dataset-path", "unused", "--inner-folds", "2", "--part-b-day-count", "3"],
    )
    xgboost.main()
    part_b_days_by_fold = [
        evaluation_dataset.get_subset(group_labels=train_labels).get_subset(recording_type="simulated_movements").index
        for train_labels, _ in captured_splitters["outer"].split(evaluation_dataset)
    ]
    assert len(part_b_days_by_fold) == 3
    assert all(len(days) == 3 for days in part_b_days_by_fold)
    assert set(part_b_days_by_fold[0][["recording_id", "recording_day"]].itertuples(index=False, name=None)) == {
        ("part_b_020", "2020-02-01"),
        ("part_b_020", "2020-02-02"),
        ("part_b_022", "2020-02-02"),
    }
    for days in part_b_days_by_fold[1:]:
        pd.testing.assert_frame_equal(days, part_b_days_by_fold[0])
