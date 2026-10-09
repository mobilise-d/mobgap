"""Check human and Part B assignments in daily ML evaluation folds."""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from types import SimpleNamespace
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
    """Import the training modules when their optional ML dependencies are installed."""
    pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "weartime_ml_training_evaluation" / "evaluation"
    monkeypatch.syspath_prepend(str(script_dir))
    return importlib.import_module("loso_daily_cnn"), importlib.import_module("loso_daily_xgboost")


def test_loso_fixed_part_b_halves_and_human_only_inner_ranking(  # noqa: PLR0915 - Cover the complete nested split flow.
    evaluation_scripts: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fixed Part B halves enter outer folds; only human inner validation ranks candidates."""
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

    captured_splitters: dict[str, BaseDatasetSplitter] = {}
    captured_datasets: dict[str, SustainWearTimeDataset] = {}
    captured_wrappers: list[object] = []
    saved_models: list[Path] = []

    def capture_run(evaluation: EvaluationCV, optimizer: object) -> EvaluationCV:
        captured_splitters["outer"] = evaluation.cv_iterator
        captured_datasets["outer"] = evaluation.dataset
        captured_wrappers.append(optimizer)
        assert not (evaluation.cv_params or {}).get("return_optimizer", False)
        evaluation.results_ = {}
        return evaluation

    monkeypatch.setattr(EvaluationCV, "run", capture_run)
    monkeypatch.setattr(EvaluationCV, "get_aggregated_results_as_df", lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(EvaluationCV, "get_single_results_as_df", lambda *_, **__: pd.DataFrame())
    cnn, xgboost = evaluation_scripts
    monkeypatch.setattr(
        cnn, "import_module", lambda _: SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda _: []))
    )

    final_training_datasets: list[SustainWearTimeDataset] = []

    def capture_final_training(optimizer: object, train_dataset: SustainWearTimeDataset) -> object:
        final_training_datasets.append(train_dataset)
        optimizer.optimized_pipeline_ = SimpleNamespace(
            algo=SimpleNamespace(
                clf={"trained": True},
                feature_names=["feature"],
                model=SimpleNamespace(_model=SimpleNamespace(save=saved_models.append)),
            )
        )
        return optimizer

    def run_script(script: ModuleType) -> None:
        with monkeypatch.context() as training:
            optimizer_class = Optimize if script is cnn else xgboost.OptimizableOptunaSearch
            training.setattr(optimizer_class, "optimize", capture_final_training)
            script.main()

    for script in evaluation_scripts:
        monkeypatch.setattr(script, "SustainWearTimeDataset", lambda *_, **__: dataset)
        monkeypatch.setattr(script, "DATASET_PATH", Path("unused"))
        monkeypatch.setattr(script, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(script, "RUN_NAME", script.__name__)
        if script is xgboost:
            monkeypatch.setattr(script, "INNER_FOLDS", 2)
        run_script(script)
        assert len(final_training_datasets[-1].index) == 23
        assert captured_datasets["outer"] is dataset
        assert captured_splitters["outer"].get_n_splits(dataset) == 3
        fixed_part_b = None
        for train_labels, test_labels in captured_splitters["outer"].split(dataset):
            train = dataset.get_subset(group_labels=train_labels)
            test = dataset.get_subset(group_labels=test_labels)
            assert len(train.index) == 14
            assert len(test.index) == 9
            human_train = train.get_subset(recording_type="human_movement")
            human_test = test.get_subset(recording_type="human_movement")
            assert len(set(human_test.index["participant_id"])) == 1
            assert not set(human_train.index["participant_id"]) & set(human_test.index["participant_id"])
            part_b_train = set(train.get_subset(recording_type="simulated_movements").group_labels)
            part_b_test = set(test.get_subset(recording_type="simulated_movements").group_labels)
            assert len(part_b_train) == len(part_b_test) == 4
            assert not part_b_train & part_b_test
            assert part_b_train | part_b_test == set(
                dataset.get_subset(recording_type="simulated_movements").group_labels
            )
            fixed_part_b = fixed_part_b or (part_b_train, part_b_test)
            assert fixed_part_b == (part_b_train, part_b_test)

        with monkeypatch.context() as selection:
            selection.setattr(script, "PARTICIPANT_IDS", ["002", "003"])
            selection.setattr(script, "MAX_PARTICIPANTS", 2)
            run_script(script)
            assert len(final_training_datasets[-1].index) == 18
            selected_folds = list(captured_splitters["outer"].split(dataset))
            assert len(selected_folds) == 2
            assert {
                dataset.get_subset(group_labels=test)
                .get_subset(recording_type="human_movement")
                .index["participant_id"]
                .iloc[0]
                for _, test in selected_folds
            } == {"002", "003"}
            assert all(len(dataset.get_subset(group_labels=train).index) == 9 for train, _ in selected_folds)

    assert saved_models == [tmp_path / "loso_daily_cnn" / "model.keras"] * 2

    with (tmp_path / "loso_daily_xgboost" / "model.pkl").open("rb") as file:
        assert pickle.load(file) == {"trained": True}
    with (tmp_path / "loso_daily_xgboost" / "feature_order.pkl").open("rb") as file:
        assert pickle.load(file) == ["feature"]
    run_script(xgboost)
    evaluation_dataset = captured_datasets["outer"]
    wrapper = captured_wrappers[-1]
    assert wrapper.return_optimized is True

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
        assert set(inner_train.index["recording_type"]) == {"human_movement", "simulated_movements"}
        assert set(inner_train.get_subset(recording_type="simulated_movements").group_labels) == set(
            outer_train.get_subset(recording_type="simulated_movements").group_labels
        )
    inner_optimizer = captured_inner["optimizer"]
    assert isinstance(inner_optimizer, Optimize)
    sample_human_days = inner_optimizer.train_dataset_transform
    assert sample_human_days is not None
    inner_train_labels, _ = next(captured_inner["splitter"].split(inner_dataset))
    inner_training_days = inner_dataset.get_subset(group_labels=inner_train_labels)
    sampled = sample_human_days(inner_training_days)
    human_training_days = inner_training_days.get_subset(recording_type="human_movement")
    assert len(sampled.get_subset(recording_type="human_movement").index) == max(
        1, round(0.4 * len(human_training_days.index))
    )
    pd.testing.assert_frame_equal(
        sampled.get_subset(recording_type="simulated_movements").index,
        inner_training_days.get_subset(recording_type="simulated_movements").index,
    )
    pd.testing.assert_frame_equal(sample_human_days(inner_training_days).index, sampled.index)

    monkeypatch.setattr(xgboost, "PART_B_DAY_COUNT", 3)
    sampled_days = sample_human_days(inner_training_days)
    assert len(sampled_days.get_subset(recording_type="simulated_movements").index) == 3
    assert set(sampled_days.get_subset(recording_type="simulated_movements").group_labels) <= set(
        outer_train.group_labels
    )
    pd.testing.assert_frame_equal(
        sampled_days.get_subset(recording_type="human_movement").index,
        sampled.get_subset(recording_type="human_movement").index,
    )
