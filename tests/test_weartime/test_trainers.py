"""Short fits exercising the public dataset optimization boundary."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.naive_bayes import GaussianNB
from tpcp import Dataset
from tpcp.optimize import Optimize

from mobgap.consts import BF_SENSOR_COLS
from mobgap.weartime import (
    KerasWeartimeModel,
    WearTimeModelConfig,
    WtdMegaritisCNN,
    WtdMegaritisCNNTrainer,
    WtdMegaritisXGBoostTrainer,
    load_keras_weartime_model,
)


class TinyDataset(Dataset):
    """Two tiny recordings with both classes and an uncertain final window."""

    def create_index(self) -> pd.DataFrame:
        """Return two distinct participant recordings."""
        return pd.DataFrame({"participant_id": ["001", "002"], "recording_id": ["a", "b"]})

    @property
    def sampling_rate_hz(self) -> float:
        """Use a non-default rate to test native input configuration."""
        return 2.0

    @property
    def data_ss(self) -> pd.DataFrame:
        """Generate deterministic body-frame input."""
        data = np.random.default_rng(42).normal(size=(80, 6)).astype(np.float32)
        data[40:] *= 3
        return pd.DataFrame(data, columns=BF_SENSOR_COLS)

    @property
    def reference_weartime_(self) -> pd.DataFrame:
        """Retain known wear, known non-wear and explicit uncertainty."""
        return pd.DataFrame(
            {
                "start": [0, 60],
                "end": [40, 80],
                "label": pd.Categorical(["wear", "uncertain"], categories=["wear", "uncertain"]),
            }
        ).rename_axis("wt_id")

    @property
    def recording_metadata(self) -> dict:
        """No optional metadata required."""
        return {}

    @property
    def participant_metadata(self) -> dict:
        """No optional metadata required."""
        return {}


@pytest.mark.parametrize("classifier", ["sklearn", "xgboost"])
def test_sklearn_trainer_through_optimize(classifier: str) -> None:
    """Fit a supported sklearn estimator without modifying its template and run a cloned winner."""
    template = (
        GaussianNB()
        if classifier == "sklearn"
        else pytest.importorskip("xgboost").XGBClassifier(n_estimators=2, max_depth=2, n_jobs=1, random_state=42)
    )
    trainer = WtdMegaritisXGBoostTrainer(
        classifier=template,
        window_sec=10,
        feature_names=("acc_pa_mean",),
        training_overlap=0,
        convert_to_body_frame=False,
    )
    trained = Optimize(trainer).optimize(TinyDataset()).optimized_pipeline_
    assert not hasattr(template, "classes_")
    assert trained.algo.model_config == WearTimeModelConfig(2, 10, feature_names=("acc_pa_mean",))
    assert trained.algo.overlap == 0.75
    result = trained.clone().run(TinyDataset()[0])
    assert len(result.algo_.window_predictions_) == 13


@pytest.mark.parametrize("architecture", ["cnn", "cnn_lstm"])
def test_tiny_cnn_optimize_and_artifact_roundtrip(architecture: Literal["cnn", "cnn_lstm"], tmp_path: Path) -> None:
    """Fit only six tiny known windows, then clone, infer and reload the serialized graph."""
    pytest.importorskip("tensorflow")
    supplied_detector = WtdMegaritisCNN(waking_hours=(time(0), time(0, 1)))
    trainer = WtdMegaritisCNNTrainer(
        supplied_detector,
        window_sec=10,
        training_overlap=0,
        filters=(2, 2, 2),
        kernel_size=3,
        dense_units=2,
        lstm_units=2,
        architecture=architecture,
        epochs=1,
        batch_size=3,
        fit_verbose=0,
        shuffle_buffer_size=4,
        convert_to_body_frame=False,
    )
    trained = Optimize(trainer).optimize(TinyDataset()).optimized_pipeline_
    assert trainer.algo.model.model is None
    assert supplied_detector.model.model is None
    assert trained.algo.waking_hours == supplied_detector.waking_hours
    config = trained.algo.model.model_config
    assert config == WearTimeModelConfig(2, 10)
    probabilities = trained.clone().run(TinyDataset()[0]).algo_.model_.window_probabilities_
    model_path = tmp_path / "model.keras"
    trained.algo.model.model.save(model_path)
    loaded = WtdMegaritisCNN(
        model=KerasWeartimeModel(
            model=load_keras_weartime_model(model_path), model_config=WearTimeModelConfig.from_dict(config.to_dict())
        )
    )
    loaded.detect(TinyDataset()[0].data_ss, sampling_rate_hz=2.00000001)
    assert_allclose(loaded.model_.window_probabilities_, probabilities, rtol=1e-6)
    with pytest.raises(ValueError, match=r"requires 2\.0 Hz"):
        loaded.detect(TinyDataset()[0].data_ss, sampling_rate_hz=4)
