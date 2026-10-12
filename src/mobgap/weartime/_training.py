"""Lazy recording access shared by wear-time trainers."""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003 - tpcp evaluates annotations at runtime.

import numpy as np

from mobgap.data.base import BaseGaitDataset  # noqa: TC001 - tpcp evaluates annotations at runtime.
from mobgap.weartime.base import TrainingRecording  # noqa: TC001 - tpcp evaluates annotations at runtime.
from mobgap.weartime.pipeline import _conditionally_to_bf


class _TrainingDataFromDataset:
    def __init__(self, dataset: BaseGaitDataset, *, convert_to_body_frame: bool) -> None:
        self.dataset = dataset
        self.convert_to_body_frame = convert_to_body_frame

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[TrainingRecording]:
        for datapoint_index in range(len(self.dataset)):
            yield self.load_recording(datapoint_index)

    def load_recording(self, datapoint_index: int) -> TrainingRecording:
        datapoint = self.dataset[datapoint_index]
        data = _conditionally_to_bf(datapoint.data_ss, self.convert_to_body_frame)
        reference_weartime = datapoint.reference_weartime_
        return data, reference_weartime


def _single_sampling_rate_hz(dataset: BaseGaitDataset) -> float:
    sampling_rates = [float(datapoint.sampling_rate_hz) for datapoint in dataset]

    if not sampling_rates:
        raise ValueError("Cannot self-optimize a wear-time detector on an empty dataset.")

    sampling_rate_hz = sampling_rates[0]
    if not all(np.isclose(sampling_rate, sampling_rate_hz) for sampling_rate in sampling_rates):
        raise ValueError("Wear-time model self-optimization requires all datapoints to use the same sampling rate.")

    return sampling_rate_hz
