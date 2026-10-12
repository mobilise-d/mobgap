"""Input requirements carried by fitted wear-time models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mobgap.consts import BF_SENSOR_COLS


@dataclass(frozen=True)
class WearTimeModelConfig:
    """Immutable input configuration produced during training and saved with the model.

    Sampling rate, window duration and channel order must match training. XGBoost additionally requires the
    ordered feature names. CNN models use an empty feature tuple because they operate on raw windows.
    Inference overlap, thresholds and execution settings are configured independently on the detector.
    """

    sampling_rate_hz: float
    window_sec: float = 5.0
    sensor_cols: tuple[str, ...] = tuple(BF_SENSOR_COLS)
    feature_names: tuple[str, ...] = ()

    def validate_sampling_rate(self, sampling_rate_hz: float) -> None:
        """Require the model's native sampling rate; data is never resampled implicitly."""
        if not np.isclose(sampling_rate_hz, self.sampling_rate_hz):
            raise ValueError(
                f"The wear-time model requires {self.sampling_rate_hz} Hz, but received {sampling_rate_hz} Hz. "
                "Resample the input explicitly or use a model trained at this rate."
            )

    def window_parameters(self, overlap: float) -> tuple[int, int]:
        """Return sample counts using the trained rate, including for nearly equal reported rates."""
        if not 0 <= overlap < 1:
            raise ValueError("`overlap` must be in [0, 1).")
        window_samples = round(self.window_sec * self.sampling_rate_hz)
        if window_samples <= 0:
            raise ValueError("The model window must contain at least one sample.")
        return window_samples, window_samples - int(window_samples * overlap)

    def to_dict(self) -> dict[str, Any]:
        """Return the input requirements for JSON artifact export."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> WearTimeModelConfig:
        """Restore tuple-valued input requirements from a JSON artifact."""
        return cls(
            sampling_rate_hz=config["sampling_rate_hz"],
            window_sec=config["window_sec"],
            sensor_cols=tuple(config["sensor_cols"]),
            feature_names=tuple(config["feature_names"]),
        )
