"""Helpful pipelines to evaluate reorientation algorithms."""

from collections.abc import Hashable
from typing import Any

import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import OptimizableParameter, OptimizablePipeline
from typing_extensions import Self, Unpack

from mobgap._gaitmap.utils.rotations import flip_dataset
from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.pipeline import iter_gs
from mobgap.re_orientation.base import BaseReorientationCorrector
from mobgap.utils.conversions import to_body_frame

REORIENTATION_ROTATIONS = {
    "identity": Rotation.identity(),
    "rot_x_180": Rotation.from_euler("x", 180, degrees=True),
    "rot_y_180": Rotation.from_euler("y", 180, degrees=True),
    "rot_z_180": Rotation.from_euler("z", 180, degrees=True),
    "rot_z_90": Rotation.from_euler("z", 90, degrees=True),
    "rot_z_-90": Rotation.from_euler("z", -90, degrees=True),
    "rot_x_180_z_90": Rotation.from_euler("xz", [180, 90], degrees=True),
    "rot_x_180_z_-90": Rotation.from_euler("xz", [180, -90], degrees=True),
}
REORIENTATION_LABELS = tuple(REORIENTATION_ROTATIONS.keys())
UNKNOWN_ORIENTATION_LABEL = "unknown"


def _orientation_class_from_result(algo: BaseReorientationCorrector) -> str:
    """Map a reorientation algorithm response to one of the simulated rotation classes."""
    if not hasattr(algo, "result_"):
        raise ValueError(
            "ReorientationEmulationPipeline requires algorithms to expose a `result_` "
            "attribute with at least `family` and `phase`."
        )

    result = algo.result_
    family = result.family
    phase = result.phase

    if family is None:
        return UNKNOWN_ORIENTATION_LABEL
    if family == 1:
        return "rot_x_180" if phase < 0 else "identity"
    if family == 2:
        return "rot_z_180" if phase > 0 else "rot_y_180"
    if family == 3:
        return "rot_z_90" if phase > 0 else "rot_x_180_z_90"
    if family == 4:
        return "rot_x_180_z_-90" if phase < 0 else "rot_z_-90"

    return UNKNOWN_ORIENTATION_LABEL


class ReorientationEmulationPipeline(OptimizablePipeline[BaseGaitDatasetWithReference]):
    """Run a reorientation algorithm on simulated sensor misorientations.

    This pipeline uses the reference walking bouts of a datapoint. For every walking bout,
    it creates one copy for each supported rough mounting orientation using
    :func:`mobgap._gaitmap.utils.rotations.flip_dataset`, runs the wrapped algorithm, and
    stores the detected orientation class.

    Parameters
    ----------
    algo
        The reorientation algorithm to be run in the pipeline.

    Attributes
    ----------
    predictions_
        Dataframe with one row per walking bout and simulated orientation. The
        dataframe is indexed by ``wb_id`` and has the columns ``label`` and
        ``prediction``.
    predictions_per_wb_
        A dict containing one ``label``/``prediction`` dataframe per walking bout.
    per_wb_algo_
        A dict of the reorientation algorithm instances run on each walking bout and
        simulated orientation. The key is ``(wb_id, label)``.

    Other Parameters
    ----------------
    datapoint
        The datapoint that was passed to the run method.
    """

    algo: OptimizableParameter[BaseReorientationCorrector]

    predictions_: pd.DataFrame
    predictions_per_wb_: dict[Hashable, pd.DataFrame]
    per_wb_algo_: dict[tuple[Hashable, str], BaseReorientationCorrector]

    def __init__(self, algo: BaseReorientationCorrector) -> None:
        self.algo = algo

    def run(self, datapoint: BaseGaitDatasetWithReference) -> Self:
        """Run the pipeline on a single datapoint."""
        self.datapoint = datapoint

        wb_list = datapoint.reference_parameters_.wb_list
        if wb_list.empty:
            self.predictions_ = pd.DataFrame(
                columns=["label", "prediction"],
                index=pd.Index([], name="wb_id"),
            )
            self.predictions_per_wb_ = {}
            self.per_wb_algo_ = {}
            return self

        data = to_body_frame(datapoint.data_ss)
        result_algo_list = {}
        predictions_per_wb = {}
        predictions = []

        for wb, wb_data in iter_gs(data, wb_list):
            wb_predictions = []
            for label, rotation in REORIENTATION_ROTATIONS.items():
                rotated_data = flip_dataset(wb_data, rotation)
                algo = self.algo.clone().detect_correct(
                    rotated_data, sampling_rate_hz=datapoint.sampling_rate_hz
                )
                result_algo_list[(wb.id, label)] = algo
                wb_predictions.append(
                    {
                        "wb_id": wb.id,
                        "label": label,
                        "prediction": _orientation_class_from_result(algo),
                    }
                )
            predictions_per_wb[wb.id] = pd.DataFrame(wb_predictions)[
                ["label", "prediction"]
            ]
            predictions.extend(wb_predictions)

        self.per_wb_algo_ = result_algo_list
        self.predictions_per_wb_ = predictions_per_wb
        self.predictions_ = pd.DataFrame(predictions).set_index("wb_id")[
            ["label", "prediction"]
        ]

        return self

    def self_optimize(self, dataset: BaseGaitDatasetWithReference, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Run a self-optimization of the wrapped algorithm if available."""
        self.algo.self_optimize(dataset, **kwargs)
        return self


__all__ = [
    "REORIENTATION_LABELS",
    "REORIENTATION_ROTATIONS",
    "ReorientationEmulationPipeline",
]
