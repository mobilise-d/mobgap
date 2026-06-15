"""Helpful pipelines to evaluate reorientation algorithms."""

from collections.abc import Hashable

import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import OptimizableParameter, Pipeline
from typing_extensions import Self

from mobgap._gaitmap.utils.rotations import flip_dataset
from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.pipeline import iter_gs
from mobgap.re_orientation.base import BaseReorientationCorrector
from mobgap.utils.conversions import to_body_frame

# Labels describe simulated mounting states: identity, rotations around PA, and PA-flipped variants.
REORIENTATION_ROTATIONS = {
    "identity": Rotation.identity(),
    "pa_normal__rot_pa_pos90": Rotation.from_euler("z", 90, degrees=True),
    "pa_normal__rot_pa_180": Rotation.from_euler("z", 180, degrees=True),
    "pa_normal__rot_pa_neg90": Rotation.from_euler("z", -90, degrees=True),
    "pa_flipped__rot_pa_0": Rotation.from_euler("x", 180, degrees=True),
    "pa_flipped__rot_pa_pos90": Rotation.from_euler("xz", [180, 90], degrees=True),
    "pa_flipped__rot_pa_180": Rotation.from_euler("y", 180, degrees=True),
    "pa_flipped__rot_pa_neg90": Rotation.from_euler("xz", [180, -90], degrees=True),
}
REORIENTATION_LABELS = tuple(REORIENTATION_ROTATIONS.keys())
UNKNOWN_ORIENTATION_LABEL = "unknown"


def _orientation_class_from_result(algo: BaseReorientationCorrector) -> str:
    """Map a reorientation algorithm response to one of the simulated orientation classes."""
    if not hasattr(algo, "result_"):
        raise ValueError(
            "ReorientationEmulationPipeline requires algorithms to expose a `result_` "
            "attribute with at least `family` and `phase`."
        )

    result = algo.result_
    family = result.family
    phase = result.phase

    if family is None or phase is None:
        return UNKNOWN_ORIENTATION_LABEL

    # phase is None when trust_gravity skips Stage 3 for is_up (intentional)
    # or when bout is too short to compute phase (treat as unknown for all families).
    # For is_up + trust_gravity, correction_action is "none" — return identity.
    # For all other phase=None cases, return unknown.
    if phase is None:
        if family == "is_up" and result.correction_action == "none":
            return "identity"
        return UNKNOWN_ORIENTATION_LABEL

    family_map = {
        "is_up": ("pa_flipped__rot_pa_0", "identity", phase < 0),
        "is_down": ("pa_normal__rot_pa_180", "pa_flipped__rot_pa_180", phase > 0),
        "ml_up": ("pa_normal__rot_pa_pos90", "pa_flipped__rot_pa_pos90", phase > 0),
        "ml_down": ("pa_flipped__rot_pa_neg90", "pa_normal__rot_pa_neg90", phase < 0),
    }

    if family not in family_map:
        return UNKNOWN_ORIENTATION_LABEL

    label_if_true, label_if_false, condition = family_map[family]
    return label_if_true if condition else label_if_false


class ReorientationEmulationPipeline(Pipeline[BaseGaitDatasetWithReference]):
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

        # TODO: This needs to be changed, once the Reorientation algorithm is properly implemented as "converting
        #  from sensor to body frame".
        data = to_body_frame(datapoint.data_ss)
        result_algo_list = {}
        predictions_per_wb = {}
        predictions = []

        for wb, wb_data in iter_gs(data, wb_list):
            wb_predictions = []
            for label, rotation in REORIENTATION_ROTATIONS.items():
                # trust_gravity intentionally does not correct Family "is_up" front-back flips
                # (pa_flipped__rot_pa_0), so skip this orientation for fair evaluation.
                if (
                    hasattr(self.algo, "correction_mode")
                    and self.algo.correction_mode == "trust_gravity"
                    and label == "pa_flipped__rot_pa_0"
                ):
                    continue
                rotated_data = flip_dataset(wb_data, rotation)
                algo = self.algo.clone().detect_correct(rotated_data, sampling_rate_hz=datapoint.sampling_rate_hz)
                result_algo_list[(wb.id, label)] = algo
                wb_predictions.append(
                    {
                        "wb_id": wb.id,
                        "label": label,
                        "prediction": _orientation_class_from_result(algo),
                    }
                )
            predictions_per_wb[wb.id] = pd.DataFrame(wb_predictions)[["label", "prediction"]]
            predictions.extend(wb_predictions)

        self.per_wb_algo_ = result_algo_list
        self.predictions_per_wb_ = predictions_per_wb
        self.predictions_ = pd.DataFrame(predictions).set_index("wb_id")[["label", "prediction"]]

        return self


__all__ = [
    "REORIENTATION_LABELS",
    "REORIENTATION_ROTATIONS",
    "ReorientationEmulationPipeline",
]
