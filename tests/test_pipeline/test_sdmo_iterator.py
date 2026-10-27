import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.spatial.transform import Rotation

from mobgap.pipeline import SdmoIterator
from mobgap.re_orientation import ReorientationMethodDM
from mobgap.signal_based import RMS


def test_wb_windows_collect_one_row_with_original_identity():
    data = pd.DataFrame({"signal": [1, 2, 3, 4, 5, 6]}, index=range(100, 106))
    sequences = pd.DataFrame(
        {"start": [3, 0], "end": [6, 2], "orientation": [2, -1]},
        index=pd.Index(["later", "earlier"], name="wb_id"),
    )
    iterator = SdmoIterator()
    for (region, window), result in iterator.iterate(data, sequences):
        # Metadata stays in the source table, and correction remains caller-owned.
        correction = sequences.loc[region.id, "orientation"]
        result.parameters = pd.DataFrame({"value": [window.signal.sum() * correction]}, index=["ignored"])

    assert_frame_equal(
        iterator.results_.parameters,
        pd.DataFrame({"value": [30, -3]}, index=sequences.index),
    )
    assert_frame_equal(iterator.raw_results_[0].input.data, data.iloc[3:6])


def test_turn_windows_preserve_composite_ids_and_do_not_offset_outcomes():
    turns = pd.DataFrame(
        {"start": [4, 1, 3], "end": [6, 3, 5]},
        index=pd.MultiIndex.from_tuples([(8, 0), (2, 0), (8, 1)], names=["gs_id", "turn_id"]),
    )
    iterator = SdmoIterator()
    seen = []
    for (region, window), result in iterator.iterate(pd.DataFrame({"yaw": [0, 10, 20, 30, 40, 50]}), turns):
        seen.append(region.id)
        result.parameters = pd.DataFrame({"peak": [window.yaw.max()], "start": [0]})

    assert seen == [(8, 0), (2, 0), (8, 1)]
    assert_frame_equal(
        iterator.results_.parameters,
        pd.DataFrame({"peak": [50, 20, 40], "start": [0, 0, 0]}, index=turns.index),
    )


def test_unset_results_omit_only_their_own_sequence():
    sequences = pd.DataFrame({"start": [0, 1, 2], "end": [1, 2, 3]}, index=pd.Index([5, 2, 8], name="wb_id"))
    iterator = SdmoIterator()
    for (region, _window), result in iterator.iterate(pd.DataFrame({"signal": [10, 20, 30]}), sequences):
        if region.id != 2:
            result.parameters = pd.DataFrame({"value": [region.id]})
    assert_frame_equal(
        iterator.results_.parameters,
        pd.DataFrame({"value": [5, 8]}, index=pd.Index([5, 8], name="wb_id")),
    )


def test_reuse_for_empty_turn_table_clears_cached_wb_results():
    iterator = SdmoIterator()
    data = pd.DataFrame({"signal": [1]})
    for _, result in iterator.iterate(data, pd.DataFrame({"start": [0], "end": [1]})):
        result.parameters = pd.DataFrame({"value": [1]})
    assert not iterator.results_.parameters.empty

    empty_turns = pd.DataFrame(
        {"start": pd.Series(dtype=int), "end": pd.Series(dtype=int)},
        index=pd.MultiIndex.from_arrays([[], []], names=["gs_id", "turn_id"]),
    )
    list(iterator.iterate(data, empty_turns))
    assert_frame_equal(iterator.results_.parameters, pd.DataFrame(index=empty_turns.index))


def test_cloned_iterator_supports_explicit_correction_and_real_outcomes():
    data = pd.DataFrame({"acc_x": [1.0, 3.0, 5.0], "acc_y": [0.0] * 3, "acc_z": [0.0] * 3})
    correction = (Rotation.identity(), Rotation.identity())
    sequences = pd.DataFrame({"start": [0], "end": [3], "orientation": [correction]}, index=pd.Index([7], name="wb_id"))
    iterator = SdmoIterator().clone()
    for (region, window), result in iterator.iterate(data, sequences):
        prepared = ReorientationMethodDM.apply_correction(window, sequences.loc[region.id, "orientation"])
        result.parameters = RMS(acc_columns=["acc_is"], gyr_columns=[]).calculate(prepared).signal_based_parameters_
    # Body IS is sensor X; centered [1, 3, 5] has RMS sqrt(8/3).
    assert_frame_equal(
        iterator.results_.parameters[["rms_acc_is"]],
        pd.DataFrame({"rms_acc_is": [(8 / 3) ** 0.5]}, index=sequences.index),
    )
