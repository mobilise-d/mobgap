def test_signal_based_algorithms(snapshot):
    from examples.signal_based._01_signal_based_algorithms import (
        log_dimensionless_jerk,
        rms_jerk_ratio,
        frequency_amplitude,
        harmonic_ratio,
        harmonic_ratio_fourier,
        regularity_symmetry,
        rms,
        sample_entropy,
        sd_range,
        stride_level,
        turn,
    )

    snapshot.assert_match(turn.signal_based_parameters_.to_frame().T, "turn_sdmo")
    snapshot.assert_match(stride_level.signal_based_parameters_.to_frame().T, "stride_level_sdmo")
    snapshot.assert_match(rms.signal_based_parameters_.to_frame().T, "rms_sdmo")
    snapshot.assert_match(regularity_symmetry.signal_based_parameters_.to_frame().T, "regularity_symmetry_sdmo")
    snapshot.assert_match(frequency_amplitude.signal_based_parameters_.to_frame().T, "frequency_amplitude_sdmo")
    snapshot.assert_match(sample_entropy.signal_based_parameters_.to_frame().T, "sample_entropy_sdmo")
    snapshot.assert_match(harmonic_ratio.signal_based_parameters_.to_frame().T, "harmonic_ratio_sdmo")
    snapshot.assert_match(harmonic_ratio_fourier.signal_based_parameters_.to_frame().T, "harmonic_ratio_fourier_sdmo")
    snapshot.assert_match(sd_range.signal_based_parameters_.to_frame().T, "sd_range_sdmo")
    snapshot.assert_match(log_dimensionless_jerk.signal_based_parameters_.to_frame().T, "ldlj_sdmo")
    snapshot.assert_match(rms_jerk_ratio.signal_based_parameters_.to_frame().T, "rms_jerk_ratio_sdmo")


def test_signal_based_mobilised_pipeline():
    from pandas.testing import assert_frame_equal, assert_index_equal

    from examples.signal_based._02_signal_based_mobilised_pipeline import (
        aggregated_sdmos,
        per_turn_sdmos,
        per_wb_sdmos,
        pipeline,
    )

    assert_index_equal(per_wb_sdmos.index, pipeline.per_wb_parameters_.index)
    assert_index_equal(per_turn_sdmos.index, pipeline.raw_turn_list_.index)
    assert_frame_equal(per_turn_sdmos[pipeline.raw_turn_list_.columns], pipeline.raw_turn_list_)
    # This trial has detected turns extending outside accepted WBs. Keep them as complete turn windows.
    assert per_turn_sdmos.loc[(1, 0), "start"] < pipeline.per_wb_parameters_.loc[1, "start"]
    assert len(per_wb_sdmos) == 7
    assert len(per_turn_sdmos) == 10
    assert (per_wb_sdmos["rms_acc_is"] > 0).all()
    assert (per_turn_sdmos["turn_peak_ang_vel"] > 0).all()
    assert not aggregated_sdmos.empty
