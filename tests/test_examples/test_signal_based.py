def test_signal_based_algorithms(snapshot):
    from examples.signal_based._01_signal_based_algorithms import (
        turn,
        stride_level,
        rms,
        regularity_symmetry,
        frequency_amplitude,
        sample_entropy,
        harmonic_ratio,
        sd_range,
        jerk,
        angular_acceleration,
    )

    snapshot.assert_match(turn.signal_based_parameters_, "turn_sdmo")
    snapshot.assert_match(stride_level.signal_based_parameters_, "stride_level_sdmo")
    snapshot.assert_match(rms.signal_based_parameters_, "rms_sdmo")
    snapshot.assert_match(regularity_symmetry.signal_based_parameters_, "regularity_symmetry_sdmo")
    snapshot.assert_match(frequency_amplitude.signal_based_parameters_, "frequency_amplitude_sdmo")
    snapshot.assert_match(sample_entropy.signal_based_parameters_, "sample_entropy_sdmo")
    snapshot.assert_match(harmonic_ratio.signal_based_parameters_, "harmonic_ratio_sdmo")
    snapshot.assert_match(sd_range.signal_based_parameters_, "sd_range_sdmo")
    snapshot.assert_match(jerk.signal_based_parameters_, "jerk_sdmo")
    snapshot.assert_match(angular_acceleration.signal_based_parameters_, "angular_acceleration_sdmo")


def test_signal_based_mobilised_pipeline(snapshot):
    from examples.signal_based._02_signal_based_mobilised_pipeline import sdmo_only_available, sdmo_full_output

    snapshot.assert_match(sdmo_only_available.signal_based_parameters_, "only_available_sdmos")
    snapshot.assert_match(sdmo_full_output.signal_based_parameters_, "full_mobilised_sdmos")
