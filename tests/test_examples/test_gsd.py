def test_gsd_iluz(snapshot):
    from examples.gsd._01_gsd_iluz import (
        categorized_intervals,
        long_trial_output,
        long_trial_output_modified,
        matches,
        short_trial_output,
    )

    snapshot.assert_match(long_trial_output.gsd_list_, "long_trial_output")
    snapshot.assert_match(long_trial_output_modified.gsd_list_, "long_trial_output_modified")
    snapshot.assert_match(short_trial_output.gsd_list_, "short_trial_output")
    snapshot.assert_match(categorized_intervals.tp_intervals, "tp_intervals")
    snapshot.assert_match(categorized_intervals.fp_intervals, "fp_intervals")
    snapshot.assert_match(categorized_intervals.fn_intervals, "fn_intervals")
    snapshot.assert_match(matches, "matches")
