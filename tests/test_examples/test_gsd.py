def test_gsd_iluz(snapshot):
    from examples.gsd._01_gsd_iluz import (
        categorized_intervals,
        long_trial_output,
        long_trial_output_modified,
        matches,
        short_trial_output,
    )

    snapshot.assert_match(long_trial_output.gs_list_, "long_trial_output")
    snapshot.assert_match(long_trial_output_modified.gs_list_, "long_trial_output_modified")
    snapshot.assert_match(short_trial_output.gs_list_, "short_trial_output")
    snapshot.assert_match(categorized_intervals.tp_intervals, "tp_intervals")
    snapshot.assert_match(categorized_intervals.fp_intervals, "fp_intervals")
    snapshot.assert_match(categorized_intervals.fn_intervals, "fn_intervals")
    snapshot.assert_match(matches, "matches")


def test_gsd_b(snapshot):
    from examples.gsd._02_gsd_pi import gsd_output

    snapshot.assert_match(gsd_output, "gsd_output")
