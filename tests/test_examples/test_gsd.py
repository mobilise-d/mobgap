def test_gsd_iluz(snapshot):
    from examples.gsd._01_gsd_iluz import long_trial_output, long_trial_output_modified, short_trial_output

    snapshot.assert_match(long_trial_output.gsd_list_, "long_trial_output")
    snapshot.assert_match(long_trial_output_modified.gsd_list_, "long_trial_output_modified")
    snapshot.assert_match(short_trial_output.gsd_list_, "short_trial_output")
