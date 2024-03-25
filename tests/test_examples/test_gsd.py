import pandas as pd


def test_gsd_iluz(snapshot):
    from examples.gsd._01_gsd_iluz import (
        long_trial_output,
        long_trial_output_modified,
        short_trial_output,
    )

    snapshot.assert_match(long_trial_output.gs_list_, "long_trial_output")
    snapshot.assert_match(long_trial_output_modified.gs_list_, "long_trial_output_modified")
    snapshot.assert_match(short_trial_output.gs_list_, "short_trial_output")


def test_gsd_evaluation(snapshot):
    from examples.gsd._02_gsd_evaluation import (
        categorized_intervals,
        matched_metrics_dict,
        matches,
        unmatched_metrics_dict,
    )

    snapshot.assert_match(categorized_intervals, "categorized_intervals")
    snapshot.assert_match(pd.DataFrame(matched_metrics_dict, index=[0]), "general_metrics_dict")
    snapshot.assert_match(pd.DataFrame(unmatched_metrics_dict, index=[0]), "mobilised_metrics_dict")
    snapshot.assert_match(matches, "matches")
