import pandas as pd


def test_gsd_iluz(snapshot):
    from examples.gait_sequences._01_gsd_iluz import (
        long_trial_output,
        long_trial_output_modified,
        short_trial_output,
    )

    snapshot.assert_match(long_trial_output.gs_list_, "long_trial_output")
    snapshot.assert_match(long_trial_output_modified.gs_list_, "long_trial_output_modified")
    snapshot.assert_match(short_trial_output.gs_list_, "short_trial_output")


def test_gsd_adaptive_ionescu(snapshot):
    from examples.gait_sequences._02_gsd_ionescu import (
        long_trial_output_adaptive,
        long_trial_output_normal,
        short_trial_output_adaptive,
        short_trial_output_normal,
    )

    snapshot.assert_match(long_trial_output_adaptive.gs_list_, "long_trial_output_adaptive")
    snapshot.assert_match(short_trial_output_adaptive.gs_list_, "short_trial_output_adaptive")
    snapshot.assert_match(long_trial_output_normal.gs_list_, "long_trial_output_normal")
    snapshot.assert_match(short_trial_output_normal.gs_list_, "short_trial_output_normal")


def test_gsd_evaluation(snapshot):
    from examples.gait_sequences._03_gsd_evaluation import (
        categorized_intervals,
        cross_validate_results,
        matched_metrics_dict,
        matches,
        unmatched_metrics_dict,
    )

    snapshot.assert_match(categorized_intervals, "categorized_intervals")
    snapshot.assert_match(pd.DataFrame(matched_metrics_dict, index=[0]), "general_metrics_dict")
    snapshot.assert_match(pd.DataFrame(unmatched_metrics_dict, index=[0]), "mobilised_metrics_dict")
    snapshot.assert_match(matches.reset_index(), "matches", check_dtype=False)
    snapshot.assert_match(
        cross_validate_results[["test__single__precision", "test__single__accuracy"]], "cross_validate_results"
    )


def test_gsd_challenges(snapshot):
    from examples.gait_sequences._04_gsd_evaluation_challenges import (
        agg_results,
        agg_results_cv,
        single_results,
        single_results_cv,
    )

    snapshot.assert_match(agg_results_cv.drop(columns="runtime_s"), "agg_results_cv")
    snapshot.assert_match(
        single_results_cv.reset_index().infer_objects().drop(columns="runtime_s"), "exploded_results_cv"
    )
    snapshot.assert_match(agg_results.drop(columns="runtime_s"), "agg_results")
    snapshot.assert_match(single_results.reset_index().drop(columns="runtime_s"), "exploded_results")
