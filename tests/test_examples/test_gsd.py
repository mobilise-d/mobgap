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
        cross_validate_results,
        matched_metrics_dict,
        matches,
        unmatched_metrics_dict,
    )

    snapshot.assert_match(categorized_intervals, "categorized_intervals")
    snapshot.assert_match(pd.DataFrame(matched_metrics_dict, index=[0]), "general_metrics_dict")
    snapshot.assert_match(pd.DataFrame(unmatched_metrics_dict, index=[0]), "mobilised_metrics_dict")
    snapshot.assert_match(matches, "matches", check_dtype=False)
    snapshot.assert_match(
        cross_validate_results[["test_single_precision", "test_single_accuracy"]], "cross_validate_results"
    )


def test_gsd_dmo_evaluation_on_wb_level(snapshot):
    from examples.gsd._03_dmo_evaluation_on_wb_level import agg_results, gs_errors, gs_matches, gs_tp_fp_fn

    snapshot.assert_match(gs_tp_fp_fn, "gs_tp_fp_fn")

    # flatten multiindex columns as they are not supported by snapshot
    gs_matches.columns = ["_".join(pair) for pair in gs_matches.columns]
    snapshot.assert_match(gs_matches, "gs_matches")

    snapshot.assert_match(gs_errors, "gs_errors")

    # check index of agg_results using snapshot
    snapshot.assert_match(agg_results.reset_index().drop(columns=[0]), "agg_results", check_dtype=False)
    # use workaround for values of agg_results because tuples in cells can't be parsed by json reader
    expected_agg_results = [
        12.25,
        4.27,
        1.0,
        0.82,
        (-0.08, 0.44),
        0.36,
        (0.06, 0.33),
        (-0.18, 0.48),
        0.47,
        (0.05, 0.36),
    ]
    _assert_float_tuples_list_almost_equal(agg_results.to_numpy(), expected_agg_results)


def _assert_float_tuples_list_almost_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_val, expected_val in zip(actual, expected):
        if isinstance(actual_val, tuple):
            assert type(actual_val) == type(expected_val)
            assert len(actual_val) == len(expected_val)
            for a, e in zip(actual_val, expected_val):
                assert round(a, 2) == e
        else:
            assert round(actual_val, 2) == expected_val
