from collections.abc import Iterable

import numpy as np
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
    snapshot.assert_match(matches.reset_index(), "matches", check_dtype=False)
    snapshot.assert_match(
        cross_validate_results[["test_single_precision", "test_single_accuracy"]], "cross_validate_results"
    )


def test_gsd_dmo_evaluation_on_wb_level(snapshot):
    from examples.gsd._03_dmo_evaluation_on_wb_level import agg_results, gs_errors, gs_errors_adapted, gs_matches, gs_matches_with_errors, gs_tp_fp_fn, default_agg_results

    snapshot.assert_match(gs_tp_fp_fn, "gs_tp_fp_fn")

    # flatten multiindex columns as they are not supported by snapshot
    gs_matches.columns = ["_".join(pair) for pair in gs_matches.columns]
    snapshot.assert_match(gs_matches.reset_index(), "gs_matches")

    # flatten multiindex columns as they are not supported by snapshot
    gs_errors.columns = ["_".join(pair) for pair in gs_errors.columns]
    snapshot.assert_match(gs_errors, "gs_errors")

    # flatten multiindex columns as they are not supported by snapshot
    gs_errors_adapted.columns = ["_".join(pair) for pair in gs_errors_adapted.columns]
    snapshot.assert_match(gs_errors_adapted, "gs_errors_adapted")

    # flatten multiindex columns as they are not supported by snapshot
    gs_matches_with_errors.columns = ["_".join(pair) for pair in gs_matches_with_errors.columns]
    snapshot.assert_match(gs_matches_with_errors, "gs_matches_with_errors")

    # check index of agg_results using snapshot
    snapshot.assert_match(agg_results.reset_index().drop(columns=[0]), "agg_results_index")
    # flatten values of agg_results because tuples can't be handled by snapshot utility
    snapshot.assert_match(np.array(_flatten_float_tuple_results(agg_results)), "agg_results_data")

    # check index of agg_results using snapshot
    snapshot.assert_match(default_agg_results.reset_index().drop(columns=[0]), "default_agg_results_index")
    # flatten values of agg_results because tuples can't be handled by snapshot utility
    snapshot.assert_match(np.array(list(_flatten_float_tuple_results(default_agg_results))), "default_agg_results_data")


def _flatten_float_tuple_results(result_list):
    for item in result_list:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from _flatten_float_tuple_results(item)
        else:
            yield item
