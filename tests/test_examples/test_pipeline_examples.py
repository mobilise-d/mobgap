import pandas as pd
from pandas.testing import assert_frame_equal


def test_gs_iterator(snapshot):
    from examples.pipeline._01_gs_iterator import custom_iterator, iterator, long_trial_gs

    assert len(custom_iterator.raw_results_) == len(long_trial_gs)

    wb_ids = tuple(v.input[0].id for v in custom_iterator.raw_results_)
    assert wb_ids == tuple(long_trial_gs.index)

    snapshot.assert_match(custom_iterator.results_.n_samples.to_frame(), "n_samples")
    filtered_data = pd.concat(custom_iterator.results_.filtered_data)
    filtered_data.index = filtered_data.index.round("ms")
    snapshot.assert_match(filtered_data, "filtered_data")

    snapshot.assert_match(iterator.results_.ic_list, "initial_contacts")
    snapshot.assert_match(iterator.results_.cadence_per_sec, "cadence")


def test_full_mobilise_pipeline(snapshot):
    from examples.pipeline._02_step_by_step_mobilised_pipeline import (
        agg_results,
        final_strides,
        per_wb_params,
        pipeline,
    )

    assert_frame_equal(pipeline.per_stride_parameters_, final_strides)
    assert_frame_equal(pipeline.per_wb_parameters_.drop(columns="rule_obj"), per_wb_params.drop(columns="rule_obj"))
    assert_frame_equal(pipeline.aggregated_parameters_, agg_results)

    snapshot.assert_match(pipeline.per_stride_parameters_, "per_stride_parameters")
    snapshot.assert_match(pipeline.per_wb_parameters_.drop(columns="rule_obj"), "per_wb_parameters")
    snapshot.assert_match(pipeline.aggregated_parameters_, "aggregated_parameters")


def test_preconfigured_mobilise_pipeline(snapshot):
    from examples.pipeline._03_preconfigured_mobilised_pipelines import (
        aggregated_paras,
        per_wb_paras,
        pipeline_ha,
        pipeline_ms,
    )

    snapshot.assert_match(per_wb_paras.drop(columns="rule_obj"), "meta_pipeline_full_per_wb_parameters")
    snapshot.assert_match(aggregated_paras, "meta_pipeline_full_aggregated_parameters")

    snapshot.assert_match(pipeline_ms.per_wb_parameters_.drop(columns="rule_obj"), "ms_per_wb_parameters")
    snapshot.assert_match(pipeline_ha.per_wb_parameters_.drop(columns="rule_obj"), "ha_per_wb_parameters")


def test_dmo_evaluation_on_wb_level(snapshot):
    from examples.pipeline._03_dmo_evaluation_on_wb_level import (
        agg_results,
        combined_dmos,
        custom_wb_errors,
        daily_matches,
        default_agg_results,
        wb_errors,
        wb_matches,
        wb_matches_with_errors,
        wb_tp_fp_fn,
    )

    # flatten multiindex columns as they are not supported by snapshot
    combined_dmos.columns = ["_".join(pair) for pair in combined_dmos.columns]
    snapshot.assert_match(combined_dmos.reset_index(), "combined_dmos")

    # flatten multiindex columns as they are not supported by snapshot
    daily_matches.columns = ["_".join(pair) for pair in daily_matches.columns]
    snapshot.assert_match(daily_matches.reset_index(), "daily_matches")

    # flatten multiindex columns as they are not supported by snapshot
    wb_matches.columns = ["_".join(pair) for pair in wb_matches.columns]
    snapshot.assert_match(wb_matches.reset_index(), "det_ref_daily")
    snapshot.assert_match(wb_matches.reset_index(), "wb_matches")

    snapshot.assert_match(wb_tp_fp_fn, "wb_tp_fp_fn")

    # flatten multiindex columns as they are not supported by snapshot
    wb_errors.columns = ["_".join(pair) for pair in wb_errors.columns]
    snapshot.assert_match(wb_errors, "wb_errors")

    # flatten multiindex columns as they are not supported by snapshot
    custom_wb_errors.columns = ["_".join(pair) for pair in custom_wb_errors.columns]
    snapshot.assert_match(custom_wb_errors, "custom_wb_errors")

    # flatten multiindex columns as they are not supported by snapshot
    wb_matches_with_errors.columns = ["_".join(pair) for pair in wb_matches_with_errors.columns]
    snapshot.assert_match(wb_matches_with_errors, "wb_matches_with_errors")

    # check index of agg_results using snapshot
    snapshot.assert_match(agg_results.reset_index().drop(columns=["values"]), "agg_results_index")
    # flatten values of agg_results because tuples can't be handled by snapshot utility
    agg_results = agg_results.map(_tuple_to_str)
    snapshot.assert_match(agg_results.map(_tuple_to_str), "agg_results_data")

    # check index of agg_results using snapshot
    snapshot.assert_match(default_agg_results.reset_index().drop(columns=["values"]), "default_agg_results_index")
    # flatten values of agg_results because tuples can't be handled by snapshot utility
    snapshot.assert_match(default_agg_results.map(_tuple_to_str), "default_agg_results_data")


def _tuple_to_str(t):
    return str(t) if isinstance(t, tuple) else t
