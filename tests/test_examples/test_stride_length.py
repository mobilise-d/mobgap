def test_zjilstra(snapshot):
    from examples.stride_length._01_sl_zijlstra import sl_zijlstra, sl_zijlstra_reoriented

    snapshot.assert_match(sl_zijlstra.stride_length_per_sec_, "sl_zijlstra")
    snapshot.assert_match(sl_zijlstra_reoriented.stride_length_per_sec_, "sl_zijlstra_reoriented")


def test_stride_length_evaluation(snapshot):
    from examples.stride_length._02_sl_evaluation import (
        agg_results,
        avg_stride_length_per_gs,
        combined_sl,
        combined_sl_with_errors,
        sl_errors,
        stride_length_result,
    )

    snapshot.assert_match(stride_length_result, "stride_length_result")
    snapshot.assert_match(avg_stride_length_per_gs, "avg_stride_length_per_gs")

    # flatten multiindex columns as they are not supported by snapshot
    combined_sl.columns = ["_".join(pair) for pair in combined_sl.columns]
    snapshot.assert_match(combined_sl.reset_index(), "combined_sl")

    # flatten multiindex columns as they are not supported by snapshot
    sl_errors.columns = ["_".join(pair) for pair in sl_errors.columns]
    snapshot.assert_match(sl_errors.reset_index(), "sl_errors")

    # flatten multiindex columns as they are not supported by snapshot
    combined_sl_with_errors.columns = ["_".join(pair) for pair in combined_sl_with_errors.columns]
    snapshot.assert_match(combined_sl_with_errors.reset_index(), "combined_sl_with_errors")

    # check index of agg_results using snapshot
    snapshot.assert_match(agg_results.reset_index().drop(columns=["values"]), "agg_results_index")
    snapshot.assert_match(agg_results.reset_index(), "agg_results_data")
