def test_cad_from_ic(snapshot):
    from examples.cadence._01_cad_from_ic import cad_from_ic, cad_from_ic_detector

    snapshot.assert_match(cad_from_ic.cadence_per_sec_)
    snapshot.assert_match(cad_from_ic_detector.cadence_per_sec_)


def test_cadence_evaluation(snapshot):
    from examples.cadence._02_cad_evaluation import (
        agg_results,
        avg_cadence_per_gs,
        cad_errors,
        cadence_result,
        combined_cad,
        combined_cad_with_errors,
    )

    snapshot.assert_match(cadence_result, "cadence_result")
    snapshot.assert_match(avg_cadence_per_gs, "avg_cadence_per_gs")

    # flatten multiindex columns as they are not supported by snapshot
    combined_cad.columns = ["_".join(pair) for pair in combined_cad.columns]
    snapshot.assert_match(combined_cad.reset_index(), "combined_cad")

    # flatten multiindex columns as they are not supported by snapshot
    cad_errors.columns = ["_".join(pair) for pair in cad_errors.columns]
    snapshot.assert_match(cad_errors.reset_index(), "cad_errors")

    # flatten multiindex columns as they are not supported by snapshot
    combined_cad_with_errors.columns = ["_".join(pair) for pair in combined_cad_with_errors.columns]
    snapshot.assert_match(combined_cad_with_errors.reset_index(), "combined_cad_with_errors")

    # check index of agg_results using snapshot
    snapshot.assert_match(agg_results.reset_index().drop(columns=["values"]), "agg_results_index")
    snapshot.assert_match(agg_results.reset_index(), "agg_results_data")
