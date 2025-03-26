import pandas as pd


def test_mccamley(snapshot):
    from examples.laterality._01_lrc_mccamley import detected_ics

    snapshot.assert_match(detected_ics)


def test_ullrich(snapshot):
    from examples.laterality._02_lrc_ullrich import detected_ics, predictions

    snapshot.assert_match(detected_ics)
    snapshot.assert_match(predictions)


def _merge_cv_nested_results(results_df):
    raw_results = results_df[["test__data_labels", "test__single__raw__predictions"]]
    merged_results = {}
    for i, row in raw_results.iterrows():
        merged_results[i] = row["test__single__raw__predictions"]

    return pd.concat(merged_results)


def test_lrc_evaluation(snapshot):
    from examples.laterality._99_lrc_evaluation import evaluation_results_pre_trained, evaluation_results_with_opti

    evaluation_results_with_opti = _merge_cv_nested_results(evaluation_results_with_opti)
    evaluation_results_pre_trained = _merge_cv_nested_results(evaluation_results_pre_trained)

    snapshot.assert_match(evaluation_results_with_opti, "evaluation_results_with_opti")
    snapshot.assert_match(evaluation_results_pre_trained, "evaluation_results_pre_trained")
