import pandas as pd


def test_mccamley(snapshot):
    from examples.lrc._01_mccamley import detected_ics

    snapshot.assert_match(detected_ics)


def _merge_cv_nested_results(results_df):
    raw_results = results_df[["test_data_labels", "test_single_raw_results"]]
    merged_results = {}
    for i, row in raw_results.iterrows():
        merged_results[i] = pd.concat(row["test_single_raw_results"], keys=row["test_data_labels"])

    return pd.concat(merged_results)


def test_lrc_evaluation(snapshot):
    from examples.lrc._99_lrc_evaluation import evaluation_results_pre_trained, evaluation_results_with_opti

    evaluation_results_with_opti = _merge_cv_nested_results(evaluation_results_with_opti)
    evaluation_results_pre_trained = _merge_cv_nested_results(evaluation_results_pre_trained)

    snapshot.assert_match(evaluation_results_with_opti, "evaluation_results_with_opti")
    snapshot.assert_match(evaluation_results_pre_trained, "evaluation_results_pre_trained")
